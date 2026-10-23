package com.albus.flatscan.sync

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.Request
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

/**
 * Against real servers started by tools/sync_it/run.sh: wsgidav (WebDAV over HTTPS with a self-signed
 * certificate, basic auth) and paperless-ngx 2.18 in Docker. Skipped when FS_WEBDAV_URL / FS_PAPERLESS_URL
 * are not set. [FlakyProxy] takes the network away mid-upload.
 */
@RunWith(AndroidJUnit4::class)
class SyncIntegrationTest {
    private val davUrl = System.getenv("FS_WEBDAV_URL")
    private val davUser = System.getenv("FS_WEBDAV_USER") ?: ""
    private val davPass = System.getenv("FS_WEBDAV_PASS") ?: ""
    private val plUrl = System.getenv("FS_PAPERLESS_URL")
    private val plToken = System.getenv("FS_PAPERLESS_TOKEN") ?: ""

    private class MemLedger : SyncLedger {
        val m = mutableMapOf<String, String?>()
        override suspend fun get(doc: String) = m[doc]
        override suspend fun set(doc: String, value: String?) { m[doc] = value }
    }

    private fun bigPdf(text: String, kb: Int = 512): File = File.createTempFile("doc", ".pdf").apply {
        // a valid PDF followed by padding (a comment), large enough to be cut mid-upload
        writeBytes(PaperlessUploader.tinyPdf(text) + ("%" + "x".repeat(1023) + "\n").repeat(kb).toByteArray())
    }

    // ---- WebDAV

    /** The server's certificate fingerprint, pinned as the user would after the first-use dialog. */
    private fun davPins(): Map<String, String> {
        val host = davUrl!!.toHttpUrl().host
        try {
            Http.client(host, emptyMap()).newCall(Request.Builder().url(davUrl).build()).execute().close()
            fail("a self-signed certificate must not be trusted without a pin")
        } catch (e: Exception) {
            val u = Http.untrusted(e)
            assertNotNull("expected an untrusted-certificate error, got $e", u)
            return mapOf(host to u!!.sha256)
        }
        error("unreachable")
    }

    private fun dav(url: String = davUrl!!, pass: String = davPass, folder: String = ""): WebDavClient {
        val u = url.toHttpUrl()
        return WebDavClient(Http.client(u.host, davPins()), url + folder, davUser, pass)
    }

    private fun freshFolder(): String {
        val f = "it-${UUID.randomUUID().toString().take(8)}/"
        dav(folder = f).makeFolder()
        return f
    }

    @Test
    fun webdavTestConnectionUploadsAndDeletes() = runBlocking {
        assumeTrue(davUrl != null)
        val folder = freshFolder()
        val msg = WebDavUploader(dav(folder = folder)).test()
        assertTrue(msg, msg.startsWith("Uploaded and deleted"))
        assertEquals(emptyList<String>(), dav(folder = folder).list())
    }

    @Test
    fun webdavWrongPasswordFailsTheTestAsAuth() = runBlocking {
        assumeTrue(davUrl != null)
        try {
            WebDavUploader(dav(pass = "wrong")).test()
            fail("a wrong password must fail the connection test")
        } catch (e: SyncException) {
            assertEquals(SyncException.Kind.Auth, e.kind)
        }
    }

    @Test
    fun webdavNeverOverwritesAndAddsASuffix() = runBlocking {
        assumeTrue(davUrl != null)
        val folder = freshFolder()
        val c = dav(folder = folder)
        c.putBytes("2026-01-02_Bill.pdf", "someone else's".toByteArray(), mustBeNew = true)
        val f = bigPdf("bill", kb = 4)
        val name = WebDavUploader(c).upload(UploadJob("d1", "2026-01-02_Bill", f, "Bill", 0, emptyList()), MemLedger())
        assertEquals("2026-01-02_Bill (2).pdf", name)
        assertEquals("someone else's", String(c.get("2026-01-02_Bill.pdf")!!))
        assertTrue(c.get(name)!!.contentEquals(f.readBytes()))
    }

    private fun proxied(folder: String, proxy: FlakyProxy): WebDavClient {
        val u = davUrl!!.toHttpUrl()
        val via = u.newBuilder().port(proxy.localPort).build().toString()
        return WebDavClient(Http.client(u.host, davPins()), via + folder, davUser, davPass)
    }

    @Test
    fun webdavOfflineMidUploadThenOnlineUploadsExactlyOnce() = runBlocking {
        assumeTrue(davUrl != null)
        val folder = freshFolder()
        FlakyProxy(davUrl!!.toHttpUrl().host, davUrl.toHttpUrl().port).use { proxy ->
            val up = WebDavUploader(proxied(folder, proxy))
            val ledger = MemLedger()
            val f = bigPdf("offline test")
            val job = UploadJob("d2", "Offline", f, "Offline", 0, emptyList())
            proxy.mode = FlakyProxy.Mode.CutRequest(afterBytes = 200_000)   // the phone goes offline part-way
            try {
                up.upload(job, ledger)
                fail("the cut upload must fail")
            } catch (e: SyncException) {
                assertTrue("a network failure is retried: $e", e.retryable)
            }
            assertEquals(1, proxy.faults.get())
            assertEquals("webdav:Offline.pdf", ledger.m["d2"])            // the name stays reserved
            val name = up.upload(job, ledger)                             // back online: the retry
            assertEquals("Offline.pdf", name)
            assertEquals(listOf("Offline.pdf"), dav(folder = folder).list())
            assertTrue(dav(folder = folder).get(name)!!.contentEquals(f.readBytes()))
            assertEquals(null, ledger.m["d2"])
        }
    }

    @Test
    fun webdavAnswerLostThenRetryDoesNotUploadTwice() = runBlocking {
        assumeTrue(davUrl != null)
        val folder = freshFolder()
        FlakyProxy(davUrl!!.toHttpUrl().host, davUrl.toHttpUrl().port).use { proxy ->
            val ledger = MemLedger()
            val f = bigPdf("lost answer", kb = 64)
            // what upload() does: reserve the name, then PUT; the PUT arrives but its answer is lost
            ledger.m["d3"] = WebDavUploader.PREFIX + "Lost.pdf"
            proxy.mode = FlakyProxy.Mode.DropResponse
            val lost = runCatching { proxied(folder, proxy).put("Lost.pdf", f, mustBeNew = true) }
            assertTrue("the answer must be lost", lost.isFailure)
            assertEquals(1, proxy.faults.get())
            // the retry finds its own complete file and stops
            val name = WebDavUploader(proxied(folder, proxy)).upload(UploadJob("d3", "Lost", f, "Lost", 0, emptyList()), ledger)
            assertEquals("Lost.pdf", name)
            assertEquals(listOf("Lost.pdf"), dav(folder = folder).list())
            assertTrue(dav(folder = folder).get(name)!!.contentEquals(f.readBytes()))
        }
    }

    // ---- paperless-ngx

    private fun paperless(url: String = plUrl!!, token: String = plToken) = PaperlessClient(Http.client(url.toHttpUrl().host, emptyMap()), url, token)

    private fun countTitled(title: String): Int {
        val u = plUrl!!.toHttpUrl().newBuilder().addPathSegments("api/documents/").addQueryParameter("title__iexact", title).build()
        val r = Http.client(u.host, emptyMap()).newCall(Request.Builder().url(u).header("Authorization", "Token $plToken").build()).execute()
        return r.use { JSONObject(it.body!!.string()).getInt("count") }
    }

    /** Waits until paperless has no consumption task running (so counts are final). */
    private fun settle() {
        val u = plUrl!!.toHttpUrl().newBuilder().addPathSegments("api/tasks/").build()
        repeat(90) {
            val body = Http.client(u.host, emptyMap()).newCall(Request.Builder().url(u).header("Authorization", "Token $plToken").build()).execute().use { it.body!!.string() }
            if (!Regex("\"status\":\\s*\"(PENDING|STARTED)\"").containsMatchIn(body)) return
            Thread.sleep(1000)
        }
    }

    @Test
    fun paperlessTestConnectionUploadsAndDeletes() = runBlocking {
        assumeTrue(plUrl != null)
        val msg = PaperlessUploader(paperless()).test()
        assertTrue(msg, msg.startsWith("Uploaded and deleted"))
        settle()
        assertEquals(0, countTitled("FlatScan connection test"))
    }

    @Test
    fun paperlessWrongTokenFailsAsAuth() = runBlocking {
        assumeTrue(plUrl != null)
        try {
            PaperlessUploader(paperless(token = "0000")).test()
            fail("a wrong token must fail")
        } catch (e: SyncException) {
            assertEquals(SyncException.Kind.Auth, e.kind)
        }
    }

    @Test
    fun paperlessUploadSendsTitleDateTagsAndCorrespondent() = runBlocking {
        assumeTrue(plUrl != null)
        val title = "IT metadata ${UUID.randomUUID().toString().take(6)}"
        val f = bigPdf(title, kb = 2)
        val created = 1_767_225_600_000L   // 2026-01-01
        val where = PaperlessUploader(paperless(), extraTags = listOf("flatscan"), correspondent = "Electric Co", waitMs = 60_000)
            .upload(UploadJob("p1", "2026-01-01_x", f, title, created, listOf("bills")), MemLedger())
        assertTrue(where, where.startsWith("paperless document"))
        val id = where.substringAfterLast(' ').toInt()
        val u = plUrl!!.toHttpUrl().newBuilder().addPathSegments("api/documents/$id/").build()
        val doc = Http.client(u.host, emptyMap()).newCall(Request.Builder().url(u).header("Authorization", "Token $plToken").build()).execute().use { JSONObject(it.body!!.string()) }
        assertEquals(title, doc.getString("title"))
        assertTrue(doc.getString("created").startsWith("2026-01-01") || doc.getString("created_date") == "2026-01-01")
        assertEquals(2, doc.getJSONArray("tags").length())
        assertTrue(doc.optInt("correspondent", -1) > 0)
    }

    @Test
    fun paperlessOfflineMidUploadThenOnlineUploadsExactlyOnce() = runBlocking {
        assumeTrue(plUrl != null)
        val u = plUrl!!.toHttpUrl()
        FlakyProxy(u.host, u.port).use { proxy ->
            val via = u.newBuilder().port(proxy.localPort).build().toString()
            val up = PaperlessUploader(paperless(via), waitMs = 60_000)
            val title = "IT offline ${UUID.randomUUID().toString().take(6)}"
            val job = UploadJob("p2", title, bigPdf(title), title, 0, emptyList())
            val ledger = MemLedger()
            // the lookups (small requests) go through; the POST is cut part-way
            proxy.mode = FlakyProxy.Mode.CutRequest(afterBytes = 200_000)
            try {
                up.upload(job, ledger)
                fail("the cut upload must fail")
            } catch (e: SyncException) {
                assertTrue("retryable: $e", e.retryable)
            }
            assertEquals(1, proxy.faults.get())
            up.upload(job, ledger)
            settle()
            assertEquals(1, countTitled(title))
        }
    }

    @Test
    fun paperlessAnswerLostThenRetryKeepsOneDocument() = runBlocking {
        assumeTrue(plUrl != null)
        val u = plUrl!!.toHttpUrl()
        FlakyProxy(u.host, u.port).use { proxy ->
            val via = u.newBuilder().port(proxy.localPort).build().toString()
            val api = paperless(via)
            val title = "IT lost ${UUID.randomUUID().toString().take(6)}"
            val f = bigPdf(title, kb = 16)
            val ledger = MemLedger()
            // what upload() does before the POST, then the POST whose answer is lost
            ledger.m["p3"] = PaperlessUploader.PREFIX + PaperlessUploader.md5(f)
            proxy.mode = FlakyProxy.Mode.DropResponse
            runCatching { api.upload(f, title, 0, emptyList(), null) }
            assertEquals(1, proxy.faults.get())
            proxy.mode = FlakyProxy.Mode.Pass
            PaperlessUploader(api, waitMs = 60_000).upload(UploadJob("p3", title, f, title, 0, emptyList()), ledger)
            settle()
            assertEquals(1, countTitled(title))
        }
    }
}
