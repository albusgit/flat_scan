package com.albus.flatscan.sync

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.work.ListenableWorker
import androidx.work.NetworkType
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.testing.TestListenableWorkerBuilder
import androidx.work.testing.WorkManagerTestInitHelper
import androidx.work.workDataOf
import com.albus.flatscan.PageMeta
import com.albus.flatscan.RenderParams
import com.albus.flatscan.data.Library
import com.albus.flatscan.data.SyncStatus
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The worker around the uploaders: an upload cut by the network is retried (status "pending" with the
 * reason) and then completes once; wrong credentials fail at once with the reason; the queue honours the
 * Wi-Fi and charging options. The network itself is the fake [CuttingUploader] here; the real-server
 * version of the cut upload is in SyncIntegrationTest.
 */
@RunWith(AndroidJUnit4::class)
class SyncWorkerTest {
    private val ctx: Context = ApplicationProvider.getApplicationContext()
    private val lib by lazy { Library.get(ctx) }

    /** Fails the first [cuts] attempts part-way (after reserving its name), then stores the file once. */
    private class CuttingUploader(var cuts: Int, val error: SyncException? = null) : Uploader {
        val stored = mutableListOf<String>()
        override suspend fun upload(job: UploadJob, ledger: SyncLedger): String {
            error?.let { throw it }
            val name = ledger.get(job.docId) ?: "${job.fileName}.pdf".also { ledger.set(job.docId, it) }
            if (cuts > 0) {
                cuts--
                throw Http.classify(java.net.SocketException("Connection reset (offline)"))
            }
            if (name !in stored) stored += name
            ledger.set(job.docId, null)
            return name
        }
        override suspend fun test() = "ok"
    }

    private suspend fun document(): String {
        val d = lib.createDocument("Worker test")
        val id = lib.store.newId()
        lib.store.saveMeta(PageMeta(id, FloatArray(8), null, 0, false, 0f, RenderParams(), FloatArray(0), 0))
        File(lib.store.dir(id), "out.jpg").writeBytes(ByteArray(100))
        lib.setPages(d.id, listOf(id))
        // the export the worker would write (PdfDocument is not available on the JVM)
        val doc = lib.document(d.id)!!
        File(ctx.noBackupFilesDir, "sync").apply { mkdirs() }.resolve("${doc.id}_${doc.modifiedAt}.pdf").writeBytes(PaperlessUploader.tinyPdf("x"))
        return d.id
    }

    private fun worker(doc: String, attempt: Int) = TestListenableWorkerBuilder<SyncWorker>(ctx)
        .setInputData(workDataOf(SyncWorker.KEY_DOC to doc)).setRunAttemptCount(attempt).build()

    @Before
    fun setUp() {
        SyncConfig(ctx).apply { kind = SyncKind.WebDav; url = "https://example.invalid/dav/"; user = "u" }
    }

    @After
    fun tearDown() {
        SyncWorker.uploaderFactory = null
    }

    @Test
    fun cutUploadIsRetriedThenStoredOnce() = runBlocking {
        val doc = document()
        val up = CuttingUploader(cuts = 1)
        SyncWorker.uploaderFactory = { up }
        assertEquals(ListenableWorker.Result.retry(), worker(doc, 0).doWork())
        lib.document(doc)!!.let {
            assertEquals(SyncStatus.Pending, it.syncStatus)
            assertTrue(it.syncError!!.startsWith("Will retry"))
            assertTrue(it.syncRemote!!.endsWith("_Worker test.pdf"))      // the template's name, reserved across the retry
        }
        assertEquals(ListenableWorker.Result.success(), worker(doc, 1).doWork())
        val d = lib.document(doc)!!
        assertEquals(SyncStatus.Uploaded, d.syncStatus)
        assertEquals(null, d.syncRemote)
        assertEquals(d.modifiedAt, d.uploadedModified)
        assertEquals(1, up.stored.size)
        assertTrue(up.stored[0].endsWith("_Worker test.pdf"))
    }

    @Test
    fun wrongCredentialsFailAtOnceWithTheReason() = runBlocking {
        val doc = document()
        SyncWorker.uploaderFactory = { CuttingUploader(0, SyncException(SyncException.Kind.Auth, "Uploading: the server refused the credentials (HTTP 401)")) }
        assertEquals(ListenableWorker.Result.failure(), worker(doc, 0).doWork())
        val d = lib.document(doc)!!
        assertEquals(SyncStatus.Failed, d.syncStatus)
        assertTrue(d.syncError!!.contains("401"))
    }

    @Test
    fun givesUpAfterTheLastAttempt() = runBlocking {
        val doc = document()
        SyncWorker.uploaderFactory = { CuttingUploader(cuts = 100) }
        assertEquals(ListenableWorker.Result.failure(), worker(doc, SyncWorker.MAX_ATTEMPTS - 1).doWork())
        assertEquals(SyncStatus.Failed, lib.document(doc)!!.syncStatus)
    }

    @Test
    fun queueHonoursWifiAndChargingOptions() = runBlocking {
        WorkManagerTestInitHelper.initializeTestWorkManager(ctx)
        SyncConfig(ctx).apply { wifiOnly = true; chargingOnly = true }
        val doc = document()
        SyncWorker.enqueue(ctx, doc)
        SyncWorker.enqueue(ctx, doc)   // queued once
        val infos = WorkManager.getInstance(ctx).getWorkInfosForUniqueWork(SyncWorker.workName(doc)).get()
        assertEquals(1, infos.size)
        assertEquals(WorkInfo.State.ENQUEUED, infos[0].state)
        assertEquals(NetworkType.UNMETERED, infos[0].constraints.requiredNetworkType)
        assertTrue(infos[0].constraints.requiresCharging())
        assertEquals(SyncStatus.Pending, lib.document(doc)!!.syncStatus)
    }
}
