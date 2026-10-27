package com.albus.flatscan.sync

import okhttp3.Credentials
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.IOException

/**
 * A minimal WebDAV client (RFC 4918) for uploads into one folder: Nextcloud / ownCloud
 * (`https://host/remote.php/dav/files/USER/FOLDER/`), Synology, Apache mod_dav, rclone... [folderUrl] is the
 * collection the files go into. Basic authentication.
 */
class WebDavClient(private val http: OkHttpClient, folderUrl: String, private val user: String, private val password: String) {
    val folder: HttpUrl = folderUrl.let { if (it.endsWith("/")) it else "$it/" }.toHttpUrl()

    private fun url(name: String): HttpUrl = folder.newBuilder().addPathSegment(name).build()

    private fun req(u: HttpUrl) = Request.Builder().url(u).header("Authorization", Credentials.basic(user, password, Charsets.UTF_8))

    private inline fun <T> call(r: Request, what: String, f: (okhttp3.Response) -> T): T = try {
        http.newCall(r).execute().use(f)
    } catch (e: SyncException) {
        throw e
    } catch (e: IOException) {
        throw Http.classify(e)
    }

    /** Size of [name] in the folder, or null when there is no such file. */
    fun size(name: String): Long? = call(
        req(url(name)).method("PROPFIND", PROPFIND_BODY.toRequestBody(XML)).header("Depth", "0").build(), "Checking $name",
    ) { r ->
        when {
            r.code == 404 -> null
            r.code == 207 || r.isSuccessful -> Regex("<(?:\\w+:)?getcontentlength>\\s*(\\d+)\\s*<").find(r.body?.string().orEmpty())?.groupValues?.get(1)?.toLongOrNull() ?: 0L
            else -> throw Http.status(r.code, "Checking $name")
        }
    }

    fun exists(name: String): Boolean = size(name) != null

    /** The folder exists (PROPFIND on it); 404 -> false. */
    fun folderExists(): Boolean = call(req(folder).method("PROPFIND", PROPFIND_BODY.toRequestBody(XML)).header("Depth", "0").build(), "Opening the folder") { r ->
        when {
            r.code == 404 -> false
            r.code == 207 || r.isSuccessful -> true
            else -> throw Http.status(r.code, "Opening the folder")
        }
    }

    /** Creates the folder (and missing parents below the server root it is given). */
    fun makeFolder() {
        val segs = folder.pathSegments.filter { it.isNotEmpty() }
        for (i in 1..segs.size) {
            val u = folder.newBuilder().encodedPath("/").apply { segs.take(i).forEach { addPathSegment(it) } }.addPathSegment("").build()
            val code = call(req(u).method("MKCOL", null).build(), "Creating the folder") { it.code }
            if (code !in listOf(201, 405, 301, 302)) {
                // 405: exists; 401/403 on a parent the user may not list is fine as long as the last one works
                if (i == segs.size) throw Http.status(code, "Creating the folder")
            }
        }
    }

    /**
     * Uploads [file] as [name]. [mustBeNew]: `If-None-Match: *`, so a file someone else created meanwhile is
     * not overwritten (412 -> false). Returns true when stored.
     */
    fun put(name: String, file: File, mustBeNew: Boolean, type: String = "application/pdf"): Boolean =
        putBody(name, file.asRequestBody(type.toMediaType()), mustBeNew)

    fun putBytes(name: String, bytes: ByteArray, mustBeNew: Boolean, type: String = "text/plain"): Boolean =
        putBody(name, bytes.toRequestBody(type.toMediaType()), mustBeNew)

    private fun putBody(name: String, body: RequestBody, mustBeNew: Boolean): Boolean {
        val r = req(url(name)).put(body).apply { if (mustBeNew) header("If-None-Match", "*") }.build()
        return call(r, "Uploading $name") { resp ->
            when {
                resp.code == 412 -> false
                resp.isSuccessful -> true
                else -> throw Http.status(resp.code, "Uploading $name")
            }
        }
    }

    fun delete(name: String) {
        call(req(url(name)).delete().build(), "Deleting $name") { r ->
            if (!r.isSuccessful && r.code != 404) throw Http.status(r.code, "Deleting $name")
        }
    }

    fun get(name: String): ByteArray? = call(req(url(name)).get().build(), "Reading $name") { r ->
        when {
            r.code == 404 -> null
            r.isSuccessful -> r.body?.bytes()
            else -> throw Http.status(r.code, "Reading $name")
        }
    }

    /** Names of the files in the folder (PROPFIND depth 1). */
    fun list(): List<String> = call(req(folder).method("PROPFIND", PROPFIND_BODY.toRequestBody(XML)).header("Depth", "1").build(), "Listing the folder") { r ->
        if (r.code != 207 && !r.isSuccessful) throw Http.status(r.code, "Listing the folder")
        Regex("<(?:\\w+:)?href>([^<]+)</(?:\\w+:)?href>").findAll(r.body?.string().orEmpty())
            .map { java.net.URLDecoder.decode(it.groupValues[1].trimEnd('/').substringAfterLast('/'), "UTF-8") }
            .filter { it.isNotEmpty() && it != folder.pathSegments.lastOrNull { s -> s.isNotEmpty() } }
            .toList()
    }

    companion object {
        private val XML = "application/xml; charset=utf-8".toMediaType()
        private const val PROPFIND_BODY = """<?xml version="1.0" encoding="utf-8"?><d:propfind xmlns:d="DAV:"><d:prop><d:getcontentlength/><d:resourcetype/></d:prop></d:propfind>"""
    }
}
