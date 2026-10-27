package com.albus.flatscan.sync

import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * The paperless-ngx REST API, token authentication (`Authorization: Token ...`): uploads
 * (`POST /api/documents/post_document/` with title, created date, tags and correspondent), the consumption
 * task's state, lookups by checksum, tags and correspondents by name (created when missing), delete.
 */
class PaperlessClient(private val http: OkHttpClient, baseUrl: String, private val token: String) {
    val base: HttpUrl = baseUrl.let { if (it.endsWith("/")) it else "$it/" }.toHttpUrl()

    private fun api(path: String): HttpUrl.Builder = base.newBuilder().addPathSegments("api/$path")

    private fun req(u: HttpUrl) = Request.Builder().url(u).header("Authorization", "Token $token").header("Accept", "application/json; version=5")

    private inline fun <T> call(r: Request, what: String, f: (okhttp3.Response) -> T): T = try {
        http.newCall(r).execute().use(f)
    } catch (e: SyncException) {
        throw e
    } catch (e: IOException) {
        throw Http.classify(e)
    }

    private fun json(r: okhttp3.Response, what: String): String {
        if (!r.isSuccessful) throw Http.status(r.code, what)
        return r.body?.string().orEmpty()
    }

    /** Checks the address and the token (an authenticated request). */
    fun ping() {
        call(req(api("documents/").addQueryParameter("page_size", "1").build()).get().build(), "Connecting") { r ->
            json(r, "Connecting")
            if (r.header("Content-Type")?.contains("json") != true) throw SyncException(SyncException.Kind.Config, "Connecting: this address is not a paperless-ngx server")
        }
    }

    /** Uploads [file]; returns the consumption task's id. */
    fun upload(file: File, title: String, created: Long, tagIds: List<Int>, correspondent: Int?): String {
        val b = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("document", file.name, file.asRequestBody("application/pdf".toMediaType()))
            .addFormDataPart("title", title)
            .addFormDataPart("created", isoDate(created))
        tagIds.forEach { b.addFormDataPart("tags", it.toString()) }
        correspondent?.let { b.addFormDataPart("correspondent", it.toString()) }
        return call(req(api("documents/post_document/").build()).post(b.build()).build(), "Uploading") { r ->
            json(r, "Uploading").trim().trim('"')
        }
    }

    /** A consumption task: its status (PENDING, STARTED, SUCCESS, FAILURE...), the document it made, its message. */
    data class Task(val status: String, val document: Int?, val result: String?)

    fun task(id: String): Task? = call(req(api("tasks/").addQueryParameter("task_id", id).build()).get().build(), "Checking the upload") { r ->
        val a = JSONArray(json(r, "Checking the upload"))
        if (a.length() == 0) return@call null
        val t = a.getJSONObject(0)
        Task(t.optString("status"), t.optString("related_document").toIntOrNull(), t.optString("result").takeIf { it.isNotEmpty() && it != "null" })
    }

    /** The id of a document whose original has MD5 [md5], or null. */
    fun documentWithChecksum(md5: String): Int? = call(
        req(api("documents/").addQueryParameter("checksum__iexact", md5).addQueryParameter("fields", "id").build()).get().build(), "Looking up",
    ) { r ->
        JSONObject(json(r, "Looking up")).optJSONArray("results")?.optJSONObject(0)?.optInt("id")
    }

    fun deleteDocument(id: Int) {
        call(req(api("documents/$id/").build()).delete().build(), "Deleting") { r ->
            if (!r.isSuccessful && r.code != 404) throw Http.status(r.code, "Deleting")
        }
    }

    /** Ids of the tags called [names] (case-insensitive), created when missing. */
    fun tagIds(names: List<String>): List<Int> = names.distinct().map { idOf("tags/", it) }

    fun correspondentId(name: String): Int = idOf("correspondents/", name)

    private fun idOf(kind: String, name: String): Int {
        val found = call(req(api(kind).addQueryParameter("name__iexact", name).build()).get().build(), "Looking up \"$name\"") { r ->
            JSONObject(json(r, "Looking up \"$name\"")).optJSONArray("results")?.optJSONObject(0)?.optInt("id")
        }
        if (found != null) return found
        val body = JSONObject().put("name", name).put("matching_algorithm", 0).toString().toRequestBody("application/json".toMediaType())
        return call(req(api(kind).build()).post(body).build(), "Creating \"$name\"") { r -> JSONObject(json(r, "Creating \"$name\"")).getInt("id") }
    }

    companion object {
        fun isoDate(t: Long): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US).apply { timeZone = TimeZone.getDefault() }.format(Date(t))
    }
}
