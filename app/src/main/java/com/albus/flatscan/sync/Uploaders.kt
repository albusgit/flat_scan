package com.albus.flatscan.sync

import com.albus.flatscan.FileNameTemplate
import java.io.File
import java.security.MessageDigest
import java.util.UUID

/** A finished document to upload: the PDF [file], its file name without extension, title, date and tags. */
data class UploadJob(val docId: String, val fileName: String, val file: File, val title: String, val created: Long, val tags: List<String>)

/**
 * Where an upload in progress stands, per document, kept across retries and process deaths (the
 * database's `sync_remote`): the remote name reserved before the first byte is sent, or the paperless
 * checksum and task. It is what makes a retry finish the same upload instead of making a second one.
 */
interface SyncLedger {
    suspend fun get(doc: String): String?
    suspend fun set(doc: String, value: String?)
}

/** One sync target. Every failure is a [SyncException] (retryable or not). */
interface Uploader {
    /** Uploads [job] exactly once, even when called again after a failure part-way; returns where it went. */
    suspend fun upload(job: UploadJob, ledger: SyncLedger): String

    /** A real authenticated upload and delete of a small file (not a ping); returns what happened. */
    suspend fun test(): String
}

/**
 * WebDAV: the file goes into the configured folder under the export file name; a name already taken
 * gets " (2)", " (3)"... (never overwritten). The chosen name is recorded before the upload; a retry uploads
 * to that same name again (replacing its own partial file) or, if the earlier attempt did arrive and only
 * the answer was lost, finds it complete and stops.
 */
class WebDavUploader(private val dav: WebDavClient) : Uploader {
    override suspend fun upload(job: UploadJob, ledger: SyncLedger): String {
        val reserved = ledger.get(job.docId)?.takeIf { it.startsWith(PREFIX) }?.removePrefix(PREFIX)
        if (reserved != null) {
            // an earlier attempt: already complete (lost answer), else upload it again over its own partial copy
            if (dav.size(reserved) != job.file.length()) dav.put(reserved, job.file, mustBeNew = false)
            ledger.set(job.docId, null)
            return reserved
        }
        if (!dav.folderExists()) throw SyncException(SyncException.Kind.Config, "The folder ${dav.folder} does not exist")
        repeat(5) {
            val name = FileNameTemplate.unique(job.fileName, "pdf") { dav.exists(it) }
            ledger.set(job.docId, PREFIX + name)
            if (dav.put(name, job.file, mustBeNew = true)) {
                ledger.set(job.docId, null)
                return name
            }
            // someone created that name in the meantime: pick another
            ledger.set(job.docId, null)
        }
        throw SyncException(SyncException.Kind.Transient, "Could not find a free file name for ${job.fileName}")
    }

    override suspend fun test(): String {
        if (!dav.folderExists()) throw SyncException(SyncException.Kind.Config, "The folder ${dav.folder} does not exist")
        val name = ".flatscan-test-${UUID.randomUUID().toString().take(8)}.txt"
        val body = "FlatScan connection test ${System.currentTimeMillis()}".toByteArray()
        if (!dav.putBytes(name, body, mustBeNew = true)) throw SyncException(SyncException.Kind.Config, "Uploading a test file was refused")
        val back = dav.get(name)
        dav.delete(name)
        if (back == null || !back.contentEquals(body)) throw SyncException(SyncException.Kind.Config, "The test file did not arrive intact")
        if (dav.exists(name)) throw SyncException(SyncException.Kind.Config, "The test file could not be deleted")
        return "Uploaded and deleted a test file in ${dav.folder}"
    }

    companion object {
        const val PREFIX = "webdav:"
    }
}

/**
 * paperless-ngx: `POST /api/documents/post_document/` with the title, created date, tags (by name, created
 * when missing) and correspondent. Exactly once: the file's MD5 is recorded before the upload; a retry
 * first asks paperless for a document with that checksum (and the recorded task), and only uploads again
 * when there is none. paperless itself also refuses a duplicate file.
 */
class PaperlessUploader(
    private val api: PaperlessClient,
    private val extraTags: List<String> = emptyList(),
    private val correspondent: String? = null,
    private val waitMs: Long = 20_000,
    private val sleep: suspend (Long) -> Unit = { kotlinx.coroutines.delay(it) },
) : Uploader {
    override suspend fun upload(job: UploadJob, ledger: SyncLedger): String {
        val md5 = md5(job.file)
        val st = ledger.get(job.docId)?.takeIf { it.startsWith("$PREFIX$md5") }
        if (st != null) {
            api.documentWithChecksum(md5)?.let { ledger.set(job.docId, null); return "paperless document $it" }
            val task = st.substringAfter(":task:", "").takeIf { it.isNotEmpty() }
            if (task != null) {
                val t = api.task(task)
                if (t != null && t.status != "FAILURE") return finish(job, ledger, task, t)
            }
        }
        ledger.set(job.docId, PREFIX + md5)
        api.documentWithChecksum(md5)?.let { ledger.set(job.docId, null); return "already in paperless (document $it)" }
        val tags = api.tagIds((job.tags + extraTags).filter { it.isNotBlank() })
        val corr = correspondent?.takeIf { it.isNotBlank() }?.let { api.correspondentId(it) }
        val task = api.upload(job.file.renamedFor(job.fileName), job.title, job.created, tags, corr)
        ledger.set(job.docId, "$PREFIX$md5:task:$task")
        return finish(job, ledger, task, api.task(task))
    }

    /** Waits a little for paperless to consume the upload, to report a refusal; accepted is enough otherwise. */
    private suspend fun finish(job: UploadJob, ledger: SyncLedger, task: String, first: PaperlessClient.Task?): String {
        var t = first
        var waited = 0L
        while ((t == null || t.status in PENDING) && waited < waitMs) {
            sleep(1000)
            waited += 1000
            t = api.task(task)
        }
        if (t != null && t.status == "FAILURE" && t.result?.contains("duplicate", ignoreCase = true) != true) {
            ledger.set(job.docId, null)
            throw SyncException(SyncException.Kind.Config, "paperless could not consume the file: ${t.result ?: "unknown error"}")
        }
        ledger.set(job.docId, null)
        return when {
            t?.document != null -> "paperless document ${t.document}"
            t?.status == "FAILURE" -> "already in paperless"
            else -> "accepted by paperless (task $task)"
        }
    }

    override suspend fun test(): String {
        api.ping()
        val f = File.createTempFile("flatscan-test", ".pdf")
        try {
            f.writeBytes(tinyPdf("FlatScan connection test ${UUID.randomUUID()}"))
            val task = api.upload(f, "FlatScan connection test", System.currentTimeMillis(), emptyList(), null)
            var t: PaperlessClient.Task? = null
            var waited = 0L
            while (waited < 90_000) {
                t = api.task(task)
                if (t != null && t.status !in PENDING) break
                sleep(1500)
                waited += 1500
            }
            return when {
                t?.status == "SUCCESS" && t.document != null -> {
                    api.deleteDocument(t.document!!)
                    "Uploaded and deleted a test document (paperless keeps it in its trash until emptied)"
                }
                t?.status == "FAILURE" -> throw SyncException(SyncException.Kind.Config, "paperless refused the test document: ${t.result}")
                else -> "The test upload was accepted (task $task) but paperless has not processed it yet: delete \"FlatScan connection test\" in paperless later"
            }
        } finally {
            f.delete()
        }
    }

    companion object {
        const val PREFIX = "paperless:"
        private val PENDING = setOf("PENDING", "STARTED", "RECEIVED", "RETRY")

        fun md5(f: File): String {
            val d = MessageDigest.getInstance("MD5")
            f.inputStream().use { s -> val b = ByteArray(65536); while (true) { val n = s.read(b); if (n < 0) break; d.update(b, 0, n) } }
            return d.digest().joinToString("") { "%02x".format(it) }
        }

        /** paperless takes the original file name from the upload: the export name. */
        private fun File.renamedFor(name: String): File {
            val want = File(parentFile, "${com.albus.flatscan.DocumentName.clean(name).ifEmpty { "scan" }}.pdf")
            if (want == this) return this
            copyTo(want, overwrite = true)
            want.deleteOnExit()
            return want
        }

        /** A one-page PDF with [text] (Helvetica): the connection test's document. */
        fun tinyPdf(text: String): ByteArray {
            val safe = text.replace("\\", "").replace("(", "").replace(")", "")
            val content = "BT /F1 18 Tf 72 720 Td ($safe) Tj ET"
            val objs = listOf(
                "<< /Type /Catalog /Pages 2 0 R >>",
                "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
                "<< /Length ${content.length} >>\nstream\n$content\nendstream",
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            )
            val sb = StringBuilder("%PDF-1.4\n")
            val offsets = mutableListOf<Int>()
            objs.forEachIndexed { i, o -> offsets += sb.length; sb.append("${i + 1} 0 obj\n$o\nendobj\n") }
            val xref = sb.length
            sb.append("xref\n0 ${objs.size + 1}\n0000000000 65535 f \n")
            offsets.forEach { sb.append("%010d 00000 n \n".format(it)) }
            sb.append("trailer\n<< /Size ${objs.size + 1} /Root 1 0 R >>\nstartxref\n$xref\n%%EOF\n")
            return sb.toString().toByteArray(Charsets.ISO_8859_1)
        }
    }
}
