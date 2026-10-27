package com.albus.flatscan.sync

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.albus.flatscan.ExportPrefs
import com.albus.flatscan.Exporter
import com.albus.flatscan.FileNameTemplate
import com.albus.flatscan.data.DocumentEntity
import com.albus.flatscan.data.Library
import com.albus.flatscan.data.SyncStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Uploads one document (input "doc") to the configured target. The PDF is written once per document
 * version (noBackupFilesDir/sync/<doc>_<modified>.pdf) and reused by retries, so every attempt sends the
 * same bytes; the uploaders make the retries idempotent ([SyncLedger]). Network trouble: retry with
 * exponential backoff (WorkManager), up to [MAX_ATTEMPTS]; wrong settings or credentials: failed at once,
 * with the reason shown in the history.
 */
class SyncWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result {
        val doc = inputData.getString(KEY_DOC) ?: return Result.failure()
        val lib = Library.get(applicationContext)
        val d = lib.document(doc) ?: return Result.success()
        if (d.deletedAt != null) {
            lib.setSync(doc, SyncStatus.None)
            return Result.success()
        }
        val cfg = SyncConfig(applicationContext)
        val up = uploaderFactory?.invoke(applicationContext) ?: cfg.uploader() ?: run {
            lib.setSync(doc, SyncStatus.Failed, "Sync is not set up")
            return Result.failure()
        }
        lib.setSync(doc, SyncStatus.Uploading)
        return try {
            val pages = lib.pageIds(doc)
            if (pages.isEmpty()) throw SyncException(SyncException.Kind.Config, "The document has no pages")
            val file = exportFile(applicationContext, lib, d, pages)
            val name = FileNameTemplate.render(cfg.nameTemplate, d.title, d.createdAt, pages.size, cfg.dateFormat)
            val where = up.upload(UploadJob(doc, name, file, d.title, d.createdAt, d.tags.split(",").filter { it.isNotBlank() }), lib.ledger)
            lib.markUploaded(doc, d.modifiedAt)
            file.delete()
            Log.i(TAG, "uploaded $doc: $where")
            Result.success()
        } catch (e: Exception) {
            val se = Http.classify(e)
            if (se.retryable && runAttemptCount + 1 < MAX_ATTEMPTS) {
                lib.setSync(doc, SyncStatus.Pending, "Will retry: ${se.message}")
                Result.retry()
            } else {
                lib.setSync(doc, SyncStatus.Failed, se.message)
                Result.failure()
            }
        }
    }

    companion object {
        const val TAG = "FlatScanSync"
        const val KEY_DOC = "doc"
        const val MAX_ATTEMPTS = 10

        /** Tests: the uploader to use instead of the configured one. */
        @Volatile var uploaderFactory: ((Context) -> Uploader)? = null

        /** The PDF of [d] as it is now; the same file (bytes) for every attempt at this version. */
        suspend fun exportFile(ctx: Context, lib: Library, d: DocumentEntity, pages: List<String>): File = withContext(Dispatchers.IO) {
            val dir = File(ctx.noBackupFilesDir, "sync").apply { mkdirs() }
            val f = File(dir, "${d.id}_${d.modifiedAt}.pdf")
            dir.listFiles()?.forEach { if (it.name.startsWith("${d.id}_") && it != f) it.delete() }
            if (!f.exists() || f.length() == 0L) {
                val tmp = File(dir, f.name + ".tmp")
                // the export options of the app (colour, size, paper, searchable text), as a PDF
                val o = ExportPrefs.output(ctx).copy(format = com.albus.flatscan.OutputFormat.Pdf)
                val p = Exporter(ctx, lib).prepare(com.albus.flatscan.ExportDoc(d.title, d.createdAt, d.modifiedAt, pages), o)
                tmp.outputStream().buffered().use { Exporter(ctx, lib).writePdf(p, it) }
                if (!tmp.renameTo(f)) error("cannot write $f")
            }
            f
        }

        fun workName(doc: String) = "sync-$doc"

        /** Queues the upload of [doc] (once: a queued or running upload of it is kept) with the settings' conditions. */
        suspend fun enqueue(ctx: Context, doc: String) {
            val cfg = SyncConfig(ctx)
            val c = Constraints.Builder()
                .setRequiredNetworkType(if (cfg.kind == SyncKind.Folder) NetworkType.NOT_REQUIRED else if (cfg.wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED)
                .setRequiresCharging(cfg.chargingOnly)
                .build()
            val req = OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(c)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
                .setInputData(workDataOf(KEY_DOC to doc))
                .addTag("sync")
                .build()
            Library.get(ctx).setSync(doc, SyncStatus.Pending)
            WorkManager.getInstance(ctx).enqueueUniqueWork(workName(doc), ExistingWorkPolicy.KEEP, req)
        }
    }
}
