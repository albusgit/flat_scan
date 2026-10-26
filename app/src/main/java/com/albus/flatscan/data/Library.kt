package com.albus.flatscan.data

import android.content.Context
import androidx.room.withTransaction
import com.albus.flatscan.DocumentStore
import com.albus.flatscan.LegacyDocument
import com.albus.flatscan.Tags
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.TimeUnit

/** Space the library takes on disk (bytes). */
data class StorageUsage(val total: Long, val originals: Long, val documents: Int, val pages: Int)

/**
 * The scan history: documents and their page order ([LibraryDao]) over the page folders ([DocumentStore]).
 * Every call does IO on [Dispatchers.IO]. [clock] is the time source (tests).
 */
class Library(
    private val db: LibraryDb,
    val store: DocumentStore,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    private val dao = db.dao()

    /** Documents not in the trash, last modified first. */
    val documents: Flow<List<DocumentSummary>> = dao.documents()
    val trash: Flow<List<DocumentSummary>> = dao.trash()

    suspend fun document(id: String): DocumentEntity? = io { dao.document(id) }

    /** A new, empty document titled [title] (default: "Scan" and the date). */
    suspend fun createDocument(title: String? = null): DocumentEntity = io {
        val now = clock()
        val d = DocumentEntity(UUID.randomUUID().toString(), title?.takeIf { it.isNotBlank() } ?: defaultTitle(now), now, now)
        dao.insertDocument(d)
        d
    }

    /** Page ids of [doc] in order (pages whose folder is missing are left out). */
    suspend fun pageIds(doc: String): List<String> = io { dao.pages(doc).map { it.id }.filter { store.exists(it) } }

    suspend fun pages(doc: String): List<PageEntity> = io { dao.pages(doc) }

    suspend fun setPages(doc: String, ids: List<String>) = io { dao.setPages(doc, ids, clock()) }

    /** The document page [id] belongs to, or null. */
    suspend fun documentOf(id: String): String? = io { dao.documentOf(id) }

    suspend fun rename(doc: String, title: String) = io { dao.rename(doc, title.trim(), clock()) }

    /** [doc]'s tags become [tags] (cleaned, [Tags.format]). */
    suspend fun setTags(doc: String, tags: List<String>) = io {
        val d = dao.document(doc) ?: return@io
        val v = Tags.format(tags)
        if (v != d.tags) dao.setTags(doc, v, clock())
    }

    /**
     * Puts [tag] on ([on]) or takes it off each of [docs] (feature round 5, the tag sheet). Only documents
     * whose tags change are touched; their modified date moves, so the next upload sends the new tags.
     */
    suspend fun setTag(docs: Collection<String>, tag: String, on: Boolean) = io {
        if (Tags.clean(tag).isEmpty()) return@io
        db.withTransaction {
            val now = clock()
            for (id in docs.distinct()) {
                val d = dao.document(id) ?: continue
                val old = Tags.parse(d.tags)
                val v = Tags.format(if (on) Tags.plus(old, tag) else Tags.minus(old, tag))
                if (v != d.tags) dao.setTags(id, v, now)
            }
        }
    }

    suspend fun touch(doc: String) = io { dao.touch(doc, clock()) }

    /** Moves [docs] to the trash (recoverable for [TRASH_DAYS] days). */
    suspend fun trash(docs: Collection<String>) = io { dao.setDeleted(docs.toList(), clock()) }

    suspend fun restore(docs: Collection<String>) = io { dao.setDeleted(docs.toList(), null) }

    /** Deletes [docs] and their page files now (no undo). */
    suspend fun deleteForever(docs: Collection<String>) = io {
        val ids = docs.toList()
        val pages = ids.flatMap { dao.pages(it) }.map { it.id }
        dao.deleteDocuments(ids)
        pages.forEach { store.delete(it) }
    }

    /** Empties trash older than [TRASH_DAYS] days; returns the documents deleted. */
    suspend fun purgeTrash(): Int {
        val old = io { dao.trashedBefore(clock() - TimeUnit.DAYS.toMillis(TRASH_DAYS)) }
        if (old.isNotEmpty()) deleteForever(old)
        return old.size
    }

    /** Deletes page folders no document refers to ([keep]: pages in use that are not saved yet). */
    suspend fun pruneOrphans(keep: Collection<String> = emptyList()) = io {
        store.prune(dao.allPages().map { it.id } + keep)
    }

    /** Documents without pages (an abandoned scan) are removed, except [keep]. */
    suspend fun dropEmpty(keep: String?) = io {
        val empty = dao.allDocuments().filter { it.id != keep && it.deletedAt == null && dao.pages(it.id).isEmpty() }.map { it.id }
        if (empty.isNotEmpty()) dao.deleteDocuments(empty)
    }

    suspend fun usage(): StorageUsage = io {
        val pages = dao.allPages()
        val (all, orig) = store.usage(pages.map { it.id })
        StorageUsage(all, orig, dao.allDocuments().size, pages.size)
    }

    /**
     * Deletes the original photos of documents last modified more than [days] days ago (their pages
     * keep the processed page and can still be filtered, rotated and exported, but not re-cropped).
     * Returns the bytes freed.
     */
    suspend fun dropOriginalsOlderThan(days: Int): Long = io {
        val cut = clock() - TimeUnit.DAYS.toMillis(days.toLong())
        val pages = dao.allDocuments().filter { it.modifiedAt < cut }.flatMap { dao.pages(it.id) }.filter { !it.originalDropped }
        // a page whose stored session is missing could not be shown at all without its photo: keep it
        val drop = pages.filter { store.sessionDir(it.id) != null && store.outputFile(it.id).exists() }
        var freed = 0L
        drop.forEach { freed += store.dropOriginal(it.id) }
        if (drop.isNotEmpty()) dao.markOriginalDropped(drop.map { it.id })
        freed
    }

    suspend fun originalDropped(page: String): Boolean = io { dao.page(page)?.originalDropped == true }

    /**
     * The document of the versions before history (cacheDir/document) moves into the library: its page
     * folders are moved to the page store, and it becomes a document (title: its name, or "Scan" and the
     * date of its oldest file). Returns the new document's id, or null when there was nothing to import.
     * The legacy folder is deleted afterwards, so this runs once.
     */
    suspend fun importLegacy(legacy: LegacyDocument?): String? = io {
        legacy ?: return@io null
        val moved = legacy.pages.filter { id ->
            val from = File(legacy.dir, id)
            val to = store.dir(id)
            store.root.mkdirs()
            to.exists() || from.renameTo(to) || runCatching { from.copyRecursively(to, overwrite = true) }.getOrDefault(false)
        }
        val result = if (moved.isNotEmpty()) {
            val oldest = moved.mapNotNull { File(store.dir(it), "meta.json").lastModified().takeIf { t -> t > 0 } }.minOrNull() ?: clock()
            val newest = moved.maxOf { File(store.dir(it), "meta.json").lastModified() }.coerceAtLeast(oldest)
            val d = DocumentEntity(UUID.randomUUID().toString(), legacy.name ?: defaultTitle(oldest), oldest, newest)
            dao.insertDocument(d)
            dao.insertPages(moved.mapIndexed { i, id -> PageEntity(id, d.id, i, oldest) })
            d.id
        } else null
        legacy.dir.deleteRecursively()
        result
    }

    // ---- OCR text (feature 3)
    suspend fun setPageText(page: String, text: String?) = io { dao.setPageText(page, text) }

    /** Document id -> its pages' recognised text (documents without any are left out). */
    suspend fun searchTexts(): Map<String, String> = io { dao.documentTexts().mapNotNull { d -> d.text?.let { d.doc to it } }.toMap() }

    // ---- sync (feature 6)
    suspend fun setSync(doc: String, status: SyncStatus, error: String? = null) = io { dao.setSync(doc, status, error) }

    suspend fun markUploaded(doc: String, modified: Long) = io { dao.markUploaded(doc, clock(), modified) }

    /** The documents' upload records (what a retry continues from). */
    val ledger: com.albus.flatscan.sync.SyncLedger = object : com.albus.flatscan.sync.SyncLedger {
        override suspend fun get(doc: String): String? = io { dao.syncRemote(doc) }
        override suspend fun set(doc: String, value: String?) = io { dao.setSyncRemote(doc, value) }
    }

    private suspend fun <T> io(f: suspend () -> T): T = withContext(Dispatchers.IO) { f() }

    companion object {
        const val TRASH_DAYS = 30L

        fun defaultTitle(at: Long): String = "Scan " + SimpleDateFormat("yyyy-MM-dd HH.mm", Locale.US).format(Date(at))

        @Volatile private var instance: Library? = null

        fun get(ctx: Context): Library = instance ?: synchronized(this) {
            instance ?: Library(LibraryDb.get(ctx), DocumentStore(ctx)).also { instance = it }
        }
    }
}
