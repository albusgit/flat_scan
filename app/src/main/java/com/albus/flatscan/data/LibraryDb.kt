package com.albus.flatscan.data

import android.content.Context
import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Transaction
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/** Where a document stands with the user's own server (sync, feature 6). */
enum class SyncStatus { None, Pending, Uploading, Uploaded, Failed }

/**
 * A scanned document. Its pages' files live in [DocumentStore] (one folder per page); this row and its
 * [PageEntity] rows hold the structure: title, dates, tags, order, sync state, trash.
 * [deletedAt] set = in the trash (kept 30 days, [Library.purgeTrash]).
 */
@Entity(tableName = "documents", indices = [Index("modified_at"), Index("deleted_at")])
data class DocumentEntity(
    @PrimaryKey val id: String,
    val title: String,
    @ColumnInfo(name = "created_at") val createdAt: Long,
    @ColumnInfo(name = "modified_at") val modifiedAt: Long,
    /** Comma-separated, trimmed; empty = none. */
    val tags: String = "",
    @ColumnInfo(name = "deleted_at") val deletedAt: Long? = null,
    @ColumnInfo(name = "sync_status") val syncStatus: SyncStatus = SyncStatus.None,
    @ColumnInfo(name = "sync_error") val syncError: String? = null,
)

/**
 * One page of a document, in [position] order. The processing parameters (corners, mode, filter,
 * rotation, erase and finger masks...) are in the page folder's meta.json ([com.albus.flatscan.PageMeta]),
 * next to the original photo, the processed page (cache) and the thumbnail.
 * [originalDropped]: the original photo was deleted to save space (the page can no longer be re-cropped).
 */
@Entity(
    tableName = "pages",
    foreignKeys = [ForeignKey(entity = DocumentEntity::class, parentColumns = ["id"], childColumns = ["document_id"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("document_id")],
)
data class PageEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "document_id") val documentId: String,
    val position: Int,
    @ColumnInfo(name = "created_at") val createdAt: Long,
    @ColumnInfo(name = "original_dropped") val originalDropped: Boolean = false,
)

/** A document row with its page count and first page (history list). */
data class DocumentSummary(
    val id: String,
    val title: String,
    @ColumnInfo(name = "created_at") val createdAt: Long,
    @ColumnInfo(name = "modified_at") val modifiedAt: Long,
    val tags: String,
    @ColumnInfo(name = "deleted_at") val deletedAt: Long?,
    @ColumnInfo(name = "sync_status") val syncStatus: SyncStatus,
    @ColumnInfo(name = "sync_error") val syncError: String?,
    @ColumnInfo(name = "page_count") val pageCount: Int,
    @ColumnInfo(name = "first_page") val firstPage: String?,
)

@Dao
interface LibraryDao {
    @Query(
        """SELECT d.*, (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) AS page_count,
           (SELECT p.id FROM pages p WHERE p.document_id = d.id ORDER BY p.position LIMIT 1) AS first_page
           FROM documents d WHERE d.deleted_at IS NULL ORDER BY d.modified_at DESC""",
    )
    fun documents(): Flow<List<DocumentSummary>>

    @Query(
        """SELECT d.*, (SELECT COUNT(*) FROM pages p WHERE p.document_id = d.id) AS page_count,
           (SELECT p.id FROM pages p WHERE p.document_id = d.id ORDER BY p.position LIMIT 1) AS first_page
           FROM documents d WHERE d.deleted_at IS NOT NULL ORDER BY d.deleted_at DESC""",
    )
    fun trash(): Flow<List<DocumentSummary>>

    @Query("SELECT * FROM documents WHERE id = :id")
    suspend fun document(id: String): DocumentEntity?

    @Query("SELECT * FROM documents")
    suspend fun allDocuments(): List<DocumentEntity>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertDocument(d: DocumentEntity)

    @Update
    suspend fun updateDocument(d: DocumentEntity)

    @Query("UPDATE documents SET title = :title, modified_at = :now WHERE id = :id")
    suspend fun rename(id: String, title: String, now: Long)

    @Query("UPDATE documents SET modified_at = :now WHERE id = :id")
    suspend fun touch(id: String, now: Long)

    @Query("UPDATE documents SET deleted_at = :at WHERE id IN (:ids)")
    suspend fun setDeleted(ids: List<String>, at: Long?)

    @Query("SELECT id FROM documents WHERE deleted_at IS NOT NULL AND deleted_at < :before")
    suspend fun trashedBefore(before: Long): List<String>

    @Query("DELETE FROM documents WHERE id IN (:ids)")
    suspend fun deleteDocuments(ids: List<String>)

    @Query("SELECT * FROM pages WHERE document_id = :doc ORDER BY position")
    suspend fun pages(doc: String): List<PageEntity>

    @Query("SELECT * FROM pages")
    suspend fun allPages(): List<PageEntity>

    @Query("SELECT * FROM pages WHERE id = :id")
    suspend fun page(id: String): PageEntity?

    @Query("SELECT document_id FROM pages WHERE id = :id")
    suspend fun documentOf(id: String): String?

    @Query("DELETE FROM pages WHERE document_id = :doc")
    suspend fun deletePagesOf(doc: String)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPages(p: List<PageEntity>)

    @Query("UPDATE pages SET original_dropped = 1 WHERE id IN (:ids)")
    suspend fun markOriginalDropped(ids: List<String>)

    /**
     * Makes [ids] the pages of [doc], in this order: rows keep their created date and dropped flag, pages
     * moved here from another document leave it, pages no longer listed are removed from [doc].
     */
    @Transaction
    suspend fun setPages(doc: String, ids: List<String>, now: Long) {
        val known = (pages(doc) + ids.mapNotNull { page(it) }).associateBy { it.id }
        deletePagesOf(doc)
        insertPages(ids.mapIndexed { i, id -> (known[id] ?: PageEntity(id, doc, i, now)).copy(documentId = doc, position = i) })
        touch(doc, now)
    }
}

@Database(entities = [DocumentEntity::class, PageEntity::class], version = 1, exportSchema = true)
abstract class LibraryDb : RoomDatabase() {
    abstract fun dao(): LibraryDao

    companion object {
        const val NAME = "library.db"

        /** Every migration, oldest first (LibraryMigrationTest runs them on the exported schemas). */
        val MIGRATIONS = arrayOf<androidx.room.migration.Migration>()

        @Volatile private var instance: LibraryDb? = null

        fun get(ctx: Context): LibraryDb = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(ctx.applicationContext, LibraryDb::class.java, NAME)
                .addMigrations(*MIGRATIONS)
                .build().also { instance = it }
        }
    }
}
