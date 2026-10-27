package com.albus.flatscan.data

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.albus.flatscan.DocumentStore
import com.albus.flatscan.PageMeta
import com.albus.flatscan.RenderParams
import com.albus.flatscan.Tags
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class LibraryTest {
    private val ctx: Context = ApplicationProvider.getApplicationContext()
    private lateinit var root: File
    private var now = 1_700_000_000_000L
    private lateinit var db: LibraryDb
    private lateinit var lib: Library

    @Before
    fun setUp() {
        root = File(ctx.filesDir, "test_pages_${System.nanoTime()}")
        db = Room.databaseBuilder(ctx, LibraryDb::class.java, "test_${System.nanoTime()}.db").allowMainThreadQueries().build()
        lib = Library(db, DocumentStore(root)) { now }
    }

    @After
    fun tearDown() {
        db.close()
        root.deleteRecursively()
    }

    private fun page(id: String, withSession: Boolean = true) {
        val s = lib.store
        s.saveMeta(PageMeta(id, FloatArray(8), null, 0, false, 0f, RenderParams(), FloatArray(0), 0))
        s.saveOriginal(id, "jpg", ByteArray(1000) { it.toByte() }.inputStream())
        File(s.dir(id), "out.jpg").writeBytes(ByteArray(300))
        if (withSession) File(s.dir(id), "session").apply { mkdirs() }.resolve("ready.json").writeText("{}")
    }

    @Test
    fun documentsAreListedNewestFirstWithCountAndFirstPage() = runBlocking {
        val a = lib.createDocument("A")
        now += 1000
        val b = lib.createDocument("B")
        page("p1"); page("p2"); page("p3")
        now += 1000
        lib.setPages(b.id, listOf("p3"))
        now += 1000
        lib.setPages(a.id, listOf("p2", "p1"))   // a changed last: it comes first
        val docs = lib.documents.first()
        assertEquals(listOf("A", "B"), docs.map { it.title })
        val da = docs.single { it.id == a.id }
        assertEquals(2, da.pageCount)
        assertEquals("p2", da.firstPage)
        assertEquals(listOf("p2", "p1"), lib.pageIds(a.id))
    }

    @Test
    fun movingAPageToAnotherDocumentTakesItOutOfTheFirst() = runBlocking {
        val a = lib.createDocument("A")
        val b = lib.createDocument("B")
        page("p1"); page("p2")
        lib.setPages(a.id, listOf("p1", "p2"))
        lib.setPages(b.id, listOf("p2"))
        assertEquals(listOf("p1"), lib.pageIds(a.id))
        assertEquals(listOf("p2"), lib.pageIds(b.id))
        assertEquals(b.id, lib.documentOf("p2"))
    }

    @Test
    fun renameAndTags() = runBlocking {
        val a = lib.createDocument()
        assertTrue(a.title.startsWith("Scan "))
        lib.rename(a.id, "  Tax 2025  ")
        lib.setTags(a.id, listOf("tax", " receipts ", "tax", ""))
        val d = lib.document(a.id)!!
        assertEquals("Tax 2025", d.title)
        assertEquals("tax,receipts", d.tags)
    }

    @Test
    fun aTagGoesOnSeveralDocumentsAndOnlyChangedOnesAreTouched() = runBlocking {
        val a = lib.createDocument("A")
        val b = lib.createDocument("B")
        val c = lib.createDocument("C")
        lib.setTags(a.id, listOf("Tax"))
        val before = now
        now += 1000
        // a has it already (in another case): only b changes; c is not in the selection
        lib.setTag(listOf(a.id, b.id), " tax ", on = true)
        assertEquals("Tax", lib.document(a.id)!!.tags)
        assertEquals(before, lib.document(a.id)!!.modifiedAt)
        assertEquals("tax", lib.document(b.id)!!.tags)
        assertEquals(now, lib.document(b.id)!!.modifiedAt)
        assertEquals("", lib.document(c.id)!!.tags)
        // one chip for both spellings, spelt as the newest document (b) has it; the filter finds both
        assertEquals(listOf("tax"), Tags.ranked(lib.documents.first()))
        assertEquals(setOf("A", "B"), Tags.filter(lib.documents.first(), listOf("TAX")).map { it.title }.toSet())
        // off again, ignoring case; a blank tag does nothing
        now += 1000
        lib.setTag(listOf(a.id, b.id, c.id), "TAX", on = false)
        lib.setTag(listOf(a.id), "  ", on = true)
        assertEquals(listOf("", "", ""), listOf(a, b, c).map { lib.document(it.id)!!.tags })
        assertEquals(before, lib.document(c.id)!!.modifiedAt)
    }

    @Test
    fun trashKeepsDocumentsThirtyDaysThenPurges() = runBlocking {
        val a = lib.createDocument("A")
        val b = lib.createDocument("B")
        page("pa"); page("pb")
        lib.setPages(a.id, listOf("pa"))
        lib.setPages(b.id, listOf("pb"))
        lib.trash(listOf(a.id))
        assertEquals(listOf("B"), lib.documents.first().map { it.title })
        assertEquals(listOf("A"), lib.trash.first().map { it.title })
        // day 29: still recoverable
        now += TimeUnit.DAYS.toMillis(29)
        assertEquals(0, lib.purgeTrash())
        lib.restore(listOf(a.id))
        assertEquals(2, lib.documents.first().size)
        lib.trash(listOf(a.id))
        now += TimeUnit.DAYS.toMillis(30) + 1
        assertEquals(1, lib.purgeTrash())
        assertNull(lib.document(a.id))
        assertFalse(lib.store.dir("pa").exists())
        assertTrue(lib.store.dir("pb").exists())
    }

    @Test
    fun deleteForeverRemovesFiles() = runBlocking {
        val a = lib.createDocument("A")
        page("p1")
        lib.setPages(a.id, listOf("p1"))
        lib.deleteForever(listOf(a.id))
        assertNull(lib.document(a.id))
        assertFalse(lib.store.dir("p1").exists())
    }

    @Test
    fun dropOriginalsOnlyOfOldDocumentsWithAStoredPage() = runBlocking {
        val old = lib.createDocument("old")
        page("o1"); page("o2", withSession = false)
        lib.setPages(old.id, listOf("o1", "o2"))
        now += TimeUnit.DAYS.toMillis(100)
        val fresh = lib.createDocument("new")
        page("n1")
        lib.setPages(fresh.id, listOf("n1"))
        val usage = lib.usage()
        assertEquals(3000L, usage.originals)
        val freed = lib.dropOriginalsOlderThan(90)
        assertEquals(1000L, freed)                       // o2 has no stored session: kept
        assertFalse(lib.store.hasSource("o1"))
        assertTrue(lib.store.hasSource("o2"))
        assertTrue(lib.store.hasSource("n1"))
        assertTrue(lib.originalDropped("o1"))
        assertTrue(lib.store.outputFile("o1").exists())
    }

    @Test
    fun legacyDocumentIsImportedOnce() = runBlocking {
        val legacyDir = File(ctx.cacheDir, "legacy_${System.nanoTime()}")
        for (id in listOf("x1", "x2")) {
            val d = File(legacyDir, id).apply { mkdirs() }
            File(d, "meta.json").writeText(PageMeta(id, FloatArray(8), null, 0, false, 0f, RenderParams(), FloatArray(0), 0).toJson().toString())
            File(d, "source.jpg").writeBytes(ByteArray(10))
        }
        File(legacyDir, "doc.json").writeText(JSONObject().put("pages", JSONArray(listOf("x2", "x1", "gone"))).toString())
        File(legacyDir, "name.txt").writeText("Old receipts\n")
        val id = lib.importLegacy(DocumentStore.legacyAt(legacyDir))
        assertNotNull(id)
        assertEquals("Old receipts", lib.document(id!!)!!.title)
        assertEquals(listOf("x2", "x1"), lib.pageIds(id))
        assertTrue(lib.store.hasSource("x1"))
        assertFalse(legacyDir.exists())
        assertNull(lib.importLegacy(DocumentStore.legacyAt(legacyDir)))
    }

    @Test
    fun libraryOutlivesTheProcess() = runBlocking {
        // a file-backed database closed and opened again, as after a process kill
        val name = "persist_${System.nanoTime()}.db"
        val db1 = Room.databaseBuilder(ctx, LibraryDb::class.java, name).build()
        val l1 = Library(db1, DocumentStore(root)) { now }
        val a = l1.createDocument("Kept")
        page("k1")
        l1.setPages(a.id, listOf("k1"))
        db1.close()
        val db2 = Room.databaseBuilder(ctx, LibraryDb::class.java, name).build()
        val l2 = Library(db2, DocumentStore(root)) { now }
        assertEquals("Kept", l2.document(a.id)!!.title)
        assertEquals(listOf("k1"), l2.pageIds(a.id))
        db2.close()
    }

    @Test
    fun emptyAbandonedDocumentsAreDropped() = runBlocking {
        val a = lib.createDocument("empty")
        val b = lib.createDocument("open")
        lib.dropEmpty(keep = b.id)
        assertNull(lib.document(a.id))
        assertNotNull(lib.document(b.id))
    }
}
