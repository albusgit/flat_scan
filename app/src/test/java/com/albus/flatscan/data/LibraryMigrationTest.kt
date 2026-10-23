package com.albus.flatscan.data

import androidx.room.testing.MigrationTestHelper
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Every exported schema version (app/schemas) is created and migrated to the latest with
 * [LibraryDb.MIGRATIONS]; Room validates the result against the latest schema.
 */
@RunWith(AndroidJUnit4::class)
class LibraryMigrationTest {
    @get:Rule
    val helper = MigrationTestHelper(InstrumentationRegistry.getInstrumentation(), LibraryDb::class.java)

    @Test
    fun version1IsCreatedWithItsSchema() {
        helper.createDatabase("mig_v1", 1).use { db ->
            db.execSQL("INSERT INTO documents (id, title, created_at, modified_at, tags, sync_status) VALUES ('d', 'T', 1, 2, '', 'None')")
            db.execSQL("INSERT INTO pages (id, document_id, position, created_at, original_dropped) VALUES ('p', 'd', 0, 1, 0)")
            db.query("SELECT COUNT(*) FROM pages").use { c -> c.moveToFirst(); assertEquals(1, c.getInt(0)) }
        }
    }

    @Test
    fun version1To2KeepsDocumentsAndAddsTheSyncColumns() {
        helper.createDatabase("mig_12", 1).use { db ->
            db.execSQL("INSERT INTO documents (id, title, created_at, modified_at, tags, sync_status) VALUES ('d', 'Kept', 1, 2, 'a,b', 'None')")
            db.execSQL("INSERT INTO pages (id, document_id, position, created_at, original_dropped) VALUES ('p', 'd', 0, 1, 0)")
        }
        val db = helper.runMigrationsAndValidate("mig_12", 2, true, *LibraryDb.MIGRATIONS)
        db.query("SELECT title, tags, sync_remote, uploaded_at, uploaded_modified FROM documents WHERE id = 'd'").use { c ->
            c.moveToFirst()
            assertEquals("Kept", c.getString(0))
            assertEquals("a,b", c.getString(1))
            assertEquals(true, c.isNull(2) && c.isNull(3) && c.isNull(4))
        }
        db.query("SELECT COUNT(*) FROM pages").use { c -> c.moveToFirst(); assertEquals(1, c.getInt(0)) }
    }
}
