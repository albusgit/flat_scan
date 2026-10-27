package com.albus.flatscan

import com.albus.flatscan.data.DocumentSummary
import com.albus.flatscan.data.SyncStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Tag cleaning, the documents.tags format, Home's chip order and its tag filter (AND). */
class TagsTest {
    private fun d(id: String, tags: String) = DocumentSummary(id, "Doc $id", 0, 0, tags, null, SyncStatus.None, null, 1, null)

    @Test
    fun cleaningTrimsCollapsesSpacesAndDropsCommas() {
        assertEquals("tax 2025", Tags.clean("  tax \t 2025 \n"))
        assertEquals("a b", Tags.clean("a,b"))
        assertEquals("", Tags.clean(" , "))
        assertEquals("חשבונות", Tags.clean(" חשבונות "))
    }

    @Test
    fun aTagIsAtMost32Characters() {
        assertEquals("x".repeat(Tags.MAX_LENGTH), Tags.clean("x".repeat(50)))
        // not cut between the halves of an emoji
        val emoji = "x".repeat(Tags.MAX_LENGTH - 1) + "📄" + "tail"
        val c = Tags.clean(emoji)
        assertEquals("x".repeat(Tags.MAX_LENGTH - 1), c)
        // and no space left at the end of the cut
        assertEquals("x".repeat(Tags.MAX_LENGTH - 1), Tags.clean("x".repeat(Tags.MAX_LENGTH - 1) + " yz"))
    }

    @Test
    fun tagsAreUniqueIgnoringCaseAndKeepTheFirstSpelling() {
        assertEquals(listOf("Tax", "receipts"), Tags.normalise(listOf("Tax", " receipts ", "TAX", "", "tax")))
        assertEquals("Tax,receipts", Tags.format(listOf("Tax", "receipts", "RECEIPTS")))
        assertEquals(listOf("Tax", "receipts"), Tags.parse("Tax, receipts,,tax"))
        assertEquals(emptyList<String>(), Tags.parse(""))
    }

    @Test
    fun addingAndRemovingIgnoreCase() {
        val t = listOf("Tax", "home")
        assertEquals(t, Tags.plus(t, "TAX"))
        assertEquals(t, Tags.plus(t, "   "))
        assertEquals(listOf("Tax", "home", "work"), Tags.plus(t, " work "))
        assertEquals(listOf("home"), Tags.minus(t, "tax"))
        assertEquals(t, Tags.minus(t, "nope"))
        assertTrue(Tags.has(t, " HOME "))
        assertFalse(Tags.has(t, "hom"))
    }

    @Test
    fun chipsAreMostUsedFirstThenAlphabetical() {
        val docs = listOf(d("1", "work,Tax"), d("2", "tax,home"), d("3", "TAX,home,bills"), d("4", ""))
        // tax 3, home 2, then bills and work once each (alphabetical); Tax as the newest document spells it
        assertEquals(listOf("Tax", "home", "bills", "work"), Tags.ranked(docs))
        assertEquals(emptyList<String>(), Tags.ranked(listOf(d("1", ""))))
    }

    @Test
    fun theFilterKeepsDocumentsWithEveryTagPicked() {
        val docs = listOf(d("1", "tax,2025"), d("2", "tax"), d("3", "home,2025"), d("4", ""))
        assertEquals(docs, Tags.filter(docs, emptyList()))
        assertEquals(listOf("1", "2"), Tags.filter(docs, listOf("tax")).map { it.id })
        // AND: both chips
        assertEquals(listOf("1"), Tags.filter(docs, listOf("TAX", "2025")).map { it.id })
        assertEquals(emptyList<String>(), Tags.filter(docs, listOf("tax", "home")).map { it.id })
        // a whole tag, not part of one
        assertEquals(emptyList<String>(), Tags.filter(docs, listOf("ta")).map { it.id })
    }
}
