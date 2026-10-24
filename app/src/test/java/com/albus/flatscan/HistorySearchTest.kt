package com.albus.flatscan

import com.albus.flatscan.data.DocumentSummary
import com.albus.flatscan.data.SyncStatus
import androidx.compose.ui.text.SpanStyle
import com.albus.flatscan.ui.filterDocuments
import com.albus.flatscan.ui.searchSnippet
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HistorySearchTest {
    private fun d(id: String, title: String, tags: String = "") = DocumentSummary(id, title, 0, 0, tags, null, SyncStatus.None, null, 1, null)

    private val docs = listOf(d("1", "Electricity bill March"), d("2", "חשבונית ארנונה"), d("3", "Lease", tags = "home,contract"))

    @Test
    fun matchesEveryWordAnyCase() {
        assertEquals(listOf("1"), filterDocuments(docs, "bill ELECTRICITY").map { it.id })
        assertEquals(docs, filterDocuments(docs, "  "))
    }

    @Test
    fun hebrewTitlesAndTags() {
        assertEquals(listOf("2"), filterDocuments(docs, "ארנונה").map { it.id })
        assertEquals(listOf("3"), filterDocuments(docs, "contract").map { it.id })
    }

    @Test
    fun recognisedTextCounts() {
        val text = mapOf("3" to "monthly rent payable")
        assertEquals(listOf("3"), filterDocuments(docs, "rent", text::get).map { it.id })
    }

    @Test
    fun snippetMarksTheWordsWhereTheyAreFound() {
        val mark = SpanStyle()
        val text = "Invoice 2026-114\nSubtotal 1,140.00   VAT 100.00\nTotal due 1,240.00 by 30 November. Thank you for your business, payment by transfer."
        val s = searchSnippet(text, "total DUE", mark)!!
        assert(s.text.contains("Total due 1,240.00")) { s.text }
        assert(!s.text.contains("\n"))
        val marked = s.spanStyles.map { s.text.substring(it.start, it.end).lowercase() }
        // "Subtotal" and "Total" both hold "total"
        assertEquals(2, marked.count { it == "total" })
        assertEquals(1, marked.count { it == "due" })
        assertNull(searchSnippet(text, "lease", mark))
        assertNull(searchSnippet(null, "total", mark))
    }
}
