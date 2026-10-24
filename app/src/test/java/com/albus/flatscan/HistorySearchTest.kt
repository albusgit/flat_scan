package com.albus.flatscan

import com.albus.flatscan.data.DocumentSummary
import com.albus.flatscan.data.SyncStatus
import com.albus.flatscan.ui.filterDocuments
import org.junit.Assert.assertEquals
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
}
