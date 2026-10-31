package com.albus.flatscan

import android.net.Uri
import com.albus.flatscan.batch.BatchItem
import com.albus.flatscan.batch.BatchProcessor
import com.albus.flatscan.batch.ItemState
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** Photos processed at once finish out of order: their pages still go into the document in batch order. */
@RunWith(RobolectricTestRunner::class)
class BatchOrderTest {
    private val uri = Uri.parse("content://x/1")
    private fun item(seq: Int, vararg pages: String) = BatchItem(seq, "d", uri, null, if (pages.isEmpty()) ItemState.Working else ItemState.Done, pages = pages.toList())

    /** Places the pages of [seq] as the batch would, among the photos finished before it. */
    private fun place(order: MutableList<String>, seq: Int, ids: List<String>, done: MutableList<BatchItem>, anchor: Int? = null) {
        order.addAll(BatchProcessor.insertionIndex(order, seq, anchor, done), ids)
        done += item(seq, *ids.toTypedArray())
    }

    @Test
    fun outOfOrderFinishesReadInBatchOrder() {
        val order = mutableListOf("old1", "old2")
        val done = mutableListOf<BatchItem>()
        place(order, 2, listOf("c"), done)
        place(order, 0, listOf("a"), done)
        place(order, 3, listOf("d"), done)
        place(order, 1, listOf("b1", "b2"), done)   // a book spread: two pages
        assertEquals(listOf("old1", "old2", "a", "b1", "b2", "c", "d"), order)
    }

    @Test
    fun insertedBeforeAPageStaysThere() {
        val order = mutableListOf("p1", "p2", "p3")
        val done = mutableListOf<BatchItem>()
        place(order, 1, listOf("b"), done, anchor = 1)
        place(order, 0, listOf("a"), done, anchor = 1)
        place(order, 2, listOf("c"), done, anchor = 1)
        assertEquals(listOf("p1", "a", "b", "c", "p2", "p3"), order)
    }

    @Test
    fun aPageTheUserMovedIsFollowed() {
        val order = mutableListOf("x")
        val done = mutableListOf<BatchItem>()
        place(order, 0, listOf("a"), done)
        // the user drags "a" before "x"; the next photo follows it
        order.remove("a")
        order.add(0, "a")
        place(order, 1, listOf("b"), done)
        assertEquals(listOf("a", "b", "x"), order)
    }

    @Test
    fun anAnchorPastTheEndIsTheEnd() {
        assertEquals(2, BatchProcessor.insertionIndex(listOf("p", "q"), 0, 9, emptyList()))
    }
}
