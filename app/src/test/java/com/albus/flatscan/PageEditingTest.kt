package com.albus.flatscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PageEditingTest {
    private fun doc(n: Int) = PagesState("d", mapOf("d" to (1..n).map { "p$it" }))
    private fun PagesState.ids() = pages.joinToString(",")

    @Test
    fun moveOneForwardAndBack() {
        assertEquals("p2,p3,p1,p4", PageOps.moveOne(doc(4), 0, 2).ids())
        assertEquals("p4,p1,p2,p3", PageOps.moveOne(doc(4), 3, 0).ids())
        assertEquals(doc(4), PageOps.moveOne(doc(4), 1, 1))
        assertEquals(doc(4), PageOps.moveOne(doc(4), 1, 9))
    }

    @Test
    fun moveSelectionAsABlock() {
        // p2 and p4 to the front, in document order
        assertEquals("p2,p4,p1,p3,p5", PageOps.move(doc(5), setOf("p4", "p2"), 0).ids())
        // to the end
        assertEquals("p1,p3,p5,p2,p4", PageOps.move(doc(5), setOf("p2", "p4"), 99).ids())
        // between p3 and p5 of the remaining list (p1, p3, p5)
        assertEquals("p1,p3,p2,p4,p5", PageOps.move(doc(5), setOf("p2", "p4"), 2).ids())
    }

    @Test
    fun reverseAll() {
        assertEquals("p5,p4,p3,p2,p1", PageOps.reverse(doc(5)).ids())
        assertEquals(doc(5).pages, PageOps.reverse(PageOps.reverse(doc(5))).pages)
    }

    @Test
    fun insertAnywhere() {
        assertEquals("n1,n2,p1,p2,p3", PageOps.insert(doc(3), 0, listOf("n1", "n2")).ids())
        assertEquals("p1,n1,p2,p3", PageOps.insert(doc(3), 1, listOf("n1")).ids())
        assertEquals("p1,p2,p3,n1", PageOps.insert(doc(3), 3, listOf("n1")).ids())
        assertEquals("p1,p2,p3,n1", PageOps.insert(doc(3), 42, listOf("n1")).ids())
    }

    @Test
    fun deleteDuplicateRotate() {
        assertEquals("p1,p3", PageOps.delete(doc(3), setOf("p2")).ids())
        val (dup, copies) = PageOps.duplicate(doc(3), setOf("p1", "p3")) { "$it'" }
        assertEquals("p1,p1',p2,p3,p3'", dup.ids())
        assertEquals(mapOf("p1" to "p1'", "p3" to "p3'"), copies)
        val r = PageOps.rotate(doc(3), setOf("p2"), 1) { 3 }
        assertEquals(mapOf("p2" to 0), r.rotations)          // 3 + 1 = 0 (mod 4)
        val r2 = PageOps.rotate(r, setOf("p2"), -1) { 3 }
        assertEquals(3, r2.rotations["p2"])
        // a copy keeps the rotation the session gave its original
        val (dup2, _) = PageOps.duplicate(r2, setOf("p2")) { "$it'" }
        assertEquals(3, dup2.rotations["p2'"])
    }

    @Test
    fun splitIntoANewDocument() {
        val s = PageOps.split(doc(5), 2, "e")
        assertEquals("p1,p2", s.ids())
        assertEquals(listOf("p3", "p4", "p5"), s.docs["e"])
        assertEquals(doc(5), PageOps.split(doc(5), 0, "e"))   // nothing to split off
        assertEquals(doc(5), PageOps.split(doc(5), 5, "e"))
    }

    @Test
    fun moveToAnotherDocument() {
        val s = PageOps.moveToDocument(doc(4), setOf("p3", "p1"), "other", listOf("o1"))
        assertEquals("p2,p4", s.ids())
        assertEquals(listOf("o1", "p1", "p3"), s.docs["other"])
    }

    @Test
    fun undoRedoCoverTheWholeSession() {
        val h = PagesHistory(doc(4))
        h.apply(PageOps.moveOne(h.state, 3, 0))                   // p4,p1,p2,p3
        h.apply(PageOps.reverse(h.state))                         // p3,p2,p1,p4
        h.apply(PageOps.rotate(h.state, setOf("p2"), 1) { 0 })
        h.apply(PageOps.split(h.state, 2, "e"))                   // p3,p2 | p1,p4
        assertEquals("p3,p2", h.state.ids())
        assertEquals(listOf("p1", "p4"), h.state.docs["e"])
        assertEquals("p3,p2,p1,p4", h.undo()!!.ids())
        assertEquals(mapOf("p2" to 1), h.state.rotations)
        h.undo()
        assertTrue(h.state.rotations.isEmpty())
        assertEquals("p4,p1,p2,p3", h.undo()!!.ids())
        assertEquals("p1,p2,p3,p4", h.undo()!!.ids())
        assertNull(h.undo())
        assertFalse(h.canUndo)
        assertEquals("p4,p1,p2,p3", h.redo()!!.ids())
        // a new edit drops the redo branch
        h.apply(PageOps.delete(h.state, setOf("p1")))
        assertFalse(h.canRedo)
        assertEquals("p4,p2,p3", h.state.ids())
        // a page deleted in this session stays referenced (its files must survive until the session ends)
        assertTrue("p1" in h.referenced())
    }

    @Test
    fun noOpEditsAreNotRecorded() {
        val h = PagesHistory(doc(3))
        h.apply(PageOps.moveOne(h.state, 1, 1))
        assertFalse(h.canUndo)
    }
}
