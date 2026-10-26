package com.albus.flatscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The tag sheet over one or several documents: each tag's state, what a tap does, the chip order. */
class TagEditTest {
    private val three = TagEdit(mapOf("a" to listOf("tax", "home"), "b" to listOf("Tax"), "c" to listOf("TAX", "work")))

    @Test
    fun aTagIsOnAllSomeOrNoneOfTheDocuments() {
        assertEquals(TagState.All, three.state("tax"))
        assertEquals(TagState.Some, three.state("home"))
        assertEquals(TagState.None, three.state("bills"))
        assertEquals(TagState.None, TagEdit(emptyMap()).state("tax"))
    }

    @Test
    fun aTapPutsItOnAllUnlessAllHaveItThenTakesItOff() {
        assertFalse(three.tapTurnsOn("tax"))
        assertTrue(three.tapTurnsOn("home"))
        assertTrue(three.tapTurnsOn("bills"))
        val on = three.set("home", true)
        assertEquals(TagState.All, on.state("home"))
        // the documents that had it keep their spelling, the others get it at the end
        assertEquals(listOf("tax", "home"), on.docs["a"])
        assertEquals(listOf("Tax", "home"), on.docs["b"])
        val off = on.set("TAX", false)
        assertEquals(TagState.None, off.state("tax"))
        assertEquals(mapOf("a" to listOf("home"), "b" to listOf("home"), "c" to listOf("work", "home")), off.docs)
    }

    @Test
    fun oneDocumentIsAllOrNone() {
        val one = TagEdit(mapOf("a" to listOf("tax")))
        assertEquals(TagState.All, one.state("Tax"))
        assertFalse(one.tapTurnsOn("tax"))
        assertEquals(listOf<String>(), one.set("tax", false).docs["a"])
        // a blank tag changes nothing
        assertEquals(one, one.set("  ", true))
    }

    @Test
    fun theDocumentsOwnTagsComeFirstInTheLibrarysOrder() {
        val known = listOf("bills", "home", "tax", "receipts")
        // own: tax, home, work (work is not in known yet); then the rest of known
        assertEquals(listOf("home", "tax", "work", "bills", "receipts"), three.choices(known))
        assertEquals(known, TagEdit(mapOf("x" to emptyList())).choices(known))
    }
}
