package com.albus.flatscan

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

class FileNameTemplateTest {
    private val at = Calendar.getInstance().apply { set(2026, Calendar.MARCH, 7, 9, 5, 0) }.timeInMillis

    @Test
    fun defaultTemplate() {
        assertEquals("2026-03-07_Electricity bill", FileNameTemplate.render(FileNameTemplate.DEFAULT, "Electricity bill", at, 3))
    }

    @Test
    fun placeholdersAndDateFormats() {
        assertEquals("07.03.2026 Bill 3p", FileNameTemplate.render("{date} {title} {pages}p", "Bill", at, 3, "dd.MM.yyyy"))
        assertEquals("Bill_09-05", FileNameTemplate.render("{title}_{time}", "Bill", at, 1))
        assertEquals("Bill_2", FileNameTemplate.render("{title}_{n}", "Bill", at, 4, page = 2))
        // {n} without a page leaves no dangling separator
        assertEquals("Bill", FileNameTemplate.render("{title}_{n}", "Bill", at, 4))
        // a broken date pattern falls back to the default one
        assertEquals("2026-03-07", FileNameTemplate.render("{date}", "x", at, 1, "''bad"))
    }

    @Test
    fun automaticNamesGetADash() {
        assertEquals("2026-03-07_Scan - Oct 30", FileNameTemplate.render(FileNameTemplate.DEFAULT, "Scan · Oct 30", at, 1))
    }

    @Test
    fun hebrewTitlesAndUnsafeCharacters() {
        assertEquals("2026-03-07_חשבון חשמל", FileNameTemplate.render(FileNameTemplate.DEFAULT, "חשבון חשמל", at, 1))
        assertEquals("2026-03-07_a b c", FileNameTemplate.render(FileNameTemplate.DEFAULT, "a/b:c", at, 1))
        assertEquals("Scan", FileNameTemplate.render("{n}", "", at, 1))
    }

    @Test
    fun neverOverwriteAddsASuffix() {
        val taken = setOf("Bill.pdf", "Bill (2).pdf")
        assertEquals("Bill (3).pdf", FileNameTemplate.unique("Bill", "pdf") { it in taken })
        assertEquals("Other.pdf", FileNameTemplate.unique("Other", "pdf") { it in taken })
    }
}
