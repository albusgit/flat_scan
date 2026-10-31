package com.albus.flatscan

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * File names of exported and uploaded documents from a template such as `{date}_{title}`.
 * Placeholders: {title}, {date} (in [dateFormat], a java.text.SimpleDateFormat pattern), {time} (HH-mm),
 * {pages} (page count), {n} (page number, for one file per page; empty otherwise). Unknown placeholders
 * stay as they are. The result is cleaned for file systems and servers ([DocumentName.clean]) and never empty.
 */
object FileNameTemplate {
    const val DEFAULT = "{date}_{title}"
    const val DEFAULT_DATE = "yyyy-MM-dd"
    val DATE_FORMATS = listOf("yyyy-MM-dd", "dd.MM.yyyy", "MM-dd-yyyy", "yyyyMMdd", "yyyy-MM-dd_HH-mm")

    fun render(template: String, title: String, at: Long, pages: Int, dateFormat: String = DEFAULT_DATE, page: Int? = null): String {
        val date = runCatching { SimpleDateFormat(dateFormat, Locale.US).format(Date(at)) }.getOrElse { SimpleDateFormat(DEFAULT_DATE, Locale.US).format(Date(at)) }
        val time = SimpleDateFormat("HH-mm", Locale.US).format(Date(at))
        val out = template.ifBlank { DEFAULT }
            // the automatic names' " · " (Scan · Oct 30) as a plain dash: not every server or tool takes the dot
            .replace("{title}", title.replace(" · ", " - "))
            .replace("{date}", date)
            .replace("{time}", time)
            .replace("{pages}", pages.toString())
            .replace("{n}", page?.toString() ?: "")
        // separators left dangling by an empty placeholder ("_{n}" without a page) go too
        val cleaned = DocumentName.clean(out).trim('_', '-', ' ', '.')
        return cleaned.ifEmpty { DocumentName.clean(title).ifEmpty { "Scan" } }
    }

    /** [name] made unique against [taken] by " (2)", " (3)"...: never overwrite, add a suffix. */
    fun unique(name: String, ext: String, taken: (String) -> Boolean): String {
        var candidate = "$name.$ext"
        var i = 2
        while (taken(candidate)) candidate = "$name ($i).${ext}".also { i++ }
        return candidate
    }
}
