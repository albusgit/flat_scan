package com.albus.flatscan

import android.content.Context
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** PDF page size: a paper size (portrait or landscape to match the page) or the page's own shape. */
enum class PdfPageSize(val label: String, val wPt: Int, val hPt: Int) {
    A4("A4", 595, 842),
    Letter("Letter", 612, 792),
    Fit("Same as the page", 595, 842),
}

/** Margin around the page image on the PDF page, in points (1/72 in). */
enum class PdfMargin(val label: String, val pt: Int) { None("None", 0), Small("Small", 18), Normal("Normal", 36) }

/** Resolution of the page images in the PDF: as scanned, or at most 150 dpi on the PDF page (smaller files). */
enum class PdfImageSize(val label: String, val maxDpi: Int) { Full("Full", 0), Compact("Smaller file", 150) }

/** PDF export options; the defaults write what earlier versions wrote (A4, no margin, full resolution). */
data class PdfOptions(
    val pageSize: PdfPageSize = PdfPageSize.A4,
    val margin: PdfMargin = PdfMargin.None,
    val imageSize: PdfImageSize = PdfImageSize.Full,
)

/** The export choices, remembered (their own preferences file). */
object ExportPrefs {
    private fun prefs(ctx: Context) = ctx.getSharedPreferences("export", Context.MODE_PRIVATE)

    fun pdf(ctx: Context): PdfOptions {
        val p = prefs(ctx)
        return PdfOptions(
            pageSize = PdfPageSize.entries.firstOrNull { it.name == p.getString("pdf_size", null) } ?: PdfPageSize.A4,
            margin = PdfMargin.entries.firstOrNull { it.name == p.getString("pdf_margin", null) } ?: PdfMargin.None,
            imageSize = PdfImageSize.entries.firstOrNull { it.name == p.getString("pdf_image", null) } ?: PdfImageSize.Full,
        )
    }

    fun setPdf(ctx: Context, o: PdfOptions) {
        prefs(ctx).edit().putString("pdf_size", o.pageSize.name).putString("pdf_margin", o.margin.name)
            .putString("pdf_image", o.imageSize.name).apply()
    }
}

/** Document names: the default name and what a file name may hold. The name itself is the document's title (Library). */
object DocumentName {
    fun default(): String = "Scan " + SimpleDateFormat("yyyy-MM-dd HH.mm", Locale.US).format(Date())

    /** No path separators or characters file providers reject. */
    fun clean(s: String): String = s.replace(Regex("[\\\\/:*?\"<>|\\n\\r\\t]"), " ").trim().take(80)
}
