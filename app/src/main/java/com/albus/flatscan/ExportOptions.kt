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

/** What an export writes. */
enum class OutputFormat(val label: String, val caption: String) {
    Pdf("PDF", "All pages in one PDF"),
    Jpeg("JPEG", "One picture per page"),
    Png("PNG", "One lossless picture per page"),
    PdfAndJpeg("PDF + JPEG", "The PDF and one picture per page"),
}

/** Colour of the exported pages. B&W pages are stored with 1 bit per pixel (CCITT G4 in the PDF, PNG for pictures). */
enum class ColorMode(val label: String) { Color("Colour"), Grayscale("Grayscale"), BlackWhite("Black & white") }

/**
 * Size presets: the most pixels per inch of the page on paper (the page is never enlarged) and the JPEG
 * quality; black and white always keeps [bwDpi] (text needs it, and 1-bit pages are small anyway).
 */
enum class SizePreset(val label: String, val dpi: Int, val jpegQuality: Int, val bwDpi: Int) {
    Small("Small", 150, 55, 200),
    Balanced("Balanced", 200, 72, 300),
    High("High", 300, 90, 300),
}

/** Every export choice, remembered ([ExportPrefs]). */
data class OutputOptions(
    val format: OutputFormat = OutputFormat.Pdf,
    val color: ColorMode = ColorMode.Color,
    val size: SizePreset = SizePreset.Balanced,
    val pageSize: PdfPageSize = PdfPageSize.A4,
    val margin: PdfMargin = PdfMargin.None,
    /** An invisible OCR text layer: the PDF can be searched and its text copied. */
    val searchable: Boolean = true,
    /** Ask where to save every time (the system file picker); off: into [folder] without asking. */
    val askWhere: Boolean = true,
    /** The default export folder (a Storage Access Framework tree URI), or null. */
    val folder: String? = null,
    val template: String = FileNameTemplate.DEFAULT,
    val dateFormat: String = FileNameTemplate.DEFAULT_DATE,
)

/** The export choices, remembered (their own preferences file). */
object ExportPrefs {
    private fun prefs(ctx: Context) = ctx.getSharedPreferences("export", Context.MODE_PRIVATE)

    private inline fun <reified E : Enum<E>> enumOf(name: String?, d: E): E = enumValues<E>().firstOrNull { it.name == name } ?: d

    fun output(ctx: Context): OutputOptions {
        val p = prefs(ctx)
        val d = OutputOptions()
        return OutputOptions(
            format = enumOf(p.getString("format", null), d.format),
            color = enumOf(p.getString("color", null), d.color),
            size = enumOf(p.getString("size", null), d.size),
            pageSize = enumOf(p.getString("pdf_size", null), d.pageSize),
            margin = enumOf(p.getString("pdf_margin", null), d.margin),
            searchable = p.getBoolean("searchable", d.searchable),
            askWhere = p.getBoolean("ask_where", d.askWhere),
            folder = p.getString("folder", null),
            template = p.getString("template", null) ?: d.template,
            dateFormat = p.getString("date_format", null) ?: d.dateFormat,
        )
    }

    fun setOutput(ctx: Context, o: OutputOptions) {
        prefs(ctx).edit().putString("format", o.format.name).putString("color", o.color.name).putString("size", o.size.name)
            .putString("pdf_size", o.pageSize.name).putString("pdf_margin", o.margin.name).putBoolean("searchable", o.searchable)
            .putBoolean("ask_where", o.askWhere).putString("folder", o.folder).putString("template", o.template)
            .putString("date_format", o.dateFormat).apply()
    }
}

/** Document names: the default name and what a file name may hold. The name itself is the document's title (Library). */
object DocumentName {
    fun default(): String = "Scan " + SimpleDateFormat("yyyy-MM-dd HH.mm", Locale.US).format(Date())

    /** No path separators or characters file providers reject. */
    fun clean(s: String): String = s.replace(Regex("[\\\\/:*?\"<>|\\n\\r\\t]"), " ").trim().take(80)
}
