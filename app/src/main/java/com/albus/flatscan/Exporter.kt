package com.albus.flatscan

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.net.Uri
import android.provider.DocumentsContract
import androidx.core.content.FileProvider
import com.albus.flatscan.data.Library
import com.albus.flatscan.output.Bilevel
import com.albus.flatscan.output.Ocr
import com.albus.flatscan.output.OcrLayout
import com.albus.flatscan.output.PdfImage
import com.albus.flatscan.output.PdfInfo
import com.albus.flatscan.output.PdfPage
import com.albus.flatscan.output.PdfWriter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import kotlin.math.max
import kotlin.math.roundToInt

/** One document to export: its title, dates and pages (ids, in order). */
data class ExportDoc(val title: String, val created: Long, val modified: Long, val pages: List<String>)

/**
 * Exports a document (feature 3): a PDF of the finished pages (out.jpg) through our own [PdfWriter]
 * (searchable with an OCR text layer, 1-bit CCITT G4 for black and white, JPEG at a preset quality for
 * colour and grayscale, metadata), and / or one picture per page (JPEG, PNG; 1-bit PNG for black and
 * white). [prepare] encodes every page once; the size estimate is the exact size of the PDF those pages
 * make. Everything here does IO and image work: call it off the main thread.
 */
class Exporter(private val ctx: Context, private val lib: Library) {
    /** A page encoded for the chosen options: the PDF image, the picture file's bytes, and its text. */
    class PreparedPage(val image: PdfImage, val picture: ByteArray, val pictureExt: String, val text: Ocr.PageText?)

    class Prepared(val doc: ExportDoc, val options: OutputOptions, val pages: List<PreparedPage>) {
        /** The PDF's exact size in bytes (written to a counter), or null for picture-only exports. */
        val pdfBytes: Long? by lazy { if (options.format == OutputFormat.Jpeg || options.format == OutputFormat.Png) null else countPdf(this) }
        val pictureBytes: Long get() = if (options.format == OutputFormat.Pdf) 0 else pages.sumOf { it.picture.size.toLong() }
        val totalBytes: Long get() = (pdfBytes ?: 0) + pictureBytes
    }

    /** Encodes the pages of [doc] for [o] ([progress]: stage, 0..1). OCR runs only for a searchable PDF and reuses cached text. */
    fun prepare(doc: ExportDoc, o: OutputOptions, progress: (String, Float) -> Unit = { _, _ -> }): Prepared {
        val ocr = if (o.searchable && o.format != OutputFormat.Jpeg && o.format != OutputFormat.Png) Ocr(ctx) else null
        val out = doc.pages.mapIndexedNotNull { i, id ->
            val f = lib.store.outputFile(id)
            if (!f.exists()) return@mapIndexedNotNull null
            val text = ocr?.let {
                progress("Recognising text, page ${i + 1} of ${doc.pages.size}", i.toFloat() / doc.pages.size)
                runCatching { it.pageWords(lib, id) }.getOrNull()
            }
            progress("Preparing page ${i + 1} of ${doc.pages.size}", (i + 0.5f) / doc.pages.size)
            encodePage(f, o).let { (img, pic, ext) -> PreparedPage(img, pic, ext, text) }
        }
        progress("Done", 1f)
        return Prepared(doc, o, out)
    }

    /** The document's file name (without extension) from the options' template; [page] for one picture per page. */
    fun fileName(doc: ExportDoc, o: OutputOptions, page: Int? = null): String {
        val base = FileNameTemplate.render(o.template, doc.title, doc.created, doc.pages.size, o.dateFormat, page)
        return if (page != null && !o.template.contains("{n}")) "$base - $page" else base
    }

    fun writePdf(p: Prepared, out: OutputStream) = writePdfTo(p, out)

    /** A new file in the folder [tree] named [name] + [ext] (a taken name gets " (2)"...), with [write]'s bytes; returns its name. */
    fun saveInFolder(tree: Uri, name: String, ext: String, mime: String, write: (OutputStream) -> Unit): String {
        val cr = ctx.contentResolver
        val treeDoc = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val taken = mutableSetOf<String>()
        cr.query(DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree)),
            arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)?.use { c -> while (c.moveToNext()) taken += c.getString(0) }
        val file = FileNameTemplate.unique(name, ext) { it in taken }
        val uri = DocumentsContract.createDocument(cr, treeDoc, mime, file) ?: error("cannot create $file")
        (cr.openOutputStream(uri, "wt") ?: error("cannot open $file")).use(write)
        return file
    }

    /** Writes the export into [tree] (the default folder, or one picked now); returns what was written. */
    fun saveToFolder(p: Prepared, tree: Uri): String {
        val o = p.options
        val written = mutableListOf<String>()
        if (o.format != OutputFormat.Jpeg && o.format != OutputFormat.Png) {
            written += saveInFolder(tree, fileName(p.doc, o), "pdf", "application/pdf") { writePdf(p, it) }
        }
        if (o.format != OutputFormat.Pdf) {
            p.pages.forEachIndexed { i, pg ->
                written += saveInFolder(tree, fileName(p.doc, o, i + 1), pg.pictureExt, mimeOf(pg.pictureExt)) { it.write(pg.picture) }
            }
        }
        return if (written.size == 1) "Saved \"${written[0]}\"" else "Saved ${written.size} files"
    }

    /** The share sheet for the export (files in cacheDir/shared, read access granted to the receiving app). */
    fun shareIntent(p: Prepared): Intent {
        val dir = File(ctx.cacheDir, "shared").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val o = p.options
        val files = mutableListOf<File>()
        if (o.format != OutputFormat.Jpeg && o.format != OutputFormat.Png) {
            files += File(dir, "${fileName(p.doc, o)}.pdf").also { f -> f.outputStream().buffered().use { writePdf(p, it) } }
        }
        if (o.format != OutputFormat.Pdf) p.pages.forEachIndexed { i, pg ->
            files += File(dir, "${fileName(p.doc, o, i + 1)}.${pg.pictureExt}").also { it.writeBytes(pg.picture) }
        }
        val uris = files.map { FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", it) }
        val send = if (uris.size == 1) Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris[0])
        else Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
        send.type = if (files.all { it.extension == "pdf" }) "application/pdf" else if (files.all { it.extension != "pdf" }) "image/*" else "*/*"
        // the ClipData carries the read grant for every file to the receiving app
        send.clipData = ClipData.newRawUri("", uris[0]).also { c -> uris.drop(1).forEach { c.addItem(ClipData.Item(it)) } }
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        return Intent.createChooser(send, null)
    }

    companion object {
        /** The pixel size a page image of [w] x [h] gets at [dpi] on its PDF page (never larger than it is). */
        fun target(w: Int, h: Int, o: OutputOptions, dpi: Int): Pair<Int, Int> {
            val (pw, ph) = pageSize(w, h, o)
            val m = if (o.pageSize == PdfPageSize.Fit) 0.0 else o.margin.pt.toDouble()
            val box = PdfWriter.place(w, h, pw, ph, m)
            val tw = (box[2] / 72.0 * dpi).roundToInt()
            if (tw >= w) return w to h
            return tw.coerceAtLeast(1) to (h.toDouble() * tw / w).roundToInt().coerceAtLeast(1)
        }

        /** [src] (a finished page, JPEG) encoded for [o]: the PDF image, the picture file and its extension. */
        fun encodePage(src: File, o: OutputOptions): Triple<PdfImage, ByteArray, String> {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }.also { BitmapFactory.decodeFile(src.absolutePath, it) }
            val w0 = bounds.outWidth
            val h0 = bounds.outHeight
            val dpi = if (o.color == ColorMode.BlackWhite) o.size.bwDpi else o.size.dpi
            val (tw, th) = target(w0, h0, o, dpi)
            // the finished page as it is: no re-encoding (High, colour, already small enough)
            if (o.color == ColorMode.Color && tw == w0 && o.size == SizePreset.High && o.format != OutputFormat.Png) {
                val bytes = src.readBytes()
                return Triple(PdfImage.Jpeg(bytes, w0, h0, 3), bytes, "jpg")
            }
            var bmp = BitmapFactory.decodeFile(src.absolutePath, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })
                ?: error("cannot read $src")
            if (tw != bmp.width) bmp = Bitmap.createScaledBitmap(bmp, tw, th, true)
            return when (o.color) {
                ColorMode.BlackWhite -> {
                    val px = IntArray(bmp.width * bmp.height).also { bmp.getPixels(it, 0, bmp.width, 0, 0, bmp.width, bmp.height) }
                    val packed = Bilevel.threshold(px, bmp.width, bmp.height)
                    val g4 = Bilevel.encodeG4(packed, bmp.width, bmp.height)
                    Triple(PdfImage.Bilevel(packed, bmp.width, bmp.height, g4), Bilevel.png(packed, bmp.width, bmp.height), "png")
                }
                ColorMode.Grayscale, ColorMode.Color -> {
                    // grayscale: a one-channel JPEG from the native library (Android writes only three-channel ones; where
                    // the library is missing, as in JVM tests, an R = G = B JPEG)
                    val grayJpeg = if (o.color == ColorMode.Grayscale) runCatching { NativeLib.encodeGrayJpeg(bmp, o.size.jpegQuality) }.getOrNull() else null
                    if (o.color == ColorMode.Grayscale) bmp = gray(bmp)
                    val jpg = grayJpeg ?: ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, o.size.jpegQuality, it) }.toByteArray()
                    val picture = if (o.format == OutputFormat.Png) ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray() else jpg
                    Triple(PdfImage.Jpeg(jpg, bmp.width, bmp.height, if (grayJpeg != null) 1 else 3), picture, if (o.format == OutputFormat.Png) "png" else "jpg")
                }
            }.also { bmp.recycle() }
        }

        private fun gray(b: Bitmap): Bitmap {
            val g = Bitmap.createBitmap(b.width, b.height, Bitmap.Config.ARGB_8888)
            Canvas(g).drawBitmap(b, 0f, 0f, Paint().apply { colorFilter = ColorMatrixColorFilter(ColorMatrix().apply { setSaturation(0f) }) })
            return g
        }

        fun mimeOf(ext: String) = when (ext) { "png" -> "image/png"; "jpg" -> "image/jpeg"; else -> "application/pdf" }

        /** PDF page size for an image of [w] x [h]: the paper turned to match it, or the image's own shape (A4's long side). */
        fun pageSize(w: Int, h: Int, o: OutputOptions): Pair<Double, Double> {
            val landscape = w > h
            return when (o.pageSize) {
                PdfPageSize.Fit -> if (landscape) 842.0 to max(1.0, 842.0 * h / w) else max(1.0, 842.0 * w / h) to 842.0
                else -> if (landscape) o.pageSize.hPt.toDouble() to o.pageSize.wPt.toDouble() else o.pageSize.wPt.toDouble() to o.pageSize.hPt.toDouble()
            }
        }

        private fun writePdfTo(p: Prepared, out: OutputStream) {
            val o = p.options
            val lines = p.pages.flatMap { it.text?.lines.orEmpty() }
            val pages = p.pages.map { pg ->
                val im = pg.image
                val (pw, ph) = pageSize(im.width, im.height, o)
                val m = if (o.pageSize == PdfPageSize.Fit) 0.0 else o.margin.pt.toDouble()
                val (x, y, w, h) = PdfWriter.place(im.width, im.height, pw, ph, m).toList()
                val t = if (o.searchable) pg.text else null
                PdfPage(pw, ph, im, x, y, w, h, t?.width ?: im.width, t?.height ?: im.height, t?.lines.orEmpty())
            }
            PdfWriter(out).write(
                pages,
                PdfInfo(
                    title = p.doc.title, creator = "FlatScan", producer = "FlatScan ${BuildConfigInfo.VERSION}",
                    created = p.doc.created, modified = p.doc.modified, lang = OcrLayout.language(lines),
                ),
            )
        }

        private fun countPdf(p: Prepared): Long {
            var n = 0L
            val counter = object : OutputStream() {
                override fun write(b: Int) { n++ }
                override fun write(b: ByteArray, off: Int, len: Int) { n += len }
            }
            writePdfTo(p, counter)
            return n
        }
    }
}
