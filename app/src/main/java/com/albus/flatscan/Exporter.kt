package com.albus.flatscan

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.provider.DocumentsContract
import androidx.core.content.FileProvider
import java.io.File
import java.io.OutputStream
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt

enum class ExportFormat(val mime: String, val ext: String) { JPG("image/jpeg", "jpg"), PDF("application/pdf", "pdf") }

/** Writes the pages of a document (their rendered out.jpg files, in order). Call off the main thread. */
object Exporter {
    /**
     * One PDF page per scan ([o]: paper size, margin, image resolution; portrait or landscape to match
     * the page). The image keeps its proportions, centred in the margins; an image within 1% of the
     * box's shape fills it exactly (the default A4 page: what earlier versions wrote).
     */
    fun writePdf(pages: List<File>, out: OutputStream, o: PdfOptions = PdfOptions()) {
        val doc = PdfDocument()
        try {
            val paint = Paint(Paint.FILTER_BITMAP_FLAG)
            pages.forEachIndexed { i, f ->
                // one decoded page in memory at a time
                var bmp = BitmapFactory.decodeFile(f.absolutePath) ?: error("cannot read $f")
                val landscape = bmp.width > bmp.height
                val (pw, ph) = when (o.pageSize) {
                    PdfPageSize.Fit -> {
                        // the page's own shape, its long side as long as A4's
                        val long = 842.0
                        if (landscape) long.toInt() to (long * bmp.height / bmp.width).roundToInt().coerceAtLeast(1)
                        else (long * bmp.width / bmp.height).roundToInt().coerceAtLeast(1) to long.toInt()
                    }
                    else -> if (landscape) o.pageSize.hPt to o.pageSize.wPt else o.pageSize.wPt to o.pageSize.hPt
                }
                val m = if (o.pageSize == PdfPageSize.Fit) 0 else o.margin.pt
                val bw = (pw - 2 * m).coerceAtLeast(1)
                val bh = (ph - 2 * m).coerceAtLeast(1)
                val k = min(bw.toDouble() / bmp.width, bh.toDouble() / bmp.height)
                val dst = if (abs(bmp.width.toDouble() / bmp.height / (bw.toDouble() / bh) - 1) < 0.01) Rect(m, m, m + bw, m + bh)
                else {
                    val dw = (bmp.width * k).roundToInt()
                    val dh = (bmp.height * k).roundToInt()
                    Rect((pw - dw) / 2, (ph - dh) / 2, (pw - dw) / 2 + dw, (ph - dh) / 2 + dh)
                }
                if (o.imageSize.maxDpi > 0) {
                    // at most maxDpi on the PDF page (72 pt per inch)
                    val wpx = (dst.width() / 72.0 * o.imageSize.maxDpi).roundToInt()
                    if (wpx in 1 until bmp.width) {
                        val hpx = (bmp.height.toDouble() * wpx / bmp.width).roundToInt().coerceAtLeast(1)
                        val small = Bitmap.createScaledBitmap(bmp, wpx, hpx, true)
                        bmp.recycle()
                        bmp = small
                    }
                }
                val page = doc.startPage(PdfDocument.PageInfo.Builder(pw, ph, i + 1).create())
                page.canvas.drawBitmap(bmp, null, dst, paint)
                doc.finishPage(page)
                bmp.recycle()
            }
            doc.writeTo(out)
        } finally {
            doc.close()
        }
    }

    fun savePdf(ctx: Context, pages: List<File>, uri: Uri, o: PdfOptions = PdfOptions()) {
        ctx.contentResolver.openOutputStream(uri)?.use { writePdf(pages, it, o) } ?: error("cannot open $uri")
    }

    /** A single page to a file the user created. */
    fun saveJpg(ctx: Context, page: File, uri: Uri) {
        ctx.contentResolver.openOutputStream(uri)?.use { o -> page.inputStream().use { it.copyTo(o) } } ?: error("cannot open $uri")
    }

    /**
     * One JPG per page in a new folder [baseName] inside the folder the user picked
     * (OpenDocumentTree); falls back to the picked folder itself. Returns the number written.
     */
    fun saveJpgsToFolder(ctx: Context, pages: List<File>, tree: Uri, baseName: String): Int {
        val cr = ctx.contentResolver
        val treeDoc = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val parent = runCatching {
            DocumentsContract.createDocument(cr, treeDoc, DocumentsContract.Document.MIME_TYPE_DIR, baseName)
        }.getOrNull() ?: treeDoc
        var n = 0
        pages.forEachIndexed { i, f ->
            val name = if (pages.size == 1) baseName else "$baseName - ${i + 1}"
            val doc = DocumentsContract.createDocument(cr, parent, ExportFormat.JPG.mime, name) ?: error("cannot create $name")
            cr.openOutputStream(doc)?.use { o -> f.inputStream().use { it.copyTo(o) } } ?: error("cannot open $doc")
            n++
        }
        return n
    }

    /** Builds the share intent: one PDF, or the JPGs (ACTION_SEND for one, ACTION_SEND_MULTIPLE for several). */
    fun shareIntent(ctx: Context, pages: List<File>, format: ExportFormat, baseName: String, o: PdfOptions = PdfOptions()): Intent {
        val dir = File(ctx.cacheDir, "shared").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val files = when (format) {
            ExportFormat.PDF -> listOf(File(dir, "$baseName.pdf").also { f -> f.outputStream().buffered().use { writePdf(pages, it, o) } })
            ExportFormat.JPG -> pages.mapIndexed { i, src ->
                val name = if (pages.size == 1) "$baseName.jpg" else "$baseName - ${i + 1}.jpg"
                File(dir, name).also { src.copyTo(it, overwrite = true) }
            }
        }
        val uris = files.map { FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", it) }
        val send = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris[0])
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
        }
        send.type = format.mime
        // the ClipData carries the read grant for every file to the receiving app
        send.clipData = ClipData.newRawUri("", uris[0]).also { c -> uris.drop(1).forEach { c.addItem(ClipData.Item(it)) } }
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        return Intent.createChooser(send, null)
    }
}
