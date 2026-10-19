package com.albus.flatscan

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File
import java.io.OutputStream

enum class ExportFormat(val mime: String, val ext: String) { JPG("image/jpeg", "jpg"), PDF("application/pdf", "pdf") }

object Exporter {
    fun write(bitmap: Bitmap, format: ExportFormat, out: OutputStream) {
        when (format) {
            ExportFormat.JPG -> bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
            ExportFormat.PDF -> writePdf(bitmap, out)
        }
    }

    /** One A4 page (portrait or landscape to match the scan); the image fills the page. */
    private fun writePdf(bitmap: Bitmap, out: OutputStream) {
        val landscape = bitmap.width > bitmap.height
        val pw = if (landscape) 842 else 595
        val ph = if (landscape) 595 else 842
        val doc = PdfDocument()
        val page = doc.startPage(PdfDocument.PageInfo.Builder(pw, ph, 1).create())
        page.canvas.drawBitmap(bitmap, null, Rect(0, 0, pw, ph), Paint(Paint.FILTER_BITMAP_FLAG))
        doc.finishPage(page)
        doc.writeTo(out)
        doc.close()
    }

    fun save(ctx: Context, bitmap: Bitmap, format: ExportFormat, uri: Uri) {
        ctx.contentResolver.openOutputStream(uri)?.use { write(bitmap, format, it) } ?: error("cannot open $uri")
    }

    fun share(ctx: Context, bitmap: Bitmap, format: ExportFormat) {
        val dir = File(ctx.cacheDir, "shared").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val file = File(dir, "scan_${System.currentTimeMillis()}.${format.ext}")
        file.outputStream().use { write(bitmap, format, it) }
        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = format.mime
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        ctx.startActivity(Intent.createChooser(send, null).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }
}
