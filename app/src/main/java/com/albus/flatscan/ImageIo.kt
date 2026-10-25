package com.albus.flatscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream
import kotlin.math.hypot
import kotlin.math.max

class LoadedPhoto(val bitmap: Bitmap, val focalPx: Float)

/** Largest photo side kept in memory; the final resample samples this image. */
const val MAX_PHOTO_SIDE = 4096

/**
 * Decodes a photo upright (EXIF orientation applied) as ARGB_8888, at most [MAX_PHOTO_SIDE]
 * on the long side. Focal length in pixels comes from EXIF (35 mm equivalent, by the frame
 * diagonal), else from [focalMmPerLongSide] (camera sensor data), else 0 (the core then uses
 * the reference default 0.72 x long side).
 */
fun loadPhoto(ctx: Context, uri: Uri, focalMmPerLongSide: Float = 0f): LoadedPhoto =
    decodePhoto({ ctx.contentResolver.openInputStream(uri) ?: error("cannot open $uri") }, focalMmPerLongSide)

/**
 * A stored original ([DocumentStore.saveOriginal]) decoded as [loadPhoto] decodes it: a new scan and a
 * later re-edit of the same page start from the same pixels.
 */
fun loadPhotoFile(f: File, focalMmPerLongSide: Float = 0f): LoadedPhoto = decodePhoto({ f.inputStream() }, focalMmPerLongSide)

private fun decodePhoto(open: () -> InputStream, focalMmPerLongSide: Float): LoadedPhoto {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    open().use { BitmapFactory.decodeStream(it, null, bounds) }
    var sample = 1
    while (max(bounds.outWidth, bounds.outHeight) / sample > MAX_PHOTO_SIDE) sample *= 2
    val opts = BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    var bmp = open().use { BitmapFactory.decodeStream(it, null, opts) }
        ?: error("cannot decode image")
    var focal = 0f
    open().use { s ->
        val exif = ExifInterface(s)
        val m = Matrix()
        when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        if (!m.isIdentity) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        val f35 = exif.getAttributeInt(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, 0)
        if (f35 > 0) focal = (f35 * hypot(bmp.width.toDouble(), bmp.height.toDouble()) / 43.2666).toFloat()
    }
    if (focal == 0f && focalMmPerLongSide > 0f) focal = focalMmPerLongSide * max(bmp.width, bmp.height)
    if (bmp.config != Bitmap.Config.ARGB_8888) bmp = bmp.copy(Bitmap.Config.ARGB_8888, false)
    return LoadedPhoto(bmp, focal)
}

fun Bitmap.rotatedQuarterTurns(q: Int): Bitmap {
    val n = ((q % 4) + 4) % 4
    if (n == 0) return this
    val m = Matrix().apply { postRotate(90f * n) }
    return Bitmap.createBitmap(this, 0, 0, width, height, m, true)
}

/** A copy no larger than [maxSide] for display. */
fun Bitmap.forDisplay(maxSide: Int = 2048): Bitmap {
    val s = maxSide.toFloat() / max(width, height)
    if (s >= 1f) return this
    return Bitmap.createScaledBitmap(this, (width * s).toInt().coerceAtLeast(1), (height * s).toInt().coerceAtLeast(1), true)
}

/** The file extension of an image [uri] (from its MIME type, else its path), "jpg" when unknown. */
fun extensionOf(ctx: Context, uri: Uri): String {
    val mime = runCatching { ctx.contentResolver.getType(uri) }.getOrNull()
    val fromMime = mime?.let { android.webkit.MimeTypeMap.getSingleton().getExtensionFromMimeType(it) }
    val fromPath = uri.lastPathSegment?.substringAfterLast('.', "")?.takeIf { it.length in 2..5 }
    return (fromMime ?: fromPath ?: "jpg").lowercase().let { if (it == "jpeg") "jpg" else it }
}

/** 1536 -> "1.5 KB" (base 1024, one decimal below 10). */
fun formatBytes(n: Long): String {
    if (n < 1024) return "$n B"
    val units = listOf("KB", "MB", "GB", "TB")
    var v = n / 1024.0
    var i = 0
    while (v >= 1024 && i < units.lastIndex) { v /= 1024; i++ }
    return if (v < 10) "%.1f %s".format(java.util.Locale.US, v, units[i]) else "%.0f %s".format(java.util.Locale.US, v, units[i])
}

/**
 * The pages of a PDF ([uri]) as PNG files in cacheDir/pdf_import/ (the folder is emptied first), white
 * background, at 200 dpi with the long side capped at 3508 px (A4 at 300 dpi). For importing PDF pages.
 * [onCount] gets the files (page order) before the first page is rendered, [onPage] a page's index once its
 * file is written. The pages render one after the other (PdfRenderer), and up to [encoders] are PNG-encoded at
 * a time beside that (Android's encoder is single-threaded; the same bytes as one at a time).
 */
suspend fun renderPdfPages(
    ctx: Context,
    uri: Uri,
    encoders: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 3),
    onCount: (List<File>) -> Unit = {},
    onPage: (Int) -> Unit = {},
): List<File> = withContext(Dispatchers.IO) {
    val dir = File(ctx.cacheDir, "pdf_import").apply { deleteRecursively(); mkdirs() }
    (ctx.contentResolver.openFileDescriptor(uri, "r") ?: error("cannot open $uri")).use { fd ->
        android.graphics.pdf.PdfRenderer(fd).use { r ->
            val files = List(r.pageCount) { File(dir, "page_%04d.png".format(java.util.Locale.US, it + 1)) }
            onCount(files)
            // a permit per page in memory, from its render to the end of its encode
            val permits = Semaphore(encoders)
            coroutineScope {
                for (i in files.indices) {
                    permits.acquire()
                    val b = try {
                        r.openPage(i).use { page ->
                            val k = minOf(200f / 72f, 3508f / max(page.width, page.height))
                            val w = (page.width * k).toInt().coerceAtLeast(1)
                            val h = (page.height * k).toInt().coerceAtLeast(1)
                            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
                                it.eraseColor(android.graphics.Color.WHITE)
                                page.render(it, null, null, android.graphics.pdf.PdfRenderer.Page.RENDER_MODE_FOR_PRINT)
                            }
                        }
                    } catch (e: Throwable) {
                        permits.release()
                        throw e
                    }
                    launch(Dispatchers.Default) {
                        try {
                            files[i].outputStream().buffered().use { b.compress(Bitmap.CompressFormat.PNG, 100, it) }
                        } finally {
                            b.recycle()
                            permits.release()
                        }
                        onPage(i)
                    }
                }
            }
            files
        }
    }
}
