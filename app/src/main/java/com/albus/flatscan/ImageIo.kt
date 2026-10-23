package com.albus.flatscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
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
