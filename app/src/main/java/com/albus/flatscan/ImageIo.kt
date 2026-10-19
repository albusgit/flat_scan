package com.albus.flatscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
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
fun loadPhoto(ctx: Context, uri: Uri, focalMmPerLongSide: Float = 0f): LoadedPhoto {
    val cr = ctx.contentResolver
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
    var sample = 1
    while (max(bounds.outWidth, bounds.outHeight) / sample > MAX_PHOTO_SIDE) sample *= 2
    val opts = BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    var bmp = cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, opts) }
        ?: error("cannot decode image")
    var focal = 0f
    cr.openInputStream(uri)?.use { s ->
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
