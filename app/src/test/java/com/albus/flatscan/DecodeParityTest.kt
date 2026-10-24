package com.albus.flatscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.exifinterface.media.ExifInterface
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer

/**
 * "Re-editing corners on an old scan gives the same result as on a fresh scan": the pipeline is
 * deterministic for the same input pixels (README, parity sections), so the requirement reduces to a
 * fresh scan and a re-edit handing the core the same bitmap. Both now decode the stored original with
 * [loadPhotoFile]. Real JPEG decoding (Robolectric native graphics).
 */
@RunWith(AndroidJUnit4::class)
class DecodeParityTest {
    private val ctx: Context = ApplicationProvider.getApplicationContext()

    private fun photoJpeg(w: Int, h: Int, orientation: Int? = null): ByteArray {
        val b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) for (x in 0 until w) b.setPixel(x, y, Color.rgb((x * 7 + y) % 256, (y * 3) % 256, (x * y) % 256))
        val out = ByteArrayOutputStream()
        b.compress(Bitmap.CompressFormat.JPEG, 92, out)
        if (orientation == null) return out.toByteArray()
        val f = File.createTempFile("exif", ".jpg", ctx.cacheDir)
        f.writeBytes(out.toByteArray())
        ExifInterface(f).apply { setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString()); saveAttributes() }
        return f.readBytes()
    }

    private fun pixels(b: Bitmap): ByteArray = ByteBuffer.allocate(b.byteCount).also { b.copyPixelsToBuffer(it) }.array()

    @Test
    fun freshScanAndReEditDecodeTheSamePixels() {
        val store = DocumentStore(File(ctx.filesDir, "parity_${System.nanoTime()}"))
        val bytes = photoJpeg(320, 240, ExifInterface.ORIENTATION_ROTATE_90)
        val f = store.saveOriginal("p", "jpg", bytes.inputStream())
        assertTrue(f.readBytes().contentEquals(bytes))            // kept byte for byte
        val fresh = loadPhotoFile(f).bitmap                        // what processing gets now
        val later = store.loadSource("p").bitmap                   // what a re-edit gets
        assertEquals(240, fresh.width)                             // EXIF turn applied
        assertEquals(320, fresh.height)
        assertTrue(pixels(fresh).contentEquals(pixels(later)))
    }

    @Test
    fun theOldReEncodedSourceWasNotTheSame() {
        // before history: the page was re-processed from the decoded photo re-encoded at q95
        val bytes = photoJpeg(320, 240)
        val f = File.createTempFile("orig", ".jpg", ctx.cacheDir).apply { writeBytes(bytes) }
        val fresh = loadPhotoFile(f).bitmap
        val out = ByteArrayOutputStream()
        fresh.compress(Bitmap.CompressFormat.JPEG, 95, out)
        val legacy = android.graphics.BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
        assertFalse(pixels(fresh).contentEquals(pixels(legacy.copy(Bitmap.Config.ARGB_8888, false))))
    }
}
