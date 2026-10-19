package com.albus.flatscan

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * The page-finder net (assets/models/pagefinder.onnx, seg/segnet.py) on ONNX Runtime. The core
 * makes its input ([NativeLib.segInput]) and post-processes its logits (core/src/pagefind.cpp).
 * [logits] returns null when the model is missing or cannot run: the caller then uses the
 * colour-threshold detector.
 */
class PageFinder(private val ctx: Context) {
    private var session: OrtSession? = null
    private var failed = false

    @Synchronized
    private fun session(): OrtSession? {
        session?.let { return it }
        if (failed) return null
        return try {
            val bytes = ctx.assets.open(MODEL).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT) }
            OrtEnvironment.getEnvironment().createSession(bytes, opts).also { session = it }
        } catch (e: Exception) {
            Log.e(ScanViewModel.TAG, "page finder unavailable ($MODEL)", e)
            failed = true
            null
        }
    }

    /** The net's logits for [photo] (float[2 * 256 * 256]: page paper, book gutter), or null. */
    @Synchronized
    fun logits(photo: Bitmap): FloatArray? {
        val s = session() ?: return null
        val x = NativeLib.segInput(photo) ?: return null
        val env = OrtEnvironment.getEnvironment()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                if (fb.remaining() != 2 * SIZE * SIZE) return null
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
    }

    companion object {
        const val MODEL = "models/pagefinder.onnx"
        const val SIZE = 256
    }
}
