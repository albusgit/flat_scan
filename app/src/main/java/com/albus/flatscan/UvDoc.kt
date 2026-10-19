package com.albus.flatscan

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * UVDoc (Verhoeven et al. 2023, github.com/tanguymagne/UVDoc, MIT): the flattening net
 * (assets/models/uvdoc.onnx: the 2D grid head, fp16 weights, fp32 compute) on ONNX Runtime. The
 * core picks the part of the photo it sees ([NativeLib.uvCrop]), makes its input
 * ([NativeLib.uvInput]) and uses its 45 x 31 grid as stage 1 of the flattening when the grid
 * agrees with the page finder (core/src/uvdoc.cpp). [grid] returns null when the model is
 * missing or cannot run: the page is then flattened from its outline as before.
 */
class UvDoc(private val ctx: Context) {
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
            Log.e(ScanViewModel.TAG, "UVDoc unavailable ($MODEL)", e)
            failed = true
            null
        }
    }

    /** The grid (float[2 * 45 * 31]) for [crop] (NativeLib.uvCrop) of [photo], or null. */
    @Synchronized
    fun grid(photo: Bitmap, crop: IntArray): FloatArray? {
        val s = session() ?: return null
        val x = NativeLib.uvInput(photo, crop) ?: return null
        val env = OrtEnvironment.getEnvironment()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, IN_H.toLong(), IN_W.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                if (fb.remaining() != 2 * GRID_W * GRID_H) return null
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
    }

    companion object {
        const val MODEL = "models/uvdoc.onnx"
        const val IN_W = 488
        const val IN_H = 712
        const val GRID_W = 31
        const val GRID_H = 45
    }
}
