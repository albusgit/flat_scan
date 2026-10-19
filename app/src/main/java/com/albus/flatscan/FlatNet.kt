package com.albus.flatscan

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * A page-flattening net on ONNX Runtime (core/src/uvdoc.h):
 *  * [UVDOC]: UVDoc (Verhoeven et al. 2023, github.com/tanguymagne/UVDoc, MIT), assets/models/uvdoc.onnx,
 *    488 x 712 in, a 45 x 31 grid out;
 *  * [SYNTHDOC]: SyntheticDoc (ECCV 2026, github.com/tanguymagne/SyntheticDoc, code MIT), the same net
 *    retrained on simulated pages, assets/models/syntheticdoc.onnx, 512 x 720 in, a 31 x 45 grid and a
 *    3 x 720 x 512 shading map (shadow removal) out.
 * Both are the 2D heads with fp16 weights / fp32 compute. The core picks the part of the photo the net
 * sees ([NativeLib.uvCrop]), makes its input ([NativeLib.uvInput]) and checks the grid against the
 * page finder. [run] returns null when the model is missing or cannot run.
 */
class FlatNet(private val ctx: Context, val model: Int) {
    private var session: OrtSession? = null
    private var failed = false
    private val file = if (model == SYNTHDOC) "models/syntheticdoc.onnx" else "models/uvdoc.onnx"
    private val inW = if (model == SYNTHDOC) 512 else 488
    private val inH = if (model == SYNTHDOC) 720 else 712

    /** The net's grid (float[2 * gh * gw]) and, for SyntheticDoc, its shading map (float[3 * inH * inW]). */
    class Output(val grid: FloatArray, val shading: FloatArray?)

    @Synchronized
    private fun session(): OrtSession? {
        session?.let { return it }
        if (failed) return null
        return try {
            val bytes = ctx.assets.open(file).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT) }
            OrtEnvironment.getEnvironment().createSession(bytes, opts).also { session = it }
        } catch (e: Exception) {
            Log.e(ScanViewModel.TAG, "flattening net unavailable ($file)", e)
            failed = true
            null
        }
    }

    /**
     * The net on [crop] (NativeLib.uvCrop with this [model]) of [photo], or null; [logits] /
     * [pageIndex]: the page finder's output, to hide the other page of a book spread.
     */
    @Synchronized
    fun run(photo: Bitmap, crop: IntArray, logits: FloatArray?, pageIndex: Int): Output? {
        val s = session() ?: return null
        val x = NativeLib.uvInput(photo, crop, logits, pageIndex) ?: return null
        if (x.size != 3 * inW * inH) return null
        val env = OrtEnvironment.getEnvironment()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, inH.toLong(), inW.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val g = (r.get(0) as OnnxTensor).floatBuffer
                if (g.remaining() != 2 * GRID_POINTS) return null
                val grid = FloatArray(g.remaining()).also { g.get(it) }
                val shading = if (model == SYNTHDOC && r.size() > 1) {
                    val b = (r.get(1) as OnnxTensor).floatBuffer
                    if (b.remaining() == 3 * inW * inH) FloatArray(b.remaining()).also { b.get(it) } else null
                } else {
                    null
                }
                Output(grid, shading)
            }
        }
    }

    companion object {
        const val UVDOC = 0
        const val SYNTHDOC = 1
        const val GRID_POINTS = 45 * 31
    }
}
