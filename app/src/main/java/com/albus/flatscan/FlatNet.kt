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
 *  * [SYNTHDOC]: SyntheticDoc high-res (ECCV 2026, github.com/tanguymagne/SyntheticDoc, code MIT), the
 *    same net retrained on 1M simulated pages at 2x the size, assets/models/syntheticdoc_hr.onnx,
 *    1024 x 1440 in, a 121 x 177 grid out (one cell ~13 px of a page: creases and folds). Its shading
 *    map is not used: it bleaches colour; shadows come from [ShadeNet].
 * Both are the 2D heads with fp16 weights / fp32 compute. The core picks the part of the photo the net
 * sees ([NativeLib.uvCrop]), makes its input ([NativeLib.uvInput]) and checks the grid against the
 * page finder. [run] returns null when the model is missing or cannot run.
 */
class FlatNet(private val ctx: Context, val model: Int) {
    private var session: OrtSession? = null
    private var failed = false
    private val file = if (model == SYNTHDOC) "models/syntheticdoc_hr.onnx" else "models/uvdoc.onnx"
    private val inW = if (model == SYNTHDOC) 1024 else 488
    private val inH = if (model == SYNTHDOC) 1440 else 712
    private val gridPoints = if (model == SYNTHDOC) 121 * 177 else 45 * 31

    /**
     * The net's grid (float[2 * gh * gw]) and, for SyntheticDoc, its shading map (float[3 * inH * inW]);
     * [ms]: how long this call took.
     */
    class Output(val grid: FloatArray, val shading: FloatArray?, val ms: NetTiming)

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
        val t0 = System.nanoTime()
        val loaded = session != null
        val s = session() ?: return null
        val loadMs = if (loaded) 0.0 else NetTiming.msSince(t0)
        val t1 = System.nanoTime()
        val x = NativeLib.uvInput(photo, crop, logits, pageIndex) ?: return null
        if (x.size != 3 * inW * inH) return null
        val inputMs = NetTiming.msSince(t1)
        val env = OrtEnvironment.getEnvironment()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, inH.toLong(), inW.toLong())).use { t ->
            val t2 = System.nanoTime()
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val runMs = NetTiming.msSince(t2)
                val g = (r.get(0) as OnnxTensor).floatBuffer
                if (g.remaining() != 2 * gridPoints) return null
                val grid = FloatArray(g.remaining()).also { g.get(it) }
                Output(grid, null, NetTiming(runMs, inputMs, loadMs, "1x3x${inH}x$inW"))
            }
        }
    }

    companion object {
        const val UVDOC = 0
        const val SYNTHDOC = 2   // the core's model id (uv_spec): SyntheticDoc high-res
    }
}
