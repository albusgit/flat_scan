package com.albus.flatscan

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * The shadow net (ShadeNet, tools/shadow_net, trained here on synthetic shadows over coloured pages),
 * assets/models/shadenet.onnx: a 0.5 M-parameter U-Net that predicts a flat page's shading map, so the
 * page divided by it loses hand, object and leaf shadows but keeps colour blocks, tints and pictures.
 * The core calls [shade] during processing with the flattened page (PipelineOptions::shade_net).
 */
class ShadeNet(private val ctx: Context) : ShadeNetRunner {
    private var session: OrtSession? = null
    private var failed = false

    @Synchronized
    private fun session(): OrtSession? {
        session?.let { return it }
        if (failed) return null
        return try {
            val bytes = ctx.assets.open(FILE).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT) }
            OrtEnvironment.getEnvironment().createSession(bytes, opts).also { session = it }
        } catch (e: Exception) {
            Log.e(ScanViewModel.TAG, "shadow net unavailable ($FILE)", e)
            failed = true
            null
        }
    }

    @Synchronized
    override fun shade(input: FloatArray, width: Int, height: Int): FloatArray? {
        val s = session() ?: return null
        if (input.size != 3 * width * height) return null
        return try {
            val env = OrtEnvironment.getEnvironment()
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, height.toLong(), width.toLong())).use { t ->
                s.run(mapOf(s.inputNames.first() to t)).use { r ->
                    val b = (r.get(0) as OnnxTensor).floatBuffer
                    if (b.remaining() != input.size) null else FloatArray(b.remaining()).also { b.get(it) }
                }
            }
        } catch (e: Exception) {
            Log.e(ScanViewModel.TAG, "shadow net failed", e)
            null
        }
    }

    companion object {
        const val FILE = "models/shadenet.onnx"
    }
}
