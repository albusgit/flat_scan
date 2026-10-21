package com.albus.flatscan

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * The rule net (RuleNet, tools/rule_net, trained here on synthetic forms), assets/models/rulenet.onnx:
 * a 0.25 M-parameter U-Net that marks the printed straight rules of the page the flattening nets
 * produced, so the line refinement follows table rules and not rows of text, letters, curves or the
 * paper's edge. The core calls [rules] in the line stage with the page at half size
 * (PipelineOptions::rule_net).
 */
class RuleNet(private val ctx: Context) : RuleNetRunner {
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
            Log.e(ScanViewModel.TAG, "rule net unavailable ($FILE)", e)
            failed = true
            null
        }
    }

    @Synchronized
    override fun rules(input: FloatArray, width: Int, height: Int): FloatArray? {
        val s = session() ?: return null
        if (input.size != 3 * width * height) return null
        return try {
            val env = OrtEnvironment.getEnvironment()
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, height.toLong(), width.toLong())).use { t ->
                s.run(mapOf(s.inputNames.first() to t)).use { r ->
                    val b = (r.get(0) as OnnxTensor).floatBuffer
                    if (b.remaining() != 2 * width * height) null else FloatArray(b.remaining()).also { b.get(it) }
                }
            }
        } catch (e: Exception) {
            Log.e(ScanViewModel.TAG, "rule net failed", e)
            null
        }
    }

    companion object {
        const val FILE = "models/rulenet.onnx"
    }
}
