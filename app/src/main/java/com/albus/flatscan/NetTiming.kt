package com.albus.flatscan

import ai.onnxruntime.OrtEnvironment
import android.os.Build
import org.json.JSONObject
import java.io.File

/**
 * Timing of one net call (Phase 0 of docs/accel/PLAN.md, measured, never estimated): [runMs] ONNX
 * Runtime's session.run, [inputMs] making the net's input in the core (uvInput / segInput), [loadMs]
 * creating the session when this call did (0: already loaded). [shape]: the input, N x C x H x W.
 */
class NetTiming(val runMs: Double, val inputMs: Double, val loadMs: Double, val shape: String) {
    fun toJson(): JSONObject = JSONObject().put("run_ms", r1(runMs)).put("input_ms", r1(inputMs))
        .put("load_ms", r1(loadMs)).put("shape", shape).put("backend", NetEnv.BACKEND)

    companion object {
        fun msSince(t0: Long): Double = (System.nanoTime() - t0) / 1e6
        private fun r1(v: Double): Double = Math.round(v * 10) / 10.0

        /** Several calls of one net (the shadow net may run more than once) as one entry. */
        fun sum(v: List<NetTiming>): NetTiming? = if (v.isEmpty()) null else
            NetTiming(v.sumOf { it.runMs }, v.sumOf { it.inputMs }, v.sumOf { it.loadMs }, v.joinToString(", ") { it.shape })
    }
}

/** Where the nets run and on what: the device facts the acceleration work compares across phones. */
object NetEnv {
    /** Every net runs on ONNX Runtime's CPU execution provider (no provider is added; fp32, default threads). */
    const val BACKEND = "CPU"

    /** SoC: Build.SOC_MANUFACTURER / SOC_MODEL (API 31+); before that Build.HARDWARE / BOARD and cpuinfo's Hardware line. */
    val soc: String by lazy {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}"
        else {
            val hw = runCatching {
                File("/proc/cpuinfo").readLines().firstOrNull { it.startsWith("Hardware") }?.substringAfter(':')?.trim()
            }.getOrNull()
            "hardware ${Build.HARDWARE}, board ${Build.BOARD}" + (if (!hw.isNullOrEmpty()) ", cpuinfo $hw" else "") + " (API < 31: no SOC_MODEL)"
        }
    }

    val ortVersion: String by lazy { runCatching { OrtEnvironment.getEnvironment().version }.getOrDefault("unknown") }

    /** Execution providers this ONNX Runtime build offers (the nets use only CPU). */
    val providers: String by lazy { runCatching { OrtEnvironment.getAvailableProviders().joinToString(", ") }.getOrDefault("unknown") }

    fun toJson(): JSONObject = JSONObject().put("soc", soc).put("backend", BACKEND).put("ort", ortVersion)
        .put("ort_providers", providers).put("cores", Runtime.getRuntime().availableProcessors())
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}").put("api", Build.VERSION.SDK_INT)
}
