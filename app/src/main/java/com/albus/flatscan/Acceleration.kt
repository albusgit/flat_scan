package com.albus.flatscan

import android.os.Build
import org.json.JSONObject
import java.io.File

/**
 * Settings > Acceleration (docs/accel/PLAN.md, Stream C step 4): the backends each group of nets tries
 * ([NativeLib.configureNets]; the CPU is always the last resort, and a net that fails on an accelerator
 * runs on the CPU for the rest of the app's life). CPU is the default: an accelerated path changes the
 * nets' outputs within the tolerances of jni_bridge.cpp (benchBackends) and is opt-in.
 *  * [AUTO]: on first use per (SoC, app version) the self-test photo's nets run on the CPU and on every
 *    accelerator this ONNX Runtime build has except NNAPI ([candidates]); per group the fastest backend whose outputs
 *    match the CPU's within the tolerances is kept ([Settings.autoChoice]);
 *  * [NPU]: Qualcomm QNN HTP (fp16) for the flattening nets and the shadow net;
 *  * [NNAPI]: Android NNAPI (fp16) for the same nets; not in the QNN build's ONNX Runtime (arm64-v8a
 *    built with -Port=qnn, the default), where it falls back to the CPU.
 * The page finder always runs on the CPU (PLAN: 34 NNAPI partitions).
 */
object Accel {
    const val AUTO = "auto"
    const val CPU = "cpu"
    const val NPU = "npu"
    const val NNAPI = "nnapi"
    val CHOICES = listOf(
        AUTO to "Auto: the fastest backend whose results match the CPU",
        CPU to "CPU (reference results)",
        NPU to "NPU (Qualcomm QNN)",
        NNAPI to "NNAPI (experimental)",
    )

    /**
     * The backends of each group of nets, comma separated ("" = CPU only): [finder] the page finder,
     * [flat] the flattening nets, [shade] the shadow net. [setting]: the Settings value it came from.
     */
    data class Plan(val setting: String, val finder: String, val flat: String, val shade: String, val note: String = "") {
        val cpuOnly: Boolean get() = finder.isEmpty() && flat.isEmpty() && shade.isEmpty()
        private fun name(b: String) = b.ifEmpty { "cpu" }
        fun summary(): String = "page finder ${name(finder)}, flattening nets ${name(flat)}, shadow net ${name(shade)}"
        fun toJson(): JSONObject = JSONObject().put("setting", setting).put("finder", name(finder)).put("flat", name(flat))
            .put("shade", name(shade)).put("note", note)
    }

    /** The plan of a fixed setting; null for [AUTO] (its plan comes from the stored benchmark). */
    fun fixed(setting: String): Plan? = when (setting) {
        CPU -> Plan(CPU, "", "", "")
        NPU -> Plan(NPU, "", "qnn", "qnn")
        NNAPI -> Plan(NNAPI, "", "nnapi", "nnapi")
        else -> null
    }

    /**
     * Auto's candidates: the accelerators this ONNX Runtime build has (backend names for benchBackends).
     * Not NNAPI: on the x86_64 emulator (nnapi-reference) ORT 1.22.0's NNAPI EP aborted the process inside
     * Run (std::terminate, which no fallback can catch), and NNAPI in Auto is PLAN Stream G (not started).
     * NNAPI stays selectable by hand as experimental.
     */
    fun candidates(): String = listOfNotNull(
        "qnn".takeIf { NetEnv.has("QNNExecutionProvider") },
        "xnnpack".takeIf { NetEnv.has("XnnpackExecutionProvider") },
    ).joinToString(",")

    /** The plan of a stored Auto benchmark ([Settings.autoChoice]) if it was made on this SoC and app version. */
    fun autoPlan(stored: String?): Plan? {
        val j = stored?.let { runCatching { JSONObject(it) }.getOrNull() } ?: return null
        if (j.optString("soc") != NetEnv.soc || j.optString("app") != BuildConfigInfo.VERSION) return null
        val c = j.optJSONObject("bench")?.optJSONObject("choice") ?: return null
        fun b(k: String) = c.optString(k, "cpu").let { if (it == "cpu") "" else it }
        return Plan(AUTO, b("finder"), b("flat"), b("shade"), "benchmarked ${j.optString("at")}")
    }
}

/** Where the nets run and on what: the device facts the acceleration work compares across phones. */
object NetEnv {
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

    private val ort: JSONObject by lazy { runCatching { JSONObject(NativeLib.ortInfo()) }.getOrDefault(JSONObject()) }

    /** The ONNX Runtime linked into libflatscan. */
    val ortVersion: String get() = ort.optString("ort", "unknown")

    /** Execution providers this ONNX Runtime build has (QNNExecutionProvider, NnapiExecutionProvider, ...). */
    val providers: List<String> by lazy { ort.optJSONArray("providers")?.let { a -> List(a.length()) { a.optString(it) } } ?: emptyList() }

    fun has(provider: String) = provider in providers

    /** The processing details' / self-test's device facts; backend: the groups' backends. */
    fun toJson(plan: Accel.Plan): JSONObject = JSONObject().put("soc", soc).put("backend", plan.summary()).put("acceleration", plan.toJson())
        .put("ort", ortVersion).put("ort_providers", providers.joinToString(", ")).put("cores", Runtime.getRuntime().availableProcessors())
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}").put("api", Build.VERSION.SDK_INT)
}
