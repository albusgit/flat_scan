package com.albus.flatscan

import android.os.Build
import org.json.JSONObject
import java.io.File

/**
 * Settings > Acceleration: the backends each group of nets tries ([NativeLib.configureNets]). Every net runs
 * on LiteRT (the TensorFlow Lite runtime, in the APK with its .tflite models): its CPU kernels are the
 * reference and always the last resort, and a net that fails on the GPU runs on the CPU for the rest of the
 * app's life. The app never depends on the GPU.
 *  * [AUTO] (the default): on first use per (SoC, app version, candidates) the self-test photo's nets run on
 *    the CPU and on the GPU when this phone can try it ([candidates]: the GPU delegate's library loads and the
 *    phone has an OpenCL library); per group the fastest backend whose outputs match the CPU's within the
 *    tolerances is kept ([Settings.autoChoice]), SyntheticDoc high-res judged on its own. A phone without
 *    OpenCL gets the CPU at once, without a benchmark;
 *  * [CPU]: LiteRT's CPU kernels (fp32; XNNPACK where it takes the ops) for every net;
 *  * [GPU]: the LiteRT GPU delegate in full precision (fp32 compute, fp16-stored weights) for UVDoc,
 *    SyntheticDoc high-res, the shadow net and the rule net on any phone with OpenCL (Mali, Adreno, Xclipse),
 *    without the CPU comparison of Auto; a net the delegate cannot take (or a shadow-net / rule-net page shape
 *    without a GPU model) runs on the CPU.
 * The page finder always runs on the CPU (PLAN: 34 NNAPI partitions). The Qualcomm NPU (ONNX Runtime QNN) and
 * NNAPI left the app with ONNX Runtime in Round 11 (docs/results/r11/runtime.md).
 */
object Accel {
    const val AUTO = "auto"
    const val CPU = "cpu"
    const val GPU = "gpu"
    val CHOICES: List<Pair<String, String>> = listOf(
        AUTO to "Auto: the fastest backend whose results match the CPU",
        CPU to "CPU (reference results)",
        GPU to "GPU (LiteRT)",
    )

    /**
     * The backends of each group of nets, comma separated ("" = CPU only): [finder] the page finder,
     * [flat] UVDoc (both passes), [hr] SyntheticDoc high-res, [shade] the shadow net, [rule] the rule net. [setting]: the
     * Settings value it came from.
     */
    data class Plan(
        val setting: String, val finder: String, val flat: String, val hr: String, val shade: String, val rule: String, val note: String = "",
    ) {
        val cpuOnly: Boolean get() = finder.isEmpty() && flat.isEmpty() && hr.isEmpty() && shade.isEmpty() && rule.isEmpty()
        private fun name(b: String) = b.ifEmpty { "cpu" }
        fun summary(): String =
            "page finder ${name(finder)}, UVDoc ${name(flat)}, SyntheticDoc high-res ${name(hr)}, shadow net ${name(shade)}, rule net ${name(rule)}"
        // part of the pages' geometry keys: the rule net only when it is not on the CPU (stored pages keep their keys)
        fun key(): String = "${name(finder)}/${name(flat)}/${name(hr)}/${name(shade)}" + if (rule.isEmpty()) "" else "/rule=$rule"
        fun toJson(): JSONObject = JSONObject().put("setting", setting).put("finder", name(finder)).put("flat", name(flat))
            .put("hr", name(hr)).put("shade", name(shade)).put("rule", name(rule)).put("note", note)
    }

    /** The plan of a fixed setting; null for [AUTO] (its plan comes from the stored benchmark). */
    fun fixed(setting: String): Plan? = when (setting) {
        CPU -> Plan(CPU, "", "", "", "", "")
        GPU -> if (gpuOff) Plan(GPU, "", "", "", "", "", "the GPU crashed the app on this phone: CPU") else Plan(GPU, "", "gpu", "gpu", "gpu", "gpu")
        else -> null
    }

    /** Auto's candidates (backend names for benchBackends): the LiteRT GPU where this phone can try it. */
    fun candidates(): String = if (NetEnv.gpuAvailable && !gpuOff) "gpu" else ""

    /** The GPU crashed the app on this phone ([CrashGuard]): the nets stay on the CPU until it is turned on again. */
    @Volatile var gpuOff = false

    /**
     * The plan of a stored Auto benchmark ([Settings.autoChoice]) if it was made on this SoC and app version
     * with today's candidates.
     */
    fun autoPlan(stored: String?): Plan? {
        val j = stored?.let { runCatching { JSONObject(it) }.getOrNull() } ?: return null
        if (j.optString("soc") != NetEnv.soc || j.optString("app") != BuildConfigInfo.VERSION) return null
        if (j.optString("candidates") != candidates()) return null
        val c = j.optJSONObject("bench")?.optJSONObject("choice") ?: return null
        fun b(k: String) = c.optString(k, "cpu").let { if (it == "cpu") "" else it }
        // a benchmark from before SyntheticDoc high-res / the rule net had their own groups: run it again
        if (!c.has("hr") || !c.has("rule")) return null
        return Plan(AUTO, b("finder"), b("flat"), b("hr"), b("shade"), b("rule"), j.optString("note").ifEmpty { "benchmarked ${j.optString("at")}" })
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

    private val rt: JSONObject by lazy { runCatching { JSONObject(NativeLib.runtimeInfo()) }.getOrDefault(JSONObject()) }

    /** The nets' CPU runtime ("LiteRT 2.20.0-dev0+selfbuilt": LiteRT 1.4.0's libraries). */
    val runtime: String get() = rt.optString("cpu", "unknown")

    /** Whether the LiteRT GPU delegate can be tried here (its library loads and the phone has an OpenCL library). */
    val gpuAvailable: Boolean get() = rt.optBoolean("gpu", false)

    /** Why not, when [gpuAvailable] is false. */
    val gpuWhy: String get() = rt.optString("gpu_why")

    /** The processing details' / self-test's device facts; backend: the groups' backends. */
    fun toJson(plan: Accel.Plan): JSONObject = JSONObject().put("soc", soc).put("backend", plan.summary()).put("acceleration", plan.toJson())
        .put("runtime", runtime).put("gpu", if (gpuAvailable) "available" else "unavailable: $gpuWhy")
        .put("cores", Runtime.getRuntime().availableProcessors())
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}").put("api", Build.VERSION.SDK_INT)
}
