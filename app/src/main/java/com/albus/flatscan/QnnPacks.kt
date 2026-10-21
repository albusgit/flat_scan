package com.albus.flatscan

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File

/**
 * The Qualcomm NPU libraries as on-demand asset packs (app/build.gradle.kts): qnn_htp (libQnnHtp,
 * libQnnHtpPrepare, libQnnSystem, the fixed-shape models) and one pack per Hexagon generation (qnn_v68 ..
 * qnn_v79: its DSP skeleton and CPU stub). The base has none of them, so a phone without a Snapdragon never
 * downloads them, and a Snapdragon downloads only its own generation, when the NPU is first used
 * (Settings > Acceleration NPU or Auto). Sources, in order:
 *  * Play Asset Delivery (Play Store installs, and `bundletool install-apks` of a --local-testing build);
 *  * a universal APK (bundletool --mode=universal) carries the packs' assets inside the APK: they are
 *    copied once to files/qnn (the DSP and dlopen need real files).
 * Without either, the NPU is unavailable and its nets run on the CPU, as on any other phone. The fetching
 * itself is [PackFetch] (Packs.kt), shared with the LiteRT GPU pack ([LiteRtPacks]).
 */
object QnnPacks {
    private const val TAG = "FlatScan"
    const val COMMON = "qnn_htp"
    private val GENERATIONS = listOf("v68", "v69", "v73", "v75", "v79")

    // SoC (Build.SOC_MODEL) -> Hexagon generation, only where it is certain; any other Snapdragon fetches
    // every generation (about 3 MB each) and QNN picks the one it needs
    private val HTP = mapOf(
        "SM8350" to "v68", "SM7325" to "v68",            // 888, 778G
        "SM8450" to "v69", "SM8475" to "v69",            // 8 Gen 1, 8+ Gen 1
        "SM8550" to "v73",                               // 8 Gen 2
        "SM8650" to "v75",                               // 8 Gen 3
        "SM8750" to "v79",                               // 8 Elite
    )

    /** Debug (batch mode `--ez qnn_packs true`): fetch every pack on any phone, to check the delivery on an emulator. */
    @Volatile var force = false

    val isSnapdragon: Boolean by lazy {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MANUFACTURER.equals("QTI", true) || Build.SOC_MANUFACTURER.equals("Qualcomm", true)
        } else {
            Build.HARDWARE.equals("qcom", true) || Build.BOARD.lowercase().let { b -> listOf("msm", "sdm", "sm", "lahaina", "taro", "kalama").any { b.startsWith(it) } }
        }
    }

    /** This phone's Hexagon generation, or null (not certain: every generation is fetched). */
    val generation: String? by lazy { if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) HTP[Build.SOC_MODEL.uppercase()] else null }

    fun packs(): List<String> = listOf(COMMON) + (generation?.let { listOf(it) } ?: GENERATIONS).map { "qnn_$it" }

    /** Where the packs' files are: [lib] the common libraries, [models] the static models, [htp] the generations' folders. */
    data class Dirs(val lib: File, val models: File, val htp: List<File>, val source: String)

    @Volatile var dirs: Dirs? = null
        private set
    @Volatile var status: String = "not needed yet"
        private set
    @Volatile var loaded: Boolean = false
        private set

    /**
     * The packs on disk and the QNN libraries loaded ([NativeLib.loadQnn]); false when this phone has no
     * Snapdragon or the packs cannot be had (status says why). Blocks while downloading: call off the main
     * thread. [stage]: progress text.
     */
    @Synchronized
    fun ensure(context: Context, stage: (String) -> Unit): Boolean {
        if (loaded) return true
        if (!isSnapdragon && !force) {
            status = "not a Snapdragon: the NPU libraries are not downloaded"
            return false
        }
        val d = dirs ?: fromPlay(context, stage) ?: fromApk(context, stage)
        if (d == null) return false
        dirs = d
        val r = runCatching { org.json.JSONObject(NativeLib.loadQnn(d.lib.absolutePath, d.models.absolutePath, d.htp.joinToString(";") { it.absolutePath })) }
            .getOrElse { org.json.JSONObject().put("ok", false).put("error", it.message) }
        loaded = r.optBoolean("ok")
        status = (if (loaded) "ready (${d.source}, ${d.htp.joinToString(", ") { it.name }})" else "libraries failed to load: ${r.optString("error")}")
        Log.i(TAG, "qnn packs: $status $r")
        return loaded
    }

    private fun fromPlay(context: Context, stage: (String) -> Unit): Dirs? = runCatching {
        val want = packs()
        val loc = PackFetch.fromPlay(context, want, "the NPU libraries", stage)
        val root = File(loc.getValue(COMMON), "qnn")
        Dirs(File(root, "lib"), File(root, "models"), want.drop(1).map { File(File(loc.getValue(it), "qnn"), it.removePrefix("qnn_")) }, "Play")
    }.onFailure { status = "Play Asset Delivery: ${it.message}"; Log.i(TAG, "qnn packs from Play: $it") }.getOrNull()

    // a universal APK: the packs' assets are in the APK; copied once to files/qnn (versioned by the app's update time)
    private fun fromApk(context: Context, stage: (String) -> Unit): Dirs? = runCatching {
        val out = PackFetch.fromApk(context, "qnn", "lib", "not in this APK either (an APK built with assembleDebug has no NPU libraries)",
            "the NPU libraries", stage)
        val gens = (generation?.let { listOf(it) } ?: GENERATIONS).map { File(out, it) }.filter { it.isDirectory }
        Dirs(File(out, "lib"), File(out, "models"), gens, "APK")
    }.onFailure { status = "$status; APK: ${it.message}"; Log.i(TAG, "qnn packs from the APK: $it") }.getOrNull()
}
