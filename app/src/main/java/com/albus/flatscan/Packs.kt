package com.albus.flatscan

import android.content.Context
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.android.play.core.assetpacks.AssetPackManagerFactory
import com.google.android.play.core.assetpacks.model.AssetPackStatus
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * An on-demand asset pack's files on disk (app/build.gradle.kts), for [QnnPacks] and [LiteRtPacks]. Sources:
 *  * [fromPlay]: Play Asset Delivery (Play Store installs, and `bundletool install-apks` of a --local-testing build);
 *  * [fromApk]: an APK that carries the packs' assets itself (bundletool --mode=universal; for LiteRT also
 *    `assembleDebug -PlitertInBase=true`): copied once to files/<root> (the DSP and dlopen need real files).
 * Both throw with why; `what` names the files in the progress text ("the NPU libraries").
 */
internal object PackFetch {
    /** The assets folder of each pack in [want], downloaded first when one is missing. Blocks: call off the main thread. */
    fun fromPlay(context: Context, want: List<String>, what: String, stage: (String) -> Unit): Map<String, File> {
        val m = AssetPackManagerFactory.getInstance(context)
        fun located() = want.associateWith { m.getPackLocation(it)?.assetsPath() }
        var loc = located()
        if (loc.values.any { it == null }) {
            stage("Downloading $what")
            Tasks.await(m.fetch(want), 30, TimeUnit.SECONDS)
            val until = System.nanoTime() + TimeUnit.MINUTES.toNanos(10)
            while (true) {
                val st = Tasks.await(m.getPackStates(want), 30, TimeUnit.SECONDS)
                val states = want.map { st.packStates()[it] }
                val total = states.sumOf { it?.totalBytesToDownload() ?: 0L }
                val got = states.sumOf { it?.bytesDownloaded() ?: 0L }
                if (states.any { it == null }) error("unknown pack in ${st.packStates().keys}")
                val bad = states.firstOrNull { it!!.status() == AssetPackStatus.FAILED || it.status() == AssetPackStatus.CANCELED }
                if (bad != null) error("pack ${bad.name()}: status ${bad.status()}, error ${bad.errorCode()}")
                if (states.all { it!!.status() == AssetPackStatus.COMPLETED }) break
                if (System.nanoTime() > until) error("download timed out")
                if (states.any { it!!.status() == AssetPackStatus.WAITING_FOR_WIFI || it.status() == AssetPackStatus.REQUIRES_USER_CONFIRMATION }) {
                    error("the download is waiting for Wi-Fi or the user's confirmation")
                }
                stage("Downloading $what: ${if (total > 0) 100 * got / total else 0}%")
                Thread.sleep(500)
            }
            loc = located()
        }
        if (loc.values.any { it == null }) error("packs missing after the download")
        return loc.mapValues { File(it.value!!) }
    }

    /**
     * assets/[root]/<folder>/<file> of this APK, copied to files/[root] once per app update (versioned by its
     * update time); [missing] is the error when the APK has no [root]/[needs] folder.
     */
    fun fromApk(context: Context, root: String, needs: String, missing: String, what: String, stage: (String) -> Unit): File {
        val am = context.assets
        val names = am.list(root)?.toList().orEmpty()
        if (needs !in names) error(missing)
        val out = File(context.filesDir, root)
        val stamp = File(out, ".stamp")
        val want = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        if (!stamp.isFile || stamp.readText() != want) {
            stage("Unpacking $what")
            out.deleteRecursively()
            for (dir in names) {
                for (f in am.list("$root/$dir").orEmpty()) {
                    File(out, "$dir/$f").also { it.parentFile?.mkdirs() }.outputStream().use { o -> am.open("$root/$dir/$f").use { it.copyTo(o) } }
                }
            }
            stamp.writeText(want)
        }
        return out
    }
}

/**
 * The LiteRT (TensorFlow Lite) GPU runtime as the on-demand asset pack litert_gpu (app/build.gradle.kts):
 * libtensorflowlite_jni and libtensorflowlite_gpu_jni (assets/litert/lib/) and the GPU models (the .tflite
 * files in assets/litert/models/). Unlike the QNN packs it is fetched on any phone (they all have a GPU),
 * when the GPU or Auto is first used, then loaded with [NativeLib.loadLiteRt]. Sources, in order: this APK's
 * assets/litert/ (-PlitertInBase=true, a universal APK), then Play Asset Delivery. Without either the GPU is
 * unavailable and its nets run on the CPU.
 */
object LiteRtPacks {
    private const val TAG = "FlatScan"
    const val PACK = "litert_gpu"
    // a failed fetch is not tried again for this long (Auto asks before every processed page)
    private val RETRY_NS = TimeUnit.MINUTES.toNanos(5)

    /** Debug (batch mode `--ez litert_pack true`): fetch the pack whatever the setting. */
    @Volatile var force = false

    /** Where the pack's files are: [lib] the two libraries, [models] the .tflite models. */
    data class Dirs(val lib: File, val models: File, val source: String)

    @Volatile var dirs: Dirs? = null
        private set
    @Volatile var status: String = "not needed yet"
        private set
    @Volatile var loaded: Boolean = false
        private set
    private var failedAt = 0L

    /**
     * The pack on disk and the LiteRT libraries loaded ([NativeLib.loadLiteRt]); false when the pack cannot
     * be had or the libraries do not load (status says why). Blocks while downloading: call off the main
     * thread. [stage]: progress text.
     */
    @Synchronized
    fun ensure(context: Context, stage: (String) -> Unit): Boolean {
        if (loaded) return true
        if (failedAt != 0L && System.nanoTime() - failedAt < RETRY_NS && !force) return false
        val d = dirs ?: fromApk(context, stage) ?: fromPlay(context, stage)
        if (d == null) {
            failedAt = System.nanoTime()
            return false
        }
        dirs = d
        val r = runCatching { org.json.JSONObject(NativeLib.loadLiteRt(d.lib.absolutePath, d.models.absolutePath)) }
            .getOrElse { org.json.JSONObject().put("ok", false).put("error", it.message) }
        loaded = r.optBoolean("ok")
        val models = d.models.list().orEmpty().count { it.endsWith(".tflite") }
        status = if (loaded) "ready (${d.source}, $models GPU models)" else "libraries failed to load: ${r.optString("error")}"
        if (!loaded) failedAt = System.nanoTime()
        Log.i(TAG, "litert pack: $status $r")
        return loaded
    }

    private fun fromApk(context: Context, stage: (String) -> Unit): Dirs? = runCatching {
        val out = PackFetch.fromApk(context, "litert", "lib", "not in this APK", "the GPU runtime", stage)
        Dirs(File(out, "lib"), File(out, "models"), "APK")
    }.onFailure { status = "APK: ${it.message}"; Log.i(TAG, "litert pack from the APK: $it") }.getOrNull()

    private fun fromPlay(context: Context, stage: (String) -> Unit): Dirs? = runCatching {
        val root = File(PackFetch.fromPlay(context, listOf(PACK), "the GPU runtime", stage).getValue(PACK), "litert")
        Dirs(File(root, "lib"), File(root, "models"), "Play")
    }.onFailure { status = "$status; Play Asset Delivery: ${it.message}"; Log.i(TAG, "litert pack from Play: $it") }.getOrNull()
}
