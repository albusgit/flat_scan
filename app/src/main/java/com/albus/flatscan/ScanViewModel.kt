package com.albus.flatscan

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.math.hypot
import kotlin.math.min

enum class Screen { Home, Camera, Crop, Processing, Result, Settings }

class ScanViewModel(app: Application) : AndroidViewModel(app) {
    val settings = Settings(app)

    var screen by mutableStateOf(Screen.Home)
    private var returnTo = Screen.Home
    var message by mutableStateOf<String?>(null)

    // ---- crop
    var photo by mutableStateOf<Bitmap?>(null)
        private set
    var photoDisplay by mutableStateOf<Bitmap?>(null)
        private set
    private var focalPx = 0f
    var corners by mutableStateOf(listOf<Offset>())
    private var detected = listOf<Offset>()
    private var detectionFailed = false
    var detecting by mutableStateOf(false)
        private set
    private var userRotated = false

    // ---- processing
    var progress by mutableFloatStateOf(0f)
        private set
    var stage by mutableStateOf("")
        private set

    // ---- result
    private var handle = 0L
    var result by mutableStateOf<Bitmap?>(null)
        private set
    var before by mutableStateOf<Bitmap?>(null)
        private set
    var showBefore by mutableStateOf(false)
    var grayscale by mutableStateOf(settings.grayscale)
        private set
    var solid by mutableStateOf(settings.solidColors)
        private set
    var rotation by mutableIntStateOf(0)
        private set
    var rendering by mutableStateOf(false)
        private set
    var details by mutableStateOf("")
        private set
    private var cleaned: Bitmap? = null
    private var renderJob: Job? = null

    val gpuInfo: String by lazy { NativeLib.gpuInfo() }

    // ---- self-test / benchmark
    var selfTestRunning by mutableStateOf(false)
        private set
    var selfTestStage by mutableStateOf("")
        private set
    var selfTestReport by mutableStateOf("")
        private set

    fun runSelfTest(runs: Int = 5) {
        if (selfTestRunning) return
        selfTestRunning = true
        selfTestReport = ""
        viewModelScope.launch {
            val report = withContext(Dispatchers.Default) {
                val app = getApplication<Application>()
                val ref = java.io.File(app.cacheDir, "photo1_reference_warpmap.npz")
                app.assets.open("selftest/photo1_reference_warpmap.npz").use { i -> ref.outputStream().use { i.copyTo(it) } }
                val opts = android.graphics.BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inScaled = false
                }
                val bmp = app.assets.open("selftest/photo1.png").use { android.graphics.BitmapFactory.decodeStream(it, null, opts) }!!
                val thermalBefore = thermalStatus(app)
                val json = NativeLib.selfTest(bmp, ref.absolutePath, runs) { pct, s ->
                    viewModelScope.launch(Dispatchers.Main) { selfTestStage = "$s ($pct%)" }
                }
                formatSelfTest(JSONObject(json), thermalBefore, thermalStatus(app))
            }
            selfTestReport = report
            selfTestRunning = false
        }
    }

    private fun thermalStatus(app: Application): String {
        if (android.os.Build.VERSION.SDK_INT < 29) return "n/a (API < 29)"
        val pm = app.getSystemService(android.content.Context.POWER_SERVICE) as android.os.PowerManager
        return when (pm.currentThermalStatus) {
            0 -> "none"; 1 -> "light"; 2 -> "moderate"; 3 -> "severe"; 4 -> "critical"; 5 -> "emergency"; 6 -> "shutdown"
            else -> "unknown"
        }
    }

    private fun formatSelfTest(j: JSONObject, thermalBefore: String, thermalAfter: String): String {
        val b = android.os.Build.VERSION.SDK_INT
        val sb = StringBuilder()
        sb.append("FlatScan ${BuildConfigInfo.VERSION} self-test (measured on this device)\n")
        sb.append("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, Android ${android.os.Build.VERSION.RELEASE} (API $b)\n")
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) sb.append("SoC: ${android.os.Build.SOC_MANUFACTURER} ${android.os.Build.SOC_MODEL}\n")
        sb.append("Thermal status: before $thermalBefore, after $thermalAfter\n")
        if (!j.optBoolean("ok")) return sb.append("FAILED: ").append(j.optString("error")).toString()
        sb.append("GPU: ${j.optString("gpu")}\n")
        sb.append("Final resample on GPU: ${j.optBoolean("gpu_resample")}\n\n")
        val mp = j.getJSONObject("map_parity")
        sb.append("1. Map parity vs reference (limit 0.1 px): max %.4f px, mean %.5f px -> %s\n".format(
            mp.optDouble("max_epe_px"), mp.optDouble("mean_epe_px"), if (mp.optBoolean("passed")) "PASS" else "FAIL"))
        val rs = j.getJSONObject("gpu_vs_cpu_resample")
        sb.append("2. GPU vs CPU resample: max %d levels, values > 1: %d -> %s\n".format(
            rs.optInt("max_levels"), rs.optInt("values_gt1"), if (rs.optInt("max_levels") <= 1) "PASS" else "FAIL"))
        val cl = j.getJSONObject("gpu_vs_cpu_cleanup")
        sb.append("   GPU vs CPU cleanup: max %d levels (GPU used: %s; %d ms GPU, %d ms CPU)\n".format(
            cl.optInt("max_levels"), cl.optBoolean("gpu_used"), cl.optLong("gpu_ms"), cl.optLong("cpu_ms")))
        sb.append("Peak memory (VmHWM): ${j.optLong("peak_rss_mb")} MB\n\n")
        sb.append("Stage times over ${j.optInt("runs")} runs after 1 warm-up (median / p95 ms), photo 1 (1659x2212):\n")
        val st = j.getJSONArray("stages_ms")
        for (i in 0 until st.length()) {
            val o = st.getJSONObject(i)
            sb.append("%-34s %6d / %6d\n".format(o.optString("stage"), o.optLong("median"), o.optLong("p95")))
        }
        return sb.toString()
    }

    fun open(s: Screen) {
        if (s == Screen.Settings) returnTo = screen
        screen = s
    }

    fun back() {
        screen = when (screen) {
            Screen.Camera, Screen.Crop -> Screen.Home
            Screen.Result -> Screen.Crop
            Screen.Settings -> returnTo
            else -> Screen.Home
        }
    }

    // ------------------------------------------------------------------ input
    fun loadUri(uri: Uri, focalMmPerLongSide: Float = 0f) {
        viewModelScope.launch {
            detecting = true
            screen = Screen.Crop
            try {
                val p = withContext(Dispatchers.IO) { loadPhoto(getApplication(), uri, focalMmPerLongSide) }
                usePhoto(p.bitmap)
                focalPx = p.focalPx
                userRotated = false
                detect()
            } catch (e: Exception) {
                Log.e(TAG, "load failed", e)
                message = "Could not open the image"
                detecting = false
                screen = Screen.Home
            }
        }
    }

    private fun usePhoto(b: Bitmap) {
        photo = b
        photoDisplay = b.forDisplay()
    }

    private suspend fun detect() {
        val p = photo ?: return
        detecting = true
        val c = withContext(Dispatchers.Default) { NativeLib.detect(p) }
        detectionFailed = c == null
        detected = if (c != null) List(4) { Offset(c[2 * it], c[2 * it + 1]) } else defaultCorners(p)
        corners = detected
        detecting = false
        if (c == null) message = "Page not found - drag the corners to the page"
    }

    private fun defaultCorners(p: Bitmap): List<Offset> {
        val dx = p.width * 0.08f
        val dy = p.height * 0.08f
        return listOf(Offset(dx, dy), Offset(p.width - dx, dy), Offset(p.width - dx, p.height - dy), Offset(dx, p.height - dy))
    }

    /** The user moved a corner (more than ~0.3% of the image size) -> homography path. */
    private fun cornersEdited(): Boolean {
        val p = photo ?: return false
        val tol = 0.003f * min(p.width, p.height)
        return detectionFailed || corners.zip(detected).any { (a, b) -> hypot(a.x - b.x, a.y - b.y) > tol }
    }

    /** Rotate the photo 90 degrees clockwise (corners follow). Disables auto orientation. */
    fun rotatePhoto() {
        val p = photo ?: return
        val h = p.height.toFloat()
        val rot = { o: Offset -> Offset(h - 1 - o.y, o.x) }
        // after a clockwise turn the old bottom-left corner is the new top-left
        val reorder = { l: List<Offset> -> listOf(rot(l[3]), rot(l[0]), rot(l[1]), rot(l[2])) }
        corners = reorder(corners)
        detected = reorder(detected)
        usePhoto(p.rotatedQuarterTurns(1))
        userRotated = true
    }

    // ------------------------------------------------------------------ processing
    fun process() {
        val p = photo ?: return
        val edited = cornersEdited()
        val c = FloatArray(8) { i -> if (i % 2 == 0) corners[i / 2].x else corners[i / 2].y }
        screen = Screen.Processing
        progress = 0f
        stage = "Starting"
        viewModelScope.launch {
            // a render of the previous session may still be running natively: wait, then free it
            renderJob?.join()
            val h = withContext(Dispatchers.Default) {
                releaseHandle()
                NativeLib.process(p, c, edited, focalPx, !userRotated, settings.outputWidth, true) { pct, s ->
                    viewModelScope.launch(Dispatchers.Main) { progress = pct / 100f; stage = s }
                }
            }
            val info = JSONObject(NativeLib.info(h))
            if (!info.optBoolean("ok")) {
                NativeLib.release(h)
                message = "Processing failed: ${info.optString("error")}"
                screen = Screen.Crop
                return@launch
            }
            handle = h
            details = describe(info, edited)
            grayscale = settings.grayscale
            solid = settings.solidColors
            rotation = 0
            showBefore = false
            before = p.forDisplay()
            render(final = true)
        }
    }

    private fun describe(info: JSONObject, edited: Boolean): String {
        val sb = StringBuilder()
        val mode = info.optString("mode")
        sb.append(
            when {
                edited -> "Geometry: homography (corners adjusted by hand)"
                info.optBoolean("fell_back") -> "Geometry: homography (surface fit rejected, outline residual %.1f px)".format(info.optDouble("fit_median_px"))
                mode == "surface" -> "Geometry: page surface model (outline residual %.2f px)".format(info.optDouble("fit_median_px"))
                else -> "Geometry: $mode"
            },
        )
        sb.append("\nResample: ").append(if (info.optBoolean("gpu_resample")) "GPU" else "CPU fallback")
        sb.append("\n").append(gpuInfo)
        val ms = info.optJSONObject("ms")
        if (ms != null) {
            sb.append("\n\nStage times (ms):")
            for (k in ms.keys()) sb.append("\n").append(k).append(": ").append(ms.getLong(k))
        }
        return sb.toString()
    }

    fun changeGray(v: Boolean) { if (v != grayscale) { grayscale = v; render() } }
    fun changeSolid(v: Boolean) { if (v != solid) { solid = v; render() } }
    fun rotateResult() {
        rotation = (rotation + 1) % 4
        cleaned?.let { result = it.rotatedQuarterTurns(rotation) }
        before = before?.rotatedQuarterTurns(1)
    }

    private fun render(final: Boolean = false) {
        val h = handle
        if (h == 0L) return
        renderJob?.cancel()
        rendering = true
        renderJob = viewModelScope.launch {
            val (bmp, json) = withContext(Dispatchers.Default) {
                val size = NativeLib.pageSize(h)
                val b = Bitmap.createBitmap(size[0], size[1], Bitmap.Config.ARGB_8888)
                b to NativeLib.render(h, grayscale, solid, false, true, b)
            }
            cleaned = bmp
            result = bmp.rotatedQuarterTurns(rotation)
            rendering = false
            val r = JSONObject(json)
            if (final) details += "\ncleanup: ${r.optLong("cleanup_ms")} (" + (if (r.optInt("gpu_tone") == 1) "GPU" else "CPU") + ")"
            if (screen == Screen.Processing) screen = Screen.Result
        }
    }

    /** The page as saved: cleaned, rotated. */
    fun output(): Bitmap? = result

    private fun releaseHandle() {
        if (handle != 0L) NativeLib.release(handle)
        handle = 0L
    }

    override fun onCleared() {
        releaseHandle()
        super.onCleared()
    }

    companion object {
        const val TAG = "FlatScan"
    }
}
