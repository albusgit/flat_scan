package com.albus.flatscan

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

enum class Screen { Home, Camera, Crop, Processing, Result, Pages, Settings }

enum class ImportSource { Camera, Gallery }

/** One page found by [NativeLib.detect]. */
class DetectedPage(
    val ok: Boolean,
    val corners: List<Offset>,
    val reasons: List<String>,          // quality gate: why the automatic crop is doubtful
    val centroid: Offset,               // outline centroid, photo px
) {
    companion object {
        fun from(j: JSONObject): DetectedPage {
            val c = j.optJSONArray("corners")
            val r = j.optJSONArray("needs_manual_crop")
            val m = j.optJSONArray("centroid")
            return DetectedPage(
                ok = j.optBoolean("ok") && c != null && c.length() == 8,
                corners = if (c != null && c.length() == 8) List(4) { Offset(c.optDouble(2 * it).toFloat(), c.optDouble(2 * it + 1).toFloat()) } else emptyList(),
                reasons = if (r != null) List(r.length()) { r.optString(it) } else emptyList(),
                centroid = if (m != null && m.length() == 2) Offset(m.optDouble(0).toFloat(), m.optDouble(1).toFloat()) else Offset.Zero,
            )
        }
    }
}

/**
 * The second page of a book spread, processed right after the first one ([firstId]) from the same
 * photo. [firstRot] is the first page's applied rotation (cv::ROTATE_* code, -1 none): it tells
 * which way "right" is in the photo.
 */
private class SpreadPage(
    val firstId: String, val photo: Bitmap, val page: DetectedPage, val firstCentroid: Offset,
    val focalPx: Float, val cropRotation: Int, val userRotated: Boolean, var firstRot: Int = -1,
)

/**
 * App state. Native sessions: at most one is alive (the page shown on the result screen) and
 * every call that touches it runs on one dedicated thread ([nativeCtx]), so renders, thumbnails,
 * the histogram and release are serialised and a released handle is never used. Every other
 * page lives on disk ([DocumentStore]).
 */
class ScanViewModel(app: Application) : AndroidViewModel(app) {
    val settings = Settings(app)
    private val store = DocumentStore(app)

    var screen by mutableStateOf(Screen.Home)
    private var returnTo = Screen.Home
    private var pagesReturnTo = Screen.Home
    var message by mutableStateOf<String?>(null)

    private val nativeExec = Executors.newSingleThreadExecutor { r -> Thread(r, "flatscan-native") }
    private val nativeCtx = nativeExec.asCoroutineDispatcher()
    /** The live session as the native thread sees it; read and written only on [nativeCtx]. */
    private var nativeHandle = 0L
    /** Page-size render targets (ping-pong, so the one on screen is never written); [nativeCtx] only. */
    private val buffers = arrayOfNulls<Bitmap>(2)

    // ---- document
    /** Page ids in document order. */
    val pages = mutableStateListOf<String>()
    /** Bumped whenever a page's out.jpg is rewritten (thumbnail cache key). */
    val pageVersions = mutableStateMapOf<String, Int>()
    /** The page with the live session (result screen). */
    var currentId by mutableStateOf<String?>(null)
        private set
    /** Imported images still to go through crop -> processing. */
    val pending = mutableStateListOf<Uri>()
    var pendingTotal by mutableIntStateOf(0)
        private set
    var lastSource by mutableStateOf(ImportSource.Camera)
        private set

    // ---- crop (the photo on the crop screen; may be a new page or the current one)
    var photo by mutableStateOf<Bitmap?>(null)
        private set
    var photoDisplay by mutableStateOf<Bitmap?>(null)
        private set
    var corners by mutableStateOf(listOf<Offset>())
    var detecting by mutableStateOf(false)
        private set
    private var photoPageId: String? = null
    private var cropMeta: PageMeta? = null          // stored page being re-processed
    private var focalPx = 0f
    private var detected: List<Offset>? = null
    private var userRotated = false
    private var cropRotation = 0
    private var sourceJob: Job? = null
    private var recropBackup: CropBackup? = null

    // ---- page finder, quality gate, book spreads
    private val finder = PageFinder(app)
    private val uvNet = FlatNet(app, FlatNet.UVDOC)
    private val sdNet = FlatNet(app, FlatNet.SYNTHDOC)
    private val shadeNet = ShadeNet(app)
    private val logitsLock = Any()
    /** Page-finder output for one photo bitmap (detection and processing use the same photo). */
    private var logitsCache: Pair<Bitmap, FloatArray?>? = null
    /** Why the automatic crop on the crop screen is doubtful (quality gate); empty = trusted. */
    var cropReasons by mutableStateOf(listOf<String>())
        private set
    /** The gate flagged this crop: the homography path is used even if the corners are kept. */
    private var forceHomography = false
    /** The page of a book spread the crop screen shows (0 = the larger page of the photo). */
    var cropPageIndex by mutableIntStateOf(0)
        private set
    /** Found with the page on the crop screen: the other page of its book spread (waits for it). */
    private var spreadNext: SpreadPage? = null
    /** The spread page being processed now (the second one). */
    private var spreadCurrent: SpreadPage? = null

    private class CropBackup(
        val photo: Bitmap, val display: Bitmap?, val corners: List<Offset>, val detected: List<Offset>?,
        val userRotated: Boolean, val cropRotation: Int, val reasons: List<String>, val forceHomography: Boolean, val pageIndex: Int,
    )

    // ---- processing
    var progress by mutableFloatStateOf(0f)
        private set
    var stage by mutableStateOf("")
        private set

    // ---- result
    /** The live session as the UI sees it (0 = none); main thread. */
    private var handle by mutableStateOf(0L)
    val hasSession: Boolean get() = handle != 0L
    private var sessionGeom: PageMeta?
        get() = sessionGeomState
        set(v) { sessionGeomState = v }
    private var sessionGeomState by mutableStateOf<PageMeta?>(null)
    var params by mutableStateOf(RenderParams())
        private set
    /** Protected picture rectangles, page fractions [x0, y0, x1, y1, ...]. */
    var protect by mutableStateOf(listOf<Float>())
        private set
    /** Erase tool rectangles (filled with paper), page fractions [x0, y0, x1, y1, ...]. */
    var erase by mutableStateOf(listOf<Float>())
        private set
    var rotation by mutableIntStateOf(0)
        private set
    /** The cleaned page, unrotated (rotation is applied when drawing and on export). */
    var page by mutableStateOf<Bitmap?>(null)
        private set
    var pageWidth by mutableIntStateOf(0)
        private set
    var pageHeight by mutableIntStateOf(0)
        private set
    var before by mutableStateOf<Bitmap?>(null)
        private set
    var overlay by mutableStateOf<Bitmap?>(null)
        private set
    var overlayMaxPx by mutableFloatStateOf(0f)
        private set
    var showWrinkles by mutableStateOf(false)
        private set
    var thumbs by mutableStateOf<Map<FilterKind, Bitmap>>(emptyMap())
        private set
    var histogram by mutableStateOf<IntArray?>(null)
        private set
    var adjustOpen by mutableStateOf(false)
        private set
    var rendering by mutableStateOf(false)
        private set
    var details by mutableStateOf("")
        private set
    var showHoldHint by mutableStateOf(!settings.holdHintShown)
        private set
    /** The lines the unwarp was fitted to (NativeLib.lines layout: kind, x0, y0, x1, y1 page fractions). */
    var lines by mutableStateOf(FloatArray(0))
        private set
    /** Areas whose lines the unwarp must not use, page fractions [x0, y0, x1, y1, ...] (being edited). */
    var ignore by mutableStateOf(listOf<Float>())
        private set
    /** The ignore areas differ from the ones the current page was flattened with. */
    val ignoreChanged: Boolean get() = ignore != (sessionGeomState?.ignore?.toList() ?: emptyList<Float>())
    /** The user turned the unwarp off for the current page (perspective correction only). */
    val perspectiveOnly: Boolean get() = sessionGeomState?.perspectiveOnly == true
    /** The page was flattened again from the line editor: the result screen opens it again. */
    var reopenLines = false
    /** Serialises writes of stored sessions (DocumentStore.saveSession). */
    private val sessionStoreMutex = Mutex()

    private data class MainKey(val h: Long, val p: RenderParams, val protect: List<Float>, val erase: Int)
    /** Bumped whenever the native page changes under the same handle (Erase tool). */
    private var pageEdit by mutableIntStateOf(0)
    private var shownKey by mutableStateOf<MainKey?>(null)
    private var thumbKey: Pair<Long, RenderParams>? = null
    private var histKey: Pair<Long, RenderParams>? = null
    private val wake = Channel<Boolean>(Channel.CONFLATED)
    private var outVersion = 0
    private var committedVersion = -1
    private val commitMutex = Mutex()
    private var firstRenderDetails = false

    val gpuInfo: String by lazy { NativeLib.gpuInfo() }

    init {
        viewModelScope.launch {
            val order = withContext(Dispatchers.IO) { store.loadOrder().also { store.prune(it) } }
            pages.clear()
            pages.addAll(order)
        }
        viewModelScope.launch { renderLoop() }
    }

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

    // ------------------------------------------------------------------ navigation
    fun open(s: Screen) {
        when (s) {
            Screen.Settings -> returnTo = screen
            Screen.Pages -> {
                pagesReturnTo = screen
                viewModelScope.launch { commitCurrent() }   // thumbnail of the current page
            }
            else -> {}
        }
        screen = s
    }

    /** Where to go when the current step is abandoned. */
    private fun fallback(): Screen = when {
        handle != 0L -> Screen.Result
        pages.isNotEmpty() -> Screen.Pages
        else -> Screen.Home
    }

    fun back() {
        screen = when (screen) {
            Screen.Camera -> fallback()
            Screen.Crop -> {
                val b = recropBackup
                if (b != null) {
                    // re-crop abandoned: the session still matches the previous corners
                    photo = b.photo
                    photoDisplay = b.display
                    corners = b.corners
                    detected = b.detected
                    userRotated = b.userRotated
                    cropRotation = b.cropRotation
                    cropReasons = b.reasons
                    forceHomography = b.forceHomography
                    cropPageIndex = b.pageIndex
                    recropBackup = null
                } else {
                    discardStaging()
                    spreadNext = null
                    spreadCurrent = null
                }
                fallback()
            }
            Screen.Result -> {
                persist()
                adjustOpen = false
                Screen.Home
            }
            Screen.Pages -> if (pagesReturnTo == Screen.Result && handle != 0L) Screen.Result else Screen.Home
            Screen.Settings -> returnTo
            else -> Screen.Home
        }
    }

    /** Writes the current page to disk (the activity stops, or the user leaves the page). */
    fun persist() {
        viewModelScope.launch { commitCurrent() }
    }

    // ------------------------------------------------------------------ input
    fun startCamera() {
        lastSource = ImportSource.Camera
        persist()
        screen = Screen.Camera
    }

    /** Gallery import: every picked image becomes a page, one after the other. */
    fun importUris(uris: List<Uri>) {
        if (uris.isEmpty()) return
        lastSource = ImportSource.Gallery
        pending.clear()
        pending.addAll(uris.drop(1))
        pendingTotal = uris.size
        persist()
        loadUri(uris[0])
    }

    /** Index (1-based) of the next imported image, for "Next page (k of n)". */
    val nextPendingIndex: Int get() = pendingTotal - pending.size + 1

    fun nextPage() {
        val u = pending.removeFirstOrNull() ?: return
        persist()
        loadUri(u)
    }

    fun loadUri(uri: Uri, focalMmPerLongSide: Float = 0f) {
        viewModelScope.launch {
            detecting = true
            recropBackup = null
            photo = null
            photoDisplay = null
            corners = emptyList()
            cropReasons = emptyList()
            spreadNext = null
            spreadCurrent = null
            screen = Screen.Crop
            try {
                val p = withContext(Dispatchers.IO) { loadPhoto(getApplication(), uri, focalMmPerLongSide) }
                discardStaging()
                val id = store.newId()
                photoPageId = id
                cropMeta = null
                usePhoto(p.bitmap)
                focalPx = p.focalPx
                userRotated = false
                cropRotation = 0
                // the upright photo is the page's source on disk (re-processing, re-crop)
                sourceJob = viewModelScope.launch(Dispatchers.IO) {
                    runCatching { store.saveSource(id, p.bitmap) }.onFailure { Log.e(TAG, "source save failed", it) }
                }
                detect(auto = true)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "load failed", e)
                message = "Could not open the image"
                detecting = false
                screen = fallback()
            }
        }
    }

    /** Deletes the source of a new page that never made it into the document. */
    private fun discardStaging() {
        val id = photoPageId ?: return
        if (id in pages || id == currentId) return
        photoPageId = null
        val job = sourceJob
        viewModelScope.launch(Dispatchers.IO) {
            job?.join()
            store.delete(id)
        }
    }

    private fun usePhoto(b: Bitmap) {
        photo = b
        photoDisplay = b.forDisplay()
    }

    /** Page-finder logits for [b] (null: the flag is off or the net cannot run -> colour threshold). */
    private fun logitsFor(b: Bitmap): FloatArray? {
        if (!settings.pageFinderNet) return null
        synchronized(logitsLock) {
            logitsCache?.let { if (it.first === b) return it.second }
            val l = runCatching { finder.logits(b) }.onFailure { Log.e(TAG, "page finder failed", it) }.getOrNull()
            logitsCache = b to l
            return l
        }
    }

    /**
     * Finds the page(s) of the photo on the crop screen. [auto]: a trusted crop (quality gate)
     * goes straight on to processing; a doubtful one stays on the crop screen with its reasons and
     * takes the homography path. A book spread queues its second page.
     */
    private suspend fun detect(auto: Boolean = false) {
        val p = photo ?: return
        detecting = true
        val focal = focalPx
        val res = withContext(Dispatchers.Default) {
            runCatching { JSONObject(NativeLib.detect(p, logitsFor(p), focal)) }
                .onFailure { Log.e(TAG, "detect failed", it) }.getOrElse { JSONObject() }
        }
        if (photo !== p) return
        val arr = res.optJSONArray("pages")
        val found = if (arr != null) List(arr.length()) { DetectedPage.from(arr.getJSONObject(it)) } else emptyList()
        val first = found.firstOrNull()?.takeIf { it.ok }
        detected = first?.corners
        corners = detected ?: defaultCorners(p)
        cropPageIndex = 0
        cropReasons = first?.reasons ?: emptyList()
        forceHomography = first == null || first.reasons.isNotEmpty()
        val second = found.getOrNull(1)?.takeIf { it.ok }
        spreadNext = if (first != null && second != null) SpreadPage(
            firstId = photoPageId ?: "", photo = p, page = second, firstCentroid = first.centroid,
            focalPx = focalPx, cropRotation = cropRotation, userRotated = userRotated,
        ) else null
        detecting = false
        if (first == null) message = "Page not found - drag the corners to the page"
        else if (auto && cropReasons.isEmpty() && !settings.reviewCrops) process()
    }

    /** The second page of a book spread: same photo, its own corners and gate, a new document page. */
    private fun startSpreadPage(sp: SpreadPage) {
        val id = store.newId()
        val firstSource = sourceJob
        discardStaging()
        photoPageId = id
        cropMeta = null
        recropBackup = null
        usePhoto(sp.photo)
        focalPx = sp.focalPx
        userRotated = sp.userRotated
        cropRotation = sp.cropRotation
        detected = sp.page.corners
        corners = sp.page.corners
        cropPageIndex = 1
        cropReasons = sp.page.reasons
        forceHomography = sp.page.reasons.isNotEmpty()
        spreadNext = null
        spreadCurrent = sp
        sourceJob = viewModelScope.launch(Dispatchers.IO) {
            firstSource?.join()
            runCatching { store.copySource(sp.firstId, id) }.onFailure { Log.e(TAG, "spread source copy failed", it) }
        }
        if (cropReasons.isEmpty() && !settings.reviewCrops) {
            process()
        } else {
            screen = Screen.Crop
            message = "Book spread: check the second page"
        }
    }

    /**
     * Where the second page of a spread goes: the right page first for right-to-left books, the
     * left page first otherwise. "Right" is taken in the upright frame of the first page.
     */
    private fun secondGoesFirst(sp: SpreadPage): Boolean {
        val d = sp.page.centroid - sp.firstCentroid
        val dxUpright = when (sp.firstRot) {
            0 -> -d.y           // cv::ROTATE_90_CLOCKWISE
            1 -> -d.x           // cv::ROTATE_180
            2 -> d.y            // cv::ROTATE_90_COUNTERCLOCKWISE
            else -> d.x
        }
        val secondIsRight = dxUpright > 0
        return if (settings.bookRightToLeft) secondIsRight else !secondIsRight
    }

    /** The other page of the current page's book spread, if it is in the document. */
    val spreadPartner: String?
        get() = sessionGeomState?.spreadWith?.takeIf { it in pages }

    /** Swaps the two pages of the current book spread in the document. */
    fun swapSpread() {
        val a = currentId ?: return
        val b = spreadPartner ?: return
        val i = pages.indexOf(a)
        val j = pages.indexOf(b)
        if (i < 0 || j < 0) return
        pages[i] = b
        pages[j] = a
        val order = pages.toList()
        viewModelScope.launch(Dispatchers.IO) { store.saveOrder(order) }
    }

    private fun defaultCorners(p: Bitmap): List<Offset> {
        val dx = p.width * 0.08f
        val dy = p.height * 0.08f
        return listOf(Offset(dx, dy), Offset(p.width - dx, dy), Offset(p.width - dx, p.height - dy), Offset(dx, p.height - dy))
    }

    /** The user moved a corner (more than ~0.3% of the image size) -> homography path. */
    private fun cornersEdited(): Boolean {
        val p = photo ?: return false
        val d = detected ?: return true
        val tol = 0.003f * min(p.width, p.height)
        return corners.zip(d).any { (a, b) -> hypot(a.x - b.x, a.y - b.y) > tol }
    }

    /** Rotate the photo 90 degrees clockwise (corners follow). Disables auto orientation. */
    fun rotatePhoto() {
        val p = photo ?: return
        val h = p.height.toFloat()
        val rot = { o: Offset -> Offset(h - 1 - o.y, o.x) }
        // after a clockwise turn the old bottom-left corner is the new top-left
        val reorder = { l: List<Offset> -> listOf(rot(l[3]), rot(l[0]), rot(l[1]), rot(l[2])) }
        corners = reorder(corners)
        detected = detected?.let(reorder)
        usePhoto(p.rotatedQuarterTurns(1))
        userRotated = true
        cropRotation = (cropRotation + 1) % 4
        // a waiting second page of a book spread turns with the photo
        spreadNext = spreadNext?.let { sp ->
            val q = sp.page
            SpreadPage(
                firstId = sp.firstId, photo = photo!!, page = DetectedPage(q.ok, reorder(q.corners), q.reasons, rot(q.centroid)),
                firstCentroid = rot(sp.firstCentroid), focalPx = sp.focalPx, cropRotation = cropRotation, userRotated = true,
            )
        }
    }

    /**
     * "Circle the page": finds the page inside the loop the user drew around it ([loop], photo
     * px). The page finder runs on that part of the photo only; when it finds no page edge there,
     * the loop itself gives the corners. Either way the corners count as set by hand (homography).
     */
    fun detectInArea(loop: List<Offset>) {
        val p = photo ?: return
        if (loop.size < 8 || detecting) return
        val focal = focalPx
        viewModelScope.launch {
            detecting = true
            val found = withContext(Dispatchers.Default) {
                runCatching { findInLoop(p, loop, focal) }.onFailure { Log.e(TAG, "loop detect failed", it) }.getOrNull()
            }
            detecting = false
            if (photo !== p) return@launch
            if (found == null) {
                message = "Draw a loop around the whole page"
                return@launch
            }
            corners = found.first
            cropReasons = emptyList()
            spreadNext = null
            message = if (found.second) "Page found inside the loop" else "No page edge found: corners from your loop"
        }
    }

    private fun findInLoop(p: Bitmap, loop: List<Offset>, focal: Float): Pair<List<Offset>, Boolean>? {
        val m = 0.04f * maxOf(p.width, p.height)
        val x0 = (loop.minOf { it.x } - m).toInt().coerceIn(0, p.width - 1)
        val y0 = (loop.minOf { it.y } - m).toInt().coerceIn(0, p.height - 1)
        val x1 = (loop.maxOf { it.x } + m).toInt().coerceIn(x0 + 1, p.width)
        val y1 = (loop.maxOf { it.y } + m).toInt().coerceIn(y0 + 1, p.height)
        if (x1 - x0 < 64 || y1 - y0 < 64) return null
        val origin = Offset(x0.toFloat(), y0.toFloat())
        val crop = Bitmap.createBitmap(p, x0, y0, x1 - x0, y1 - y0)
        val logits = if (settings.pageFinderNet) runCatching { finder.logits(crop) }.getOrNull() else null
        val res = JSONObject(NativeLib.detect(crop, logits, focal))
        val arr = res.optJSONArray("pages")
        val centre = Offset(loop.sumOf { it.x.toDouble() }.toFloat() / loop.size, loop.sumOf { it.y.toDouble() }.toFloat() / loop.size) - origin
        // a page that fills the whole crop is the finder giving up, not a page edge
        val best = (0 until (arr?.length() ?: 0)).map { arr!!.getJSONObject(it) }
            .filter { !it.optBoolean("whole_frame") }
            .map { DetectedPage.from(it) }.filter { it.ok }
            .minByOrNull { (it.centroid - centre).getDistance() }
        if (best != null) {
            return best.corners.map { (it + origin).let { q -> Offset(q.x.coerceIn(0f, p.width - 1f), q.y.coerceIn(0f, p.height - 1f)) } } to true
        }
        // the loop's extreme points along the diagonals
        val tl = loop.minBy { it.x + it.y }
        val br = loop.maxBy { it.x + it.y }
        val tr = loop.maxBy { it.x - it.y }
        val bl = loop.minBy { it.x - it.y }
        return listOf(tl, tr, br, bl).map { Offset(it.x.coerceIn(0f, p.width - 1f), it.y.coerceIn(0f, p.height - 1f)) } to false
    }

    private fun setCrop(m: PageMeta, bmp: Bitmap) {
        photoPageId = m.id
        cropMeta = m
        usePhoto(bmp)
        corners = List(4) { Offset(m.corners[2 * it], m.corners[2 * it + 1]) }
        detected = m.detected?.let { d -> List(4) { Offset(d[2 * it], d[2 * it + 1]) } }
        userRotated = m.userRotated
        cropRotation = m.cropRotation
        focalPx = m.focalPx
        cropPageIndex = m.pageIndex
        forceHomography = m.forceHomography
        cropReasons = emptyList()
        spreadNext = null
        spreadCurrent = null
    }

    /** Back to the crop screen with the current page's corners; Continue keeps all its settings. */
    fun recrop() {
        val id = currentId ?: return
        val geom = currentMeta() ?: return
        val p = photo
        if (photoPageId == id && p != null) {
            recropBackup = CropBackup(p, photoDisplay, corners, detected, userRotated, cropRotation, cropReasons, forceHomography, cropPageIndex)
            screen = Screen.Crop
            return
        }
        viewModelScope.launch {
            detecting = true
            photo = null
            photoDisplay = null
            screen = Screen.Crop
            try {
                val bmp = withContext(Dispatchers.IO) { store.loadSource(id).rotatedQuarterTurns(geom.cropRotation) }
                discardStaging()
                setCrop(geom, bmp)
                recropBackup = CropBackup(bmp, photoDisplay, corners, detected, userRotated, cropRotation, cropReasons, forceHomography, cropPageIndex)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "re-crop load failed", e)
                message = "Could not open the page"
                screen = Screen.Result
            } finally {
                detecting = false
            }
        }
    }

    // ------------------------------------------------------------------ pages
    private fun geometryKey(m: PageMeta) =
        m.geometryKey(settings.outputWidth, settings.autoRotate && !m.userRotated, settings.pageFinderNet) +
            "|uv=" + settings.uvDoc + "|nets=" + settings.flatNets + "|shadows=" + settings.neuralShadows + "|shadenet=1|hr=1|lines=7"

    /**
     * Opens a page of the document: from its stored session when its geometry is unchanged (no
     * re-processing), otherwise re-processed from its source.
     */
    fun openPage(id: String) {
        if (id == currentId && handle != 0L) {
            screen = Screen.Result
            return
        }
        viewModelScope.launch {
            screen = Screen.Processing
            progress = 0f
            stage = "Opening page"
            commitCurrent()
            try {
                val m = withContext(Dispatchers.IO) { store.loadMeta(id) }
                val stored = withContext(Dispatchers.IO) { store.loadSession(id, geometryKey(m)) }
                if (stored != null) {
                    handle = 0L
                    resetResultState()
                    val h = withContext(nativeCtx) {
                        if (nativeHandle != 0L) NativeLib.release(nativeHandle)
                        nativeHandle = NativeLib.restoreSession(stored.first.absolutePath, true)
                        nativeHandle
                    }
                    if (h != 0L) {
                        discardStaging()
                        recropBackup = null
                        photo = null
                        photoDisplay = null
                        photoPageId = null
                        cropMeta = null
                        spreadNext = null
                        spreadCurrent = null
                        activate(h, m, m.params, m.protect.toList(), m.rotation, stored.third, restored = true)
                        return@launch
                    }
                    Log.w(TAG, "stored session of $id unreadable: re-processing")
                }
                val bmp = withContext(Dispatchers.IO) { store.loadSource(id).rotatedQuarterTurns(m.cropRotation) }
                discardStaging()
                recropBackup = null
                setCrop(m, bmp)
                process()
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "open page failed", e)
                message = "Could not open the page"
                screen = fallback()
            }
        }
    }

    fun deletePage(id: String) {
        viewModelScope.launch {
            if (id == currentId) closeSession()
            if (photoPageId == id) photoPageId = null
            pages.remove(id)
            pageVersions.remove(id)
            val order = pages.toList()
            withContext(Dispatchers.IO) {
                store.delete(id)
                store.saveOrder(order)
            }
            if (pages.isEmpty() && screen == Screen.Pages) screen = Screen.Home
        }
    }

    fun movePage(id: String, delta: Int) {
        val i = pages.indexOf(id)
        val j = i + delta
        if (i < 0 || j !in pages.indices) return
        pages[i] = pages[j]
        pages[j] = id
        val order = pages.toList()
        viewModelScope.launch(Dispatchers.IO) { store.saveOrder(order) }
    }

    /** Clears the document (all pages and the live session). */
    fun newDocument() {
        viewModelScope.launch {
            closeSession()
            pages.clear()
            pageVersions.clear()
            pending.clear()
            pendingTotal = 0
            photo = null
            photoDisplay = null
            photoPageId = null
            sourceJob?.join()
            withContext(Dispatchers.IO) { store.deleteAll() }
        }
    }

    suspend fun thumbnail(id: String): Bitmap? = withContext(Dispatchers.IO) { runCatching { store.loadThumbnail(id, 480) }.getOrNull() }

    /** The rendered pages in order, for export (writes the current page first). */
    suspend fun exportPages(): List<File> {
        commitCurrent()
        return pages.map { store.outputFile(it) }.filter { withContext(Dispatchers.IO) { it.exists() } }
    }

    private suspend fun closeSession() {
        if (handle == 0L) {
            currentId = null
            return
        }
        commitCurrent()
        handle = 0L
        currentId = null
        sessionGeom = null
        resetResultState()
        withContext(nativeCtx) {
            if (nativeHandle != 0L) NativeLib.release(nativeHandle)
            nativeHandle = 0L
        }
    }

    private fun resetResultState() {
        page = null
        before = null
        overlay = null
        thumbs = emptyMap()
        histogram = null
        lines = FloatArray(0)
        shownKey = null
        thumbKey = null
        histKey = null
    }

    // ------------------------------------------------------------------ processing
    /**
     * Processes the photo on the crop screen. [ignoreAreas] / [perspective]: the line editor's
     * settings; null keeps the page's own (its ignore areas only while the corners are unchanged).
     */
    fun process(ignoreAreas: List<Float>? = null, perspective: Boolean? = null) {
        val p = photo ?: return
        val id = photoPageId ?: return
        if (corners.size != 4) return
        val edited = cornersEdited()
        val c = FloatArray(8) { i -> if (i % 2 == 0) corners[i / 2].x else corners[i / 2].y }
        val force = forceHomography
        val pageIndex = cropPageIndex
        val spread = spreadCurrent?.takeIf { pageIndex == 1 }
        val same = id == currentId
        val stored = cropMeta?.takeIf { it.id == id }
        val prev = if (same) sessionGeom else stored
        val ig = ignoreAreas
            ?: prev?.takeIf { it.corners.contentEquals(c) && it.cropRotation == cropRotation }?.ignore?.toList()
            ?: emptyList()
        val po = perspective ?: prev?.perspectiveOnly ?: false
        val geom = PageMeta(
            id = id, corners = c, detected = detected?.let { d -> FloatArray(8) { i -> if (i % 2 == 0) d[i / 2].x else d[i / 2].y } },
            cropRotation = cropRotation, userRotated = userRotated, focalPx = focalPx,
            params = RenderParams(), protect = FloatArray(0), rotation = 0,
            pageIndex = pageIndex, forceHomography = force,
            spreadWith = spread?.firstId ?: stored?.spreadWith ?: (if (same) sessionGeom?.spreadWith else null),
            ignore = ig.toFloatArray(), perspectiveOnly = po,
        )
        val key = geometryKey(geom)
        val defaults = settings.defaultParams()
        val startParams = when {
            same -> params
            stored != null -> stored.params
            else -> defaults
        }
        val startProtect = when {
            same -> protect
            stored != null -> stored.protect.toList()
            else -> emptyList()
        }
        val startRotation = when {
            same -> rotation
            stored != null -> stored.rotation
            else -> 0
        }
        val autoOrient = settings.autoRotate && !userRotated
        val width = settings.outputWidth
        val useUv = settings.uvDoc
        val nets = settings.flatNets
        val shadows = settings.neuralShadows
        val focal = focalPx
        recropBackup = null
        screen = Screen.Processing
        progress = 0f
        stage = "Starting"
        viewModelScope.launch {
            if (!same) commitCurrent()
            sourceJob?.join()
            // from here on the UI no longer uses the old session; the native thread frees it
            handle = 0L
            resetResultState()
            val (h, info) = withContext(nativeCtx) {
                if (nativeHandle != 0L) NativeLib.release(nativeHandle)
                nativeHandle = 0L
                val hh = try {
                    val logits = logitsFor(p)
                    // the flattening nets on the page's box (the core checks them against the page finder
                    // and keeps the straighter page: UVDoc unless SyntheticDoc high-res is clearly straighter)
                    fun runNet(net: FlatNet): Pair<IntArray, FlatNet.Output>? {
                        val cr = NativeLib.uvCrop(p.width, p.height, c, net.model)
                        val out = runCatching { net.run(p, cr, logits, pageIndex) }.onFailure { Log.e(TAG, "flattening net ${net.model} failed", it) }.getOrNull()
                        return out?.let { cr to it }
                    }
                    if (useUv || shadows) viewModelScope.launch(Dispatchers.Main) { stage = "Flattening (neural nets)" }
                    val uv = if (useUv && !po && nets != 2) runNet(uvNet) else null
                    val sdGrid = if (useUv && !po && nets != 1) runNet(sdNet) else null
                    val first = uv ?: sdGrid
                    // the second slot: SyntheticDoc's grid next to UVDoc's
                    val secondGrid = if (uv != null) sdGrid?.second?.grid else null
                    val secondCrop = if (uv != null) sdGrid?.first else null
                    NativeLib.process(
                        p, c, edited, focal, autoOrient, width, true, logits, pageIndex, force, po, ig.toFloatArray(),
                        first?.second?.grid, first?.first, secondGrid, secondCrop,
                        null, if (shadows) shadeNet else null,
                    ) { pct, s ->
                        viewModelScope.launch(Dispatchers.Main) { progress = pct / 100f; stage = s }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "process failed", e)
                    0L
                }
                if (hh == 0L) return@withContext 0L to JSONObject().put("ok", false).put("error", "no session")
                val j = JSONObject(NativeLib.info(hh))
                if (j.optBoolean("ok")) nativeHandle = hh else NativeLib.release(hh)
                hh to j
            }
            if (!info.optBoolean("ok")) {
                currentId = null
                sessionGeom = null
                message = "Processing failed: ${info.optString("error")}"
                screen = Screen.Crop
                return@launch
            }
            if (id !in pages) {
                val i = spread?.let { pages.indexOf(it.firstId) } ?: -1
                if (spread != null && i >= 0) pages.add(if (secondGoesFirst(spread)) i else i + 1, id) else pages.add(id)
                pageVersions[id] = 0
            }
            val order = pages.toList()
            val meta = geom.copy(params = startParams, protect = startProtect.toFloatArray(), rotation = startRotation)
            withContext(Dispatchers.IO) {
                runCatching {
                    store.saveMeta(meta)
                    store.saveOrder(order)
                    // the first page of the spread learns its partner
                    if (spread != null) store.saveMeta(store.loadMeta(spread.firstId).copy(spreadWith = id))
                }.onFailure { Log.e(TAG, "meta save failed", it) }
            }
            if (spread != null) {
                spreadCurrent = null
                message = "Book spread: added as 2 pages (${if (settings.bookRightToLeft) "right" else "left"} page first)"
            }
            val text = describe(info, edited, force, po, ig.size / 4)
            // the processed page is kept on disk: opening it again needs no re-processing
            storeSession(id, key, info, text, withContext(nativeCtx) { if (h == nativeHandle) NativeLib.snapshot(h) else 0L })
            activate(h, geom, startParams, startProtect, startRotation, text, restored = false)
            // a book spread: its second page follows from the same photo
            val next = spreadNext
            if (next != null && pageIndex == 0) {
                next.firstRot = info.optInt("rot", -1)
                startSpreadPage(next)
            }
        }
    }

    /** Makes [h] the live session of page [geom] and shows it on the result screen. */
    private suspend fun activate(h: Long, geom: PageMeta, p: RenderParams, prot: List<Float>, rot: Int, text: String, restored: Boolean) {
        val (size, ln) = withContext(nativeCtx) { NativeLib.pageSize(h) to NativeLib.lines(h) }
        pageWidth = size[0]
        pageHeight = size[1]
        currentId = geom.id
        sessionGeom = geom
        params = p
        protect = prot
        rotation = rot
        lines = ln
        ignore = geom.ignore.toList()
        erase = geom.erase.toList()
        if (erase.isNotEmpty()) {
            val rects = erase.toFloatArray()
            withContext(nativeCtx) { if (h == nativeHandle) NativeLib.setErase(h, rects) }
        }
        outVersion++
        details = text
        firstRenderDetails = true
        handle = h
        requestRender(false)
        awaitRender()
        // a stored session renders what out.jpg already holds
        if (restored) committedVersion = outVersion
        if (screen == Screen.Processing) screen = Screen.Result
        fetchBefore(h)
        if (showWrinkles) fetchOverlay(h)
    }

    /** Writes a session snapshot as page [id]'s stored session (frees the snapshot). */
    private fun storeSession(id: String, key: String, info: JSONObject, text: String, snap: Long) {
        if (snap == 0L) return
        viewModelScope.launch(Dispatchers.IO) {
            sessionStoreMutex.withLock {
                var used = false
                runCatching {
                    store.saveSession(id, key, info, text) { dir ->
                        used = true
                        NativeLib.writeSnapshot(snap, dir.absolutePath)
                    }
                }.onFailure { Log.e(TAG, "session save failed", it) }
                // the page is gone: writing to a missing directory fails but frees the snapshot
                if (!used) NativeLib.writeSnapshot(snap, File(store.root, "missing").absolutePath)
            }
        }
    }

    /**
     * Flattens the current page again with the line editor's ignore areas and/or the unwarp
     * turned on or off ([perspective]); the filters, protected areas and rotation are kept.
     */
    fun reflatten(perspective: Boolean? = null) {
        val id = currentId ?: return
        val geom = currentMeta() ?: return
        val areas = ignore
        reopenLines = true
        val p = photo
        if (photoPageId == id && p != null && recropBackup == null) {
            process(areas, perspective)
            return
        }
        viewModelScope.launch {
            screen = Screen.Processing
            progress = 0f
            stage = "Opening the photo"
            try {
                val bmp = withContext(Dispatchers.IO) { store.loadSource(id).rotatedQuarterTurns(geom.cropRotation) }
                discardStaging()
                recropBackup = null
                setCrop(geom, bmp)
                process(areas, perspective)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "reflatten failed", e)
                message = "Could not open the photo"
                screen = Screen.Result
            }
        }
    }

    /** Line editor: ignores line [i] of [lines] (or uses it again when it is ignored). */
    fun toggleLine(i: Int) {
        val l = lines
        if (5 * i + 4 >= l.size) return
        val cx = (l[5 * i + 1] + l[5 * i + 3]) / 2
        val cy = (l[5 * i + 2] + l[5 * i + 4]) / 2
        if (ignoreAt(cx, cy) >= 0) {
            removeIgnoreAt(cx, cy)
            return
        }
        val pad = 0.008f
        ignore = ignore + listOf(
            (minOf(l[5 * i + 1], l[5 * i + 3]) - pad).coerceIn(0f, 1f), (minOf(l[5 * i + 2], l[5 * i + 4]) - pad).coerceIn(0f, 1f),
            (maxOf(l[5 * i + 1], l[5 * i + 3]) + pad).coerceIn(0f, 1f), (maxOf(l[5 * i + 2], l[5 * i + 4]) + pad).coerceIn(0f, 1f),
        )
    }

    /** Index of the first ignore area containing the page point (fractions), or -1. */
    fun ignoreAt(x: Float, y: Float): Int {
        val a = ignore
        var k = 0
        while (k + 3 < a.size) {
            if (x >= a[k] && x <= a[k + 2] && y >= a[k + 1] && y <= a[k + 3]) return k / 4
            k += 4
        }
        return -1
    }

    /** Removes every ignore area containing the page point. */
    fun removeIgnoreAt(x: Float, y: Float) {
        ignore = ignore.chunked(4).filterNot { x >= it[0] && x <= it[2] && y >= it[1] && y <= it[3] }.flatten()
    }

    /** Line editor: ignores every line inside the rectangle (page fractions, any corner order). */
    fun addIgnore(x0: Float, y0: Float, x1: Float, y1: Float) {
        val a = minOf(x0, x1).coerceIn(0f, 1f)
        val b = minOf(y0, y1).coerceIn(0f, 1f)
        val c = maxOf(x0, x1).coerceIn(0f, 1f)
        val d = maxOf(y0, y1).coerceIn(0f, 1f)
        if (c - a < 0.01f || d - b < 0.01f) return
        ignore = ignore + listOf(a, b, c, d)
    }

    fun clearIgnore() {
        ignore = emptyList()
    }

    /** Line editor closed without flattening again: back to the areas the page uses. */
    fun revertIgnore() {
        ignore = sessionGeomState?.ignore?.toList() ?: emptyList()
    }

    private fun describe(info: JSONObject, edited: Boolean, forced: Boolean, perspectiveOnly: Boolean, nIgnore: Int): String {
        val sb = StringBuilder()
        if (perspectiveOnly) sb.append("Unwarp: off (perspective correction only)\n")
        else if (nIgnore > 0) sb.append("Unwarp: lines in $nIgnore area${if (nIgnore == 1) "" else "s"} ignored\n")
        val mode = info.optString("mode")
        val gate = info.optJSONArray("needs_manual_crop")
        val reasons = if (gate != null) List(gate.length()) { gate.optString(it) } else emptyList()
        sb.append(
            when {
                mode == "uvdoc" -> "Geometry: %s neural flattening (outline agreement %.2f)".format(
                    if (info.optInt("uv_model") == 2) "SyntheticDoc high-res" else "UVDoc", info.optDouble("uv_agreement")) +
                    (info.optJSONArray("uv_spread")?.let { a ->
                        if (a.optDouble(0, -1.0) >= 0 && a.optDouble(1, -1.0) >= 0)
                            "\nNets compared: text-row spread %.2f / %.2f deg".format(a.optDouble(0), a.optDouble(1)) else ""
                    } ?: "") +
                    if (info.optBoolean("lines_rejected")) "\nLine refinement: skipped (it would have distorted the page, %.2f)".format(info.optDouble("line_distortion"))
                    else "\nLine refinement: applied"
                edited -> "Geometry: homography (corners adjusted by hand)"
                reasons.isNotEmpty() || forced -> "Geometry: homography (doubtful automatic crop${if (reasons.isNotEmpty()) ": " + reasons.joinToString("; ") else ""})"
                info.optBoolean("uv_rejected") && mode == "surface" ->
                    "Geometry: page surface model (UVDoc disagreed with the page outline, %.2f)".format(info.optDouble("uv_agreement"))
                info.optBoolean("fell_back") -> "Geometry: homography (surface fit rejected, outline residual %.1f px)".format(info.optDouble("fit_median_px"))
                mode == "surface" -> "Geometry: page surface model (outline residual %.2f px)".format(info.optDouble("fit_median_px"))
                else -> "Geometry: $mode"
            },
        )
        if (info.has("rot")) sb.append("\nOrientation: ").append(
            when (info.optInt("rot")) { 0 -> "turned 90 degrees clockwise"; 2 -> "turned 90 degrees counter-clockwise"; 1 -> "turned 180 degrees"; else -> "as photographed" },
        )
        if (info.has("finder")) {
            sb.append("\nPage finder: ").append(if (info.optString("finder") == "net") "net + edge snapping" else "colour threshold")
            if (info.optBoolean("whole_frame")) sb.append(" (no page found: whole frame)")
            if (info.optInt("n_pages", 1) == 2) sb.append(", book spread page ").append(info.optInt("page_index") + 1).append(" of 2")
            if (info.optString("finder") == "net") sb.append("\n  net margin %.2f, edge strength %.2f, at limit %.2f".format(
                info.optDouble("net_margin"), info.optDouble("edge_strength"), info.optDouble("edge_at_limit")))
        }
        if (info.optBoolean("shadows_removed")) sb.append(if (info.optBoolean("shade_net")) "\nShadows: removed (shadow net)" else "\nShadows: removed (shading map)")
        if (info.optDouble("fingers_erased", 0.0) > 0) sb.append("\nFingers: erased (%.1f%% of the page)".format(100 * info.optDouble("fingers_erased")))
        sb.append("\nResample: ").append(if (info.optBoolean("gpu_resample")) "GPU" else "CPU fallback")
        sb.append("\n").append(gpuInfo)
        val ms = info.optJSONObject("ms")
        if (ms != null) {
            sb.append("\n\nStage times (ms):")
            for (k in ms.keys()) sb.append("\n").append(k).append(": ").append(ms.optLong(k))
        }
        return sb.toString()
    }

    private fun currentMeta(): PageMeta? =
        sessionGeom?.copy(params = params, protect = protect.toFloatArray(), rotation = rotation, erase = erase.toFloatArray())

    /**
     * Writes the current page's settings and (if it changed) its rendered output to disk.
     * Waits for the render of the latest parameters first.
     */
    private suspend fun commitCurrent() = withContext(NonCancellable) {
        commitMutex.withLock {
            val id = currentId ?: return@withLock
            if (id !in pages || handle == 0L) return@withLock
            awaitRender()
            val meta = currentMeta() ?: return@withLock
            val v = outVersion
            val pg = page
            val rot = rotation
            // copy on the native thread: no render can write the buffer meanwhile
            val snap = if (v != committedVersion && pg != null) withContext(nativeCtx) { pg.rotatedCopy(rot) } else null
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    store.saveMeta(meta)
                    if (snap != null) store.saveOutput(id, snap)
                }.onFailure { Log.e(TAG, "commit failed", it) }.isSuccess
            }
            snap?.recycle()
            if (ok && snap != null) {
                committedVersion = v
                pageVersions[id] = (pageVersions[id] ?: 0) + 1
            }
        }
    }

    // ------------------------------------------------------------------ result edits
    fun selectFilter(f: FilterKind) {
        if (f == params.filter) return
        params = params.copy(filter = f)
        settings.defaultFilter = f
        requestRender(false)
    }

    fun setStrength(v: Float) {
        params = params.copy(strength = v.coerceIn(0f, 1f))
        requestRender(true)
    }

    /** The strength slider was released: remember it for the next scan. */
    fun strengthDone() {
        settings.defaultStrength = params.strength
    }

    fun setSolid(v: Boolean) {
        if (v == params.solid) return
        params = params.copy(solid = v)
        requestRender(false)
    }

    fun setBgCleanup(v: Float) {
        params = params.copy(bgCleanup = v.coerceIn(0f, 1f))
        requestRender(true)
    }

    fun setLevels(black: Float, white: Float, gamma: Float) {
        params = params.copy(black = black.coerceIn(0f, 0.5f), white = white.coerceIn(0.5f, 1f), gamma = gamma.coerceIn(0.3f, 3f))
        requestRender(true)
    }

    fun resetAdjustments() {
        val d = RenderParams()
        params = params.copy(bgCleanup = d.bgCleanup, black = d.black, white = d.white, gamma = d.gamma)
        requestRender(false)
    }

    fun openAdjust(open: Boolean) {
        adjustOpen = open
        if (open) requestRender(false)    // histogram
    }

    /** Adds a protected rectangle (page fractions, any corner order). */
    fun addProtect(x0: Float, y0: Float, x1: Float, y1: Float) {
        val a = minOf(x0, x1).coerceIn(0f, 1f)
        val b = minOf(y0, y1).coerceIn(0f, 1f)
        val c = maxOf(x0, x1).coerceIn(0f, 1f)
        val d = maxOf(y0, y1).coerceIn(0f, 1f)
        if (c - a < 0.01f || d - b < 0.01f) return
        protect = protect + listOf(a, b, c, d)
        requestRender(false)
    }

    /** Removes the protected rectangles containing the page point (fractions). */
    fun removeProtectAt(x: Float, y: Float) {
        val next = protect.chunked(4).filterNot { x >= it[0] && x <= it[2] && y >= it[1] && y <= it[3] }.flatten()
        if (next.size == protect.size) return
        protect = next
        requestRender(false)
    }

    /** Removes the protected rectangle drawn last. */
    fun undoProtect() {
        if (protect.size < 4) return
        protect = protect.dropLast(4)
        requestRender(false)
    }

    // ---- Erase tool
    /** Fills a rectangle (page fractions, any corner order) with the paper around it. */
    fun addErase(x0: Float, y0: Float, x1: Float, y1: Float) {
        val a = minOf(x0, x1).coerceIn(0f, 1f)
        val b = minOf(y0, y1).coerceIn(0f, 1f)
        val c = maxOf(x0, x1).coerceIn(0f, 1f)
        val d = maxOf(y0, y1).coerceIn(0f, 1f)
        if (c - a < 0.005f || d - b < 0.005f) return
        setEraseRects(erase + listOf(a, b, c, d))
    }

    fun undoErase() {
        if (erase.size >= 4) setEraseRects(erase.dropLast(4))
    }

    fun clearErase() {
        if (erase.isNotEmpty()) setEraseRects(emptyList())
    }

    /** Removes the erased rectangles containing the page point (fractions). */
    fun removeEraseAt(x: Float, y: Float) {
        val next = erase.chunked(4).filterNot { x >= it[0] && x <= it[2] && y >= it[1] && y <= it[3] }.flatten()
        if (next.size != erase.size) setEraseRects(next)
    }

    private fun setEraseRects(rects: List<Float>) {
        val h = handle
        if (h == 0L) return
        erase = rects
        val arr = rects.toFloatArray()
        viewModelScope.launch {
            rendering = true
            withContext(nativeCtx) { if (h == nativeHandle) NativeLib.setErase(h, arr) }
            if (handle != h) return@launch
            thumbKey = null
            histKey = null
            pageEdit++
            outVersion++
            requestRender(false)
        }
    }

    fun clearProtect() {
        if (protect.isEmpty()) return
        protect = emptyList()
        requestRender(false)
    }

    fun rotateResult() {
        rotation = (rotation + 1) % 4
        outVersion++
    }

    /** One tap for an upside-down page (there is no automatic 180-degree turn). */
    fun rotateResult180() {
        rotation = (rotation + 2) % 4
        outVersion++
    }

    fun setWrinkles(v: Boolean) {
        showWrinkles = v
        val h = handle
        if (v && overlay == null && h != 0L) fetchOverlay(h)
    }

    fun holdStarted() {
        if (showHoldHint) {
            showHoldHint = false
            settings.holdHintShown = true
        }
    }

    // ------------------------------------------------------------------ rendering
    private fun requestRender(debounce: Boolean) {
        wake.trySend(debounce)
    }

    private fun wantedKey() = MainKey(handle, params, protect, pageEdit)

    /** Waits (bounded) until the page on screen matches the current parameters. */
    private suspend fun awaitRender() {
        if (handle == 0L) return
        requestRender(false)
        withTimeoutOrNull(30_000) { snapshotFlow { handle == 0L || shownKey == wantedKey() }.first { it } }
    }

    private suspend fun renderLoop() {
        for (debounce in wake) {
            if (debounce) delay(RENDER_DEBOUNCE_MS)
            try {
                renderPending()
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "render failed", e)
            } finally {
                rendering = false
            }
        }
    }

    /**
     * Brings the main render, then the filter thumbnails, then (when the adjustments panel is
     * open) the histogram up to date with the latest parameters; a change made meanwhile is
     * picked up before the remaining lower-priority work.
     */
    private suspend fun renderPending() {
        while (true) {
            val h = handle
            if (h == 0L) return
            val key = wantedKey()
            if (key != shownKey) {
                rendering = true
                val shown = page
                val r = withContext(nativeCtx) {
                    if (h != nativeHandle) return@withContext null
                    val size = NativeLib.pageSize(h)
                    val target = pickBuffer(size[0], size[1], shown)
                    val json = NativeLib.render(h, key.p.toArray(), key.protect.toFloatArray(), true, target)
                    target to json
                } ?: return
                if (handle != h) continue
                page = r.first
                shownKey = key
                outVersion++
                if (firstRenderDetails) {
                    firstRenderDetails = false
                    runCatching {
                        val j = JSONObject(r.second)
                        val sb = StringBuilder("\n\nRender:")
                        for (k in j.keys()) sb.append("\n").append(k).append(": ").append(j.opt(k))
                        details += sb.toString()
                    }
                }
                continue
            }
            rendering = false
            val tp = key.p.copy(filter = FilterKind.Original, solid = false)
            if (thumbKey != (h to tp)) {
                val tw = THUMB_W
                val th = (THUMB_W.toFloat() * pageHeight / pageWidth.coerceAtLeast(1)).roundToInt().coerceIn(32, 4 * THUMB_W)
                var complete = true
                for (f in FilterKind.entries) {
                    if (handle != h || wantedKey() != key) { complete = false; break }
                    val b = withContext(nativeCtx) {
                        if (h != nativeHandle) return@withContext null
                        val t = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888)
                        NativeLib.renderThumb(h, key.p.copy(filter = f, solid = false).toArray(), t)
                        t
                    } ?: return
                    if (handle != h) return
                    thumbs = thumbs + (f to b)
                }
                if (complete) thumbKey = h to tp
                continue
            }
            if (adjustOpen) {
                val hp = key.p.copy(black = 0f, white = 1f, gamma = 1f, solid = false)
                if (histKey != (h to hp)) {
                    val hist = withContext(nativeCtx) { if (h == nativeHandle) NativeLib.histogram(h, hp.toArray()) else null } ?: return
                    if (handle != h) return
                    histogram = hist
                    histKey = h to hp
                    continue
                }
            }
            return
        }
    }

    /** A page-size target that is not the bitmap on screen ([nativeCtx] only). */
    private fun pickBuffer(w: Int, h: Int, shown: Bitmap?): Bitmap {
        for (b in buffers) if (b != null && b !== shown && b.width == w && b.height == h) return b
        val slot = if (buffers[0] === shown) 1 else 0
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
            it.setHasMipMap(true)       // smooth downscaling of the full page on screen
            buffers[slot] = it
        }
    }

    private fun fetchBefore(h: Long) {
        viewModelScope.launch {
            val b = withContext(nativeCtx) {
                if (h != nativeHandle) return@withContext null
                val size = NativeLib.pageSize(h)
                Bitmap.createBitmap(size[0], size[1], Bitmap.Config.ARGB_8888).also {
                    it.setHasMipMap(true)
                    NativeLib.renderBefore(h, it)
                }
            }
            if (b != null && handle == h) before = b
        }
    }

    private fun fetchOverlay(h: Long) {
        viewModelScope.launch {
            val r = withContext(nativeCtx) {
                if (h != nativeHandle) return@withContext null
                val size = NativeLib.pageSize(h)
                val b = Bitmap.createBitmap(size[0], size[1], Bitmap.Config.ARGB_8888)
                b.setHasMipMap(true)
                val top = NativeLib.renderOverlay(h, true, true, b)
                b to top
            }
            if (r != null && handle == h) {
                overlay = r.first
                overlayMaxPx = r.second
            }
        }
    }

    override fun onCleared() {
        // release on the native thread, after any render still running there
        nativeExec.execute {
            if (nativeHandle != 0L) NativeLib.release(nativeHandle)
            nativeHandle = 0L
        }
        nativeExec.shutdown()
        super.onCleared()
    }

    companion object {
        const val TAG = "FlatScan"
        const val RENDER_DEBOUNCE_MS = 120L
        const val THUMB_W = 96
    }
}

/** A new bitmap: this one turned [q] clockwise quarter turns (a plain copy for 0). */
fun Bitmap.rotatedCopy(q: Int): Bitmap {
    val n = ((q % 4) + 4) % 4
    return if (n == 0) copy(Bitmap.Config.ARGB_8888, false) else rotatedQuarterTurns(n)
}
