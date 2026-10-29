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
import kotlinx.coroutines.CompletableDeferred
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
import com.albus.flatscan.data.Library
import com.albus.flatscan.data.StorageUsage
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

enum class Screen { Home, Camera, Crop, Processing, Result, Pages, Settings, Licenses, Trash, Sync, Advanced, Storage }

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
    val library = Library.get(app)
    private val store = library.store

    var screen by mutableStateOf(Screen.Home)

    /** The paper screens' theme (Settings > Appearance): Settings.THEME_LIGHT, THEME_DARK or THEME_SYSTEM. */
    var themeMode by mutableStateOf(settings.themeMode)
        private set

    fun setTheme(mode: String) {
        settings.themeMode = mode
        themeMode = settings.themeMode
    }
    private var returnTo = Screen.Home
    private var pagesReturnTo = Screen.Home
    var message by mutableStateOf<String?>(null)

    /** An action offered with one message ([showMessage]); shown only while that message is. */
    class MessageAction(val message: String, val label: String, val run: () -> Unit)
    var messageAction by mutableStateOf<MessageAction?>(null)
        private set

    /** [text] with an action button ([label], e.g. Undo) for 5 s. */
    fun showMessage(text: String, label: String, run: () -> Unit) {
        messageAction = MessageAction(text, label, run)
        message = text
    }

    private val nativeExec = Executors.newSingleThreadExecutor { r -> Thread(r, "flatscan-native") }
    private val nativeCtx = nativeExec.asCoroutineDispatcher()
    /** The live session as the native thread sees it; read and written only on [nativeCtx]. */
    private var nativeHandle = 0L
    /** Page-size render targets (ping-pong, so the one on screen is never written); [nativeCtx] only. */
    private val buffers = arrayOfNulls<Bitmap>(2)

    // ---- document
    /** The open document (null: none yet; one is created when its first page is added). */
    var docId by mutableStateOf<String?>(null)
        private set
    /** Its title. */
    var docTitle by mutableStateOf<String?>(null)
        private set
    /** Page ids in document order. */
    val pages = mutableStateListOf<String>()
    /** Bumped whenever a page's out.jpg is rewritten (thumbnail cache key). */
    val pageVersions = mutableStateMapOf<String, Int>()
    /** The page with the live session (result screen). */
    var currentId by mutableStateOf<String?>(null)
        private set
    /** Document or whiteboard mode for new scans and imports (the camera screen's switch, remembered). */
    var captureMode by mutableStateOf(settings.captureMode)
        private set

    fun chooseCaptureMode(m: CaptureMode) {
        captureMode = m
        settings.captureMode = m
    }

    /** The mode of the page on the result screen. */
    val pageMode: CaptureMode get() = sessionGeomState?.mode ?: CaptureMode.Document

    /** The mode of the photo on the crop screen: its page's, or [captureMode] for a new page. */
    private var cropMode = CaptureMode.Document
    /** Whiteboard mode: the board detection saw a dark slide (the Dark slide filter for the new page). */
    private var boardDark = false

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
    private val logitsLock = Any()
    /**
     * Page-finder output for one photo bitmap (detection and processing use the same photo), with its
     * processing-details entry (NativeLib.pageFinder timing).
     */
    private var logitsCache: Triple<Bitmap, FloatArray?, String?>? = null
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

    /** The page being processed is one page of a book spread (the processing animation curls it at the gutter). */
    val processingBookPage: Boolean get() = spreadCurrent != null

    /** The processing animation has landed (PaperUnfold's onFinished) for this run: the result screen waits for it. */
    var unfoldLanded by mutableStateOf(false)

    /** A page's finger and erase edits (PageMeta's), carried over when it is processed again. */
    private class PageEdits(val erase: FloatArray, val fingersAuto: Boolean, val strokes: FloatArray, val fill: Int)

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
    /** Light, colour and page shape on top of the core's render (PageLook; default: unchanged). */
    var look by mutableStateOf(PageLook())
        private set
    /** Undo / redo of the current page's edits ([trackEdits]). */
    val history = EditHistory()
    /** "Apply to all pages" is re-rendering the other pages. */
    var applyingAll by mutableStateOf(false)
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
    /** Opening a stored page: its last rendering (out.jpg), shown while the session is restored ([openPage]). */
    var openingPreview by mutableStateOf<Bitmap?>(null)
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
    /** Lines drawn along rules the unwarp missed (PageMeta.userLines layout, being edited). */
    var userLines by mutableStateOf(listOf<Float>())
        private set
    /** The line brush's strokes (PageMeta.linePaint layout, being edited). */
    var linePaint by mutableStateOf(listOf<Float>())
        private set
    /** The unwarp straightens every line it finds (off: only painted and drawn lines; being edited). */
    var autoLines by mutableStateOf(true)
        private set
    /** The line editor's state before one of its edits (its own undo). */
    private data class LineEdit(val ignore: List<Float>, val drawn: List<Float>, val paint: List<Float>, val auto: Boolean)
    /** The line editor's own undo: its state before each of its edits, since it opened. */
    private val lineHistory = ArrayList<LineEdit>()
    /** How many line edits Undo can take back. */
    var lineUndoSteps by mutableIntStateOf(0)
        private set
    /** The line editor has changes to flatten with: ignore areas, drawn lines, the brush or the automatic lines switch. */
    val lineEditsChanged: Boolean
        get() = ignoreChanged || userLines != (sessionGeomState?.userLines?.toList() ?: emptyList<Float>()) ||
            linePaint != (sessionGeomState?.linePaint?.toList() ?: emptyList<Float>()) || autoLines != (sessionGeomState?.autoLines ?: true)
    /** The page's shadows are removed (Page panel switch). */
    val removeShadows: Boolean get() = sessionGeomState?.removeShadows != false
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
            // the single document of earlier versions (in the cache) becomes the first document of the history
            val legacy = runCatching { library.importLegacy(DocumentStore.legacy(app)) }
                .onFailure { Log.e(TAG, "legacy import failed", it) }.getOrNull()
            runCatching { library.purgeTrash() }.onFailure { Log.e(TAG, "trash purge failed", it) }
            refreshSearchText()
            val doc = (legacy ?: settings.lastDocument)?.let { library.document(it) }?.takeIf { it.deletedAt == null }
            if (doc != null && docId == null && pages.isEmpty()) {
                docId = doc.id
                docTitle = doc.title
                pages.addAll(library.pageIds(doc.id))
            }
            runCatching {
                library.dropEmpty(docId)
                library.pruneOrphans(keep = pages + listOfNotNull(photoPageId, currentId))
            }.onFailure { Log.e(TAG, "library clean-up failed", it) }
        }
        viewModelScope.launch { renderLoop() }
        // dispatched (not Main.immediate): trackEdits reads editState() at once, whose fields (fingersAuto...) are
        // declared further down and not yet initialised while this init block runs
        viewModelScope.launch(Dispatchers.Main) { trackEdits(this@ScanViewModel, history) }
    }

    // ---- self-test / benchmark
    var selfTestRunning by mutableStateOf(false)
        private set
    var selfTestStage by mutableStateOf("")
        private set
    var selfTestReport by mutableStateOf("")
        private set

    /** [report] (debug batch mode `--ez selftest true`): the report is also written there. */
    fun runSelfTest(runs: Int = 5, report: java.io.File? = null) {
        if (selfTestRunning || !hasSelfTest) return
        selfTestRunning = true
        selfTestReport = ""
        viewModelScope.launch {
            val app = getApplication<Application>()
            val text = withContext(Dispatchers.Default) { CrashGuard.guard(app, "the self-test", gpu = !currentPlan().cpuOnly || Accel.candidates().isNotEmpty()) {
                val ref = java.io.File(app.cacheDir, "photo1_reference_warpmap.npz")
                app.assets.open("selftest/photo1_reference_warpmap.npz").use { i -> ref.outputStream().use { i.copyTo(it) } }
                val bmp = selfTestPhoto()
                val thermalBefore = thermalStatus(app)
                val json = NativeLib.selfTest(bmp, ref.absolutePath, runs) { pct, s ->
                    viewModelScope.launch(Dispatchers.Main) { selfTestStage = "$s ($pct%)" }
                }
                val thermalMid = thermalStatus(app)
                val nets = runCatching {
                    benchNets(bmp, runs) { s -> viewModelScope.launch(Dispatchers.Main) { selfTestStage = s } }
                }.onFailure { Log.e(TAG, "net benchmark failed", it) }
                formatSelfTest(JSONObject(json), nets, thermalBefore, thermalMid, thermalStatus(app))
            } }
            report?.let { f -> runCatching { f.parentFile?.mkdirs(); f.writeText(text) } }
            selfTestReport = text
            selfTestRunning = false
        }
    }

    /**
     * The self-test's photo (assets/selftest/photo1.webp, 1659 x 2212: lossless WebP, the pixels of the earlier
     * photo1.png) and its reference warp map: debug builds only (src/debug/assets), release builds have no self-test.
     */
    val hasSelfTest: Boolean by lazy {
        runCatching { getApplication<Application>().assets.list("selftest").orEmpty().contains("photo1.webp") }.getOrDefault(false)
    }

    private fun selfTestPhoto(): Bitmap = assetPhoto("selftest/photo1.webp")

    /**
     * Auto's benchmark photo (assets/bench/photo1.webp): the self-test photo as lossy WebP (quality 85, 0.5 MB
     * instead of 2.3), in every build. The benchmark compares each backend's nets with the CPU's on the same
     * input, so it needs no exact pixels.
     */
    private fun benchPhoto(): Bitmap = assetPhoto("bench/photo1.webp")

    private fun assetPhoto(path: String): Bitmap {
        val opts = android.graphics.BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
        }
        return getApplication<Application>().assets.open(path).use { android.graphics.BitmapFactory.decodeStream(it, null, opts) }!!
    }

    /**
     * Settings > Acceleration > "Run the benchmark": Auto's benchmark now (its result is kept as Auto's
     * choice for this SoC and app version), with its report in the self-test area.
     */
    fun runAccelBenchmark(candidates: String = Accel.candidates(), reportFile: java.io.File? = null) {
        if (selfTestRunning) return
        enableGpu()   // asked for explicitly: the GPU is tried again even after a crash turned it off
        selfTestRunning = true
        selfTestReport = ""
        viewModelScope.launch {
            val report = withContext(Dispatchers.Default) {
                runCatching {
                    synchronized(netsLock) {
                        val stage = { s: String -> viewModelScope.launch(Dispatchers.Main) { selfTestStage = s }; Unit }
                        val stored = runAutoBenchmark(stage, candidates)
                        if (settings.acceleration == Accel.AUTO) {
                            configureNets(currentPlan())
                            if (!currentPlan().cpuOnly) warmGpu(stage)
                        }
                        formatBench(stored)
                    }
                }.getOrElse { "Acceleration benchmark FAILED: ${it.message ?: it}" }
            }
            reportFile?.let { f -> runCatching { f.parentFile?.mkdirs(); f.writeText(report) } }
            selfTestReport = report
            selfTestRunning = false
        }
    }

    // ---- the nets' backends (Settings > Acceleration)
    /** The groups' backends as last given to [NativeLib.configureNets]. */
    @Volatile private var netPlan = Accel.Plan(Accel.CPU, "", "", "", "", "")

    /** Debug (`--ei threads N`): the CPU nets' threads; 0: the default, min(cores, 4). */
    @Volatile var netThreads = 0
    private val netsLock = Any()
    /** [configureNets] ran at least once. */
    @Volatile private var netsConfigured = false
    /** [configureNets]' own lock (not [netsLock]: a benchmark under it may be waiting for the first configuration). Before
     *  init, which starts the first configuration on the native thread. */
    private val configLock = Any()
    /**
     * Debug (`--ez gpu_warm true`): every group on the GPU, the GPU sessions made as after Auto's benchmark
     * ([warmGpu]), the setup ms of each net in [report]; then the setting's backends again.
     */
    fun debugWarmGpu(report: File) {
        viewModelScope.launch(Dispatchers.Default) {
            val lines = mutableListOf<String>()
            synchronized(netsLock) {
                configureNets(Accel.fixed(Accel.GPU)!!)
                val t0 = System.nanoTime()
                warmGpu { lines += it }
                lines += "total ${msSince(t0).roundToInt()} ms; " + NativeLib.prewarmGpu(null).lineSequence().filter { it.isNotBlank() }.joinToString("; ", "again: ")
                configureNets(currentPlan())
            }
            runCatching { report.parentFile?.mkdirs(); report.writeText(lines.joinToString("\n") + "\n") }
        }
    }

    /** The launch's GPU set-up ([setUpGpu]) while it runs: its stage text; null: none running. Before init, which starts it. */
    @Volatile private var gpuSetup: String? = null
    /** Processing waiting for that set-up ([ensureNets]): where its stage texts go. */
    @Volatile private var gpuSetupSink: ((String) -> Unit)? = null

    init {
        // a crash in guarded native work last time (CrashGuard): report it, and keep the nets off a GPU that crashed
        val app = getApplication<Application>()
        CrashGuard.installHandler(app)
        CrashGuard.check(app)?.let { r ->
            selfTestReport = r
            message = r.lineSequence().first() + (if (CrashGuard.gpuDisabled(app)) " - the GPU is off now" else "") + " (details: Settings > Self-test)"
        }
        Accel.gpuOff = CrashGuard.gpuDisabled(app)
        // the CPU until processing asks for the setting's backends (ensureNets). On the native thread: the first
        // NativeLib call loads libflatscan.so (0.85 s on the emulator), which kept the first frame waiting on the
        // main thread; a net asked for before this ran waits for it (jni_bridge.cpp assets_locked)
        viewModelScope.launch(nativeCtx) { synchronized(configLock) { if (!netsConfigured) configureNets(netPlan) } }
        setUpGpu()
    }

    /** Settings: the user asks for the GPU again after a crash turned it off. */
    fun enableGpu() {
        val app = getApplication<Application>()
        if (!CrashGuard.gpuDisabled(app)) return
        CrashGuard.enableGpu(app)
        Accel.gpuOff = false
    }

    /** The current setting's plan: a fixed setting's, Auto's stored choice, or the CPU while Auto has not run. */
    fun currentPlan(): Accel.Plan = Accel.fixed(settings.acceleration) ?: Accel.autoPlan(settings.autoChoice)
        ?: Accel.Plan(Accel.AUTO, "", "", "", "", "", "not benchmarked yet")

    private fun configureNets(plan: Accel.Plan, staticModels: Boolean = false) = synchronized(configLock) {
        val app = getApplication<Application>()
        // the LiteRT GPU's serialised kernels (the folder kept its earlier name)
        val cache = File(app.noBackupFilesDir, "qnn_cache").apply { mkdirs() }
        NativeLib.configureNets(app.assets, cache.absolutePath, plan.finder, plan.flat, plan.hr, plan.shade, plan.rule, netThreads, staticModels)
        netPlan = plan
        netsConfigured = true
    }

    /**
     * The nets on the current setting's backends; with Auto and no benchmark for this SoC and app
     * version yet, the benchmark runs first ([stage]: its progress; it takes seconds to a minute, as the GPU
     * compiles its kernels on first use; none on a phone without a GPU candidate) and its choice is kept.
     */
    private fun ensureNets(stage: (String) -> Unit): Accel.Plan {
        // the launch's set-up (setUpGpu) may hold netsLock: its stage texts meanwhile
        gpuSetup?.let { stage(it) }
        gpuSetupSink = stage
        try {
            return synchronized(netsLock) { benchAndWarm(stage) }
        } finally {
            gpuSetupSink = null
        }
    }

    /**
     * Auto's benchmark when this SoC and app version has none, the setting's backends, and after a benchmark the
     * app's GPU sessions (NativeLib.prewarmGpu: the benchmark left the groups empty, and the GPU would compile and
     * tune on the page). Caller holds [netsLock].
     */
    private fun benchAndWarm(stage: (String) -> Unit): Accel.Plan {
        var benched = false
        if (settings.acceleration == Accel.AUTO && Accel.autoPlan(settings.autoChoice) == null) {
            stage(GPU_SETUP)
            runAutoBenchmark(stage)
            benched = true
        }
        val plan = currentPlan().also { if (it != netPlan) configureNets(it) }
        if (benched && !plan.cpuOnly) warmGpu(stage)
        return plan
    }

    private fun warmGpu(stage: (String) -> Unit) {
        val r = runCatching {
            CrashGuard.guard(getApplication(), "setting up the GPU", gpu = true) { NativeLib.prewarmGpu { _, s -> stage("$GPU_SETUP: $s") } }
        }.onFailure { Log.e(TAG, "GPU warm-up failed", it) }.getOrNull()
        Log.i(TAG, "GPU warm-up: ${r?.trim()}")
        r?.let { stage(it.trim()) }
    }

    /**
     * After an install or a system update (Auto with no benchmark for this SoC and app version): Auto's benchmark
     * and the app's GPU sessions at launch, in the background, instead of inside the first page's processing (where
     * the user watched it for minutes). Processing that comes first waits for it, with its stage text. Nothing without
     * Auto, with a stored choice, or without a GPU candidate (ensureNets then stores the CPU at once).
     */
    private fun setUpGpu() {
        if (settings.acceleration != Accel.AUTO || Accel.autoPlan(settings.autoChoice) != null) return
        viewModelScope.launch(Dispatchers.Default) {
            runCatching {
                if (Accel.candidates().isEmpty()) return@runCatching
                gpuSetup = GPU_SETUP
                synchronized(netsLock) {
                    benchAndWarm { s ->
                        gpuSetup = s
                        gpuSetupSink?.invoke(s)
                    }
                }
            }.onFailure { Log.e(TAG, "GPU set-up at launch failed", it) }
            gpuSetup = null
        }
    }

    /**
     * Auto's benchmark on [benchPhoto] ([NativeLib.benchBackends]); keeps it as [Settings.autoChoice].
     * No candidate (no GPU on this phone): the CPU at once, without running anything.
     */
    private fun runAutoBenchmark(stage: (String) -> Unit, cand: String = Accel.candidates()): JSONObject {
        val at = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.ROOT).format(java.util.Date())
        val bench = if (cand.isEmpty()) JSONObject().put("ok", true).put("runs", 0)
            .put("choice", JSONObject().put("finder", "cpu").put("flat", "cpu").put("hr", "cpu").put("shade", "cpu").put("rule", "cpu"))
        else CrashGuard.guard(getApplication(), "the acceleration benchmark (GPU)", gpu = true) {
            // one timed run per net: the backends differ by large factors; the stage says what it is (not the page's progress)
            JSONObject(NativeLib.benchBackends(benchPhoto(), cand, 1) { pct, s -> stage("$GPU_SETUP: ${s.removePrefix("Auto: ")}, $pct %") })
        }
        val stored = JSONObject().put("soc", NetEnv.soc).put("app", BuildConfigInfo.VERSION).put("at", at).put("candidates", cand).put("bench", bench)
        if (cand.isEmpty()) stored.put("note", "no GPU on this phone: ${NetEnv.gpuWhy}")
        // the fp16 backends (gpu16, cpu16: debug `--es bench`) are measured, not kept as Auto's choice
        if (cand.split(',').none { it.trim().endsWith("16") }) settings.autoChoice = stored.toString()
        else stored.put("note", "fp16 candidates: measured only, Auto's choice unchanged")
        Log.i(TAG, "auto acceleration $stored")
        return stored
    }

    /** A stored Auto benchmark ([runAutoBenchmark]) as report text, with its JSON. */
    fun formatBench(stored: JSONObject): String {
        val sb = StringBuilder()
        val b = stored.optJSONObject("bench") ?: JSONObject()
        sb.append("FlatScan ${BuildConfigInfo.VERSION} acceleration benchmark (measured on this device, ${stored.optString("at")})\n")
        sb.append("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, SoC: ${stored.optString("soc")}\n")
        sb.append("${NetEnv.runtime}; GPU: ${if (NetEnv.gpuAvailable) "available" else "unavailable (${NetEnv.gpuWhy})"}; candidates: ${stored.optString("candidates").ifEmpty { "none" }}\n")
        if (!b.optBoolean("ok")) sb.append("FAILED: ").append(b.optString("error")).append("\n")
        val tol = b.optJSONObject("tolerance")
        if (tol != null) sb.append(("Tolerance vs CPU: grids p99 <= %.2f px, max <= %.1f px (1600 x 2263 page); shadow net p99 <= %.1f, max <= %.1f levels; " +
            "rule net p99 <= %.2f, max <= %.2f (probability)\n").format(tol.optDouble("grid_p99_px"), tol.optDouble("grid_max_px"),
            tol.optDouble("shade_p99_levels"), tol.optDouble("shade_max_levels"), tol.optDouble("rule_p99"), tol.optDouble("rule_max")))
        sb.append("%d timed runs after a first run (first: session creation, GPU kernel compile or cached kernels). ms; diff vs CPU:\n".format(b.optInt("runs")))
        val arr = b.optJSONArray("backends")
        for (i in 0 until (arr?.length() ?: 0)) {
            val be = arr!!.getJSONObject(i)
            sb.append("\n").append(be.optString("backend")).append(":\n")
            val nets = be.optJSONObject("nets") ?: continue
            for (k in nets.keys()) {
                val n = nets.getJSONObject(k)
                if (!n.optBoolean("ran")) {
                    sb.append("  %-14s did not run on this backend (first call %d ms)\n".format(k, n.optInt("first_ms")))
                    continue
                }
                val unit = when (k) { "shadenet" -> "levels"; "rulenet" -> "prob"; else -> "px" }
                sb.append("  %-14s first %6d  median %7.1f  p95 %7.1f  diff p99 %.3f max %.3f %s  %s\n".format(k, n.optInt("first_ms"),
                    n.optDouble("median_ms"), n.optDouble("p95_ms"), n.optDouble("p99_diff"), n.optDouble("max_diff"), unit,
                    if (n.optBoolean("pass")) "PASS" else "FAIL"))
            }
        }
        b.optJSONObject("choice")?.let { c ->
            sb.append("\nAuto's choice: page finder ${c.optString("finder")}, UVDoc ${c.optString("flat")}, SyntheticDoc high-res ${c.optString("hr")}, " +
                "shadow net ${c.optString("shade")}, rule net ${c.optString("rule", "cpu")}\n")
        }
        sb.append("\nJSON (send this back with the report):\n").append(stored.toString())
        return sb.toString()
    }

    private fun thermalStatus(app: Application): String {
        if (android.os.Build.VERSION.SDK_INT < 29) return "n/a (API < 29)"
        val pm = app.getSystemService(android.content.Context.POWER_SERVICE) as android.os.PowerManager
        return when (pm.currentThermalStatus) {
            0 -> "none"; 1 -> "light"; 2 -> "moderate"; 3 -> "severe"; 4 -> "critical"; 5 -> "emergency"; 6 -> "shutdown"
            else -> "unknown"
        }
    }

    /**
     * Phase 0 of docs/accel/PLAN.md: the five nets on the self-test photo [bmp] exactly as automatic
     * processing runs them (page finder, detection, UVDoc on the whole photo, UVDoc and SyntheticDoc
     * high-res on the page it found, the shadow net inside NativeLib.process), [runs] times after 1
     * warm-up, every net on whatever the settings say. Times in ms, measured on this device.
     */
    private fun benchNets(bmp: Bitmap, runs: Int, stage: (String) -> Unit): JSONObject {
        val plan = ensureNets(stage)
        val per = linkedMapOf<String, MutableList<JSONObject>>()
        val warm = linkedMapOf<String, JSONObject>()
        val walls = linkedMapOf<String, MutableList<Double>>()
        var geometry = ""
        val width = settings.outputWidth
        for (k in 0..runs) {
            stage(if (k == 0) "Nets: warm-up run" else "Nets: run $k of $runs")
            val got = linkedMapOf<String, JSONObject>()
            val t0 = System.nanoTime()
            val ft = arrayOfNulls<String>(1)
            val lg = NativeLib.pageFinder(bmp, ft) ?: throw IllegalStateException("page finder net unavailable")
            ft[0]?.let { got["page_finder"] = JSONObject(it) }
            val td = System.nanoTime()
            val pages = JSONObject(NativeLib.detect(bmp, lg, 0f)).optJSONArray("pages")
            val pg = pages?.optJSONObject(0)?.let { DetectedPage.from(it) }
            if (pg == null || !pg.ok) throw IllegalStateException("no page found on the test photo")
            val detectMs = msSince(td)
            val c = FloatArray(8) { i -> if (i % 2 == 0) pg.corners[i / 2].x else pg.corners[i / 2].y }
            val h = NativeLib.process(
                bmp, c, false, 0f, true, width, true, lg, 0, pg.reasons.isNotEmpty(), false, FloatArray(0), true, 0, true, true, false, true, null,
                true, null, null,
            )
            val pageMs = msSince(t0)
            val info = if (h != 0L) JSONObject(NativeLib.info(h)).also { NativeLib.release(h) } else null
            if (info == null || !info.optBoolean("ok")) throw IllegalStateException("processing failed: ${info?.optString("error")}")
            geometry = info.optString("mode") + (if (info.optString("mode") == "uvdoc") " (model ${info.optInt("uv_model")})" else "") +
                (if (info.optBoolean("shade_net")) ", shadow net applied" else "")
            val a = info.optJSONObject("accel")
            val n = a?.optJSONObject("nets")
            for ((key, _) in NET_LABELS) n?.optJSONObject(key)?.let { got[key] = it }
            if (k == 0) { warm.putAll(got); continue }
            for ((key, v) in got) per.getOrPut(key) { mutableListOf() }.add(v)
            for ((key, v) in listOf("detect" to detectMs, "flattening_nets" to (a?.optDouble("flattening_nets_ms") ?: 0.0),
                "process" to (a?.optDouble("process_ms") ?: 0.0), "page" to pageMs))
                walls.getOrPut(key) { mutableListOf() }.add(v)
        }
        fun stat(v: List<Double>) = JSONObject().put("median", pct(v, 50.0).roundToInt()).put("p95", pct(v, 95.0).roundToInt())
        val nets = JSONObject()
        for ((key, _) in NET_LABELS) {
            val v = per[key] ?: continue
            val w = warm[key]
            // warmup_input_ms: on a net's first use it includes creating its session (QNN: compiling the graph)
            nets.put(key, JSONObject().put("run_ms", stat(v.map { it.optDouble("run_ms") })).put("input_ms", stat(v.map { it.optDouble("input_ms") }))
                .put("warmup_run_ms", w?.optDouble("run_ms")?.roundToInt()).put("warmup_input_ms", w?.optDouble("input_ms")?.roundToInt())
                .put("warmup_first_use", w?.optBoolean("first_use")).put("shape", v.last().optString("shape"))
                .put("backend", v.last().optString("backend")))
        }
        val wall = JSONObject()
        for ((key, v) in walls) wall.put(key, stat(v))
        return NetEnv.toJson(plan).put("runs", runs).put("render_width", width).put("geometry", geometry).put("nets", nets)
            .put("wall_ms", wall).put("peak_rss_mb", peakRssMb())
    }

    private fun msSince(t0: Long): Double = (System.nanoTime() - t0) / 1e6

    // ---- debug batch mode (docs/accel/PLAN.md Stream C step 5)
    var batchRunning by mutableStateOf(false)
        private set
    var batchStatus by mutableStateOf("")
        private set

    /**
     * Debug builds ([MainActivity]: `adb shell am start -S -n com.albus.flatscan/.MainActivity --es batch SET`):
     * every image (.jpg, .jpeg, .png) in files/batch/SET of the app's internal storage (written with
     * `adb exec-in run-as com.albus.flatscan sh -c 'cat > files/batch/SET/NAME.jpg' < NAME.jpg`: files that
     * adb pushes into the external directory are not readable by the app), else of its external files
     * directory, processed as tools/desktop_check/pipeline_cli does with the nets in-process
     * ([NativeLib.batchPhoto]), into batch_out/[out] (default SET_ACCEL) of the external files directory
     * (/sdcard/Android/data/com.albus.flatscan/files, which adb pull reads): per photo
     * NAME.json, the page NAME + [ext] and the nets' outputs NAME_logits / _p1 / _rc0 / _rc2.npz; run.json
     * (device, backends, options), times_phone.txt (wall ms per photo) and, when finished, done.txt.
     * Photos whose JSON exists are skipped, so a stopped run resumes. [accel]: an [Accel] setting for this
     * run (default: the current one); [shade]: the shadow net (default on, off for SET "docunet", as
     * gate_b.py); [rules]: the rule net (default: Settings > Table rules); [ext]: ".jpg" or ".png" (default ".png" for "docunet", else ".jpg"); [staticModels]: the
     * CPU runs the fixed-shape UVDoc / SyntheticDoc copies (HUMAN GATE A-1); [only]: comma-separated names.
     */
    /**
     * Debug (MainActivity `--es open_test SET`): imports the photos of files/batch/SET (internal storage, as
     * [runBatch]) as a gallery batch, waits for its pages, then opens each stored page [rounds] times in turn
     * and writes to [report] per open: ms to the stored preview on screen ([openingPreview]) and to the result
     * screen with its first render.
     */
    fun runOpenTest(set: String, report: java.io.File, rounds: Int = 3) {
        val dir = java.io.File(getApplication<Application>().filesDir, "batch/$set")
        val photos = dir.listFiles { f -> f.extension.lowercase() in setOf("jpg", "jpeg", "png") }.orEmpty().sortedBy { it.name }
        if (photos.size < 2) {
            runCatching { report.parentFile?.mkdirs(); report.writeText("open test: need >= 2 photos in $dir\n") }
            return
        }
        viewModelScope.launch {
            val sb = StringBuilder("open test: ${photos.size} photos of $dir\n")
            importUris(photos.map { Uri.fromFile(it) })
            val done = withTimeoutOrNull(900_000) {
                snapshotFlow { screen == Screen.Pages && !batch && pages.size >= photos.size }.first { it }
            }
            if (done == null) sb.append("TIMEOUT waiting for the batch (screen $screen, ${pages.size} pages)\n")
            for (round in 0 until rounds)
                for (id in pages.toList()) {
                    delay(1500)   // the last page's session store and render settle
                    if (id == currentId) open(Screen.Pages)
                    val t0 = android.os.SystemClock.elapsedRealtime()
                    var tPreview = -1L
                    val watch = launch {
                        snapshotFlow { openingPreview != null }.first { it }
                        tPreview = android.os.SystemClock.elapsedRealtime() - t0
                    }
                    openPage(id)
                    val ok = withTimeoutOrNull(120_000) {
                        snapshotFlow { screen == Screen.Result && currentId == id && page != null }.first { it }
                    } != null
                    watch.cancel()
                    sb.append("round $round page $id: preview ${tPreview} ms, result ${if (ok) android.os.SystemClock.elapsedRealtime() - t0 else -1} ms\n")
                    Log.i(TAG, "open test: round $round page $id preview $tPreview")
                }
            sb.append("details of the last open:\n").append(details).append('\n')
            runCatching { report.parentFile?.mkdirs(); report.writeText(sb.toString()) }
        }
    }

    fun runBatch(set: String, out: String?, accel: String?, shade: Boolean?, ext: String?, staticModels: Boolean, only: String?, rules: Boolean? = null) {
        if (batchRunning) return
        batchRunning = true
        batchStatus = "Batch $set: starting"
        viewModelScope.launch {
            val msg = withContext(Dispatchers.Default) {
                runCatching { batch(set, out, accel, shade, ext, staticModels, only, rules) }
                    .getOrElse { Log.e(TAG, "batch failed", it); "Batch $set FAILED: ${it.message ?: it}" }
            }
            Log.i(TAG, msg)
            batchStatus = msg
            batchRunning = false
        }
    }

    private fun batch(set: String, out: String?, accel: String?, shade: Boolean?, ext: String?, staticModels: Boolean, only: String?, rules: Boolean?): String {
        fun status(t: String) { viewModelScope.launch(Dispatchers.Main) { batchStatus = t } }
        val root = getApplication<Application>().getExternalFilesDir(null) ?: error("no external files directory")
        val inDir = File(getApplication<Application>().filesDir, "batch/$set").takeIf { it.isDirectory } ?: File(root, "batch/$set")
        val keep = only?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet()
        val photos = inDir.listFiles { f -> f.isFile && f.extension.lowercase() in setOf("jpg", "jpeg", "png") }.orEmpty()
            .filter { keep == null || it.nameWithoutExtension in keep }.sortedBy { it.name }
        if (photos.isEmpty()) error("no images in ${inDir.absolutePath}")
        val setting = accel?.lowercase() ?: settings.acceleration
        val doShade = shade ?: (set != "docunet")
        val doRules = rules ?: settings.ruleNet
        val pageExt = ext ?: if (set == "docunet") ".png" else ".jpg"
        if (pageExt != ".jpg" && pageExt != ".png") error("ext must be .jpg or .png")
        val plan = synchronized(netsLock) {
            val pl = Accel.fixed(setting) ?: (Accel.autoPlan(settings.autoChoice)
                ?: run { runAutoBenchmark({ status("Batch $set: $it") }); Accel.autoPlan(settings.autoChoice) } ?: Accel.fixed(Accel.CPU)!!)
            configureNets(pl, staticModels)
            pl
        }
        val outDir = File(root, "batch_out/" + (out ?: "${set}_$setting")).apply { mkdirs() }
        File(outDir, "done.txt").delete()
        File(outDir, "run.json").writeText(NetEnv.toJson(plan).put("set", set).put("photos", photos.size).put("shade", doShade).put("rule_net", doRules)
            .put("ext", pageExt).put("static_models", staticModels).put("threads", netThreads).put("app", BuildConfigInfo.VERSION)
            .put("started", java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.ROOT).format(java.util.Date())).toString(1))
        val timesFile = File(outDir, "times_phone.txt")
        val times = runCatching { JSONObject(timesFile.readText()) }.getOrDefault(JSONObject())
        var done = 0
        var ok = 0
        val t0 = System.nanoTime()
        try {
            for ((i, f) in photos.withIndex()) {
                val name = f.nameWithoutExtension
                if (File(outDir, "$name.json").exists()) continue
                status("Batch $set (${plan.summary()}): ${i + 1} of ${photos.size}, $name")
                val t = System.nanoTime()
                val line = NativeLib.batchPhoto(f.absolutePath, File(outDir, name).absolutePath, pageExt, doShade, doRules)
                val ms = msSince(t)
                val good = runCatching { JSONObject(line).optBoolean("ok") }.getOrDefault(false)
                done++
                if (good) ok++
                times.put(name, JSONObject().put("wall", ms.roundToInt()))
                timesFile.writeText(times.toString(1))
                Log.i(TAG, "batch $set ${i + 1}/${photos.size} $name ok=$good ${ms.roundToInt()} ms")
            }
        } finally {
            synchronized(netsLock) { configureNets(currentPlan()) }
        }
        val msg = "Batch $set done: $done processed, ${photos.size - done} already there, $ok ok, %.0f s; %s -> %s".format(
            msSince(t0) / 1000, plan.summary(), outDir.absolutePath)
        File(outDir, "done.txt").writeText(msg + "\n")
        return msg
    }

    /** Linear-interpolation percentile (jni_bridge.cpp pct). */
    private fun pct(v: List<Double>, q: Double): Double {
        if (v.isEmpty()) return 0.0
        val s = v.sorted()
        val idx = q / 100.0 * (s.size - 1)
        val lo = idx.toInt()
        if (lo + 1 >= s.size) return s.last()
        return s[lo] + (s[lo + 1] - s[lo]) * (idx - lo)
    }

    /** VmHWM of this process, MB (-1 if unreadable). */
    private fun peakRssMb(): Long = runCatching {
        File("/proc/self/status").readLines().first { it.startsWith("VmHWM:") }.substringAfter(':').trim().split(' ').first().toLong() / 1024
    }.getOrDefault(-1L)

    private fun formatSelfTest(j: JSONObject, nets: Result<JSONObject>, thermalBefore: String, thermalMid: String, thermalAfter: String): String {
        val b = android.os.Build.VERSION.SDK_INT
        val sb = StringBuilder()
        sb.append("FlatScan ${BuildConfigInfo.VERSION} self-test (measured on this device)\n")
        sb.append("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, Android ${android.os.Build.VERSION.RELEASE} (API $b)\n")
        sb.append("SoC: ${NetEnv.soc}\n")
        sb.append("Thermal status: before $thermalBefore, after the core checks $thermalMid, after the nets $thermalAfter\n")
        if (!j.optBoolean("ok")) sb.append("FAILED: ").append(j.optString("error")).append("\n")
        else formatCoreSelfTest(j, sb)
        formatNets(nets, sb)
        val raw = JSONObject().put("self_test", j).put("nets", nets.getOrNull() ?: JSONObject().put("error", nets.exceptionOrNull()?.toString()))
        sb.append("\nJSON (send this back with the report):\n").append(raw.toString())
        return sb.toString()
    }

    private fun formatNets(nets: Result<JSONObject>, sb: StringBuilder) {
        val n = nets.getOrElse {
            sb.append("\nNeural nets: FAILED: ").append(it.message ?: it.toString()).append("\n")
            return
        }
        sb.append("\nNeural nets: %s (acceleration %s; %s; GPU %s; %d cores)\n".format(
            n.optString("backend"), n.optJSONObject("acceleration")?.optString("setting"), n.optString("runtime"), n.optString("gpu"), n.optInt("cores")))
        sb.append("photo 1, automatic corners, every net on, render width %d, %d runs after 1 warm-up (median / p95 ms;\n".format(
            n.optInt("render_width"), n.optInt("runs")))
        sb.append("other: input and checks; warm-up other includes creating the session, GPU: compiling its kernels):\n")
        val nj = n.optJSONObject("nets")
        for ((key, label) in NET_LABELS) {
            val o = nj?.optJSONObject(key) ?: continue
            val r = o.optJSONObject("run_ms")
            val i = o.optJSONObject("input_ms")
            sb.append("%-28s %6d / %6d  other %4d  warm-up %6d + %6d  %s  %s\n".format(label, r?.optInt("median") ?: 0, r?.optInt("p95") ?: 0,
                i?.optInt("median") ?: 0, o.optInt("warmup_run_ms"), o.optInt("warmup_input_ms"), o.optString("backend"), o.optString("shape")))
        }
        val w = n.optJSONObject("wall_ms")
        for ((key, label) in listOf("detect" to "Detection (core)", "flattening_nets" to "Flattening nets (wall)",
            "process" to "Processing incl. shadow, rule nets", "page" to "Page total")) {
            val o = w?.optJSONObject(key) ?: continue
            sb.append("%-28s %6d / %6d\n".format(label, o.optInt("median"), o.optInt("p95")))
        }
        sb.append("Geometry: ${n.optString("geometry")}\n")
        sb.append("Peak memory after the nets (VmHWM): ${n.optLong("peak_rss_mb")} MB\n")
    }

    private fun formatCoreSelfTest(j: JSONObject, sb: StringBuilder) {
        sb.append("GPU: ${j.optString("gpu")}\n")
        sb.append("Final resample on GPU: ${j.optBoolean("gpu_resample")}\n\n")
        val mp = j.getJSONObject("map_parity")
        sb.append("1. Map parity vs reference (limit 0.1 px): max %.4f px, mean %.5f px -> %s\n".format(
            mp.optDouble("max_epe_px"), mp.optDouble("mean_epe_px"), if (mp.optBoolean("passed")) "PASS" else "FAIL"))
        val rs = j.getJSONObject("gpu_vs_cpu_resample")
        sb.append("2. GPU vs CPU resample: max %d levels, values > 1: %d -> %s\n".format(
            rs.optInt("max_levels"), rs.optInt("values_gt1"), if (rs.optInt("max_levels") <= 1) "PASS" else "FAIL"))
        // the core writes "gpu_vs_cpu_filter" (render_page, Auto color 0.7)
        val cl = j.optJSONObject("gpu_vs_cpu_filter") ?: j.optJSONObject("gpu_vs_cpu_cleanup") ?: JSONObject()
        sb.append("   GPU vs CPU cleanup: max %d levels (GPU used: %s; %d ms GPU, %d ms CPU)\n".format(
            cl.optInt("max_levels"), cl.optBoolean("gpu_used"), cl.optLong("gpu_ms"), cl.optLong("cpu_ms")))
        sb.append("Peak memory (VmHWM): ${j.optLong("peak_rss_mb")} MB\n\n")
        sb.append("Stage times over ${j.optInt("runs")} runs after 1 warm-up (median / p95 ms), photo 1 (1659x2212):\n")
        val st = j.getJSONArray("stages_ms")
        for (i in 0 until st.length()) {
            val o = st.getJSONObject(i)
            sb.append("%-34s %6d / %6d\n".format(o.optString("stage"), o.optLong("median"), o.optLong("p95")))
        }
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
                batch = false
                fallback()
            }
            Screen.Result -> {
                persist()
                adjustOpen = false
                documentDone()
                Screen.Home
            }
            Screen.Pages -> if (pagesReturnTo == Screen.Result && handle != 0L) Screen.Result else Screen.Home.also { documentDone() }
            Screen.Settings -> returnTo
            Screen.Licenses -> Screen.Settings
            Screen.Trash -> Screen.Home
            Screen.Sync -> Screen.Settings
            Screen.Advanced -> Screen.Settings
            Screen.Storage -> Screen.Settings
            else -> {
                batch = false
                Screen.Home
            }
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
        prewarmFinder()
    }

    private var finderWarm = false

    /** The page finder's session, made while the camera is open (its first photo would otherwise wait for it). */
    private fun prewarmFinder() {
        if (finderWarm || !settings.pageFinderNet) return
        finderWarm = true
        viewModelScope.launch(Dispatchers.IO) { runCatching { NativeLib.prewarmFinder() }.onFailure { Log.e(TAG, "page finder warm-up failed", it) } }
    }

    /** Home > Scan: the camera, into a new document. */
    fun scanNew() {
        viewModelScope.launch {
            closeDocument()
            startCamera()
        }
    }

    /** Home > Import: the picked photos become a new document. */
    fun importNew(uris: List<Uri>) {
        viewModelScope.launch {
            closeDocument()
            importUris(uris)
        }
    }

    /** Recognised text of a document for search (none until OCR has run on it), and its change counter. */
    fun searchText(doc: String): String? = searchTexts[doc]
    val hasOcrText: Boolean get() = searchTexts.isNotEmpty()

    /** Gallery import: every picked image becomes a page, one after the other. */
    fun importUris(uris: List<Uri>) {
        if (uris.isEmpty()) return
        lastSource = ImportSource.Gallery
        pending.clear()
        pending.addAll(uris.drop(1))
        pendingTotal = uris.size
        batch = uris.size > 1
        persist()
        loadUri(uris[0])
    }

    /**
     * Several photos picked at once: each is found, flattened and added as a page without stopping (a photo
     * whose page is doubtful or not found is flattened from the best crop there is; re-crop it afterwards),
     * then the pages screen opens. With Settings > "Check every crop" each crop still stops for a look, and
     * the batch goes on after it. A failure stops at that photo's crop; back there ends the batch.
     */
    var batch by mutableStateOf(false)
        private set

    /** Index (1-based) of the next imported image, for "Next page (k of n)". */
    val nextPendingIndex: Int get() = pendingTotal - pending.size + 1

    /** Index (1-based) of the imported image being worked on. */
    val importIndex: Int get() = pendingTotal - pending.size

    /** The batch's next photo, or its end: the pages screen with every page. */
    private fun continueBatch() {
        if (pending.isNotEmpty()) {
            nextPage()
            return
        }
        val n = pendingTotal
        batch = false
        flatImport = false
        insertAt = null
        open(Screen.Pages)
        message = "$n photos imported"
    }

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
            screen = if (batch && !settings.reviewCrops) Screen.Processing else Screen.Crop
            try {
                // a PDF page still being rendered: its file first (a failed render fails the read below)
                pdfPages[uri]?.let { runCatching { it.await() } }
                val id = store.newId()
                // the photo is kept byte for byte and processed from that copy (re-edits later start from the same pixels)
                val p = withContext(Dispatchers.IO) {
                    val app = getApplication<Application>()
                    val f = (app.contentResolver.openInputStream(uri) ?: error("cannot open $uri")).use { store.saveOriginal(id, extensionOf(app, uri), it) }
                    runCatching { loadPhotoFile(f, focalMmPerLongSide) }.onFailure { store.delete(id) }.getOrThrow()
                }
                discardStaging()
                photoPageId = id
                cropMeta = null
                cropMode = captureMode
                usePhoto(p.bitmap)
                focalPx = p.focalPx
                userRotated = false
                cropRotation = 0
                sourceJob = null
                detect(auto = true)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "load failed", e)
                message = "Could not open the image"
                detecting = false
                if (batch) continueBatch() else screen = fallback()
            }
        }
    }

    /** Deletes the source of a new page that never made it into the document. */
    private fun discardStaging() {
        val id = photoPageId ?: return
        if (id in pages || id == currentId) return
        photoPageId = null
        runCatching { NativeLib.cancelPrecompute() }
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
            val t = arrayOfNulls<String>(1)
            val l = runCatching { NativeLib.pageFinder(b, t) }.onFailure { Log.e(TAG, "page finder failed", it) }.getOrNull()
            logitsCache = Triple(b, l, t[0])
            return l
        }
    }

    /** The page finder's processing-details entry for [b] (it ran when [logitsFor] first saw [b]), or null. */
    private fun finderTiming(b: Bitmap): String? = synchronized(logitsLock) { logitsCache?.takeIf { it.first === b }?.third }

    /**
     * Finds the page(s) of the photo on the crop screen. [auto]: a trusted crop (quality gate)
     * goes straight on to processing; a doubtful one stays on the crop screen with its reasons and
     * takes the homography path. A book spread queues its second page.
     */
    private suspend fun detect(auto: Boolean = false) {
        val p = photo ?: return
        if (cropMode == CaptureMode.Whiteboard) {
            detectBoard(p, auto)
            return
        }
        if (flatImport) {
            // a PDF page: the whole image is the page, already flat
            detected = listOf(Offset(0f, 0f), Offset(p.width - 1f, 0f), Offset(p.width - 1f, p.height - 1f), Offset(0f, p.height - 1f))
            corners = detected!!
            cropPageIndex = 0
            cropReasons = emptyList()
            forceHomography = true
            spreadNext = null
            detecting = false
            if (auto) process(perspective = true)
            if (pending.isEmpty()) flatImport = false
            return
        }
        detecting = true
        val focal = focalPx
        // with automatic corners the flattening nets need only the photo and the page finder's output: they
        // start before the detection, beside it and the crop screen
        netsAhead(p)
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
        // no page found: the corners will be set by hand, the nets ahead are of no use
        if (first == null) runCatching { NativeLib.cancelPrecompute() }
        detecting = false
        val doubtful = first == null || cropReasons.isNotEmpty()
        if (first == null && !(batch && !settings.reviewCrops)) message = "Page not found - drag the corners to the page"
        if (auto && !settings.reviewCrops && (!doubtful || batch)) process()
    }

    /**
     * The flattening nets for the photo [detect] is about to find the page of, started at once
     * ([NativeLib.precomputeNets]): processing it with the detected corners then takes their grids. Not where
     * processing would run no flattening nets, nor before Auto's benchmark has chosen the backends
     * (processing runs it first). Registers them before it returns, so that processing cannot miss them.
     */
    /** Debug builds (MainActivity `--ez no_ahead true`): [netsAhead] off, for A/B checks. */
    var netsAheadOff = false

    private suspend fun netsAhead(p: Bitmap) {
        if (netsAheadOff || !settings.uvDoc || cropMode != CaptureMode.Document) return
        if (settings.acceleration == Accel.AUTO && Accel.autoPlan(settings.autoChoice) == null) return
        val nets = settings.flatNets
        val plan = withContext(Dispatchers.Default) {
            runCatching {
                val pl = synchronized(netsLock) { currentPlan().also { if (it != netPlan) configureNets(it) } }
                NativeLib.precomputeNets(p, logitsFor(p), 0, nets)
                pl
            }.onFailure { Log.e(TAG, "nets ahead failed", it) }.getOrNull()
        } ?: return
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                CrashGuard.guard(getApplication(), "the flattening nets on the crop screen", gpu = !plan.cpuOnly) { NativeLib.runPrecompute() }
            }.onFailure { Log.e(TAG, "nets ahead failed", it) }
        }
    }

    /**
     * Whiteboard mode: the board's or screen's corners (core/src/board.cpp). Found: processed straight away
     * (unless Settings > Check every crop). Not found: the whole photo, on the crop screen to set the
     * corners by hand. Corners always lie inside the photo, so nothing from outside it is ever shown.
     */
    private suspend fun detectBoard(p: Bitmap, auto: Boolean) {
        val b = withContext(Dispatchers.Default) { runCatching { NativeLib.detectBoard(p) }.onFailure { Log.e(TAG, "board detection failed", it) }.getOrNull() }
        if (photo !== p) return
        val frame = listOf(Offset(1f, 1f), Offset(p.width - 2f, 1f), Offset(p.width - 2f, p.height - 2f), Offset(1f, p.height - 2f))
        val found = b?.let { List(4) { i -> Offset(it[2 * i], it[2 * i + 1]) } }
        detected = found ?: frame
        corners = detected!!
        boardDark = b != null && b[10] > 0.5f
        cropPageIndex = 0
        cropReasons = if (found == null) listOf("No board edges found: drag the corners to the board or screen") else emptyList()
        forceHomography = true
        spreadNext = null
        detecting = false
        if (found == null && !(batch && !settings.reviewCrops)) message = "Board not found - drag the corners"
        if (auto && !settings.reviewCrops && (found != null || batch)) process()
    }

    /** Page panel: the page's shadows removed or left as photographed (processed again). */
    fun setRemoveShadows(on: Boolean) {
        val g = sessionGeomState ?: return
        if (g.removeShadows == on || !sourceAvailable(g.id)) return
        sessionGeomState = g.copy(removeShadows = on)
        reflatten()
    }

    /** Switches the page on the result screen between document, whiteboard and screen processing (processed again). */
    fun setPageMode(m: CaptureMode) {
        val g = sessionGeomState ?: return
        if (g.mode == m || !sourceAvailable(g.id)) return
        sessionGeomState = g.copy(mode = m)
        params = params.copy(
            filter = when (m) {
                CaptureMode.Whiteboard -> FilterKind.Whiteboard
                CaptureMode.Screen -> FilterKind.AutoColor
                CaptureMode.Document -> settings.defaultFilter
            },
        )
        // back to Look, where the kind was chosen (not the line tool, as after Flatten again)
        reflatten(reopen = false)
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
        if ((cropReasons.isEmpty() || batch) && !settings.reviewCrops) {
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
        saveOrder()
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

    /** Crop screen > Whole photo: the corners on the photo's corners (the photo is the page; homography path). */
    fun cropWholePhoto() {
        val p = photo ?: return
        val w = p.width - 1f
        val h = p.height - 1f
        corners = listOf(Offset(0f, 0f), Offset(w, 0f), Offset(w, h), Offset(0f, h))
    }

    /** Rotate the photo 90 degrees clockwise (corners follow). Disables auto orientation. */
    private var rotating = false

    fun rotatePhoto() {
        val p = photo ?: return
        if (rotating) return
        rotating = true
        viewModelScope.launch {
            try {
                // the turn of a full-size photo and its display copy: off the main thread
                val (turned, display) = withContext(Dispatchers.Default) { p.rotatedQuarterTurns(1).let { it to it.forDisplay() } }
                if (photo !== p) return@launch
                val h = p.height.toFloat()
                val rot = { o: Offset -> Offset(h - 1 - o.y, o.x) }
                // after a clockwise turn the old bottom-left corner is the new top-left
                val reorder = { l: List<Offset> -> listOf(rot(l[3]), rot(l[0]), rot(l[1]), rot(l[2])) }
                corners = reorder(corners)
                detected = detected?.let(reorder)
                photo = turned
                photoDisplay = display
                userRotated = true
                cropRotation = (cropRotation + 1) % 4
                // a waiting second page of a book spread turns with the photo
                spreadNext = spreadNext?.let { sp ->
                    val q = sp.page
                    SpreadPage(
                        firstId = sp.firstId, photo = turned, page = DetectedPage(q.ok, reorder(q.corners), q.reasons, rot(q.centroid)),
                        firstCentroid = rot(sp.firstCentroid), focalPx = sp.focalPx, cropRotation = cropRotation, userRotated = true,
                    )
                }
                // the nets ahead were for the photo as it was
                if (detected != null && cropMode == CaptureMode.Document) netsAhead(turned) else runCatching { NativeLib.cancelPrecompute() }
            } finally {
                rotating = false
            }
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
        val logits = if (settings.pageFinderNet) runCatching { NativeLib.pageFinder(crop, null) }.getOrNull() else null
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
        cropMode = m.mode
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
        if (!sourceAvailable(id)) return
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
                val bmp = withContext(Dispatchers.IO) { store.loadSource(id).bitmap.rotatedQuarterTurns(geom.cropRotation) }
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
    /** The page still has its original photo (re-crop, unwarp changes); otherwise says why not. */
    private fun sourceAvailable(id: String): Boolean {
        if (photoPageId == id && photo != null) return true
        if (store.hasSource(id)) return true
        message = "The original photo of this page was removed to save space: it can no longer be re-cropped"
        return false
    }

    private fun geometryKey(m: PageMeta) =
        m.geometryKey(settings.outputWidth, settings.autoRotate && !m.userRotated, settings.pageFinderNet) +
            "|uv=" + settings.uvDoc + "|nets=" + settings.flatNets + "|shadows=" + settings.neuralShadows + "|shadenet=1|hr=1|lines=9|rulenet=" + settings.ruleNet + "|pass1=1|hrfirst=2|fold=1" +
            currentPlan().let { if (it.cpuOnly) "" else "|accel=${it.key()}" }

    /**
     * Opens a page of the document: from its stored session when its geometry is unchanged (no
     * re-processing), otherwise re-processed from its source. Its last rendering is on screen at once
     * ([openingPreview]) while that happens.
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
            val preview = launch {
                val b = withContext(Dispatchers.IO) { runCatching { store.loadThumbnail(id, PREVIEW_MAX_SIDE) }.getOrNull() }
                if (screen == Screen.Processing) openingPreview = b
            }
            try {
                commitCurrent()
                openStored(id)
            } finally {
                preview.cancel()
                openingPreview = null
            }
        }
    }

    /**
     * The page [step] pages after the current one in the document (-1: the one before), opened as [openPage] does: a
     * swipe on the result screen. False at either end of the document, or when no page of it is open.
     */
    fun openAdjacentPage(step: Int): Boolean {
        val i = pages.indexOf(currentId ?: return false)
        val j = i + step
        if (i < 0 || j !in pages.indices || screen != Screen.Result) return false
        openPage(pages[j])
        return true
    }

    private suspend fun openStored(id: String) {
        try {
            val m = withContext(Dispatchers.IO) { store.loadMeta(id) }
            // without its original photo (Settings > Storage) a page is shown as it was processed
            val stored = withContext(Dispatchers.IO) {
                store.loadSession(id, geometryKey(m)) ?: if (!store.hasSource(id)) store.loadAnySession(id) else null
            }
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
                    return
                }
                Log.w(TAG, "stored session of $id unreadable: re-processing")
            }
            val bmp = withContext(Dispatchers.IO) { store.loadSource(id).bitmap.rotatedQuarterTurns(m.cropRotation) }
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

    fun deletePage(id: String) = deletePages(setOf(id))

    fun movePage(id: String, delta: Int) {
        val i = pages.indexOf(id)
        editPages { PageOps.moveOne(it, i, i + delta) }
    }

    // ------------------------------------------------------------------ page editor (feature 2)
    /**
     * Undo / redo of the page editor: one history per open document, covering every structural edit of
     * the session (order, delete, insert, duplicate, rotate, split, move to another document). Pages a
     * state still refers to keep their files until the session ends ([endPageSession]).
     */
    private var pagesHistory: PagesHistory? = null
    /** Result rotation of each page the session turned, before its first turn (undo target). */
    private val baseRotations = mutableMapOf<String, Int>()
    var canUndoPages by mutableStateOf(false)
        private set
    var canRedoPages by mutableStateOf(false)
        private set
    /** New pages (camera, photos, PDF) go here instead of the end; null: the end. */
    var insertAt by mutableStateOf<Int?>(null)
        private set
    private val pagesMutex = Mutex()

    private fun pagesSession(): PagesHistory? {
        val d = docId ?: return null
        val h = pagesHistory
        if (h != null && h.state.current == d) return h
        return PagesHistory(PagesState(d, mapOf(d to pages.toList()))).also {
            pagesHistory = it
            baseRotations.clear()
            updateUndoFlags()
        }
    }

    private fun updateUndoFlags() {
        canUndoPages = pagesHistory?.canUndo == true
        canRedoPages = pagesHistory?.canRedo == true
    }

    /** Applies one structural edit [f] to the open document as an undo step, and writes it. */
    private fun editPages(f: suspend (PagesState) -> PagesState) {
        viewModelScope.launch {
            pagesMutex.withLock {
                val h = pagesSession() ?: return@withLock
                // a live change (drag and drop, a new page) becomes its own step first
                val folded = h.state.pages != pages.toList()
                if (folded) h.apply(h.state.copy(docs = h.state.docs + (h.state.current to pages.toList())))
                val next = f(h.state)
                if (next != h.state) h.apply(next) else if (!folded) return@withLock
                commitPagesState(h.state)
                updateUndoFlags()
            }
        }
    }

    fun undoPages() = walkPages { it.undo() }
    fun redoPages() = walkPages { it.redo() }

    private fun walkPages(step: (PagesHistory) -> PagesState?) {
        viewModelScope.launch {
            pagesMutex.withLock {
                val h = pagesSession() ?: return@withLock
                val s = step(h) ?: return@withLock
                commitPagesState(s)
                updateUndoFlags()
            }
        }
    }

    /** Makes [s] true on screen, in the database and in the pages' rotations. */
    private suspend fun commitPagesState(s: PagesState) {
        val now = s.pages
        if (currentId != null && currentId !in now) closeSession()
        if (pages.toList() != now) {
            pages.clear()
            pages.addAll(now)
        }
        for ((doc, ids) in s.docs) runCatching { library.setPages(doc, ids) }.onFailure { Log.e(TAG, "page order of $doc failed", it) }
        // rotations: every page the session ever turned goes to the state's value, or back to where it started
        for ((id, base) in baseRotations) {
            val want = s.rotations[id] ?: base
            if (id == currentId && handle != 0L) {
                if (rotation != want) { rotation = want; outVersion++ }
            } else if (store.exists(id)) {
                val m = withContext(Dispatchers.IO) { runCatching { store.loadMeta(id) }.getOrNull() } ?: continue
                if (m.rotation != want && rotateStoredPage(id, m, want)) pageVersions[id] = (pageVersions[id] ?: 0) + 1
            }
        }
    }

    /** Page [id] (not the live one) with result rotation [rot]: rendered again from its session, else its out.jpg turned. */
    private suspend fun rotateStoredPage(id: String, m: PageMeta, rot: Int): Boolean = runCatching {
        if (renderStoredPage(id, m.params, m.look, rot)) return@runCatching true
        withContext(Dispatchers.IO) {
            val b = android.graphics.BitmapFactory.decodeFile(store.outputFile(id).absolutePath) ?: return@withContext false
            val t = b.rotatedQuarterTurns(rot - m.rotation)
            store.saveOutput(id, t)
            store.saveMeta(m.copy(rotation = rot))
            true
        }
    }.onFailure { Log.e(TAG, "rotate $id failed", it) }.getOrDefault(false)

    /** Ends the editing session of the open document: pages no state refers to any more are deleted. */
    private suspend fun endPageSession() {
        pagesHistory = null
        baseRotations.clear()
        insertAt = null
        updateUndoFlags()
        runCatching {
            library.dropEmpty(docId)
            library.pruneOrphans(keep = pages + listOfNotNull(photoPageId, currentId))
        }.onFailure { Log.e(TAG, "page clean-up failed", it) }
    }

    /** Drag and drop on the pages screen: the order changes live ([dragMove]), one undo step at the drop ([dragEnd]). */
    fun dragMove(from: Int, to: Int) {
        if (from !in pages.indices || to !in pages.indices || from == to) return
        pages.add(to, pages.removeAt(from))
    }

    fun dragEnd() = editPages { it }   // records the live order (editPages folds it in as one step)

    fun deletePages(sel: Set<String>) {
        if (photoPageId in sel && currentId !in sel) photoPageId = null
        editPages { PageOps.delete(it, sel) }
    }

    fun reversePages() = editPages { PageOps.reverse(it) }

    fun movePagesTo(sel: Set<String>, to: Int) = editPages { PageOps.move(it, sel, to) }

    fun rotatePages(sel: Set<String>, quarters: Int) = editPages { s ->
        commitCurrent()
        for (id in sel) if (id !in baseRotations) baseRotations[id] = if (id == currentId) rotation
            else withContext(Dispatchers.IO) { runCatching { store.loadMeta(id).rotation }.getOrDefault(0) }
        PageOps.rotate(s, sel, quarters) { baseRotations[it] ?: 0 }
    }

    fun duplicatePages(sel: Set<String>) = editPages { s ->
        commitCurrent()
        val (next, copies) = PageOps.duplicate(s, sel) { store.newId() }
        withContext(Dispatchers.IO) { copies.forEach { (from, to) -> store.copyPage(from, to) } }
        copies.forEach { (from, to) -> baseRotations[from]?.let { baseRotations[to] = it } }
        next
    }

    /** Pages from [page] on become a new document ("<title> (2)"). */
    fun splitAt(page: String) = editPages { s ->
        val at = s.pages.indexOf(page)
        if (at <= 0) return@editPages s
        commitCurrent()
        val d = library.createDocument("${docTitle ?: DocumentName.default()} (2)")
        message = "Pages ${at + 1}-${s.pages.size} moved to \"${d.title}\""
        PageOps.split(s, at, d.id)
    }

    /** The selected pages go to the end of document [target] (null: a new document). */
    fun movePagesToDocument(sel: Set<String>, target: String?) = editPages { s ->
        commitCurrent()
        val t = target?.let { library.document(it) } ?: library.createDocument()
        message = if (sel.size == 1) "Page moved to \"${t.title}\"" else "${sel.size} pages moved to \"${t.title}\""
        PageOps.moveToDocument(s, sel, t.id, library.pageIds(t.id))
    }

    /** New pages are inserted before [page] (null: at the end) until the import or scan is done. */
    fun insertBefore(page: String?) {
        insertAt = page?.let { pages.indexOf(it).takeIf { i -> i >= 0 } }
    }

    /** Pages of a PDF ([uri]) imported as pages (rendered at up to 200 dpi, perspective only, no page finding). */
    fun importPdf(uri: Uri) {
        // the import starts with the first page written; the others render (and encode) meanwhile, and loadUri
        // waits for a page whose file is not written yet (pdfPages)
        pdfJob?.cancel()
        pdfPages.values.forEach { it.cancel() }
        pdfPages.clear()
        val first = CompletableDeferred<List<Uri>>()
        var uris = emptyList<Uri>()
        pdfJob = viewModelScope.launch {
            try {
                renderPdfPages(
                    getApplication(), uri,
                    onCount = { files ->
                        uris = files.map { Uri.fromFile(it) }
                        uris.forEach { pdfPages[it] = CompletableDeferred() }
                        if (uris.isEmpty()) first.complete(uris)
                    },
                    onPage = { i ->
                        pdfPages[uris[i]]?.complete(Unit)
                        if (i == 0) first.complete(uris)
                    },
                )
            } catch (e: Throwable) {
                if (e !is CancellationException) Log.e(TAG, "PDF import failed", e)
                first.completeExceptionally(e)
                pdfPages.values.forEach { it.completeExceptionally(e) }
            }
        }
        viewModelScope.launch {
            val files = runCatching { first.await() }.getOrNull()
            if (files.isNullOrEmpty()) {
                message = "Could not read the PDF"
                return@launch
            }
            flatImport = true
            importUris(files)
        }
    }

    /** The PDF pages being rendered ([importPdf]); null job: none. */
    private var pdfJob: Job? = null
    /** A PDF page's file is written (or failed): loadUri waits for it. */
    private val pdfPages = java.util.concurrent.ConcurrentHashMap<Uri, CompletableDeferred<Unit>>()

    /** The photos being imported are already flat pages (PDF pages): no detection, perspective only. */
    private var flatImport = false

    /** Writes the page order of the open document (creating the document for its first page). */
    private fun saveOrder() {
        viewModelScope.launch { saveOrderNow() }
    }

    private val orderMutex = Mutex()

    private suspend fun saveOrderNow() = orderMutex.withLock {
        val order = pages.toList()
        val d = docId ?: if (order.isEmpty()) return@withLock else ensureDocument()
        runCatching { library.setPages(d, order) }.onFailure { Log.e(TAG, "page order save failed", it) }
    }

    private suspend fun ensureDocument(): String {
        docId?.let { return it }
        val d = library.createDocument()
        docId = d.id
        docTitle = d.title
        settings.lastDocument = d.id
        return d.id
    }

    /** Closes the open document (it stays in the history); the next scan starts a new one. */
    fun newDocument() {
        viewModelScope.launch { closeDocument() }
    }

    private suspend fun closeDocument() {
        docId?.let { backgroundOcr(it); autoUpload(it) }
        closeSession()
        endPageSession()
        val old = docId
        // before the pages go: a staged photo that is one of them (a re-crop) must not be deleted
        discardStaging()
        pages.clear()
        pageVersions.clear()
        pending.clear()
        pendingTotal = 0
        photo = null
        photoDisplay = null
        photoPageId = null
        docId = null
        docTitle = null
        settings.lastDocument = null
        // a document whose scan was abandoned before its first page has nothing to keep
        if (old != null) runCatching { library.dropEmpty(null) }
    }

    // ------------------------------------------------------------------ sync (feature 6)
    val syncConfig by lazy { com.albus.flatscan.sync.SyncConfig(app) }

    /** Back to the history from the open document: it is finished for now (automatic upload). */
    private fun documentDone() {
        val d = docId ?: return
        viewModelScope.launch {
            commitCurrent()
            backgroundOcr(d)
            autoUpload(d)
        }
    }

    /** Queues [doc] when automatic upload is on and it changed since its last upload. */
    /** Recognises the text of [doc]'s pages in the background (search), when Settings > Recognise text is on. */
    private fun backgroundOcr(doc: String) {
        if (!settings.ocrAfterScan) return
        runCatching { com.albus.flatscan.output.Ocr.enqueue(getApplication(), doc) }.onFailure { Log.e(TAG, "OCR queue failed", it) }
    }

    /** The text recognition pack (output/OcrPack.kt): download size of this phone's pack, null when there is none. */
    val ocrPackBytes: Long? by lazy { com.albus.flatscan.output.OcrPack.pack(getApplication())?.size }

    /**
     * Downloads the text recognition pack (Settings, the export sheet's searchable text); then, with Recognise text
     * after scanning on, every document's pages are recognised in the background (search).
     */
    fun downloadOcr() {
        val app = getApplication<Application>()
        viewModelScope.launch {
            if (!com.albus.flatscan.output.OcrPack.download(app)) return@launch
            if (settings.ocrAfterScan) {
                val docs = runCatching { library.documents.first() }.getOrDefault(emptyList())
                for (d in docs) backgroundOcr(d.id)
            }
        }
    }

    /** Removes the text recognition pack (text already recognised stays searchable). */
    fun removeOcr() = com.albus.flatscan.output.OcrPack.remove(getApplication())

    /**
     * Debug batch mode (`--es ocr_test PHOTO`): the pack downloaded if needed, then [photo] (a file under
     * files/batch, internal or external files) recognised; the lines, the time and the pack state in files/batch_out/ocr_test.txt.
     */
    fun runOcrTest(photo: String, report: java.io.File) {
        val app = getApplication<Application>()
        viewModelScope.launch {
            val text = withContext(Dispatchers.IO) {
                val ok = com.albus.flatscan.output.OcrPack.download(app)
                val sb = StringBuilder("pack: ${com.albus.flatscan.output.OcrPack.state} (download ok $ok)\n")
                if (ok) runCatching {
                    val f = java.io.File(app.filesDir, "batch/$photo").takeIf { it.isFile } ?: java.io.File(app.getExternalFilesDir(null), "batch/$photo")
                    val bmp = android.graphics.BitmapFactory.decodeFile(f.absolutePath) ?: error("cannot read $f")
                    val t0 = System.nanoTime()
                    val lines = com.albus.flatscan.output.Ocr(app).recognise(bmp)
                    sb.append("photo ${f.name} ${bmp.width} x ${bmp.height}: ${lines.size} lines, ${lines.sumOf { it.words.size }} words in ${(System.nanoTime() - t0) / 1_000_000} ms\n")
                    for (l in lines) sb.append(if (l.rtl) "R " else "L ").append(l.words.joinToString(" ") { it.text }).append('\n')
                }.onFailure { sb.append("FAILED: ").append(Log.getStackTraceString(it)) }
                sb.toString()
            }
            runCatching { report.parentFile?.mkdirs(); report.writeText(text) }
            Log.i(TAG, "OCR test: ${text.lineSequence().take(2).joinToString(" | ")}")
        }
    }

    private suspend fun autoUpload(doc: String) {
        val c = syncConfig
        if (!c.autoUpload || !c.configured) return
        val d = library.document(doc) ?: return
        if (d.deletedAt != null || library.pageIds(doc).isEmpty()) return
        if (d.uploadedModified == d.modifiedAt) return
        runCatching { com.albus.flatscan.sync.SyncWorker.enqueue(getApplication(), doc) }.onFailure { Log.e(TAG, "upload queue failed", it) }
    }

    /** Uploads the documents now (history selection, pages screen). */
    fun uploadDocuments(ids: Collection<String>) {
        if (!syncConfig.configured) {
            message = "Set up sync first: Settings > Sync to your server"
            return
        }
        viewModelScope.launch {
            if (docId in ids) commitCurrent()
            ids.forEach { runCatching { com.albus.flatscan.sync.SyncWorker.enqueue(getApplication(), it) } }
            message = if (ids.size == 1) "Upload queued" else "${ids.size} uploads queued"
        }
    }

    /** Opens document [id] of the history on the pages screen. */
    fun openDocument(id: String) {
        if (id == docId) {
            open(Screen.Pages)
            return
        }
        viewModelScope.launch {
            closeDocument()
            val d = library.document(id) ?: return@launch
            docId = d.id
            docTitle = d.title
            settings.lastDocument = d.id
            pages.addAll(library.pageIds(d.id))
            open(Screen.Pages)
        }
    }

    /** File name of exports of the open document: its title, else "Scan" and the date. */
    fun exportName(): String = DocumentName.clean(docTitle ?: DocumentName.default()).ifEmpty { DocumentName.default() }

    // ------------------------------------------------------------------ output (feature 3)
    val exporter by lazy { Exporter(app, library) }

    /** The file name exports of the open document get with [o] (no extension). */
    fun exportFileName(o: OutputOptions): String =
        FileNameTemplate.render(o.template, docTitle ?: DocumentName.default(), System.currentTimeMillis(), pages.size, o.dateFormat)

    /**
     * The open document's pages encoded for [o] (and recognised when [o] wants searchable text; the text
     * also goes to the database for search). Null without pages.
     */
    suspend fun prepareExport(o: OutputOptions, progress: (String, Float) -> Unit): Exporter.Prepared? {
        commitCurrent()
        if (pages.isEmpty()) return null
        val d = library.document(ensureDocument()) ?: return null
        val doc = ExportDoc(d.title, d.createdAt, d.modifiedAt, pages.toList())
        val p = withContext(Dispatchers.Default) { exporter.prepare(doc, o, progress) }
        doc.pages.zip(p.pages).forEach { (id, pg) -> pg.text?.let { t -> runCatching { library.setPageText(id, t.text) } } }
        if (p.pages.any { it.text != null }) refreshSearchText()
        return p
    }

    /** Recognised text per document, for the history search. */
    private var searchTexts by mutableStateOf<Map<String, String>>(emptyMap())
    var searchTextVersion by mutableIntStateOf(0)
        private set

    fun refreshSearchText() {
        viewModelScope.launch {
            searchTexts = runCatching { library.searchTexts() }.getOrDefault(emptyMap())
            searchTextVersion++
        }
    }

    /** Renames the open document. */
    fun renameDocument(title: String?) {
        val t = title?.let { DocumentName.clean(it) }?.takeIf { it.isNotEmpty() }
        viewModelScope.launch {
            val d = ensureDocument()
            val name = t ?: Library.defaultTitle(System.currentTimeMillis())
            library.rename(d, name)
            docTitle = name
        }
    }

    fun renameDocument(id: String, title: String) {
        val t = DocumentName.clean(title).takeIf { it.isNotEmpty() } ?: return
        viewModelScope.launch {
            library.rename(id, t)
            if (id == docId) docTitle = t
        }
    }

    /** Moves documents to the trash (the open one is closed first). */
    fun trashDocuments(ids: Collection<String>) {
        viewModelScope.launch {
            if (docId in ids) closeDocument()
            library.trash(ids)
            message = if (ids.size == 1) "Moved to the trash" else "${ids.size} documents moved to the trash"
        }
    }

    fun restoreDocuments(ids: Collection<String>) {
        viewModelScope.launch { library.restore(ids) }
    }

    fun deleteForever(ids: Collection<String>) {
        viewModelScope.launch { library.deleteForever(ids) }
    }

    // ------------------------------------------------------------------ tags (feature round 5)
    /** Home's tag filter: the chips picked; the list shows the documents that have all of them. */
    var tagFilter by mutableStateOf<List<String>>(emptyList())

    /** Puts [tag] on ([on]) or takes it off documents [ids] (the tag sheet). */
    fun setTag(ids: Collection<String>, tag: String, on: Boolean) {
        viewModelScope.launch { runCatching { library.setTag(ids, tag, on) }.onFailure { Log.e(TAG, "tagging failed", it) } }
    }

    /** The same for the open document (the pages screen; made now if it has none yet, as a rename does). */
    fun setOpenDocumentTag(tag: String, on: Boolean) {
        viewModelScope.launch { runCatching { library.setTag(listOf(ensureDocument()), tag, on) }.onFailure { Log.e(TAG, "tagging failed", it) } }
    }

    /** Space used, for Settings > Storage (null while it is measured). */
    var storage by mutableStateOf<StorageUsage?>(null)
        private set

    fun measureStorage() {
        viewModelScope.launch { storage = library.usage() }
    }

    /** Settings > Storage: deletes the original photos of documents older than [days] days. */
    fun dropOriginals(days: Int) {
        viewModelScope.launch {
            commitCurrent()
            val freed = library.dropOriginalsOlderThan(days)
            message = if (freed == 0L) "No original photos to remove" else "Freed ${formatBytes(freed)}"
            storage = library.usage()
        }
    }

    suspend fun thumbnail(id: String): Bitmap? = withContext(Dispatchers.IO) {
        runCatching {
            // the lists ask again whenever they are composed: decoded once per file version (its time and size)
            val f = store.thumbFile(id).takeIf { it.exists() } ?: store.outputFile(id)
            val key = "$id ${f.lastModified()} ${f.length()}"
            thumbCache.get(key) ?: store.loadThumbnail(id, 480)?.also { thumbCache.put(key, it) }
        }.getOrNull()
    }

    /** Decoded thumbnails ([thumbnail]), up to 24 MB. */
    private val thumbCache = object : android.util.LruCache<String, Bitmap>(24 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.allocationByteCount
    }

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
    fun process(
        ignoreAreas: List<Float>? = null, perspective: Boolean? = null, drawnLines: List<Float>? = null,
        paintStrokes: List<Float>? = null, autoLinesOn: Boolean? = null,
    ) {
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
        val samePlace = prev?.takeIf { it.corners.contentEquals(c) && it.cropRotation == cropRotation }
        val ig = ignoreAreas ?: samePlace?.ignore?.toList() ?: emptyList()
        // drawn lines sit where the user saw them on this crop: a new crop drops them
        val drawn = (if (samePlace != null) drawnLines ?: samePlace.userLines.toList() else null) ?: emptyList()
        // so does the line brush; the automatic lines switch is the page's own, whatever the crop
        val paint = (if (samePlace != null) paintStrokes ?: samePlace.linePaint.toList() else null) ?: emptyList()
        val autoLn = autoLinesOn ?: prev?.autoLines ?: true
        val removeSh = prev?.removeShadows ?: true
        // the page's mode: a page keeps its own (setPageMode changes it); a new page takes the crop's
        val mode = prev?.mode ?: cropMode
        val board = mode == CaptureMode.Whiteboard
        val screenPage = mode == CaptureMode.Screen
        // a board and a screen are flat: perspective correction only, whatever the unwarp switch says
        val po = board || screenPage || (perspective ?: prev?.perspectiveOnly ?: false)
        val geom = PageMeta(
            id = id, corners = c, detected = detected?.let { d -> FloatArray(8) { i -> if (i % 2 == 0) d[i / 2].x else d[i / 2].y } },
            cropRotation = cropRotation, userRotated = userRotated, focalPx = focalPx,
            params = RenderParams(), protect = FloatArray(0), rotation = 0,
            pageIndex = pageIndex, forceHomography = force,
            spreadWith = spread?.firstId ?: stored?.spreadWith ?: (if (same) sessionGeom?.spreadWith else null),
            ignore = ig.toFloatArray(), perspectiveOnly = po, mode = mode, removeShadows = removeSh, userLines = drawn.toFloatArray(),
            linePaint = paint.toFloatArray(), autoLines = autoLn,
        )
        val key = geometryKey(geom)
        val defaults = settings.defaultParams()
        val startParams = when {
            same -> params
            stored != null -> stored.params
            board -> defaults.copy(filter = if (boardDark) FilterKind.ProjectorDark else FilterKind.Whiteboard, solid = false, screen = ScreenMode.Off)
            screenPage -> defaults.copy(filter = FilterKind.AutoColor, solid = false)
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
        val startLook = when {
            same -> look
            stored != null -> stored.look
            // a board or a screen keeps its own proportions (measured from its corners), not A4's
            board || screenPage -> PageLook(shape = PageShape.Measured)
            else -> LookPrefs.newPageLook(getApplication())
        }
        // the page's finger and erase edits stay when it is processed again (the iron switched off or on, lines
        // added) on the same crop: page fractions, so they sit on the same parts of the page; a new crop drops them
        val startEdits = when {
            samePlace == null -> null
            same -> PageEdits(erase.toFloatArray(), fingersAuto, fingerStrokes.toFloatArray(), fingerFill)
            stored != null -> PageEdits(stored.erase, stored.fingersAuto, stored.fingerStrokes, stored.fingerFill)
            else -> null
        }
        val quadHw = measuredPageHw(corners, p.width, p.height, focalPx)
        // whiteboard mode: no orientation decision (its cue is tuned on printed text), no shadow net (trained
        // on paper), no rule net, no finger erase (core: PipelineOptions::erase_fingers)
        val autoOrient = settings.autoRotate && !userRotated && !board
        val width = settings.outputWidth
        val useUv = settings.uvDoc && !board
        val nets = settings.flatNets
        val shadows = settings.neuralShadows && !board && !screenPage && removeSh
        val rules = settings.ruleNet && !board
        val focal = focalPx
        recropBackup = null
        screen = Screen.Processing
        progress = 0f
        stage = "Starting"
        unfoldLanded = false
        viewModelScope.launch {
            if (!same) commitCurrent()
            sourceJob?.join()
            // from here on the UI no longer uses the old session; the native thread frees it
            handle = 0L
            resetResultState()
            val (h, info) = withContext(nativeCtx) {
                if (nativeHandle != 0L) NativeLib.release(nativeHandle)
                nativeHandle = 0L
                // per-net timings and backends, SoC, acceleration (docs/accel/PLAN.md Phase 0, Stream C)
                var accel: (JSONObject) -> Unit = {}
                val hh = try {
                    val plan = if ((useUv && !po) || shadows) ensureNets { s -> viewModelScope.launch(Dispatchers.Main) { stage = s } } else netPlan
                    val reused = synchronized(logitsLock) { logitsCache?.first === p }
                    val logits = if (board) null else logitsFor(p)
                    val finder = if (logits != null) finderTiming(p) else null
                    accel = { j -> j.put("accel", accelJson(j.optJSONObject("accel"), finder, reused, plan)) }
                    CrashGuard.guard(getApplication(), "processing a page", gpu = !plan.cpuOnly) {
                        NativeLib.process(
                            p, c, edited || board, focal, autoOrient, width, true, if (board) null else logits, pageIndex, force, po, ig.toFloatArray(), useUv, nets, shadows, rules, board,
                            removeSh, drawn.toFloatArray(), autoLn, paint.toFloatArray(),
                        ) { pct, s ->
                            viewModelScope.launch(Dispatchers.Main) { progress = pct / 100f; stage = s }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "process failed", e)
                    0L
                }
                if (hh == 0L) return@withContext 0L to JSONObject().put("ok", false).put("error", "no session")
                val j = JSONObject(NativeLib.info(hh))
                runCatching { accel(j) }.onFailure { Log.e(TAG, "net timings failed", it) }
                j.optJSONObject("accel")?.let { Log.i(TAG, "accel $it") }
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
            // the core's last report is 95 % (Finishing): done, so the animation plays its last beats
            progress = 1f
            if (id !in pages) {
                val i = spread?.let { pages.indexOf(it.firstId) } ?: -1
                val at = insertAt
                when {
                    spread != null && i >= 0 -> pages.add(if (secondGoesFirst(spread)) i else i + 1, id)
                    at != null && at <= pages.size -> { pages.add(at, id); insertAt = at + 1 }
                    else -> pages.add(id)
                }
                pagesHistory?.let { h -> if (h.state.current == docId) { h.apply(h.state.copy(docs = h.state.docs + (h.state.current to pages.toList()))); updateUndoFlags() } }
                pageVersions[id] = 0
            }
            saveOrderNow()
            // the measured shape in the page's frame: the core's, measured along its map (already in the page's
            // frame; a curled or crumpled sheet is longer than the chord between its corners), else from the
            // corners (a perspective-only page), where a sideways page was turned a quarter (rot 0 / 2)
            val mapHw = info.optDouble("page_hw", -1.0).toFloat()
            val hw = when {
                mapHw > 0f -> mapHw
                quadHw > 0f && info.optInt("rot", -1).let { it == 0 || it == 2 } -> 1f / quadHw
                else -> quadHw
            }
            val geomL = geom.copy(look = startLook.copy(measuredHw = hw)).let { g ->
                if (startEdits == null) g
                else g.copy(erase = startEdits.erase, fingersAuto = startEdits.fingersAuto, fingerStrokes = startEdits.strokes, fingerFill = startEdits.fill)
            }
            val meta = geomL.copy(params = startParams, protect = startProtect.toFloatArray(), rotation = startRotation)
            withContext(Dispatchers.IO) {
                runCatching {
                    store.saveMeta(meta)
                    // the first page of the spread learns its partner
                    if (spread != null) store.saveMeta(store.loadMeta(spread.firstId).copy(spreadWith = id))
                }.onFailure { Log.e(TAG, "meta save failed", it) }
            }
            if (spread != null) {
                spreadCurrent = null
                message = "Book spread: added as 2 pages (${if (settings.bookRightToLeft) "right" else "left"} page first)"
            }
            val text = describe(info, edited, force, po, ig.size / 4)
            val drawnCount = countDrawnLines(drawn)
            if (!po && drawnCount > info.optInt("user_lines_used", drawnCount)) {
                message = if (drawnCount == 1) "The drawn line found no printed line to follow" else "A drawn line found no printed line to follow"
            }
            // the processed page is kept on disk: opening it again needs no re-processing
            storeSession(id, key, info, text, withContext(nativeCtx) { if (h == nativeHandle) NativeLib.snapshot(h) else 0L })
            activate(h, geomL, startParams, startProtect, startRotation, text, restored = false)
            // a book spread: its second page follows from the same photo
            val next = spreadNext
            if (next != null && pageIndex == 0) {
                next.firstRot = info.optInt("rot", -1)
                startSpreadPage(next)
            } else if (batch) {
                continueBatch()
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
        look = geom.look
        lines = ln
        ignore = geom.ignore.toList()
        userLines = geom.userLines.toList()
        linePaint = geom.linePaint.toList()
        autoLines = geom.autoLines
        forgetLineEdits()
        erase = geom.erase.toList()
        fingersAuto = geom.fingersAuto
        fingerStrokes = geom.fingerStrokes.toList()
        fingerFill = geom.fingerFill
        fingerOverlay = null
        val pageInfo = runCatching { JSONObject(withContext(nativeCtx) { NativeLib.info(h) }) }.getOrNull()
        fingersOverText = pageInfo?.optBoolean("fingers_over_text") ?: false
        edgeDoubt = pageInfo?.optBoolean("edge_doubt") ?: false
        if (geom.hasEdits) {
            val rects = erase.toFloatArray()
            val fa = fingersAuto
            val st = fingerStrokes.toFloatArray()
            val ff = fingerFill
            withContext(nativeCtx) { if (h == nativeHandle) NativeLib.setEdits(h, rects, fa, st, ff) }
        }
        outVersion++
        details = text
        firstRenderDetails = true
        handle = h
        requestRender(false)
        awaitRender()
        // a stored session renders what out.jpg already holds
        if (restored) committedVersion = outVersion
        // the processing animation ends 0.4-0.9 s after the core (the page thumps flat, the light band, the landing)
        if (!restored && screen == Screen.Processing) withTimeoutOrNull(1500) { snapshotFlow { unfoldLanded }.first { it } }
        if (screen == Screen.Processing && !batch) screen = Screen.Result
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
     * turned on or off ([perspective]); the filters, protected areas and rotation are kept. [reopen]: the result
     * screen opens on the line tool again (Flatten again), not on the tab it was left on.
     */
    fun reflatten(perspective: Boolean? = null, reopen: Boolean = true) {
        val id = currentId ?: return
        val geom = currentMeta() ?: return
        if (!sourceAvailable(id)) return
        val areas = ignore
        val drawn = userLines
        val paint = linePaint
        val auto = autoLines
        reopenLines = reopen
        val p = photo
        if (photoPageId == id && p != null && recropBackup == null) {
            process(areas, perspective, drawn, paint, auto)
            return
        }
        viewModelScope.launch {
            screen = Screen.Processing
            progress = 0f
            stage = "Opening the photo"
            try {
                val bmp = withContext(Dispatchers.IO) { store.loadSource(id).bitmap.rotatedQuarterTurns(geom.cropRotation) }
                discardStaging()
                recropBackup = null
                setCrop(geom, bmp)
                process(areas, perspective, drawn, paint, auto)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                Log.e(TAG, "reflatten failed", e)
                message = "Could not open the photo"
                screen = Screen.Result
            }
        }
    }

    /** Remembers the line edits as they are, before a change (Undo goes back to them). */
    private fun noteLineEdit() {
        lineHistory += LineEdit(ignore, userLines, linePaint, autoLines)
        lineUndoSteps = lineHistory.size
    }

    /**
     * Line editor: takes back the last line edit (a line left out or used again, a box, a line added or removed, a brush
     * stroke, the automatic lines switch).
     */
    fun undoLineEdit() {
        val last = lineHistory.removeLastOrNull() ?: return
        ignore = last.ignore
        userLines = last.drawn
        linePaint = last.paint
        autoLines = last.auto
        lineUndoSteps = lineHistory.size
    }

    /** Line editor: the automatic lines on or off (off: only the painted and drawn lines straighten the page). */
    fun setAutoLinesOn(on: Boolean) {
        if (on == autoLines) return
        noteLineEdit()
        autoLines = on
    }

    /**
     * Line brush: a stroke over the page ([points]: page fractions x0, y0, ...) of [radius] (page-width fraction).
     * [straighten]: only the lines under the Straighten paint straighten the page; else Keep as is: the lines there are
     * left out (a photo, a drawing, a logo).
     */
    fun addLineStroke(straighten: Boolean, radius: Float, points: List<Float>) {
        if (points.size < 2) return
        // at most 64 points along it (as a drawn line)
        val n = points.size / 2
        val step = maxOf(1, (n + 63) / 64)
        val kept = (0 until n step step).toMutableList().also { if (it.last() != n - 1) it += n - 1 }
        noteLineEdit()
        linePaint = linePaint + listOf(if (straighten) 0f else 1f, radius, kept.size.toFloat()) + kept.flatMap { listOf(points[2 * it], points[2 * it + 1]) }
    }

    /** Line brush: removes every stroke. */
    fun clearLinePaint() {
        if (linePaint.isEmpty()) return
        noteLineEdit()
        linePaint = emptyList()
    }

    private fun forgetLineEdits() {
        lineHistory.clear()
        lineUndoSteps = 0
    }

    /** Line editor: ignores line [i] of [lines] (or uses it again when it is ignored). */
    fun toggleLine(i: Int) {
        val l = lines
        if (5 * i + 4 >= l.size) return
        noteLineEdit()
        val cx = (l[5 * i + 1] + l[5 * i + 3]) / 2
        val cy = (l[5 * i + 2] + l[5 * i + 4]) / 2
        if (ignoreAt(cx, cy) >= 0) {
            ignore = ignoreWithout(cx, cy)
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
        val kept = ignoreWithout(x, y)
        if (kept.size == ignore.size) return
        noteLineEdit()
        ignore = kept
    }

    private fun ignoreWithout(x: Float, y: Float) = ignore.chunked(4).filterNot { x >= it[0] && x <= it[2] && y >= it[1] && y <= it[3] }.flatten()

    /** Line editor: ignores every line inside the rectangle (page fractions, any corner order). */
    fun addIgnore(x0: Float, y0: Float, x1: Float, y1: Float) {
        val a = minOf(x0, x1).coerceIn(0f, 1f)
        val b = minOf(y0, y1).coerceIn(0f, 1f)
        val c = maxOf(x0, x1).coerceIn(0f, 1f)
        val d = maxOf(y0, y1).coerceIn(0f, 1f)
        if (c - a < 0.01f || d - b < 0.01f) return
        noteLineEdit()
        ignore = ignore + listOf(a, b, c, d)
    }

    fun clearIgnore() {
        if (ignore.isEmpty()) return
        noteLineEdit()
        ignore = emptyList()
    }

    /** Line editor closed without flattening again: back to the areas and drawn lines the page uses. */
    fun revertIgnore() {
        ignore = sessionGeomState?.ignore?.toList() ?: emptyList()
        userLines = sessionGeomState?.userLines?.toList() ?: emptyList()
        linePaint = sessionGeomState?.linePaint?.toList() ?: emptyList()
        autoLines = sessionGeomState?.autoLines ?: true
        forgetLineEdits()
    }

    /**
     * Line editor > Add line: a line drawn along a rule the unwarp missed ([points]: page fractions x0, y0, ...).
     * Lines drawn since the page was last flattened share a group (they were all drawn on the same page).
     */
    fun addUserLine(points: List<Float>) {
        if (points.size < 4) return
        val xs = points.filterIndexed { i, _ -> i % 2 == 0 }
        val ys = points.filterIndexed { i, _ -> i % 2 == 1 }
        if (maxOf(xs.max() - xs.min(), ys.max() - ys.min()) < 0.04f) return   // a tap or a short wiggle
        val group = 1 + (drawnLineGroups(sessionGeomState?.userLines?.toList() ?: emptyList()).maxOrNull() ?: 0)
        // at most 64 points along it
        val n = points.size / 2
        val step = maxOf(1, (n + 63) / 64)
        val kept = (0 until n step step).toMutableList().also { if (it.last() != n - 1) it += n - 1 }
        noteLineEdit()
        userLines = userLines + listOf(group.toFloat(), kept.size.toFloat()) + kept.flatMap { listOf(points[2 * it], points[2 * it + 1]) }
    }

    /**
     * Line editor > Add line: the printed line a rough stroke was swept along ([points]: page fractions x0, y0, ...),
     * found on the page (NativeLib.findLine: the stroke only has to come near it, and the line is followed past the
     * stroke's ends) and added along its length. No line there: a message and no line (flattening could not use it).
     */
    fun traceUserLine(points: List<Float>) {
        val h = handle
        if (h == 0L || points.size < 4) return
        viewModelScope.launch {
            val found = withContext(nativeCtx) { if (h == nativeHandle) NativeLib.findLine(h, points.toFloatArray()) else FloatArray(0) }
            if (handle != h) return@launch
            if (found.size < 4) {
                message = "No printed line along that stroke: sweep along the line itself"
                return@launch
            }
            addUserLine(found.toList())
        }
    }

    /** Line editor: removes every drawn line. */
    fun clearUserLines() {
        if (userLines.isEmpty()) return
        noteLineEdit()
        userLines = emptyList()
    }

    /** Line editor: removes the drawn line nearest the page point (a tap on a straightened drawn line). */
    fun removeUserLineNear(x: Float, y: Float) {
        val l = userLines
        val starts = drawnLineStarts(l)
        var best = -1
        var bestD = Float.MAX_VALUE
        for ((k, s) in starts.withIndex()) {
            val n = l[s + 1].toInt()
            for (i in 0 until n) {
                val dx = l[s + 2 + 2 * i] - x
                val dy = l[s + 3 + 2 * i] - y
                val d = dx * dx + dy * dy
                if (d < bestD) { bestD = d; best = k }
            }
        }
        if (best < 0) return
        val a = starts[best]
        val b = starts.getOrNull(best + 1) ?: l.size
        noteLineEdit()
        userLines = l.take(a) + l.drop(b)
    }

    /** Drawn lines not flattened yet (page fractions x0, y0, ... each): the line editor draws them as drawn. */
    val pendingUserLines: List<FloatArray>
        get() {
            val stored = sessionGeomState?.userLines?.toList() ?: emptyList()
            val done = drawnLineGroups(stored).maxOrNull() ?: 0
            val l = userLines
            return drawnLineStarts(l).filter { l[it].toInt() > done }.map { s -> l.subList(s + 2, s + 2 + 2 * l[s + 1].toInt()).toFloatArray() }
        }

    /**
     * The processing details' "accel" object: the core's (nets: per net, keys of [NET_LABELS], run_ms,
     * input_ms, first_use, backend, shape; flattening_nets_ms; process_ms) with the page finder's entry
     * ([finder], run on the crop screen when [reused]) and the device facts and [plan] ([NetEnv]).
     */
    private fun accelJson(core: JSONObject?, finder: String?, reused: Boolean, plan: Accel.Plan): JSONObject {
        val a = core ?: JSONObject()
        val n = a.optJSONObject("nets") ?: JSONObject().also { a.put("nets", it) }
        finder?.let { n.put("page_finder", JSONObject(it).put("reused", reused)) }
        val env = NetEnv.toJson(plan)
        for (k in env.keys()) a.put(k, env.get(k))
        return a
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
                    else "\nLine refinement: applied" + (if (info.optBoolean("spots_dropped")) " (lines left out where they overstretched the page)" else "") +
                        (info.optJSONArray("net")?.takeIf { info.optBoolean("rule_net") }?.let { a ->
                            "\n  rule net: %d lines dropped, %d added, %d pieces joined, %d bent rules kept".format(a.optInt(0), a.optInt(1), a.optInt(2), a.optInt(3))
                        } ?: "")
                edited -> "Geometry: homography (corners adjusted by hand)"
                reasons.isNotEmpty() || forced -> "Geometry: homography (doubtful automatic crop${if (reasons.isNotEmpty()) ": " + reasons.joinToString("; ") else ""})"
                info.optBoolean("uv_rejected") && mode == "surface" ->
                    "Geometry: page surface model (UVDoc disagreed with the page outline, %.2f)".format(info.optDouble("uv_agreement"))
                info.optBoolean("fell_back") -> "Geometry: homography (surface fit rejected, outline residual %.1f px)".format(info.optDouble("fit_median_px"))
                mode == "surface" -> "Geometry: page surface model (outline residual %.2f px)".format(info.optDouble("fit_median_px"))
                else -> "Geometry: $mode"
            },
        )
        if (info.optBoolean("edge_doubt")) sb.append("\nEdges: the two flattening nets disagree on where an edge is (check the corners)")
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
        info.optJSONObject("accel")?.let { a ->
            val ac = a.optJSONObject("acceleration")
            sb.append("\n\nAcceleration: %s%s".format(Accel.CHOICES.firstOrNull { it.first == ac?.optString("setting") }?.second?.substringBefore(':') ?: "?",
                ac?.optString("note")?.takeIf { it.isNotEmpty() }?.let { " ($it)" } ?: ""))
            sb.append("\nNets (%s), run / other ms, backend:".format(a.optString("runtime")))
            val n = a.optJSONObject("nets")
            for ((k, label) in NET_LABELS) n?.optJSONObject(k)?.let { t ->
                sb.append("\n%s: %.0f / %.0f ms, %s".format(label, t.optDouble("run_ms"), t.optDouble("input_ms"), t.optString("backend")))
                if (t.optBoolean("reused")) sb.append(" (ran on the crop screen)")
                else if (t.optBoolean("first_use")) sb.append(" (first use: other includes loading)")
            }
            sb.append("\nFlattening nets: %d ms; processing (incl. shadow net): %d ms".format(a.optInt("flattening_nets_ms"), a.optInt("process_ms")))
            sb.append("\nSoC: ").append(a.optString("soc"))
        }
        val ms = info.optJSONObject("ms")
        if (ms != null) {
            sb.append("\n\nStage times (ms):")
            for (k in ms.keys()) sb.append("\n").append(k).append(": ").append(ms.optLong(k))
        }
        return sb.toString()
    }

    private fun currentMeta(): PageMeta? =
        sessionGeom?.copy(
            params = params, protect = protect.toFloatArray(), rotation = rotation, erase = erase.toFloatArray(), look = look,
            fingersAuto = fingersAuto, fingerStrokes = fingerStrokes.toFloatArray(), fingerFill = fingerFill,
        )

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
            val lk = look
            // copy on the native thread: no render can write the buffer meanwhile
            val snap = if (v != committedVersion && pg != null) withContext(nativeCtx) { finishPage(pg, rot, lk) } else null
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
                docId?.let { d -> runCatching { library.touch(d) } }
            }
        }
    }

    // ------------------------------------------------------------------ result edits
    fun selectFilter(f: FilterKind) {
        if (f == params.filter) return
        params = params.copy(filter = f)
        // the default for new pages is a document filter; whiteboard pages start from the board filters
        if (pageMode == CaptureMode.Document) settings.defaultFilter = f
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

    /** Screen mode (descreen before the filter): Off / On / Auto. The descreened page is made once per page. */
    fun setScreen(v: ScreenMode) {
        if (v == params.screen) return
        params = params.copy(screen = v)
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
        erase = rects
        applyPageEdits()
    }

    // ---- finger brush (feature 5)
    /** The automatic finger removal applies to the page on the result screen. */
    var fingersAuto by mutableStateOf(true)
        private set
    /** Brush strokes: [mode (1 add, 0 remove), radius (page-width fraction), n, x0, y0, ...] repeated. */
    var fingerStrokes by mutableStateOf(listOf<Float>())
        private set
    /** The finger mask shown while brushing (translucent orange, page size), and its share of the page. */
    var fingerOverlay by mutableStateOf<Bitmap?>(null)
        private set
    var fingerShare by mutableFloatStateOf(0f)
        private set

    /**
     * Finger tool > Smart select: the finger (or other area) under the page point ([x], [y], fractions) is found
     * and filled ([add]) or brought back.
     */
    fun smartSelectFinger(x: Float, y: Float, add: Boolean) {
        val h = handle
        if (h == 0L) return
        viewModelScope.launch {
            rendering = true
            val poly = withContext(nativeCtx) { if (h == nativeHandle) NativeLib.fingerSelect(h, x, y) else FloatArray(0) }
            rendering = false
            if (handle != h) return@launch
            if (poly.size < 6) {
                message = "Nothing to select there: tap on the finger"
                return@launch
            }
            fingerStrokes = fingerStrokes + listOf(if (add) 2f else 3f, 0f, (poly.size / 2).toFloat()) + poly.toList()
            applyPageEdits()
        }
    }

    /** A brush stroke over the page ([points]: page fractions x0, y0, ...): [add] fills it as a finger, else brings the photo back there. */
    fun addFingerStroke(add: Boolean, radius: Float, points: List<Float>) {
        if (points.size < 2) return
        fingerStrokes = fingerStrokes + listOf(if (add) 1f else 0f, radius, (points.size / 2).toFloat()) + points
        applyPageEdits()
    }

    fun undoFingerStroke() {
        var i = 0
        var last = -1
        val s = fingerStrokes
        while (i + 3 <= s.size) { last = i; i += 3 + 2 * s[i + 2].toInt() }
        if (last >= 0) { fingerStrokes = s.take(last); applyPageEdits() }
    }

    fun clearFingerStrokes() {
        if (fingerStrokes.isNotEmpty()) { fingerStrokes = emptyList(); applyPageEdits() }
    }

    /** Removed fingers: 0 filled with the paper around them, 1 painted in by the inpainting net (MI-GAN). */
    var fingerFill by mutableStateOf(0)
        private set

    fun chooseFingerFill(f: Int) {
        if (f != fingerFill) { fingerFill = f; applyPageEdits() }
    }

    fun setFingersAutoOn(on: Boolean) {
        if (on != fingersAuto) { fingersAuto = on; applyPageEdits() }
    }

    /** The finger tool is closed: its overlay goes. */
    fun closeFingerTool() {
        fingerOverlay = null
    }

    /** Text ran under a removed finger on this page (its area is outlined in the page). */
    var fingersOverText by mutableStateOf(false)
        private set

    /**
     * The two flattening nets disagree on where an edge of this page is (core PipelineResult::edge_doubt): a strip of
     * the table may show, or a corner may be cut. The result screen offers the corner screen; dismissed per page.
     */
    var edgeDoubt by mutableStateOf(false)
        private set

    fun dismissEdgeDoubt() {
        edgeDoubt = false
    }

    /** Refreshes the brush overlay (the finger tool is open). */
    fun refreshFingerOverlay() {
        val h = handle
        if (h == 0L || pageWidth == 0) return
        val fa = fingersAuto
        val st = fingerStrokes.toFloatArray()
        viewModelScope.launch {
            val r = withContext(nativeCtx) {
                if (h != nativeHandle) return@withContext null
                val b = Bitmap.createBitmap(pageWidth, pageHeight, Bitmap.Config.ARGB_8888)
                b to NativeLib.fingerOverlay(h, fa, st, b)
            }
            if (r != null && handle == h) { fingerOverlay = r.first; fingerShare = r.second }
        }
    }

    /** The Erase rectangles and the finger mask applied to the live page (from its unedited state). */
    private fun applyPageEdits() {
        val h = handle
        if (h == 0L) return
        val arr = erase.toFloatArray()
        val fa = fingersAuto
        val st = fingerStrokes.toFloatArray()
        val ff = fingerFill
        val overlayShown = fingerOverlay != null
        viewModelScope.launch {
            rendering = true
            withContext(nativeCtx) { if (h == nativeHandle) NativeLib.setEdits(h, arr, fa, st, ff) }
            if (overlayShown) refreshFingerOverlay()
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

    /** A quarter turn the other way (the result screen's Page tab). */
    fun rotateResultLeft() {
        rotation = (rotation + 3) % 4
        outVersion++
    }

    /** One tap for an upside-down page (there is no automatic 180-degree turn). */
    fun rotateResult180() {
        rotation = (rotation + 2) % 4
        outVersion++
    }

    /** Light, colour or page shape of the current page (PageLook); the shape is remembered for new pages. */
    fun updateLook(v: PageLook) {
        if (v == look) return
        if (v.shape != look.shape) LookPrefs.rememberShape(getApplication(), v.shape)
        look = v
        outVersion++
    }

    // ---- Page > Perspective (PageLook.transform, as Lightroom's Transform panel)
    /** The page's perspective on screen and on export. */
    fun setTransform(t: PageTransform) = updateLook(look.copy(transform = t))

    /** The page's height / width as shown (its shape): the frame the perspective works in. */
    val shownHw: Float get() = look.targetHw(pageWidth, pageHeight) ?: if (pageWidth > 0) pageHeight.toFloat() / pageWidth else 1f

    /**
     * Upright [mode] (0 Level, 1 Vertical, 2 Full, 3 Auto, 4 Guided by the transform's guides, -1 Off): the sliders that
     * make the page's lines level and plumb (NativeLib.upright), the other sliders and the corners back to none.
     */
    fun upright(mode: Int) {
        val t0 = look.transform
        if (mode < 0) {
            setTransform(PageTransform(constrain = t0.constrain, guides = t0.guides))
            return
        }
        // Guided waits for its second guide
        if (mode == 4 && t0.guides.size < 8) {
            setTransform(t0.copy(upright = 4))
            return
        }
        val h = handle
        if (h == 0L) return
        val hw = shownHw
        val guides = if (mode == 4) t0.guides.toFloatArray() else null
        viewModelScope.launch {
            rendering = true
            val r = withContext(nativeCtx) { if (h == nativeHandle) NativeLib.upright(h, mode, guides, hw) else FloatArray(0) }
            rendering = false
            if (handle != h) return@launch
            if (r.size < 3) {
                message = "No lines found to straighten the page by"
                return@launch
            }
            val t = look.transform
            setTransform(PageTransform(rotate = r[0], vertical = r[1], horizontal = r[2], constrain = t.constrain, upright = mode, guides = t.guides))
        }
    }

    /**
     * Guided: a guide drawn on the page as it is shown, from ([x0], [y0]) to ([x1], [y1]) (fractions there), kept on the
     * core's page (at most 4, the oldest dropped); from the second on the page is solved again.
     */
    fun addGuide(x0: Float, y0: Float, x1: Float, y1: Float) {
        val t = look.transform
        val hw = shownHw
        val inv = t.matrix(hw)?.let { PageTransform.invert(it) }
        fun back(x: Float, y: Float): List<Float> {
            val p = inv?.let { PageTransform.apply(it, x.toDouble(), y.toDouble()) } ?: return listOf(x, y)
            return listOf(p.first.toFloat(), p.second.toFloat())
        }
        val g = back(x0, y0) + back(x1, y1)
        if (kotlin.math.hypot(g[2] - g[0], (g[3] - g[1]) * hw) < 0.05f) return   // a tap or a short wiggle
        val all = (t.guides + g).takeLast(16)
        setTransform(t.copy(guides = all, upright = 4))
        if (all.size >= 8) upright(4)
    }

    fun clearGuides() = setTransform(look.transform.copy(guides = emptyList()))

    /** The edits [history] tracks, as they are now. */
    fun editState() = EditState(params, look, rotation, protect, erase, fingersAuto, fingerStrokes, fingerFill)

    fun undo() {
        // a change still inside the debounce is the one to undo
        if (handle != 0L) history.record(editState())
        history.undo()?.let { applyEditState(it) }
    }

    fun redo() { history.redo()?.let { applyEditState(it) } }

    private fun applyEditState(s: EditState) {
        if (handle == 0L) return
        if (s.params != params || s.protect != protect) {
            params = s.params
            protect = s.protect
            requestRender(false)
        }
        if (s.look != look || s.rotation != rotation) {
            look = s.look
            rotation = s.rotation
            outVersion++
        }
        if (s.erase != erase || s.fingersAuto != fingersAuto || s.fingerStrokes != fingerStrokes || s.fingerFill != fingerFill) {
            erase = s.erase
            fingersAuto = s.fingersAuto
            fingerStrokes = s.fingerStrokes
            fingerFill = s.fingerFill
            applyPageEdits()
        }
    }

    /**
     * "Apply to all pages": the current page's filter, strength, Solid colors, adjustments and colour
     * sliders on every other page of the document (each keeps its shape, rotation, protected and
     * erased areas). Each page is rendered again from its stored session (the geometry its out.jpg
     * was made with) on the native thread, one at a time, and its out.jpg and meta are rewritten.
     */
    fun applyToAllPages() {
        val id = currentId ?: return
        if (applyingAll) return
        val p = params
        val lk = look
        val others = pages.filter { it != id }
        if (others.isEmpty()) return
        applyingAll = true
        viewModelScope.launch {
            var done = 0
            for (o in others) {
                val ok = runCatching { renderStoredPage(o, p, lk) }.onFailure { Log.e(TAG, "apply to $o failed", it) }.getOrDefault(false)
                if (ok) {
                    done++
                    pageVersions[o] = (pageVersions[o] ?: 0) + 1
                }
            }
            applyingAll = false
            message = when {
                done == others.size -> "Applied to all ${others.size + 1} pages"
                done == 0 -> "Could not update the other pages"
                else -> "Applied to ${done + 1} of ${others.size + 1} pages"
            }
        }
    }

    /** Page [id] (not the live one) rendered again with [p], the colours of [lk] and result rotation [rot] (null: its own): its meta and out.jpg. */
    private suspend fun renderStoredPage(id: String, p: RenderParams, lk: PageLook, rot: Int? = null): Boolean {
        val m = withContext(Dispatchers.IO) { store.loadMeta(id) }
        val dir = withContext(Dispatchers.IO) { store.sessionDir(id) } ?: return false
        val look2 = m.look.withColorsOf(lk)
        val bmp = withContext(nativeCtx) {
            val h = NativeLib.restoreSession(dir.absolutePath, true)
            if (h == 0L) return@withContext null
            try {
                if (m.hasEdits) NativeLib.setEdits(h, m.erase, m.fingersAuto, m.fingerStrokes, m.fingerFill)
                val size = NativeLib.pageSize(h)
                val b = Bitmap.createBitmap(size[0], size[1], Bitmap.Config.ARGB_8888)
                val ok = JSONObject(NativeLib.render(h, p.toArray(), m.protect, true, b)).optBoolean("ok")
                if (ok) b else null.also { b.recycle() }
            } finally {
                NativeLib.release(h)
            }
        } ?: return false
        val r = rot ?: m.rotation
        val fin = finishPage(bmp, r, look2)
        if (fin !== bmp) bmp.recycle()
        withContext(Dispatchers.IO) {
            store.saveMeta(m.copy(params = p, look = look2, rotation = r))
            store.saveOutput(id, fin)
        }
        fin.recycle()
        return true
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
                for (f in FilterKind.forMode(pageMode)) {
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
        /** The stage text of the GPU's first set-up (Auto's benchmark, the GPU sessions). */
        const val GPU_SETUP = "Setting up the GPU for this phone (once per install)"
        const val RENDER_DEBOUNCE_MS = 120L
        const val THUMB_W = 96
        /** [openingPreview]: out.jpg decoded at full size up to this side (a page is 1600 x 2263 by default) */
        const val PREVIEW_MAX_SIDE = 2400
        /** The nets in pipeline order: key in the "accel" JSON (processing details, self-test) -> label. */
        val NET_LABELS = listOf(
            "page_finder" to "Page finder", "uvdoc_pass1" to "UVDoc pass 1 (whole photo)", "uvdoc" to "UVDoc",
            "synthdoc_hr" to "SyntheticDoc high-res", "shadenet" to "Shadow net", "rulenet" to "Rule net",
        )
    }
}

/** A new bitmap: this one turned [q] clockwise quarter turns (a plain copy for 0). */
fun Bitmap.rotatedCopy(q: Int): Bitmap {
    val n = ((q % 4) + 4) % 4
    return if (n == 0) copy(Bitmap.Config.ARGB_8888, false) else rotatedQuarterTurns(n)
}

/** Where each drawn line starts in a PageMeta.userLines list ([group, n, x0, y0, ...] repeated). */
internal fun drawnLineStarts(l: List<Float>): List<Int> {
    val out = ArrayList<Int>()
    var i = 0
    while (i + 2 <= l.size) {
        val n = l[i + 1].toInt()
        if (n < 0 || i + 2 + 2 * n > l.size) break
        out += i
        i += 2 + 2 * n
    }
    return out
}

internal fun drawnLineGroups(l: List<Float>): List<Int> = drawnLineStarts(l).map { l[it].toInt() }

internal fun countDrawnLines(l: List<Float>): Int = drawnLineStarts(l).size
