package com.albus.flatscan.batch

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.compose.ui.geometry.Offset
import androidx.core.content.ContextCompat
import com.albus.flatscan.Accel
import com.albus.flatscan.CaptureMode
import com.albus.flatscan.CrashGuard
import com.albus.flatscan.DetectedPage
import com.albus.flatscan.FilterKind
import com.albus.flatscan.LookPrefs
import com.albus.flatscan.NativeLib
import com.albus.flatscan.PageLook
import com.albus.flatscan.PageMeta
import com.albus.flatscan.PageShape
import com.albus.flatscan.PageTransform
import com.albus.flatscan.RenderParams
import com.albus.flatscan.ScreenMode
import com.albus.flatscan.Settings
import com.albus.flatscan.data.Library
import com.albus.flatscan.describePage
import com.albus.flatscan.extensionOf
import com.albus.flatscan.finishPage
import com.albus.flatscan.loadPhotoFile
import com.albus.flatscan.measuredPageHw
import com.albus.flatscan.pageGeometryKey
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.concurrent.Executors

/** Where a photo of a batch is. */
enum class ItemState { Queued, Working, Done, Failed }

/** One photo of a batch: [seq] its place in the batch, [pages] the pages it became (two for a book spread). */
data class BatchItem(
    val seq: Int, val doc: String, val uri: Uri, val anchor: Int?,
    val state: ItemState = ItemState.Queued, val progress: Float = 0f, val pages: List<String> = emptyList(),
)

/** The batch as the screens and the notification show it. */
data class BatchState(val items: List<BatchItem> = emptyList(), val title: String = "", val workers: Int = 1, val stopped: Boolean = false) {
    val total: Int get() = items.size
    val done: Int get() = items.count { it.state == ItemState.Done }
    val failed: Int get() = items.count { it.state == ItemState.Failed }
    val finished: Int get() = done + failed
    val running: Boolean get() = items.any { it.state == ItemState.Queued || it.state == ItemState.Working }
    /** The share processed, the photos being worked on counted by their progress. */
    val fraction: Float get() = if (items.isEmpty()) 0f else (finished + items.filter { it.state == ItemState.Working }.sumOf { it.progress.toDouble() }.toFloat()) / total
    fun waiting(doc: String): Int = items.count { it.doc == doc && (it.state == ItemState.Queued || it.state == ItemState.Working) }
}

/** Pages a photo of [doc] became, in their order, for the document's place [seq] of its batch. */
class PagesAdded(val doc: String, val seq: Int, val ids: List<String>)

/** What the screens that started a batch fix for it: the kind of page, and the nets' backends set up for it. */
class BatchConfig(val mode: CaptureMode, val plan: Accel.Plan, val gpuInfo: String)

/**
 * Batch processing: photos picked together are found, flattened and stored as pages in the background, several
 * at once ([workers]), while the app stays usable; [BatchService] keeps the process alive and shows the progress in
 * a notification. It lives for the process, not for a screen: a batch goes on after the app is left or swiped away.
 *
 * Each photo goes the way ScanViewModel.process takes a new imported page when nothing stops it: the page finder and
 * detection (a doubtful crop flattened from the best crop there is, as a batch always did), a book spread's second
 * page, the flattening, Upright Auto, the page's first render (out.jpg, thumbnail) and its stored session, so opening
 * it later needs no processing. The native pipeline takes several pages at once: each net run holds its own net's
 * lock (core/src/infer.cpp) and the GPU stages theirs (core/src/gpu.cpp), so one page's CPU stages run beside
 * another's nets. On a desktop held to 4 cores, 8 photos took 41.0 s one at a time, 30.6 s two at a time (1.34x)
 * and 27.7 s three at a time (1.48x).
 */
class BatchProcessor private constructor(private val app: Application) {
    private val library = Library.get(app)
    private val store = library.store
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _state = MutableStateFlow(BatchState())
    val state: StateFlow<BatchState> = _state.asStateFlow()
    private val pick = Mutex()
    private val order = Mutex()
    private var run: Job? = null
    private var cfg: BatchConfig? = null
    private var nextSeq = 0

    /**
     * The open document's owner (ScanViewModel) places finished pages itself while it lives; it returns false for
     * a document it does not have open, which then gets them here, in the library. Called on the main thread.
     */
    @Volatile var onPages: ((PagesAdded) -> Boolean)? = null

    /** [uris] become pages of [doc] (titled [title]), inserted at [anchor] (null: the end) in batch order. */
    fun enqueue(doc: String, title: String, uris: List<Uri>, anchor: Int?, config: BatchConfig) {
        if (uris.isEmpty()) return
        cfg = config
        val n = workers(app, Settings(app).batchWorkers)
        _state.update { s ->
            // a finished batch is cleared by the next one
            val keep = if (s.running) s.items else emptyList()
            s.copy(items = keep + uris.map { BatchItem(nextSeq++, doc, it, anchor) }, title = title, workers = n, stopped = false)
        }
        ContextCompat.startForegroundService(app, android.content.Intent(app, BatchService::class.java))
        if (run?.isActive == true) return
        run = scope.launch {
            val pool: ExecutorCoroutineDispatcher = Executors.newFixedThreadPool(n) { r -> Thread(r, "flatscan-batch") }.asCoroutineDispatcher()
            try {
                List(n) { launch(pool) { work() } }.joinAll()
            } finally {
                pool.close()
            }
        }
    }

    /** Stops after the photos being worked on: the ones still waiting are dropped. */
    fun stop() {
        _state.update { s -> s.copy(items = s.items.filter { it.state != ItemState.Queued }, stopped = true) }
    }

    private suspend fun work() {
        while (true) {
            val item = pick.withLock {
                val next = _state.value.items.firstOrNull { it.state == ItemState.Queued } ?: return
                set(next.seq) { it.copy(state = ItemState.Working) }
                next
            }
            val pages = runCatching { processPhoto(item) }.onFailure { Log.e(TAG, "batch photo ${item.seq} failed", it) }.getOrNull()
            if (pages.isNullOrEmpty()) {
                set(item.seq) { it.copy(state = ItemState.Failed) }
                continue
            }
            set(item.seq) { it.copy(state = ItemState.Done, pages = pages, progress = 1f) }
            place(PagesAdded(item.doc, item.seq, pages), item)
        }
    }

    private fun set(seq: Int, f: (BatchItem) -> BatchItem) = _state.update { s -> s.copy(items = s.items.map { if (it.seq == seq) f(it) else it }) }

    /** The finished pages into their document: by its open screen if it has one, else into the library here. */
    private suspend fun place(p: PagesAdded, item: BatchItem) {
        val handled = withContext(Dispatchers.Main) { onPages?.invoke(p) ?: false }
        if (handled) return
        order.withLock {
            runCatching {
                val now = library.pageIds(p.doc).toMutableList()
                val at = insertionIndex(now, p.seq, item.anchor, _state.value.items.filter { it.doc == p.doc && it.seq != p.seq })
                now.addAll(at, p.ids)
                library.setPages(p.doc, now)
            }.onFailure { Log.e(TAG, "batch page order save failed", it) }
        }
    }

    /** A photo found, flattened and stored; its page ids (empty: it failed). */
    private fun processPhoto(item: BatchItem): List<String> {
        val c = cfg ?: return emptyList()
        val s = Settings(app)
        val id = store.newId()
        val file = (app.contentResolver.openInputStream(item.uri) ?: error("cannot open ${item.uri}")).use { store.saveOriginal(id, extensionOf(app, item.uri), it) }
        val photo = runCatching { loadPhotoFile(file) }.onFailure { store.delete(id) }.getOrThrow()
        val bmp = photo.bitmap
        try {
            val board = c.mode == CaptureMode.Whiteboard
            val frame = floatArrayOf(1f, 1f, bmp.width - 2f, 1f, bmp.width - 2f, bmp.height - 2f, 1f, bmp.height - 2f)
            if (board) {
                val b = runCatching { NativeLib.detectBoard(bmp) }.onFailure { Log.e(TAG, "board detection failed", it) }.getOrNull()
                val corners = b?.let { FloatArray(8) { i -> it[i] } } ?: frame
                val dark = b != null && b[10] > 0.5f
                val r = processPage(item, bmp, id, photo.focalPx, corners, true, 0, null, null, c, s, dark) ?: return emptyList<String>().also { store.delete(id) }
                return listOf(r.id)
            }
            val logits = if (s.pageFinderNet) runCatching { NativeLib.pageFinder(bmp, null) }.onFailure { Log.e(TAG, "page finder failed", it) }.getOrNull() else null
            val res = runCatching { JSONObject(NativeLib.detect(bmp, logits, photo.focalPx)) }.onFailure { Log.e(TAG, "detect failed", it) }.getOrElse { JSONObject() }
            val arr = res.optJSONArray("pages")
            val found = if (arr != null) List(arr.length()) { DetectedPage.from(arr.getJSONObject(it)) } else emptyList()
            val first = found.firstOrNull()?.takeIf { it.ok }
            val corners = first?.corners?.let { flat(it) } ?: insetCorners(bmp)
            val force = first == null || first.reasons.isNotEmpty()
            val second = found.getOrNull(1)?.takeIf { it.ok && first != null }
            // a book spread: two pages, the second from a copy of the same photo
            val id2 = second?.let { store.newId().also { n -> store.copySource(id, n) } }
            val r1 = processPage(item, bmp, id, photo.focalPx, corners, force, 0, logits, id2, c, s, false, if (second != null) 0.5f else 1f)
            if (r1 == null) {
                store.delete(id)
                id2?.let { store.delete(it) }
                return emptyList()
            }
            if (second == null || id2 == null) return listOf(r1.id)
            val r2 = processPage(item, bmp, id2, photo.focalPx, flat(second.corners), second.reasons.isNotEmpty(), 1, logits, id, c, s, false, 0.5f, 0.5f)
            if (r2 == null) {
                store.delete(id2)
                store.saveMeta(store.loadMeta(id).copy(spreadWith = null))
                return listOf(r1.id)
            }
            return if (secondGoesFirst(second.centroid - first!!.centroid, r1.rot, s.bookRightToLeft)) listOf(r2.id, r1.id) else listOf(r1.id, r2.id)
        } finally {
            bmp.recycle()
        }
    }

    private class PageDone(val id: String, val rot: Int)

    /** One page of [bmp] processed and stored as page [id] (ScanViewModel.process for a new page, without a screen). */
    private fun processPage(
        item: BatchItem, bmp: Bitmap, id: String, focal: Float, corners: FloatArray, force: Boolean, pageIndex: Int, logits: FloatArray?,
        spreadWith: String?, c: BatchConfig, s: Settings, boardDark: Boolean, share: Float = 1f, base: Float = 0f,
    ): PageDone? {
        val board = c.mode == CaptureMode.Whiteboard
        val screenPage = c.mode == CaptureMode.Screen
        val po = board || screenPage
        val h = CrashGuard.guard(app, "processing a page in a batch", gpu = !c.plan.cpuOnly) {
            NativeLib.process(
                bmp, corners, board, focal, s.autoRotate && !board, s.outputWidth, true, if (board) null else logits, pageIndex, force, po,
                FloatArray(0), s.uvDoc && !board, s.flatNets, s.neuralShadows && !board && !screenPage, s.ruleNet && !board, board,
                true, FloatArray(0), false, FloatArray(0), null,
            ) { pct, _ -> set(item.seq) { it.copy(progress = (base + share * pct / 100f).coerceAtMost(0.99f)) } }
        }
        if (h == 0L) return null
        try {
            val info = JSONObject(NativeLib.info(h))
            if (!info.optBoolean("ok")) {
                Log.e(TAG, "batch page failed: ${info.optString("error")}")
                return null
            }
            val defaults = s.defaultParams()
            val params = when {
                board -> defaults.copy(filter = if (boardDark) FilterKind.ProjectorDark else FilterKind.Whiteboard, solid = false, screen = ScreenMode.Off)
                screenPage -> defaults.copy(filter = FilterKind.AutoColor, solid = false)
                else -> defaults
            }
            // the measured shape in the page's frame, as ScanViewModel.process measures it
            val quadHw = measuredPageHw(List(4) { Offset(corners[2 * it], corners[2 * it + 1]) }, bmp.width, bmp.height, focal)
            val mapHw = info.optDouble("page_hw", -1.0).toFloat()
            val hw = when {
                mapHw > 0f -> mapHw
                quadHw > 0f && info.optInt("rot", -1).let { it == 0 || it == 2 } -> 1f / quadHw
                else -> quadHw
            }
            var look = (if (po) PageLook(shape = PageShape.Measured) else LookPrefs.newPageLook(app)).copy(measuredHw = hw)
            val size = NativeLib.pageSize(h)
            // Upright Auto on a new document page, as on the result screen
            if (c.mode == CaptureMode.Document) {
                val shown = look.targetHw(size[0], size[1]) ?: if (size[0] > 0) size[1].toFloat() / size[0] else 1f
                val r = runCatching { NativeLib.upright(h, 3, null, shown) }.getOrNull()
                if (r != null && r.size >= 3) look = look.copy(transform = PageTransform(rotate = r[0], vertical = r[1], horizontal = r[2], upright = 3))
            }
            val meta = PageMeta(
                id = id, corners = corners, detected = corners, cropRotation = 0, userRotated = false, focalPx = focal,
                params = params, protect = FloatArray(0), rotation = 0, pageIndex = pageIndex, forceHomography = force, spreadWith = spreadWith,
                perspectiveOnly = po, look = look, mode = c.mode, autoLines = false,
            )
            store.saveMeta(meta)
            // the first render: out.jpg and the thumbnail the document's screens show
            val out = Bitmap.createBitmap(size[0], size[1], Bitmap.Config.ARGB_8888)
            try {
                if (JSONObject(NativeLib.render(h, params.toArray(), FloatArray(0), true, out)).optBoolean("ok")) {
                    val fin = finishPage(out, 0, look)
                    store.saveOutput(id, fin)
                    if (fin !== out) fin.recycle()
                }
            } finally {
                out.recycle()
            }
            // the stored session: opening the page later needs no processing
            val snap = NativeLib.snapshot(h)
            if (snap != 0L) {
                val text = describePage(info, false, force, po, 0, c.gpuInfo)
                var used = false
                runCatching {
                    store.saveSession(id, pageGeometryKey(meta, s, c.plan), info, text) { dir -> used = true; NativeLib.writeSnapshot(snap, dir.absolutePath) }
                }.onFailure { Log.e(TAG, "batch session save failed", it) }
                if (!used) NativeLib.writeSnapshot(snap, java.io.File(store.root, "missing").absolutePath)
            }
            return PageDone(id, info.optInt("rot", -1))
        } finally {
            NativeLib.release(h)
        }
    }

    companion object {
        private const val TAG = "FlatScanBatch"

        @Volatile private var instance: BatchProcessor? = null

        fun get(ctx: Context): BatchProcessor = instance ?: synchronized(this) {
            instance ?: BatchProcessor(ctx.applicationContext as Application).also { instance = it }
        }

        /**
         * Photos processed at once: [setting] 1-3, or 0 for the phone's choice: two with 6 GB of memory or more
         * (each page in flight holds its photo and the pipeline's maps, ~0.6-1 GB at 12 MP), else one.
         */
        fun workers(ctx: Context, setting: Int): Int {
            if (setting in 1..3) return setting
            val mi = ActivityManager.MemoryInfo()
            (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
            return if (mi.totalMem >= 5_500_000_000L) 2 else 1
        }

        /**
         * Where the pages of photo [seq] go in a document ordered [order]: after the pages of the batch's last
         * earlier photo already there, else before those of its first later one, else at [anchor] (null: the end).
         * Photos finish out of order when several are processed at once; the document still reads in batch order.
         */
        fun insertionIndex(order: List<String>, seq: Int, anchor: Int?, others: List<BatchItem>): Int {
            val before = others.filter { it.seq < seq }.flatMap { it.pages }.map { order.indexOf(it) }.filter { it >= 0 }.maxOrNull()
            if (before != null) return before + 1
            val after = others.filter { it.seq > seq }.flatMap { it.pages }.map { order.indexOf(it) }.filter { it >= 0 }.minOrNull()
            if (after != null) return after
            return (anchor ?: order.size).coerceIn(0, order.size)
        }

        private fun flat(c: List<Offset>) = FloatArray(8) { i -> if (i % 2 == 0) c[i / 2].x else c[i / 2].y }

        /** No page found: the photo less 8 % at each side (ScanViewModel.defaultCorners). */
        private fun insetCorners(b: Bitmap): FloatArray {
            val dx = b.width * 0.08f
            val dy = b.height * 0.08f
            return floatArrayOf(dx, dy, b.width - dx, dy, b.width - dx, b.height - dy, dx, b.height - dy)
        }

        /** A book spread's second page goes first for a right-to-left book when it is the right page (ScanViewModel.secondGoesFirst). */
        private fun secondGoesFirst(d: Offset, firstRot: Int, rtl: Boolean): Boolean {
            val dxUpright = when (firstRot) {
                0 -> -d.y
                1 -> -d.x
                2 -> d.y
                else -> d.x
            }
            val secondIsRight = dxUpright > 0
            return if (rtl) secondIsRight else !secondIsRight
        }
    }
}
