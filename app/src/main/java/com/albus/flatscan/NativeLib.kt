package com.albus.flatscan

import android.content.res.AssetManager
import android.graphics.Bitmap

/** Progress callback from the native pipeline (called on the processing thread). */
fun interface ProgressListener {
    fun onProgress(percent: Int, stage: String)
}

/** Filters of the filter strip; [id] is the native enum value (core/src/cleanup.h Filter). */
enum class FilterKind(val id: Int, val label: String) {
    Original(0, "Original"),
    Scan(6, "Scan"),                     // round 11: flattened white paper at any strength, black ink, colours kept, sharpened
    Lighten(1, "Lighten"),
    AutoColor(2, "Auto color"),
    Grayscale(3, "Grayscale"),
    HighContrast(4, "High contrast"),
    BlackWhite(5, "B&W"),
    Whiteboard(7, "Whiteboard"),         // whiteboard mode: board white, marker colours kept and boosted (core/src/board.cpp)
    ProjectorDark(8, "Dark slide"),      // whiteboard mode: a projected slide with a dark background
    ;

    companion object {
        /** The filters offered for a page of [mode]. */
        fun forMode(mode: CaptureMode): List<FilterKind> = when (mode) {
            CaptureMode.Document -> listOf(Original, Scan, Lighten, AutoColor, Grayscale, HighContrast, BlackWhite)
            CaptureMode.Whiteboard -> listOf(Whiteboard, ProjectorDark, Original, AutoColor)
            // a screen glows in its own colours (dark and blue backgrounds): no filter that whitens the page
            CaptureMode.Screen -> listOf(AutoColor, Original, Lighten, Grayscale, HighContrast)
        }
    }
}

/**
 * What is being scanned: a document (the full pipeline), a whiteboard / projected slide (the board's corners,
 * perspective only, its own cleanup) or a screen (a monitor, laptop or phone: the document's outline finder, perspective
 * only, colour filters). Screens were taken as boards before: on 41 photos of screens the board's corners cut into the
 * screen or missed it on most, the document's found it, and its iron bent the flat screen on 15 (the iron off: worse on 1).
 */
enum class CaptureMode(val label: String) { Document("Document"), Whiteboard("Whiteboard"), Screen("Screen") }

/**
 * Everything the user can change on the result screen; switching any of these never re-runs
 * the geometry. Defaults are the tested defaults (identity adjustments).
 */
/** Screen mode (core/src/screen.h): descreen a photo of a monitor / laptop / phone before the filter. */
enum class ScreenMode(val id: Int, val label: String) {
    Off(0, "Off"),
    On(1, "On"),
    Auto(2, "Auto"),                     // on when the core detects a screen photo (info() "screen")
}

/**
 * Shadows on a page: Off (left as photographed: the page is processed again without its shadow removal), Auto (the
 * page's lighting flattened in full) or Manual (how much, and how much ink that lay in shadow is deepened). The
 * shadows are found automatically in every mode (core/src/cleanup.h ShadowAdjust).
 */
enum class ShadowMode { Off, Auto, Manual }

data class RenderParams(
    val filter: FilterKind = FilterKind.AutoColor,   // Scan whitened pale colour areas on 9 of 34 real pages (docs/results/r11/filters.md): a choice, not the default
    val strength: Float = 0.7f,          // 0..1, 0 = Original
    val solid: Boolean = false,          // Solid colors (ignored for B&W)
    val bgCleanup: Float = 0f,           // 0..1: push near-paper grey artefacts to white (0 = off)
    val black: Float = 0f,               // levels black point, 0..0.5 (output fraction)
    val white: Float = 1f,               // levels white point, 0.5..1
    val gamma: Float = 1f,               // levels midtones, 0.3..3 (> 1 brightens mid-tones)
    val screen: ScreenMode = ScreenMode.Auto,
    val shadow: ShadowMode = ShadowMode.Auto,
    val shadowAmount: Float = 1f,        // Manual: 0..1, how far the lighting is flattened
    val shadowContrast: Float = 0.3f,    // Manual: 0..1, how much ink that lay in shadow is deepened
) {
    /** Layout read by jni_bridge.cpp: [filter, strength, solid, bgCleanup, black, white, gamma, screen, shadow amount, shadow contrast]. */
    fun toArray(): FloatArray {
        val (a, c) = when (shadow) {
            ShadowMode.Off -> 0f to 0f
            ShadowMode.Auto -> 1f to 0f
            ShadowMode.Manual -> shadowAmount to shadowContrast
        }
        return floatArrayOf(filter.id.toFloat(), strength, if (solid) 1f else 0f, bgCleanup, black, white, gamma, screen.id.toFloat(), a, c)
    }
}

/** JNI entry points of the C++ core (app/src/main/cpp/jni_bridge.cpp). */
object NativeLib {
    init {
        System.loadLibrary("flatscan")
    }

    /** A one-channel JPEG of [bitmap] (ARGB_8888) at [quality] (export, grayscale pages); null on failure. */
    @JvmStatic external fun encodeGrayJpeg(bitmap: Bitmap, quality: Int): ByteArray?

    /** Which backend the heavy image passes use, with the GLES driver string. */
    @JvmStatic external fun gpuInfo(): String

    /**
     * The nets' backends (core/src/infer.h, LiteRT in the APK): per group, the backends to try in order, comma
     * separated ("cpu", "gpu"; "" = CPU only; the CPU is always the last resort): [finder] the page finder,
     * [flat] UVDoc (both passes), [hr] SyntheticDoc high-res, [shade] the shadow net, [rule] the rule net. Models come from
     * [assets] (models/, the .tflite files); [cacheDir] keeps the GPU's compiled kernels; [threads]: CPU threads (0: the
     * default, min(cores, 4)). [staticModels]: no effect since the .onnx models left the app. A group whose
     * backends did not change keeps its loaded sessions.
     */
    @JvmStatic external fun configureNets(
        assets: AssetManager, cacheDir: String, finder: String, flat: String, hr: String, shade: String, rule: String, threads: Int,
        staticModels: Boolean,
    )

    /**
     * JSON: cpu (the nets' CPU runtime, "LiteRT <version>"), litert (its version), gpu (whether the LiteRT GPU
     * delegate can be tried: its library loads and the phone has an OpenCL library), gpu_why, with_ort, ort, providers.
     */
    @JvmStatic external fun runtimeInfo(): String

    /**
     * The page-finder net's logits for [photo] (float[2 * 256 * 256]: page paper, book gutter), or null
     * (net unavailable: the colour-threshold detector is used). [timing] (1 element, optional) receives
     * its processing-details entry: JSON run_ms, input_ms (the rest of the call; on first use also the
     * session's creation), first_use, backend, shape.
     */
    @JvmStatic external fun pageFinder(photo: Bitmap, timing: Array<String?>?): FloatArray?

    /** Makes the page finder's session ahead of [pageFinder]'s first photo (one run on a blank input; off the main thread). */
    @JvmStatic external fun prewarmFinder()

    /**
     * Camera guidance (ui/CameraGuide.kt): one camera analysis frame, RGBA_8888 rows of [rowStride] bytes in the
     * direct buffer [frame], turned upright by [rotation] (ImageInfo.rotationDegrees). [findPage]: the page finder
     * at 256 on its own 2-thread CPU session and the detection's quad step (no edge snapping); off: luma and
     * sharpness only. -> [found, x0, y0 .. x3, y3 (TL TR BR BL, upright frame px), upright width, height, paper
     * share, luma 0..1, glare (share of the page's pixels blown out), sharpness (Laplacian variance), ms]
     * ([PreviewFrame.from]), or null.
     */
    @JvmStatic external fun previewPage(frame: java.nio.ByteBuffer, width: Int, height: Int, rowStride: Int, rotation: Int, findPage: Boolean): FloatArray?

    /**
     * The app's GPU sessions made ahead of the first page: one run on a blank input of every net and shape its
     * group runs on the GPU (the rule net portrait and landscape), so the GPU compiles and tunes there. -> one line
     * per net: setup ms and backend.
     */
    @JvmStatic external fun prewarmGpu(listener: ProgressListener?): String

    /** Debug: the LiteRT GPU's exhaustive work-group tuning instead of the fast one (the default), from the next sessions. */
    @JvmStatic external fun setGpuTuning(exhaustive: Boolean)

    /**
     * Page detection for the crop screen. logits: the page-finder net's output for this photo
     * (float[2 * 256 * 256]) or null for the colour-threshold detector. JSON: ok, error, pages
     * (1, or 2 for a book spread): [{ok, corners [x0, y0 ... x3, y3] (TL TR BR BL, bitmap px),
     * fit_median_px, needs_manual_crop [quality-gate reasons], finder, n_pages, page_index,
     * whole_frame, spread, net_margin, edge_strength, edge_at_limit, centroid [x, y]}].
     */
    @JvmStatic external fun detect(photo: Bitmap, logits: FloatArray?, focalPx: Float): String

    /**
     * Runs the whole pipeline; returns a session handle (release it with [release]).
     * autoOrient: sideways pages are turned upright (no automatic 180-degree turn).
     * logits / pageIndex: page-finder output (null: colour threshold) and the page of a book spread.
     * forceHomography: the quality gate flagged the crop, use the homography path.
     * perspectiveOnly: the user turned the unwarp off (homography, no warp map).
     * ignore: areas whose lines the warp map must not use, page fractions [x0, y0, x1, y1, ...].
     * flatten: the flattening nets (UVDoc on the whole photo with automatic corners, then UVDoc and
     * SyntheticDoc high-res on the page it found; core/src/nets.cpp) unless perspectiveOnly; flatNets:
     * 0 both (the straighter page wins), 1 UVDoc only, 2 SyntheticDoc high-res only. shadows: the shadow
     * net on the flattened page. rules: the rule net (rulenet.onnx, Settings > Table rules) in the line
     * refinement, on the CPU. The nets run on [configureNets]' backends; their timings and backends
     * are in [info]'s "accel". removeShadows: false leaves the page's shadows (neither the shading map
     * nor the shadow net). userLines: the line editor's drawn lines, [group, n, x0, y0, ...] repeated as
     * output-page fractions (PageMeta.userLines), straightened after the line stage. autoLines: the line stage
     * straightens every line it finds (off: only those in linePaint's Straighten areas, and the drawn lines);
     * linePaint: the line brush, [mode (0 Straighten, 1 Keep as is), radius, n, x0, y0, ...] repeated
     * (PageMeta.linePaint). edgePts: with edited corners, the crop screen's outline points between them, [side, x, y]
     * repeated (side k from corner k to corner k + 1, bitmap px): the page is then fitted to that outline.
     */
    @JvmStatic external fun process(
        photo: Bitmap, corners: FloatArray?, cornersEdited: Boolean, focalPx: Float,
        autoOrient: Boolean, renderWidth: Int, useGpu: Boolean, logits: FloatArray?, pageIndex: Int,
        forceHomography: Boolean, perspectiveOnly: Boolean, ignore: FloatArray?,
        flatten: Boolean, flatNets: Int, shadows: Boolean, rules: Boolean, board: Boolean, removeShadows: Boolean, userLines: FloatArray?,
        autoLines: Boolean, linePaint: FloatArray?, edgePts: FloatArray?,
        listener: ProgressListener?,
    ): Long

    /**
     * The flattening nets for [photo] with automatic corners, ahead of [process] (the crop screen: they depend
     * only on the photo, [logits], [pageIndex] and [flatNets] there). Registers them and returns; [runPrecompute]
     * runs them. From here on a [process] of the same photo with those inputs and unedited corners takes their
     * grids instead of running the nets, also while they still run (the same page, sooner); anything else
     * cancels them. Replaces an earlier call's.
     */
    @JvmStatic external fun precomputeNets(photo: Bitmap, logits: FloatArray?, pageIndex: Int, flatNets: Int)

    /** Runs [precomputeNets]' nets on the calling thread (off the main thread); returns when they are done or cancelled. */
    @JvmStatic external fun runPrecompute()

    /** Cancels [precomputeNets]' nets (those not yet started do not run). */
    @JvmStatic external fun cancelPrecompute()

    /**
     * The finger tool's smart select: the outline of the area around page point ([x], [y], fractions) that
     * belongs with it (a finger, a thumb), page fractions [x0, y0, x1, y1, ...]; empty if none was found.
     */
    @JvmStatic external fun fingerSelect(handle: Long, x: Float, y: Float): FloatArray

    /**
     * The line editor's Add line: the printed line a rough [stroke] (page fractions x0, y0, x1, y1, ...) was drawn
     * along, followed along its length (page fractions, every 4 px); empty if there is no line under most of it.
     */
    @JvmStatic external fun findLine(handle: Long, stroke: FloatArray): FloatArray

    /**
     * Page > Perspective's Upright (core/src/upright.h): the Transform sliders [rotate, vertical, horizontal] that make
     * the page's lines ([mode] 0 Level, 1 Vertical, 2 Full, 3 Auto) or the user's [guides] (4 Guided: x0, y0, x1, y1
     * page fractions, repeated) level and plumb on the page as shown, [hw] its height / width; empty when there is
     * nothing to go by.
     */
    @JvmStatic external fun upright(handle: Long, mode: Int, guides: FloatArray?, hw: Float): FloatArray

    /**
     * Whiteboard mode: the board or screen in [photo] (core/src/board.cpp): [x0, y0 .. x3, y3 (TL, TR, BR,
     * BL), confidence, area share, dark (1: a dark slide)], or null when no reliable quadrilateral is found.
     */
    @JvmStatic external fun detectBoard(photo: Bitmap): FloatArray?

    /**
     * JSON: ok, error, mode, fell_back, fit_median_px, rot, gpu_resample, legend_px (wrinkle
     * map scale top, px), has_paper_mask, needs_manual_crop, fit_median_full_px, the finder
     * fields of [detect] (when detection ran), accel (nets: per net run_ms, input_ms, first_use,
     * backend, shape; flattening_nets_ms: the flattening nets' own time, which runs beside the CPU
     * stages; nets_overlap: 1 beside them, 0 after the detection, -1 not run; nets_wait_ms: how long
     * processing waited for them; process_ms), per-stage ms.
     */
    @JvmStatic external fun info(handle: Long): String

    /** [width, height] of the page (and of every full-size render of this session). */
    @JvmStatic external fun pageSize(handle: Long): IntArray

    /**
     * Full-size render of the page with [params] (RenderParams.toArray()) into [out]
     * (pageSize). protect: picture rectangles as page fractions [x0, y0, x1, y1, ...] that
     * Solid colors must not touch (may be empty). Returns JSON timings.
     */
    @JvmStatic external fun render(handle: Long, params: FloatArray, protect: FloatArray, useGpu: Boolean, out: Bitmap): String

    /** Live preview of any size (filter strip thumbnails): same params, no Solid colors. */
    @JvmStatic external fun renderThumb(handle: Long, params: FloatArray, out: Bitmap)

    /** Perspective-corrected-only page (press and hold), pageSize. */
    @JvmStatic external fun renderBefore(handle: Long, out: Bitmap)

    /**
     * Wrinkle map overlay (pageSize, transparent ARGB): heat map of how far each area moved
     * and/or the deformed grid of the flat page. Returns the legend top value in px.
     */
    @JvmStatic external fun renderOverlay(handle: Long, heat: Boolean, grid: Boolean, out: Bitmap): Float

    /** 256-bin luminance histogram of the filtered page with [params] (levels editor). */
    @JvmStatic external fun histogram(handle: Long, params: FloatArray): IntArray

    @JvmStatic external fun release(handle: Long)

    /**
     * Erase tool: the page as processed with every rectangle (page fractions [x0, y0, x1, y1, ...])
     * filled with the paper around it; an empty array restores the page. Re-render afterwards.
     */
    @JvmStatic external fun setEdits(handle: Long, rects: FloatArray, fingersAuto: Boolean, strokes: FloatArray, fill: Int): Boolean

    /**
     * The finger brush's view: [out] (ARGB) filled with the mask a render with these settings fills, translucent
     * orange on transparent. Returns the share of the page in the mask.
     */
    @JvmStatic external fun fingerOverlay(handle: Long, fingersAuto: Boolean, strokes: FloatArray, out: Bitmap): Float

    /**
     * The lines the warp map was fitted to: [kind, x0, y0, x1, y1, ...], ends as page fractions;
     * kind 0 = ruled line, 1 = text row, 2 = grid line (global stage only). Empty: no warp map.
     */
    @JvmStatic external fun lines(handle: Long): FloatArray

    /** A cheap copy of what [restoreSession] needs (0: none); [writeSnapshot] writes and frees it. */
    @JvmStatic external fun snapshot(handle: Long): Long

    /** Writes a [snapshot] into the existing directory [dir] and frees it (any thread). */
    @JvmStatic external fun writeSnapshot(snapshot: Long, dir: String): Boolean

    /** A session from a written snapshot without re-running the geometry; 0 if it cannot be read. */
    @JvmStatic external fun restoreSession(dir: String, useGpu: Boolean): Long

    /**
     * On-device acceptance check: runs the bundled photo [runs] times (+1 warm-up), compares the
     * final map with the reference grid at [refGridPath], GPU vs CPU resample and cleanup, and
     * returns per-stage median / p95 timings and peak memory as JSON.
     */
    @JvmStatic external fun selfTest(photo: Bitmap, refGridPath: String, runs: Int, listener: ProgressListener?): String

    /**
     * Settings > Acceleration > Auto: the nets on the self-test [photo] on the CPU and on each of
     * [candidates] (backend names, comma separated), [runs] timed runs after a first one, compared with the
     * CPU (accelerator tolerances in jni_bridge.cpp). JSON: ok, tolerance, backends [{backend, nets:
     * {key: {ran, first_ms, median_ms, p95_ms, p99_diff, max_diff, pass}}}], choice {finder, flat, hr, shade, rule}.
     */
    @JvmStatic external fun benchBackends(photo: Bitmap, candidates: String, runs: Int, listener: ProgressListener?): String

    /**
     * Debug batch mode: the image file [photoPath] as tools/desktop_check/pipeline_cli runs it with the nets
     * in-process (shade: with the shadow net), on [configureNets]' backends; writes [outBase].json (the
     * returned JSON line), the page [outBase] + [pageExt] (".jpg" q85 or ".png") and the nets' outputs
     * ([outBase]_logits.npz, _p1.npz, _rc0.npz, _rc2.npz).
     */
    @JvmStatic external fun batchPhoto(photoPath: String, outBase: String, pageExt: String, shade: Boolean, rules: Boolean): String
}
