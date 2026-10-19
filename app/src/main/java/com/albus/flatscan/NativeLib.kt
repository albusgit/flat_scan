package com.albus.flatscan

import android.graphics.Bitmap

/** Progress callback from the native pipeline (called on the processing thread). */
fun interface ProgressListener {
    fun onProgress(percent: Int, stage: String)
}

/** Filters of the filter strip; [id] is the native enum value (core/src/cleanup.h Filter). */
enum class FilterKind(val id: Int, val label: String) {
    Original(0, "Original"),
    Lighten(1, "Lighten"),
    AutoColor(2, "Auto color"),
    Grayscale(3, "Grayscale"),
    HighContrast(4, "High contrast"),
    BlackWhite(5, "B&W"),
}

/**
 * Everything the user can change on the result screen; switching any of these never re-runs
 * the geometry. Defaults are the tested defaults (identity adjustments).
 */
data class RenderParams(
    val filter: FilterKind = FilterKind.AutoColor,
    val strength: Float = 0.7f,          // 0..1, 0 = Original
    val solid: Boolean = false,          // Solid colors (ignored for B&W)
    val bgCleanup: Float = 0f,           // 0..1: push near-paper grey artefacts to white (0 = off)
    val black: Float = 0f,               // levels black point, 0..0.5 (output fraction)
    val white: Float = 1f,               // levels white point, 0.5..1
    val gamma: Float = 1f,               // levels midtones, 0.3..3 (> 1 brightens mid-tones)
) {
    /** Layout read by jni_bridge.cpp: [filter, strength, solid, bgCleanup, black, white, gamma]. */
    fun toArray(): FloatArray = floatArrayOf(filter.id.toFloat(), strength, if (solid) 1f else 0f, bgCleanup, black, white, gamma)
}

/** JNI entry points of the C++ core (app/src/main/cpp/jni_bridge.cpp). */
object NativeLib {
    init {
        System.loadLibrary("flatscan")
    }

    /** Which backend the heavy image passes use, with the GLES driver string. */
    @JvmStatic external fun gpuInfo(): String

    /** Page corners TL, TR, BR, BL as [x0, y0, ... x3, y3] in bitmap pixels, or null. */
    @JvmStatic external fun detect(photo: Bitmap): FloatArray?

    /**
     * Runs the whole pipeline; returns a session handle (release it with [release]).
     * autoOrient: 0/90/180/270 decision (180 only when the cue is confident).
     */
    @JvmStatic external fun process(
        photo: Bitmap, corners: FloatArray?, cornersEdited: Boolean, focalPx: Float,
        autoOrient: Boolean, renderWidth: Int, useGpu: Boolean, listener: ProgressListener?,
    ): Long

    /**
     * JSON: ok, error, mode, fell_back, fit_median_px, rot, gpu_resample, legend_px (wrinkle
     * map scale top, px), has_paper_mask, per-stage ms.
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
     * On-device acceptance check: runs the bundled photo [runs] times (+1 warm-up), compares the
     * final map with the reference grid at [refGridPath], GPU vs CPU resample and cleanup, and
     * returns per-stage median / p95 timings and peak memory as JSON.
     */
    @JvmStatic external fun selfTest(photo: Bitmap, refGridPath: String, runs: Int, listener: ProgressListener?): String
}
