package com.albus.flatscan

import android.graphics.Bitmap

/** Progress callback from the native pipeline (called on the processing thread). */
fun interface ProgressListener {
    fun onProgress(percent: Int, stage: String)
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

    /** Runs the whole pipeline; returns a session handle (release it with [release]). */
    @JvmStatic external fun process(
        photo: Bitmap, corners: FloatArray?, cornersEdited: Boolean, focalPx: Float,
        autoOrient: Boolean, renderWidth: Int, useGpu: Boolean, listener: ProgressListener?,
    ): Long

    /** JSON: ok, error, mode, fell_back, fit_median_px, rot, gpu_resample, per-stage ms. */
    @JvmStatic external fun info(handle: Long): String

    @JvmStatic external fun pageSize(handle: Long): IntArray

    /** Cleanup into [out] (size [pageSize]); raw copies the unwarped page as is. */
    @JvmStatic external fun render(handle: Long, gray: Boolean, solid: Boolean, raw: Boolean, useGpu: Boolean, out: Bitmap): String

    @JvmStatic external fun release(handle: Long)

    /**
     * On-device acceptance check: runs the bundled photo [runs] times (+1 warm-up), compares the
     * final map with the reference grid at [refGridPath], GPU vs CPU resample and cleanup, and
     * returns per-stage median / p95 timings and peak memory as JSON.
     */
    @JvmStatic external fun selfTest(photo: Bitmap, refGridPath: String, runs: Int, listener: ProgressListener?): String
}
