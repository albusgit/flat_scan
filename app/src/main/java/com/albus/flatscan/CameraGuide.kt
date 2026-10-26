package com.albus.flatscan

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** The camera screen's hint, in the order they are checked (the first that applies is the one shown). */
enum class Hint { TooDark, FitPage, MoveCloser, HoldStill, Ready }

/**
 * One analysed camera frame ([NativeLib.previewPage]) in the upright frame. [quad]: the page's corners TL, TR,
 * BR, BL as fractions of the frame (x0, y0 .. x3, y3; outside 0..1 when the page is cut off), null when no page
 * was found; [searched]: the page was looked for (Document mode with the page-finder net on); [area] the paper's
 * share of the frame, [luma] its mean luma (0..1), [glare] the share of the page's pixels blown out, [sharpness]
 * the frame's Laplacian variance (scene dependent: [CameraGuide] compares it with recent frames only).
 */
class PreviewFrame(
    val width: Int, val height: Int, val quad: FloatArray?, val area: Float, val luma: Float, val glare: Float, val sharpness: Float,
    val searched: Boolean, val ms: Float = 0f,
) {
    companion object {
        /** From [NativeLib.previewPage]'s array; null for an array of another layout. */
        fun from(a: FloatArray, searched: Boolean): PreviewFrame? {
            if (a.size < 16) return null
            val w = a[9].toInt()
            val h = a[10].toInt()
            if (w <= 0 || h <= 0) return null
            val quad = if (a[0] > 0.5f) FloatArray(8) { i -> a[1 + i] / if (i % 2 == 0) w else h } else null
            return PreviewFrame(w, h, quad, a[11], a[12], a[13], a[14], searched, a[15])
        }
    }
}

/**
 * What the guide shows after a frame: the [hint] (held for [CameraGuide.HYSTERESIS] frames before it changes),
 * the page's [quad] as last detected (frame fractions; null once it has been missing for the hysteresis), and
 * [readyFrames], how many frames in a row were Ready on their own (auto-capture waits for several).
 */
class GuideState(val hint: Hint = Hint.FitPage, val quad: FloatArray? = null, val readyFrames: Int = 0, val frame: PreviewFrame? = null) {
    val ready: Boolean get() = hint == Hint.Ready
}

/**
 * The camera guidance: turns analysed frames ([PreviewFrame]) into a hint. Pure (no Android, no threads): the camera
 * screen calls [update] on the main thread for every frame.
 *  * [Hint.TooDark]: the frame's mean luma below [Thresholds.dark];
 *  * [Hint.FitPage]: no page found, or a corner within [Thresholds.edge] of the frame's border (cut off);
 *    also the hint whenever the page is not looked for and nothing else applies;
 *  * [Hint.MoveCloser]: the paper covers less than [Thresholds.minArea] of the frame;
 *  * [Hint.HoldStill]: a corner moved more than [Thresholds.maxMove] (frame fractions) since the last frame, or the
 *    sharpness dropped below [Thresholds.blurDrop] of the recent best (motion blur, the focus hunting);
 *  * [Hint.Ready]: none of these.
 * A new hint is shown once it held for [HYSTERESIS] frames in a row, so one odd frame never makes the hint flicker.
 */
class CameraGuide(private val t: Thresholds = Thresholds()) {
    /**
     * [dark]: mean luma (0..1); [edge]: a corner's distance from the border (frame fractions); [minArea]: the paper's
     * share; [maxMove]: a corner's move per frame (frame fractions); [blurDrop]: sharpness against the recent best,
     * which decays by [sharpDecay] per frame; [minSharp]: below it the sharpness test is off (a featureless, dim scene).
     * First choices, not tuned on a phone yet.
     */
    data class Thresholds(
        val dark: Float = 0.16f,
        val edge: Float = 0.01f,
        val minArea: Float = 0.25f,
        val maxMove: Float = 0.025f,
        val blurDrop: Float = 0.45f,
        val sharpDecay: Float = 0.8f,
        val minSharp: Float = 4f,
    )

    var state = GuideState()
        private set
    private var pending = Hint.FitPage
    private var pendingFrames = 0
    private var lastQuad: FloatArray? = null   // the previous frame's quad (null: no page in it)
    private var missing = 0                     // frames in a row without a page
    private var sharpRef = 0f                   // the recent best sharpness (decaying)

    fun update(f: PreviewFrame): GuideState {
        sharpRef = max(f.sharpness, sharpRef * t.sharpDecay)
        val raw = rawHint(f, lastQuad, sharpRef, t)
        if (raw == pending) pendingFrames++ else { pending = raw; pendingFrames = 1 }
        val hint = if (pendingFrames >= HYSTERESIS) raw else state.hint
        missing = if (f.quad == null) missing + 1 else 0
        val quad = f.quad ?: state.quad.takeIf { missing < HYSTERESIS }
        lastQuad = f.quad
        state = GuideState(hint, quad, if (raw == Hint.Ready) state.readyFrames + 1 else 0, f)
        return state
    }

    /** Back to the start (the camera rebinds, the mode changes). */
    fun reset() {
        state = GuideState()
        pending = Hint.FitPage
        pendingFrames = 0
        lastQuad = null
        missing = 0
        sharpRef = 0f
    }

    companion object {
        /** Frames a new hint must hold before it is shown. */
        const val HYSTERESIS = 2

        /** The hint of one frame on its own; [previous]: the last frame's quad, [sharpRef]: the recent best sharpness. */
        fun rawHint(f: PreviewFrame, previous: FloatArray?, sharpRef: Float, t: Thresholds = Thresholds()): Hint {
            if (f.luma < t.dark) return Hint.TooDark
            val q = f.quad
            val blurred = sharpRef >= t.minSharp && f.sharpness < t.blurDrop * sharpRef
            if (!f.searched) return if (blurred) Hint.HoldStill else Hint.FitPage
            if (q == null || cutOff(q, t.edge)) return Hint.FitPage
            if (f.area < t.minArea) return Hint.MoveCloser
            if (blurred || previous != null && moved(previous, q) > t.maxMove) return Hint.HoldStill
            return Hint.Ready
        }

        /** A corner within [margin] of the frame's border, or outside it. */
        fun cutOff(q: FloatArray, margin: Float): Boolean = q.any { it < margin || it > 1f - margin }

        /** The largest move of a corner between two quads (frame fractions, the larger of x and y). */
        fun moved(a: FloatArray, b: FloatArray): Float {
            var m = 0f
            for (i in 0 until 8) m = max(m, abs(a[i] - b[i]))
            return m
        }

        /**
         * A quad in frame fractions ([GuideState.quad]) in px of a view of [viewW] x [viewH] that shows the
         * [frameW] x [frameH] picture (the preview, the same 4:3 field of view as the analysis frames) centred:
         * whole (PreviewView FIT_CENTER) or, [fill], cropped to cover the view (FILL_CENTER).
         */
        fun toView(q: FloatArray, frameW: Int, frameH: Int, viewW: Float, viewH: Float, fill: Boolean = false): FloatArray {
            val s = if (fill) max(viewW / frameW, viewH / frameH) else min(viewW / frameW, viewH / frameH)
            val ox = (viewW - frameW * s) / 2
            val oy = (viewH - frameH * s) / 2
            return FloatArray(8) { i -> if (i % 2 == 0) ox + q[i] * frameW * s else oy + q[i] * frameH * s }
        }

        /** The words of a hint; [board]: Whiteboard mode, [torch]: the camera has a torch (and it is off). */
        fun text(h: Hint, board: Boolean, torch: Boolean): String = when (h) {
            Hint.TooDark -> if (torch) "Too dark: turn on the torch" else "Too dark: find more light"
            Hint.FitPage -> if (board) "Fit the whole board in view" else "Fit the whole page in view"
            Hint.MoveCloser -> "Move closer"
            Hint.HoldStill -> "Hold still"
            Hint.Ready -> "Ready"
        }
    }
}
