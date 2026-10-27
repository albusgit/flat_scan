package com.albus.flatscan

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraGuideTest {
    // a page well inside a 240 x 320 frame, covering about half of it
    private val page = floatArrayOf(0.15f, 0.15f, 0.85f, 0.15f, 0.85f, 0.85f, 0.15f, 0.85f)

    private fun frame(
        quad: FloatArray? = page, area: Float = 0.5f, luma: Float = 0.5f, sharpness: Float = 100f, searched: Boolean = true,
    ) = PreviewFrame(240, 320, quad, area, luma, 0f, sharpness, searched)

    private fun shifted(q: FloatArray, d: Float) = FloatArray(8) { q[it] + d }

    @Test
    fun hintOfOneFrame() {
        fun raw(f: PreviewFrame, previous: FloatArray? = null, ref: Float = f.sharpness) = CameraGuide.rawHint(f, previous, ref)
        assertEquals(Hint.Ready, raw(frame()))
        // too dark comes first, page or not
        assertEquals(Hint.TooDark, raw(frame(luma = 0.1f)))
        assertEquals(Hint.TooDark, raw(frame(quad = null, luma = 0.1f)))
        // no page, or a corner at or past the border: fit the page
        assertEquals(Hint.FitPage, raw(frame(quad = null)))
        assertEquals(Hint.FitPage, raw(frame(quad = page.copyOf().also { it[0] = 0.005f })))
        assertEquals(Hint.FitPage, raw(frame(quad = page.copyOf().also { it[5] = 1.04f })))
        // small: closer; framing before steadiness
        assertEquals(Hint.MoveCloser, raw(frame(area = 0.2f)))
        assertEquals(Hint.MoveCloser, raw(frame(area = 0.2f), previous = shifted(page, 0.1f)))
        // moving or blurred: hold still
        assertEquals(Hint.HoldStill, raw(frame(), previous = shifted(page, 0.05f)))
        assertEquals(Hint.Ready, raw(frame(), previous = shifted(page, 0.01f)))
        assertEquals(Hint.HoldStill, raw(frame(sharpness = 30f), ref = 100f))
        // a featureless scene is never "blurred"
        assertEquals(Hint.Ready, raw(frame(sharpness = 1f), ref = 3f))
        // not looked for (Whiteboard mode, the page-finder net off): the framing instruction, or hold still
        assertEquals(Hint.FitPage, raw(frame(quad = null, searched = false)))
        assertEquals(Hint.HoldStill, raw(frame(quad = null, searched = false, sharpness = 10f), ref = 100f))
    }

    @Test
    fun aHintChangesAfterTwoFrames() {
        val g = CameraGuide()
        assertEquals(Hint.FitPage, g.state.hint)
        assertEquals(Hint.FitPage, g.update(frame()).hint)        // Ready once: not yet
        assertEquals(Hint.Ready, g.update(frame()).hint)          // twice: shown
        assertEquals(2, g.state.readyFrames)
        // one odd frame does not flicker the hint, but auto-capture's count starts again
        assertEquals(Hint.Ready, g.update(frame(area = 0.1f)).hint)
        assertEquals(0, g.state.readyFrames)
        assertEquals(Hint.Ready, g.update(frame()).hint)
        assertEquals(1, g.state.readyFrames)
        assertEquals(Hint.Ready, g.update(frame(luma = 0.05f)).hint)
        assertEquals(Hint.TooDark, g.update(frame(luma = 0.05f)).hint)
        // alternating hints never settle: the shown one stays
        g.update(frame(area = 0.1f))
        g.update(frame(quad = null))
        g.update(frame(area = 0.1f))
        assertEquals(Hint.TooDark, g.state.hint)
        g.reset()
        assertEquals(Hint.FitPage, g.state.hint)
        assertNull(g.state.quad)
    }

    @Test
    fun theQuadOutlastsOneMissedFrame() {
        val g = CameraGuide()
        g.update(frame())
        assertArrayEquals(page, g.state.quad, 0f)
        g.update(frame(quad = null))
        assertArrayEquals(page, g.state.quad, 0f)
        g.update(frame(quad = null))
        assertNull(g.state.quad)
        // a newly found page is shown at once, where it is now
        val q = shifted(page, 0.02f)
        assertArrayEquals(q, g.update(frame(quad = q)).quad, 0f)
    }

    @Test
    fun movementAndBlurAcrossFrames() {
        val g = CameraGuide()
        g.update(frame())
        g.update(frame())
        // the page sweeps across the frame: hold still
        g.update(frame(quad = shifted(page, 0.05f)))
        assertEquals(Hint.HoldStill, g.update(frame(quad = shifted(page, 0.1f))).hint)
        // it settles (one frame to compare with, then two steady ones)
        g.update(frame(quad = shifted(page, 0.1f)))
        assertEquals(Hint.Ready, g.update(frame(quad = shifted(page, 0.1f))).hint)
        // a blurred frame pair (motion blur against the recent sharp ones)
        g.update(frame(quad = shifted(page, 0.1f), sharpness = 20f))
        assertEquals(Hint.HoldStill, g.update(frame(quad = shifted(page, 0.1f), sharpness = 20f)).hint)
        // a blank page stays low: the recent best decays to it and the hint comes back
        var h = Hint.HoldStill
        repeat(15) { h = g.update(frame(quad = shifted(page, 0.1f), sharpness = 20f)).hint }
        assertEquals(Hint.Ready, h)
    }

    @Test
    fun previewToView() {
        // a portrait 3:4 frame in a 1080 x 2400 view: FIT_CENTER letterboxes it (1080 x 1440 from y 480)
        val corners = floatArrayOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f)
        assertArrayEquals(floatArrayOf(0f, 480f, 1080f, 480f, 1080f, 1920f, 0f, 1920f), CameraGuide.toView(corners, 240, 320, 1080f, 2400f), 1e-3f)
        val mid = floatArrayOf(0.5f, 0.5f, 0.25f, 0.75f, 0f, 0f, 0f, 0f)
        val v = CameraGuide.toView(mid, 240, 320, 1080f, 2400f)
        assertEquals(540f, v[0], 1e-3f)
        assertEquals(1200f, v[1], 1e-3f)
        assertEquals(270f, v[2], 1e-3f)
        assertEquals(480f + 1080f, v[3], 1e-3f)
        // a landscape view: pillarboxed
        assertArrayEquals(floatArrayOf(510f, 0f, 1290f, 0f, 1290f, 1040f, 510f, 1040f), CameraGuide.toView(corners, 240, 320, 1800f, 1040f), 1e-3f)
        // FILL_CENTER: cropped to cover (1800 x 2400 from x -360)
        assertArrayEquals(floatArrayOf(-360f, 0f, 1440f, 0f, 1440f, 2400f, -360f, 2400f), CameraGuide.toView(corners, 240, 320, 1080f, 2400f, fill = true), 1e-3f)
    }

    @Test
    fun framesFromTheNativeArray() {
        val a = floatArrayOf(1f, 24f, 32f, 216f, 32f, 216f, 288f, 24f, 288f, 240f, 320f, 0.6f, 0.4f, 0.01f, 85f, 12f)
        val f = PreviewFrame.from(a, searched = true)
        assertNotNull(f)
        f!!
        assertEquals(240, f.width)
        assertEquals(320, f.height)
        assertArrayEquals(floatArrayOf(0.1f, 0.1f, 0.9f, 0.1f, 0.9f, 0.9f, 0.1f, 0.9f), f.quad, 1e-6f)
        assertEquals(0.6f, f.area)
        assertEquals(0.4f, f.luma)
        assertEquals(85f, f.sharpness)
        assertTrue(f.searched)
        assertNull(PreviewFrame.from(a.copyOf().also { it[0] = 0f }, true)!!.quad)
        assertNull(PreviewFrame.from(FloatArray(10), true))
        assertNull(PreviewFrame.from(FloatArray(16), true))
    }

    @Test
    fun hintWords() {
        assertEquals("Fit the whole page in view", CameraGuide.text(Hint.FitPage, board = false, torch = true))
        assertEquals("Fit the whole board in view", CameraGuide.text(Hint.FitPage, board = true, torch = true))
        assertEquals("Too dark: turn on the torch", CameraGuide.text(Hint.TooDark, board = false, torch = true))
        assertEquals("Too dark: find more light", CameraGuide.text(Hint.TooDark, board = false, torch = false))
    }
}
