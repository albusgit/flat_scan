package com.albus.flatscan

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.atan2

/** Page > Perspective (PageTransform, as Lightroom's Transform panel) and the line brush's page record. */
@RunWith(AndroidJUnit4::class)
class PageTransformTest {
    private val hw = 297f / 210f

    private fun at(h: DoubleArray, x: Double, y: Double) = PageTransform.apply(h, x, y)

    @Test
    fun noChangeIsNoMatrix() {
        assertNull(PageTransform().matrix(hw))
        assertNull(PageTransform(constrain = true, upright = 3).matrix(hw))
    }

    @Test
    fun everySliderSettingRoundTripsThroughItsInverse() {
        val t = PageTransform(vertical = 0.3f, horizontal = -0.2f, rotate = 4f, aspect = 0.25f, scale = -0.1f, offsetX = 0.2f, offsetY = -0.3f)
        val m = t.matrix(hw)!!
        val inv = PageTransform.invert(m)!!
        for ((x, y) in listOf(0.0 to 0.0, 1.0 to 0.0, 0.3 to 0.7, 1.0 to 1.0, 0.5 to 0.5)) {
            val (u, v) = at(m, x, y)
            val (bx, by) = at(inv, u, v)
            assertEquals(x, bx, 1e-9)
            assertEquals(y, by, 1e-9)
        }
    }

    @Test
    fun theSlidersMoveThePageTheWayTheirNamesSay() {
        // Vertical > 0 widens the top, Horizontal > 0 the left side
        val v = PageTransform(vertical = 0.5f).matrix(hw)!!
        val top = at(v, 1.0, 0.0).first - at(v, 0.0, 0.0).first
        val bottom = at(v, 1.0, 1.0).first - at(v, 0.0, 1.0).first
        assertTrue(top > bottom)
        val h = PageTransform(horizontal = 0.5f).matrix(hw)!!
        val left = at(h, 0.0, 1.0).second - at(h, 0.0, 0.0).second
        val right = at(h, 1.0, 1.0).second - at(h, 1.0, 0.0).second
        assertTrue(left > right)
        // Rotate 5: a row turns 5 degrees (clockwise on screen), measured in the page's own proportions
        val r = PageTransform(rotate = 5f).matrix(hw)!!
        val (ax, ay) = at(r, 0.2, 0.5)
        val (bx, by) = at(r, 0.8, 0.5)
        assertEquals(5.0, Math.toDegrees(atan2((by - ay) * hw, bx - ax)), 1e-6)
        // the centre stays where it is under every tilt and turn
        for (t in listOf(v, h, r)) {
            val (cx, cy) = at(t, 0.5, 0.5)
            assertEquals(0.5, cx, 1e-9)
            assertEquals(0.5, cy, 1e-9)
        }
    }

    @Test
    fun theCornersQuadBecomesTheWholePage() {
        val q = listOf(0.1f, 0.05f, 0.92f, 0.1f, 0.95f, 0.9f, 0.05f, 0.97f)
        val m = PageTransform.fromQuad(q).matrix(hw)!!
        val corners = listOf(0.0 to 0.0, 1.0 to 0.0, 1.0 to 1.0, 0.0 to 1.0)
        for (i in 0 until 4) {
            val (x, y) = at(m, q[2 * i].toDouble(), q[2 * i + 1].toDouble())
            assertEquals(corners[i].first, x, 1e-6)
            assertEquals(corners[i].second, y, 1e-6)
        }
        // and quadOf gives it back
        val back = PageTransform.quadOf(m)!!
        for (i in 0 until 8) assertEquals(q[i], back[i], 1e-5f)
    }

    @Test
    fun constrainCropLeavesNoBlankCorner() {
        for (t in listOf(PageTransform(vertical = 0.6f), PageTransform(rotate = 7f, horizontal = -0.4f), PageTransform(scale = -0.5f))) {
            val inv = PageTransform.invert(t.copy(constrain = true).matrix(hw)!!)!!
            var tight = false
            for ((x, y) in listOf(0.0 to 0.0, 1.0 to 0.0, 1.0 to 1.0, 0.0 to 1.0)) {
                val (u, v) = at(inv, x, y)
                assertTrue("corner from ($u, $v)", u >= -1e-6 && u <= 1 + 1e-6 && v >= -1e-6 && v <= 1 + 1e-6)
                if (u < 1e-4 || u > 1 - 1e-4 || v < 1e-4 || v > 1 - 1e-4) tight = true
            }
            assertTrue("zoomed in no more than needed", tight)
        }
    }

    @Test
    fun theTransformRoundTripsThroughThePagesLook() {
        val t = PageTransform(
            vertical = 0.2f, horizontal = -0.1f, rotate = 2.5f, aspect = 0.1f, scale = 0.05f, offsetX = -0.2f, offsetY = 0.3f,
            quad = listOf(0f, 0.01f, 1f, 0f, 0.99f, 1f, 0f, 1f), constrain = true, upright = 4, guides = listOf(0.1f, 0.2f, 0.1f, 0.8f),
        )
        val look = PageLook(shape = PageShape.Measured, measuredHw = 1.5f, transform = t)
        assertEquals(look, PageLook.fromJson(JSONObject(look.toJson().toString())))
        // a page without a perspective writes none, and pages from before it read as none
        assertFalse(PageLook().toJson().has("transform"))
        assertEquals(PageTransform(), PageLook.fromJson(JSONObject().put("shape", "A4")).transform)
    }

    @Test
    fun theLineBrushAndTheAutomaticLinesSwitchAreSavedAndPartOfTheGeometry() {
        val base = PageMeta("p", FloatArray(8) { it.toFloat() }, null, 0, false, 0f, RenderParams(), FloatArray(0), 0)
        val m = base.copy(linePaint = floatArrayOf(1f, 0.03f, 2f, 0.1f, 0.2f, 0.4f, 0.2f), autoLines = false)
        val back = PageMeta.fromJson(JSONObject(m.toJson().toString()))
        assertEquals(m.linePaint.toList(), back.linePaint.toList())
        assertFalse(back.autoLines)
        // older pages: automatic lines, no paint
        val old = PageMeta.fromJson(JSONObject(base.toJson().toString()))
        assertTrue(old.autoLines)
        assertEquals(0, old.linePaint.size)
        val k = base.geometryKey(1600, true, true)
        assertFalse(k == m.geometryKey(1600, true, true))
        assertFalse(k == base.copy(autoLines = false).geometryKey(1600, true, true))
        assertEquals(k, base.copy(linePaint = FloatArray(0)).geometryKey(1600, true, true))
    }
}
