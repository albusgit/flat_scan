package com.albus.flatscan

import androidx.compose.ui.geometry.Offset
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.cos
import kotlin.math.sin

/** Auto page shape: A4 or US Letter when the measured page is close to one, its own proportions otherwise. */
@RunWith(AndroidJUnit4::class)
class PageShapeTest {
    // the core's page: A4, portrait
    private val w = 1600
    private val h = 2263

    private fun auto(measuredHw: Float) = PageLook(shape = PageShape.Auto, measuredHw = measuredHw)

    @Test
    fun aPageMeasuredNearA4KeepsTheCoresShape() {
        assertNull(auto(1.39f).targetHw(w, h))
        assertNull(auto(1.44f).targetHw(w, h))
    }

    @Test
    fun aPageMeasuredNearLetterBecomesLetter() {
        assertEquals(11f / 8.5f, auto(1.27f).targetHw(w, h)!!, 1e-6f)
        // landscape: the same, turned
        assertEquals(8.5f / 11f, auto(0.78f).targetHw(h, w)!!, 1e-6f)
    }

    @Test
    fun otherShapesKeepTheirOwnProportions() {
        // a coupon strip, a receipt, a card
        assertEquals(1.75f, auto(1.75f).targetHw(w, h)!!, 1e-6f)
        assertEquals(3.2f, auto(3.2f).targetHw(w, h)!!, 1e-6f)
        assertEquals(0.63f, auto(0.63f).targetHw(w, h)!!, 1e-6f)
    }

    @Test
    fun unmeasuredPagesKeepTheCoresShape() {
        assertNull(auto(0f).targetHw(w, h))
        assertEquals(w to h, auto(0f).outputSize(w, h))
    }

    @Test
    fun pagesStoredWithoutAShapeStayA4() {
        assertEquals(PageShape.AsScanned, PageLook.fromJson(JSONObject()).shape)
    }

    // the corners (TL, TR, BR, BL) of a w x h mm sheet 900 mm from a camera of focal f (px), turned by `tilt`
    // about the photo's x axis (towards / away), `turn` about its y axis and `spin` in its plane, rounded to float
    private fun photographed(w: Double, h: Double, tilt: Double, turn: Double, spin: Double = 0.0): List<Offset> {
        val a = Math.toRadians(tilt)
        val b = Math.toRadians(turn)
        return listOf(-w / 2 to -h / 2, w / 2 to -h / 2, w / 2 to h / 2, -w / 2 to h / 2).map { (x0, y0) ->
            val x1 = x0 * cos(spin) - y0 * sin(spin)
            val y1 = x0 * sin(spin) + y0 * cos(spin)
            val y2 = y1 * cos(a)
            val z2 = y1 * sin(a)
            val x3 = x1 * cos(b) + z2 * sin(b)
            val z3 = -x1 * sin(b) + z2 * cos(b) + 900.0
            Offset((F * (x3 + 40) / z3 + PW / 2).toFloat(), (F * (y2 - 30) / z3 + PH / 2).toFloat())
        }
    }

    @Test
    fun aPageTiltedOneWayKeepsItsTrueShape() {
        // a sheet tilted towards the camera only (the usual shot): its sides converge one way only, and the
        // other direction is still foreshortened (this measured an A4 tilted 15 deg as 1.62)
        for ((w, h) in listOf(210.0 to 297.0, 156.0 to 66.3, 80.0 to 250.0))
            for (tilt in listOf(0.0, 15.0, 35.0, 45.0)) {
                val c = photographed(w, h, tilt, 0.0)
                assertEquals("$w x $h at $tilt deg, focal known", h / w, measuredPageHw(c, PW, PH, F.toFloat()).toDouble(), 0.01 * h / w)
                // no convergence across: the focal length cannot be measured from the quad and the default (0.72 x
                // the long side: 2880 here, the camera's is 3000) is used - 1.3 % off at 35 deg
                assertEquals("$w x $h at $tilt deg, focal unknown", h / w, measuredPageHw(c, PW, PH, 0f).toDouble(), 0.025 * h / w)
            }
    }

    @Test
    fun aPageTurnedBothWaysKeepsItsTrueShape() {
        for ((w, h) in listOf(210.0 to 297.0, 156.0 to 66.3, 80.0 to 250.0))
            for ((tilt, turn) in listOf(25.0 to 20.0, 40.0 to 10.0, 0.0 to 30.0)) {
                val c = photographed(w, h, tilt, turn, 0.05)
                assertEquals("$w x $h at $tilt / $turn deg", h / w, measuredPageHw(c, PW, PH, F.toFloat()).toDouble(), 0.01 * h / w)
                assertEquals("$w x $h at $tilt / $turn deg, focal unknown", h / w, measuredPageHw(c, PW, PH, 0f).toDouble(), 0.01 * h / w)
            }
    }

    private companion object {
        const val PW = 3000
        const val PH = 4000
        const val F = 3000.0
    }
}
