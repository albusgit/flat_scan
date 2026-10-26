package com.albus.flatscan

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith

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
}
