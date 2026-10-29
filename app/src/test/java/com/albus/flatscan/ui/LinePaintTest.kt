package com.albus.flatscan.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The line brush on screen: which of the page's lines it leaves out (as the core does after flattening again). */
class LinePaintTest {
    // a page 1.4 times as high as wide; a Keep as is stroke across the middle, radius 0.05 page widths
    private val hw = 1.4f
    private val keep = listOf(1f, 0.05f, 2f, 0.2f, 0.5f, 0.8f, 0.5f)
    private val straighten = listOf(0f, 0.05f, 2f, 0.2f, 0.2f, 0.8f, 0.2f)

    @Test
    fun aLineUnderKeepAsIsIsLeftOut() {
        assertTrue(leftOutByPaint(keep, 0.5f, 0.5f, hw))
        // 0.03 page heights off the stroke = 0.042 page widths: inside the radius
        assertTrue(leftOutByPaint(keep, 0.5f, 0.53f, hw))
        // 0.05 page heights = 0.07 page widths: outside it
        assertFalse(leftOutByPaint(keep, 0.5f, 0.55f, hw))
        // past the stroke's end by more than the radius
        assertFalse(leftOutByPaint(keep, 0.9f, 0.5f, hw))
    }

    @Test
    fun withStraightenPaintOnlyTheLinesUnderItAreUsed() {
        assertFalse(leftOutByPaint(straighten, 0.5f, 0.2f, hw))
        assertTrue(leftOutByPaint(straighten, 0.5f, 0.6f, hw))
        // both: Keep as is wins where they overlap
        val both = straighten + listOf(1f, 0.05f, 2f, 0.4f, 0.2f, 0.6f, 0.2f)
        assertTrue(leftOutByPaint(both, 0.5f, 0.2f, hw))
        assertFalse(leftOutByPaint(both, 0.25f, 0.2f, hw))
    }

    @Test
    fun noPaintLeavesEveryLine() {
        assertFalse(leftOutByPaint(emptyList(), 0.5f, 0.5f, hw))
    }
}
