package com.albus.flatscan.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.grid.LazyGridState
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/**
 * The plan's UI test: in a 20-page document, page 5 is held and dragged to position 1. The page sits below
 * the visible part of the grid at first, so the drag also has to make the grid scroll up by itself (the
 * finger held at the top edge). Real pointer input on the grid the pages screen uses.
 */
@RunWith(AndroidJUnit4::class)
@Config(qualifiers = "w411dp-h891dp")
class PageGridDragTest {
    @get:Rule
    val rule = createComposeRule()

    @Test
    fun dragPage5ToPosition1InA20PageDocument() {
        val pages = mutableStateListOf<String>().apply { addAll((1..20).map { "p$it" }) }
        var drops = 0
        val state = LazyGridState()
        rule.setContent {
            ReorderablePageGrid(
                pages = pages,
                onMove = { from, to -> pages.add(to, pages.removeAt(from)) },
                onDragEnd = { drops++ },
                state = state,
                modifier = Modifier.fillMaxWidth().height(600.dp),
            ) { id, _, _, m ->
                Box(m.fillMaxWidth().height(210.dp).background(Color.White)) { Text(id) }
            }
        }
        // page 5 at the top of the viewport (rows of two: pages 1-4 are scrolled out above)
        rule.runOnIdle { runBlocking { state.scrollToItem(4) } }
        rule.waitForIdle()
        rule.onNodeWithTag("page_p5").performTouchInput {
            down(center)
            advanceEventTime(viewConfiguration.longPressTimeoutMillis + 100)
            // up to the top edge of the grid, where it scrolls by itself
            val steps = 20
            val target = Offset(center.x, -center.y + 10f)   // a few px inside the grid's top edge
            for (i in 1..steps) {
                moveTo(center + (target - center) * (i / steps.toFloat()))
                advanceEventTime(16)
            }
        }
        // hold there while the grid scrolls up through pages 1-4
        rule.mainClock.advanceTimeBy(3000)
        rule.onNodeWithTag("page_p5").performTouchInput {
            // then onto page 1's slot (top-left cell) and drop
            moveTo(Offset(width * 0.25f, 40f))
            advanceEventTime(16)
            up()
        }
        rule.waitForIdle()
        assertEquals("p5", pages[0])
        assertEquals(listOf("p1", "p2", "p3", "p4") + (6..20).map { "p$it" }, pages.drop(1))
        assertEquals(1, drops)
    }

    /**
     * Holding a page at the bottom edge scrolls the grid down, so it can be carried past the visible pages.
     * Page 3, not page 1: the grid keeps its first visible item in place, so dragging that one would move
     * the grid even without the edge scrolling.
     */
    @Test
    fun holdingAtTheBottomEdgeScrollsDown() {
        val pages = mutableStateListOf<String>().apply { addAll((1..20).map { "p$it" }) }
        val state = LazyGridState()
        rule.setContent {
            ReorderablePageGrid(
                pages = pages, onMove = { from, to -> pages.add(to, pages.removeAt(from)) }, onDragEnd = {},
                state = state, modifier = Modifier.fillMaxWidth().height(600.dp),
            ) { id, _, _, m -> Box(m.fillMaxWidth().height(210.dp).background(Color.White)) { Text(id) } }
        }
        val visibleAtStart = rule.runOnIdle { state.layoutInfo.visibleItemsInfo.size }
        var gridHeight = 0f
        rule.onNodeWithTag("page_grid").performTouchInput { gridHeight = height.toFloat() }
        rule.onNodeWithTag("page_p3").performTouchInput {
            down(center)
            advanceEventTime(viewConfiguration.longPressTimeoutMillis + 100)
            for (i in 1..20) {
                moveTo(Offset(center.x, center.y + (gridHeight - 30f - center.y) * i / 20f))
                advanceEventTime(16)
            }
        }
        rule.mainClock.advanceTimeBy(2500)
        rule.onNodeWithTag("page_p3").performTouchInput { up() }
        rule.waitForIdle()
        val at = pages.indexOf("p3")
        assertTrue("the grid did not scroll", state.firstVisibleItemIndex > 0)
        assertTrue("p3 went to $at, not past the $visibleAtStart pages visible at the start", at >= visibleAtStart)
        assertEquals(((1..20).map { "p$it" } - "p3"), pages.filter { it != "p3" })
    }
}
