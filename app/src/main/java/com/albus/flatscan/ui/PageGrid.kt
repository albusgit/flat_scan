package com.albus.flatscan.ui

import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyGridScope
import androidx.compose.foundation.lazy.grid.LazyGridState
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.toOffset
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.zIndex
import kotlinx.coroutines.delay

/**
 * The pages as a grid that reorders by hold-and-drag: the card follows the finger, the order changes
 * live through [onMove] (from index, to index) as it passes other cards, and [onDragEnd] marks the drop
 * (one undo step). Near the top or bottom edge the grid scrolls by itself, faster closer to the edge, so a
 * page can be carried through a long document. [dragEnabled] false (selection mode): no dragging.
 * [card] draws one page (its id, index, whether it is being dragged, and the modifier to apply).
 */
@Composable
fun ReorderablePageGrid(
    pages: List<String>,
    onMove: (Int, Int) -> Unit,
    onDragEnd: () -> Unit,
    modifier: Modifier = Modifier,
    dragEnabled: Boolean = true,
    state: LazyGridState = rememberLazyGridState(),
    minCell: Dp = 150.dp,
    footer: LazyGridScope.() -> Unit = {},
    card: @Composable (id: String, index: Int, dragging: Boolean, modifier: Modifier) -> Unit,
) {
    var dragId by remember { mutableStateOf<String?>(null) }
    var dragPos by remember { mutableStateOf(Offset.Zero) }
    var grab by remember { mutableStateOf(Offset.Zero) }
    var speed by remember { mutableFloatStateOf(0f) }
    val currentPages by rememberUpdatedState(pages)
    val move by rememberUpdatedState(onMove)
    val edge = with(LocalDensity.current) { 96.dp.toPx() }
    val maxSpeed = with(LocalDensity.current) { 22.dp.toPx() }

    // the card under the finger takes the dragged card's place
    fun retarget() {
        val id = dragId ?: return
        val target = state.layoutInfo.visibleItemsInfo.firstOrNull {
            val k = it.key
            k is String && k != id && k in currentPages && Rect(it.offset.toOffset(), it.size.toSize()).contains(dragPos)
        } ?: return
        val from = currentPages.indexOf(id)
        val to = currentPages.indexOf(target.key as String)
        if (from >= 0 && to >= 0) move(from, to)
    }

    fun updateSpeed() {
        val h = state.layoutInfo.viewportSize.height.toFloat()
        val y = dragPos.y
        speed = when {
            y < edge -> -maxSpeed * ((edge - y) / edge).coerceIn(0f, 1f)
            y > h - edge -> maxSpeed * ((y - (h - edge)) / edge).coerceIn(0f, 1f)
            else -> 0f
        }
    }

    LaunchedEffect(dragId) {
        while (dragId != null) {
            if (speed != 0f) {
                val used = state.scrollBy(speed)
                if (used != 0f) retarget()
            }
            delay(16)
        }
    }

    LazyVerticalGrid(
        columns = GridCells.Adaptive(minCell),
        state = state,
        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = modifier.testTag("page_grid"),
    ) {
        itemsIndexed(pages, key = { _, id -> id }) { i, id ->
            val dragging = dragId == id
            val m = Modifier
                .zIndex(if (dragging) 1f else 0f)
                .graphicsLayer {
                    if (dragging) {
                        // follows the finger wherever the grid has moved the item to
                        val info = state.layoutInfo.visibleItemsInfo.firstOrNull { it.key == id }
                        if (info != null) {
                            translationX = dragPos.x - grab.x - info.offset.x
                            translationY = dragPos.y - grab.y - info.offset.y
                        }
                        scaleX = 1.04f
                        scaleY = 1.04f
                    }
                }
                .then(
                    if (!dragEnabled) Modifier else Modifier.pointerInput(id) {
                        detectDragGesturesAfterLongPress(
                            onDragStart = { local ->
                                val info = state.layoutInfo.visibleItemsInfo.firstOrNull { it.key == id } ?: return@detectDragGesturesAfterLongPress
                                grab = local
                                dragPos = info.offset.toOffset() + local
                                dragId = id
                            },
                            onDrag = { change, d ->
                                change.consume()
                                dragPos += d
                                updateSpeed()
                                retarget()
                            },
                            onDragEnd = { dragId = null; speed = 0f; onDragEnd() },
                            onDragCancel = { dragId = null; speed = 0f; onDragEnd() },
                        )
                    },
                )
                .testTag("page_$id")
            card(id, i, dragging, m)
        }
        footer()
    }
}
