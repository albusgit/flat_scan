package com.albus.flatscan.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Stable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.toSize
import kotlin.math.max
import kotlin.math.min

/** Zoom (1x..6x) and pan of the page viewer, in screen pixels relative to the centred fit. */
@Stable
class ViewerState {
    var scale by mutableFloatStateOf(1f)
    var pan by mutableStateOf(Offset.Zero)

    fun reset() {
        scale = 1f
        pan = Offset.Zero
    }
}

private const val MAX_ZOOM = 6f
private const val DOUBLE_TAP_ZOOM = 2.5f

private enum class Mode { Pending, Hold, Zoom, Pan, Draw, Idle }

/**
 * Maps page pixels to the screen: the page (turned [rotation] clockwise quarter turns) is fitted
 * into the canvas, then zoomed by [scale] around the centre and moved by [pan].
 */
private class PageGeometry(val pw: Int, val ph: Int, val rotation: Int, val box: Size, val scale: Float, val pan: Offset) {
    val rw = if (rotation % 2 == 0) pw else ph
    val rh = if (rotation % 2 == 0) ph else pw
    val base = if (box.width <= 0f || rw == 0 || rh == 0) 1f else min(box.width / rw, box.height / rh)
    val k = base * scale        // screen px per page px
    val center = Offset(box.width / 2, box.height / 2)

    fun toPage(q: Offset): Offset {
        val v = (q - center - pan) / k
        // inverse of the clockwise quarter turns
        val u = when (rotation) {
            1 -> Offset(v.y, -v.x)
            2 -> Offset(-v.x, -v.y)
            3 -> Offset(-v.y, v.x)
            else -> v
        }
        return u + Offset(pw / 2f, ph / 2f)
    }

    fun clampPan(p: Offset, s: Float): Offset {
        val mx = max(0f, (rw * base * s - box.width) / 2)
        val my = max(0f, (rh * base * s - box.height) / 2)
        return Offset(p.x.coerceIn(-mx, mx), p.y.coerceIn(-my, my))
    }

    fun DrawScope.inPage(block: DrawScope.() -> Unit) {
        withTransform({
            translate(center.x + pan.x, center.y + pan.y)
            scale(k, k, pivot = Offset.Zero)
            rotate(90f * rotation, pivot = Offset.Zero)
            translate(-pw / 2f, -ph / 2f)
        }, block)
    }
}

/** Draws [image] stretched over the page rectangle (page pixel coordinates). */
private fun DrawScope.drawPageImage(image: ImageBitmap, pw: Int, ph: Int) {
    drawImage(
        image, srcOffset = IntOffset.Zero, srcSize = IntSize(image.width, image.height),
        dstOffset = IntOffset.Zero, dstSize = IntSize(pw, ph), filterQuality = FilterQuality.Medium,
    )
}

/**
 * The result page with pinch zoom / pan (double tap toggles 1x / 2.5x), press and hold for the
 * original ([before]), an optional [overlay] (wrinkle map) and protected rectangles. In
 * [protectMode] a one-finger drag draws a rectangle, reported as page fractions to [onRect].
 */
@Composable
fun PageViewer(
    page: Bitmap?,
    before: Bitmap?,
    overlay: Bitmap?,
    pageWidth: Int,
    pageHeight: Int,
    rotation: Int,
    protect: List<Float>,
    protectMode: Boolean,
    state: ViewerState,
    onHold: (Boolean) -> Unit,
    onRect: (Float, Float, Float, Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val pageImg = remember(page) { page?.asImageBitmap() }
    val beforeImg = remember(before) { before?.asImageBitmap() }
    val overlayImg = remember(overlay) { overlay?.asImageBitmap() }
    var holding by remember { mutableStateOf(false) }
    var box by remember { mutableStateOf(Size.Zero) }
    // the rectangle being drawn, page pixels
    var drawFrom by remember { mutableStateOf<Offset?>(null) }
    var drawTo by remember { mutableStateOf(Offset.Zero) }

    val geomNow = { PageGeometry(pageWidth, pageHeight, rotation, box, state.scale, state.pan) }
    val geometry by rememberUpdatedState(geomNow)
    val protecting by rememberUpdatedState(protectMode)
    val hold by rememberUpdatedState(onHold)
    val rect by rememberUpdatedState(onRect)
    val outline = androidx.compose.material3.MaterialTheme.colorScheme.primary

    fun zoomBy(factor: Float, centroid: Offset, panDelta: Offset) {
        val g = geometry()
        val s = (state.scale * factor).coerceIn(1f, MAX_ZOOM)
        val f = s / state.scale
        val c = centroid - g.center
        val p = c - (c - state.pan) * f + panDelta
        state.scale = s
        state.pan = g.clampPan(p, s)
    }

    Canvas(
        modifier
            .onSizeChanged { box = it.toSize() }
            .semantics { contentDescription = if (holding) "Original page" else "Cleaned page. Press and hold to see the original" }
            .pointerInput(state) {
                var lastTapTime = 0L
                var lastTapPos = Offset.Zero
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    val start = down.position
                    val longPress = viewConfiguration.longPressTimeoutMillis
                    val slop = viewConfiguration.touchSlop
                    var lastTime = down.uptimeMillis
                    var mode = Mode.Pending
                    while (true) {
                        val ev: PointerEvent? = if (mode == Mode.Pending && !protecting) {
                            val remaining = longPress - (lastTime - down.uptimeMillis)
                            withTimeoutOrNull(max(1L, remaining)) { awaitPointerEvent() }
                        } else {
                            awaitPointerEvent()
                        }
                        if (ev == null) {
                            // one finger, no movement, long enough: show the original
                            mode = Mode.Hold
                            holding = true
                            hold(true)
                            continue
                        }
                        lastTime = ev.changes.maxOf { it.uptimeMillis }
                        val pressed = ev.changes.filter { it.pressed }
                        if (pressed.isEmpty()) break
                        if (pressed.size >= 2) {
                            if (mode == Mode.Hold) {
                                holding = false
                                hold(false)
                            }
                            drawFrom = null
                            mode = Mode.Zoom
                            zoomBy(ev.calculateZoom(), ev.calculateCentroid(useCurrent = true), ev.calculatePan())
                            ev.changes.forEach { it.consume() }
                            continue
                        }
                        val ch = pressed[0]
                        when (mode) {
                            Mode.Pending -> if ((ch.position - start).getDistance() > slop) {
                                mode = when {
                                    protecting -> {
                                        drawFrom = geometry().toPage(start)
                                        drawTo = geometry().toPage(ch.position)
                                        Mode.Draw
                                    }
                                    state.scale > 1.01f -> Mode.Pan
                                    else -> Mode.Idle
                                }
                                ch.consume()
                            }
                            Mode.Pan, Mode.Zoom -> {
                                val g = geometry()
                                state.pan = g.clampPan(state.pan + (ch.position - ch.previousPosition), state.scale)
                                ch.consume()
                            }
                            Mode.Draw -> {
                                drawTo = geometry().toPage(ch.position)
                                ch.consume()
                            }
                            Mode.Hold, Mode.Idle -> {}
                        }
                    }
                    when (mode) {
                        Mode.Hold -> {
                            holding = false
                            hold(false)
                        }
                        Mode.Draw -> {
                            val a = drawFrom
                            val g = geometry()
                            if (a != null && g.pw > 0 && g.ph > 0) {
                                rect(a.x / g.pw, a.y / g.ph, drawTo.x / g.pw, drawTo.y / g.ph)
                            }
                            drawFrom = null
                        }
                        Mode.Pending -> {
                            val near = (start - lastTapPos).getDistance() < 48.dp.toPx()
                            if (lastTime - lastTapTime < viewConfiguration.doubleTapTimeoutMillis && near) {
                                if (state.scale > 1.01f) state.reset() else zoomBy(DOUBLE_TAP_ZOOM, start, Offset.Zero)
                                lastTapTime = 0L
                            } else {
                                lastTapTime = lastTime
                                lastTapPos = start
                            }
                        }
                        else -> {}
                    }
                }
            },
    ) {
        val g = PageGeometry(pageWidth, pageHeight, rotation, size, state.scale, state.pan)
        val shown = if (holding && beforeImg != null) beforeImg else pageImg
        if (shown == null || pageWidth == 0 || pageHeight == 0) return@Canvas
        with(g) {
            inPage {
                drawPageImage(shown, pageWidth, pageHeight)
                if (!holding && overlayImg != null) drawPageImage(overlayImg, pageWidth, pageHeight)
                if (protectMode) {
                    val sw = 2.dp.toPx() / g.k
                    var i = 0
                    while (i + 3 < protect.size) {
                        val tl = Offset(protect[i] * pageWidth, protect[i + 1] * pageHeight)
                        val sz = Size((protect[i + 2] - protect[i]) * pageWidth, (protect[i + 3] - protect[i + 1]) * pageHeight)
                        drawRect(outline.copy(alpha = 0.18f), tl, sz)
                        drawRect(outline, tl, sz, style = Stroke(sw))
                        i += 4
                    }
                    val a = drawFrom
                    if (a != null) {
                        val tl = Offset(min(a.x, drawTo.x).coerceIn(0f, pageWidth.toFloat()), min(a.y, drawTo.y).coerceIn(0f, pageHeight.toFloat()))
                        val br = Offset(max(a.x, drawTo.x).coerceIn(0f, pageWidth.toFloat()), max(a.y, drawTo.y).coerceIn(0f, pageHeight.toFloat()))
                        drawRect(outline.copy(alpha = 0.12f), tl, Size(br.x - tl.x, br.y - tl.y))
                        drawRect(Color.White, tl, Size(br.x - tl.x, br.y - tl.y), style = Stroke(sw * 2))
                        drawRect(outline, tl, Size(br.x - tl.x, br.y - tl.y), style = Stroke(sw))
                    }
                }
            }
        }
    }
}

/** A small page preview (filter strip): [image] turned [rotation] quarter turns, fitted. */
@Composable
fun PageThumb(image: Bitmap?, rotation: Int, modifier: Modifier = Modifier) {
    val img = remember(image) { image?.asImageBitmap() }
    Canvas(modifier) {
        val i = img ?: return@Canvas
        val g = PageGeometry(i.width, i.height, rotation, size, 1f, Offset.Zero)
        with(g) { inPage { drawPageImage(i, i.width, i.height) } }
    }
}
