package com.albus.flatscan.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
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
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.random.Random
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

private enum class Mode { Pending, Hold, Zoom, Pan, Draw, Split, Idle }

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
private fun DrawScope.drawPageImage(image: ImageBitmap, pw: Int, ph: Int, colorFilter: ColorFilter? = null) {
    drawImage(
        image, srcOffset = IntOffset.Zero, srcSize = IntSize(image.width, image.height),
        dstOffset = IntOffset.Zero, dstSize = IntSize(pw, ph), filterQuality = FilterQuality.Medium, colorFilter = colorFilter,
    )
}

/** What the page viewer edits besides zooming: nothing, protected pictures or the unwarp's lines. */
enum class AreaMode { None, Protect, Lines, Erase }

private val LineColors = listOf(Color(0xFF8FB0FF), Color(0xFF5BD18B), Color(0xFFC79BFF))   // ruled, text row, grid
private val IgnoreColor = Color(0xFFFF6B5E)
private val EraseColor = Color(0xFFF2B84B)

/**
 * The result page with pinch zoom / pan (double tap toggles 1x / 2.5x), press and hold for the
 * original ([before]) and an optional [overlay] (wrinkle map). With an [areaMode] the [areas]
 * (page fractions x0, y0, x1, y1 ...) are drawn, a one-finger drag draws a new one ([onRect],
 * page fractions) and a tap is reported to [onTap] (page fractions and the index of the line of
 * [lines] under the finger, or -1). In [AreaMode.Lines] the unwarp's [lines] (NativeLib.lines
 * layout) are drawn; a line whose middle is in an area is drawn as ignored. [pageHeight] may differ
 * from the bitmap's (PageLook shape): the bitmap is stretched over the page. [colorFilter] (PageLook
 * colours) applies to the cleaned page only. [compare] (0..1, the split across the page): the
 * original on the left of the split, the cleaned page on the right; a one-finger drag moves it
 * ([onCompare]).
 */
@Composable
fun PageViewer(
    page: Bitmap?,
    before: Bitmap?,
    overlay: Bitmap?,
    pageWidth: Int,
    pageHeight: Int,
    rotation: Int,
    areas: List<Float>,
    areaMode: AreaMode,
    lines: FloatArray,
    state: ViewerState,
    onHold: (Boolean) -> Unit,
    onRect: (Float, Float, Float, Float) -> Unit,
    onTap: (Float, Float, Int) -> Unit,
    modifier: Modifier = Modifier,
    colorFilter: ColorFilter? = null,
    compare: Float? = null,
    onCompare: (Float) -> Unit = {},
) {
    val protectMode = areaMode != AreaMode.None
    val pageImg = remember(page) { page?.asImageBitmap() }
    val beforeImg = remember(before) { before?.asImageBitmap() }
    val overlayImg = remember(overlay) { overlay?.asImageBitmap() }
    var holding by remember { mutableStateOf(false) }
    // holding for the original briefly crinkles the page back into the photo
    val crinkle = remember { Animatable(0f) }
    val reduced = rememberReducedMotion()
    LaunchedEffect(holding) {
        if (holding && !reduced) {
            crinkle.snapTo(1f)
            crinkle.animateTo(0f, tween(420))
        } else {
            crinkle.snapTo(0f)
        }
    }
    var box by remember { mutableStateOf(Size.Zero) }
    // the rectangle being drawn, page pixels
    var drawFrom by remember { mutableStateOf<Offset?>(null) }
    var drawTo by remember { mutableStateOf(Offset.Zero) }

    val geomNow = { PageGeometry(pageWidth, pageHeight, rotation, box, state.scale, state.pan) }
    val geometry by rememberUpdatedState(geomNow)
    val protecting by rememberUpdatedState(protectMode)
    val comparing by rememberUpdatedState(compare != null && !protectMode)
    val splitTo by rememberUpdatedState(onCompare)
    val hold by rememberUpdatedState(onHold)
    val rect by rememberUpdatedState(onRect)
    val tap by rememberUpdatedState(onTap)
    val shownLines by rememberUpdatedState(if (areaMode == AreaMode.Lines) lines else FloatArray(0))
    val outline = androidx.compose.material3.MaterialTheme.colorScheme.primary

    // the split as a fraction across the page on screen
    fun splitAt(x: Float): Float {
        val g = geometry()
        val w = g.rw * g.k
        val left = g.center.x + state.pan.x - w / 2
        return if (w <= 0f) 0.5f else ((x - left) / w).coerceIn(0f, 1f)
    }

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
                        val ev: PointerEvent? = if (mode == Mode.Pending && !protecting && !comparing) {
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
                                    comparing -> {
                                        splitTo(splitAt(ch.position.x))
                                        Mode.Split
                                    }
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
                            Mode.Split -> {
                                splitTo(splitAt(ch.position.x))
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
                        Mode.Pending -> if (comparing) {
                            splitTo(splitAt(start.x))
                        } else if (protecting) {
                            // a tap while editing areas: the line under the finger (24 dp), if any
                            val g = geometry()
                            val q = g.toPage(start)
                            val l = shownLines
                            val tol = 24.dp.toPx() / g.k
                            var best = -1
                            var bestD = tol
                            var i = 0
                            while (5 * i + 4 < l.size) {
                                val d = segmentDistance(q, Offset(l[5 * i + 1] * g.pw, l[5 * i + 2] * g.ph), Offset(l[5 * i + 3] * g.pw, l[5 * i + 4] * g.ph))
                                if (d < bestD) { bestD = d; best = i }
                                i++
                            }
                            if (g.pw > 0 && g.ph > 0) tap(q.x / g.pw, q.y / g.ph, best)
                        } else {
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
        // a soft shadow under the page lifts it off the dark ground
        val pageSize = Size(g.rw * g.k, g.rh * g.k)
        val pageTl = g.center + state.pan - Offset(pageSize.width / 2, pageSize.height / 2)
        val blur = 18.dp.toPx()
        for (i in 1..6) {
            val grow = blur * i / 6f
            drawRoundRect(
                Color.Black.copy(alpha = 0.07f), pageTl + Offset(-grow, -grow + blur * 0.5f), Size(pageSize.width + 2 * grow, pageSize.height + 2 * grow),
                androidx.compose.ui.geometry.CornerRadius(grow),
            )
        }
        with(g) {
            inPage {
                drawPageImage(shown, pageWidth, pageHeight, if (shown === pageImg) colorFilter else null)
                if (!holding && overlayImg != null) drawPageImage(overlayImg, pageWidth, pageHeight)
                val c = crinkle.value
                if (holding && c > 0.01f) drawCreases(pageWidth, pageHeight, c, 2.dp.toPx() / g.k)
                if (protectMode) {
                    val sw = 2.dp.toPx() / g.k
                    val areaColor = when (areaMode) {
                        AreaMode.Lines -> IgnoreColor
                        AreaMode.Erase -> EraseColor
                        else -> outline
                    }
                    if (areaMode == AreaMode.Lines) {
                        // the lines the unwarp used; ignored ones dashed in red
                        val lw = 3.dp.toPx() / g.k
                        var j = 0
                        while (5 * j + 4 < lines.size) {
                            val a = Offset(lines[5 * j + 1] * pageWidth, lines[5 * j + 2] * pageHeight)
                            val b = Offset(lines[5 * j + 3] * pageWidth, lines[5 * j + 4] * pageHeight)
                            val mid = Offset((lines[5 * j + 1] + lines[5 * j + 3]) / 2, (lines[5 * j + 2] + lines[5 * j + 4]) / 2)
                            if (inAreas(areas, mid.x, mid.y)) {
                                drawLine(IgnoreColor, a, b, strokeWidth = lw, cap = StrokeCap.Round, pathEffect = PathEffect.dashPathEffect(floatArrayOf(3 * lw, 2 * lw)))
                            } else {
                                drawLine(Color.Black.copy(alpha = 0.35f), a, b, strokeWidth = lw * 1.8f, cap = StrokeCap.Round)
                                drawLine(LineColors[lines[5 * j].toInt().coerceIn(0, 2)], a, b, strokeWidth = lw, cap = StrokeCap.Round)
                            }
                            j++
                        }
                    }
                    var i = 0
                    while (i + 3 < areas.size) {
                        val tl = Offset(areas[i] * pageWidth, areas[i + 1] * pageHeight)
                        val sz = Size((areas[i + 2] - areas[i]) * pageWidth, (areas[i + 3] - areas[i + 1]) * pageHeight)
                        // a white rim under a dashed line keeps the box visible on any picture
                        drawRect(areaColor.copy(alpha = 0.18f), tl, sz)
                        drawRect(Color.White, tl, sz, style = Stroke(sw * 2.4f))
                        drawRect(areaColor, tl, sz, style = Stroke(sw * 1.4f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(5 * sw, 3 * sw))))
                        i += 4
                    }
                    val a = drawFrom
                    if (a != null) {
                        val tl = Offset(min(a.x, drawTo.x).coerceIn(0f, pageWidth.toFloat()), min(a.y, drawTo.y).coerceIn(0f, pageHeight.toFloat()))
                        val br = Offset(max(a.x, drawTo.x).coerceIn(0f, pageWidth.toFloat()), max(a.y, drawTo.y).coerceIn(0f, pageHeight.toFloat()))
                        drawRect(areaColor.copy(alpha = 0.12f), tl, Size(br.x - tl.x, br.y - tl.y))
                        drawRect(Color.White, tl, Size(br.x - tl.x, br.y - tl.y), style = Stroke(sw * 2))
                        drawRect(areaColor, tl, Size(br.x - tl.x, br.y - tl.y), style = Stroke(sw))
                    }
                }
            }
            val split = compare
            if (split != null && !protectMode && !holding && beforeImg != null) {
                // the original on the left of the split, a divider with a round handle
                val x = pageTl.x + pageSize.width * split
                clipRect(left = pageTl.x, top = pageTl.y, right = x, bottom = pageTl.y + pageSize.height) {
                    inPage { drawPageImage(beforeImg, pageWidth, pageHeight) }
                }
                val top = maxOf(pageTl.y, 0f)
                val bottom = minOf(pageTl.y + pageSize.height, size.height)
                drawLine(Color.Black.copy(alpha = 0.35f), Offset(x, top), Offset(x, bottom), strokeWidth = 4.dp.toPx())
                drawLine(Color.White, Offset(x, top), Offset(x, bottom), strokeWidth = 2.dp.toPx())
                val c = Offset(x, (top + bottom) / 2)
                drawCircle(Color.Black.copy(alpha = 0.3f), radius = 17.dp.toPx(), center = c + Offset(0f, 1.dp.toPx()))
                drawCircle(Color.White, radius = 16.dp.toPx(), center = c)
                val a = 5.dp.toPx()
                val ink = Color(0xFF17181B)
                // two small arrows: drag sideways
                drawLine(ink, c + Offset(-a * 1.6f, 0f), c + Offset(-a * 0.6f, -a), strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
                drawLine(ink, c + Offset(-a * 1.6f, 0f), c + Offset(-a * 0.6f, a), strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
                drawLine(ink, c + Offset(a * 1.6f, 0f), c + Offset(a * 0.6f, -a), strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
                drawLine(ink, c + Offset(a * 1.6f, 0f), c + Offset(a * 0.6f, a), strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
            }
        }
    }
}

/** Crease lines across the page (fractions: x0, y0, x1, y1), a dark fold beside a lit one. */
private val Creases = Random(11).let { r ->
    List(14) {
        val x = r.nextFloat()
        val y = r.nextFloat()
        val a = r.nextFloat() * Math.PI.toFloat()
        val l = 0.25f + r.nextFloat() * 0.45f
        floatArrayOf(x - cos(a) * l / 2, y - sin(a) * l / 2, x + cos(a) * l / 2, y + sin(a) * l / 2)
    }
}

/** The creases of a crumpled sheet over a [pw] x [ph] page, at strength [k] (0..1). */
private fun DrawScope.drawCreases(pw: Int, ph: Int, k: Float, width: Float) {
    for (c in Creases) {
        val a = Offset(c[0] * pw, c[1] * ph)
        val b = Offset(c[2] * pw, c[3] * ph)
        val n = Offset(-(b.y - a.y), b.x - a.x).let { it / max(1e-3f, it.getDistance()) } * width
        drawLine(Color.Black.copy(alpha = 0.28f * k), a, b, strokeWidth = width * 1.5f, cap = StrokeCap.Round)
        drawLine(Color.White.copy(alpha = 0.45f * k), a + n, b + n, strokeWidth = width, cap = StrokeCap.Round)
    }
    drawRect(Color(0xFF8A7A55).copy(alpha = 0.10f * k), size = Size(pw.toFloat(), ph.toFloat()))
}

/** Whether the page point (fractions) lies in one of the rectangles [x0, y0, x1, y1, ...]. */
fun inAreas(a: List<Float>, x: Float, y: Float): Boolean {
    var k = 0
    while (k + 3 < a.size) {
        if (x >= a[k] && x <= a[k + 2] && y >= a[k + 1] && y <= a[k + 3]) return true
        k += 4
    }
    return false
}

private fun segmentDistance(p: Offset, a: Offset, b: Offset): Float {
    val ab = b - a
    val l2 = ab.x * ab.x + ab.y * ab.y
    val t = if (l2 <= 0f) 0f else (((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / l2).coerceIn(0f, 1f)
    return (p - (a + ab * t)).getDistance()
}

/**
 * A small page preview (filter strip): [image] turned [rotation] quarter turns, fitted; [hw] (height
 * / width, PageLook shape) stretches it, [colorFilter] tints it as the page.
 */
@Composable
fun PageThumb(image: Bitmap?, rotation: Int, modifier: Modifier = Modifier, hw: Float? = null, colorFilter: ColorFilter? = null) {
    val img = remember(image) { image?.asImageBitmap() }
    Canvas(modifier) {
        val i = img ?: return@Canvas
        val h = if (hw != null) (i.width * hw).toInt().coerceAtLeast(1) else i.height
        val g = PageGeometry(i.width, h, rotation, size, 1f, Offset.Zero)
        with(g) { inPage { drawPageImage(i, i.width, h, colorFilter) } }
    }
}
