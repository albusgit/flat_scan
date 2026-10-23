package com.albus.flatscan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.ui.draw.shadow
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.outlined.RotateRight
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Gesture
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.ScanViewModel
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt


/**
 * Corner adjustment: draggable corners and edges (an edge slides along its neighbours), a
 * magnifier loupe while dragging, rotate, and "circle the page" when the automatic crop failed.
 */
@Composable
fun CropScreen(vm: ScanViewModel) = EditorTheme {
    val photo = vm.photo
    val display = vm.photoDisplay
    var circling by remember { mutableStateOf(false) }
    // the quality-gate note stays until it is closed or a corner is moved
    var noteClosed by remember(vm.cropReasons) { mutableStateOf(false) }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        Row(
            Modifier.fillMaxWidth().height(64.dp).padding(start = 8.dp, end = 8.dp, top = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { if (circling) circling = false else vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(if (circling) "Circle the page" else "Adjust corners", style = MaterialTheme.typography.titleMedium)
                Text(
                    when {
                        vm.detecting -> "Finding the page"
                        circling -> "Draw a loop around the whole page"
                        vm.cropPageIndex == 1 -> "Second page of the book spread"
                        else -> "Fit the corners to the page"
                    },
                    style = MaterialTheme.typography.bodySmall, color = Editor.Muted,
                )
            }
            IconButton(
                onClick = { circling = !circling },
                enabled = !vm.detecting && photo != null,
                colors = if (circling) IconButtonDefaults.iconButtonColors(containerColor = Editor.Accent.copy(alpha = 0.2f), contentColor = Editor.Accent)
                else IconButtonDefaults.iconButtonColors(),
            ) {
                Icon(Icons.Outlined.Gesture, "Circle the page to find it")
            }
            IconButton(onClick = { vm.rotatePhoto() }, enabled = !vm.detecting && photo != null && !circling) {
                Icon(Icons.AutoMirrored.Outlined.RotateRight, "Rotate 90 degrees")
            }
        }
        if (!vm.detecting && !circling && !noteClosed && vm.cropReasons.isNotEmpty()) {
            // quality gate: the automatic crop is doubtful (it will use the homography path); above the
            // photo, never over a corner
            // on a yellow sticky note
            val note = RoundedCornerShape(2.dp)
            Row(
                Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, top = 4.dp)
                    .shadow(3.dp, note).background(Palette.Highlighter, note)
                    .padding(start = 12.dp, top = 4.dp, bottom = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Outlined.WarningAmber, null, tint = Palette.Ink, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(10.dp))
                Text(
                    "Check the crop: " + vm.cropReasons.joinToString("; "),
                    style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold), color = Palette.Ink,
                    maxLines = 2, modifier = Modifier.weight(1f),
                )
                IconButton(onClick = { noteClosed = true }, modifier = Modifier.size(40.dp)) {
                    Icon(Icons.Outlined.Close, "Close", Modifier.size(18.dp), tint = Palette.Ink)
                }
            }
        }
        Box(
            Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)
                .clip(RoundedCornerShape(8.dp)).background(Editor.Raised),
            contentAlignment = Alignment.Center,
        ) {
            if (photo != null && display != null) {
                CornerEditor(
                    vm, photo.width, photo.height, display, circling,
                    onEdit = { noteClosed = true },
                    onLoop = { circling = false; vm.detectInArea(it) },
                )
            }
            if (vm.detecting) CircularProgressIndicator(color = Editor.Accent)
            if (!vm.detecting && photo != null) {
                Text(
                    if (circling) "Draw around the page; it is found inside your loop" else "Drag a corner or an edge",
                    style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold),
                    modifier = Modifier.align(Alignment.BottomStart).padding(12.dp)
                        .background(Editor.Ground.copy(alpha = 0.72f), ChipShape)
                        .padding(horizontal = 12.dp, vertical = 7.dp),
                )
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 16.dp, bottom = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryPill(onClick = { vm.back() }, container = Editor.Raised, border = Editor.Outline, modifier = Modifier.weight(1f)) {
                PillText("Cancel")
            }
            PrimaryPill(
                onClick = { vm.process() },
                enabled = !vm.detecting && !circling && photo != null && vm.corners.size == 4,
                container = Editor.Accent, content = Editor.Ground,
                modifier = Modifier.weight(1.6f),
            ) {
                PillText("Flatten page")
                Spacer(Modifier.width(8.dp))
                Icon(Icons.AutoMirrored.Filled.ArrowForward, null, Modifier.size(18.dp))
            }
        }
    }
}

/**
 * Corners after edge [k] (from corner k to corner k + 1) moved by [d] photo px along its normal:
 * each end slides along its other edge, so the neighbouring edges keep their direction (a plain
 * shift where an edge is nearly parallel to the moved one).
 */
private fun moveEdge(c: List<Offset>, k: Int, d: Float): List<Offset> {
    val a = k
    val b = (k + 1) % 4
    val e = c[b] - c[a]
    val len = e.getDistance().coerceAtLeast(1e-3f)
    val n = Offset(-e.y / len, e.x / len)
    val shift = n * d
    fun slide(i: Int, other: Int): Offset {
        val dir = c[i] - c[other]
        val den = dir.x * n.x + dir.y * n.y
        // the corner moves along its other edge until it reaches the moved line
        return if (abs(den) < 0.2f * dir.getDistance()) c[i] + shift else c[i] + dir * (d / den)
    }
    val out = c.toMutableList()
    out[a] = slide(a, (a + 3) % 4)
    out[b] = slide(b, (b + 1) % 4)
    return out
}

@Composable
private fun CornerEditor(
    vm: ScanViewModel, pw: Int, ph: Int, display: android.graphics.Bitmap, circling: Boolean,
    onEdit: () -> Unit, onLoop: (List<Offset>) -> Unit,
) {
    val image = remember(display) { display.asImageBitmap() }
    var box by remember { mutableStateOf(IntSize.Zero) }
    // 0..3: a corner, 4..7: edge (k - 4), from corner k - 4 to the next one
    var active by remember { mutableIntStateOf(-1) }
    val corners by rememberUpdatedState(vm.corners)
    val edit by rememberUpdatedState(onEdit)
    val loopDone by rememberUpdatedState(onLoop)
    val loop = remember { mutableStateListOf<Offset>() }
    val primary = Editor.Accent
    val density = LocalDensity.current
    val margin = with(density) { 28.dp.toPx() }
    val handleR = with(density) { 12.dp.toPx() }
    val touchR = with(density) { 44.dp.toPx() }
    val loupeR = with(density) { 64.dp.toPx() }
    // photo pixels -> screen: fit inside the box with a margin so corner handles stay reachable
    val scale = if (box.width == 0) 1f else min((box.width - 2 * margin) / pw, (box.height - 2 * margin) / ph)
    val ox = (box.width - pw * scale) / 2
    val oy = (box.height - ph * scale) / 2
    fun toScreen(p: Offset) = Offset(ox + p.x * scale, oy + p.y * scale)
    fun toPhoto(q: Offset) = Offset(((q.x - ox) / scale).coerceIn(0f, pw - 1f), ((q.y - oy) / scale).coerceIn(0f, ph - 1f))
    fun clampPhoto(p: Offset) = Offset(p.x.coerceIn(0f, pw - 1f), p.y.coerceIn(0f, ph - 1f))
    var dragStart by remember { mutableStateOf(listOf<Offset>()) }
    var dragTotal by remember { mutableStateOf(Offset.Zero) }
    val gestures = if (circling) {
        Modifier.pointerInput(pw, ph, scale, circling) {
            detectDragGestures(
                onDragStart = { pos -> loop.clear(); loop.add(toPhoto(pos)) },
                onDragEnd = {
                    val pts = loop.toList()
                    loop.clear()
                    loopDone(pts)
                },
                onDragCancel = { loop.clear() },
                onDrag = { change, _ ->
                    change.consume()
                    loop.add(toPhoto(change.position))
                },
            )
        }
    } else {
        Modifier.pointerInput(pw, ph, scale, circling) {
            detectDragGestures(
                onDragStart = { pos ->
                    val pts = corners.map { toScreen(it) }
                    if (pts.size != 4) return@detectDragGestures
                    val dc = pts.map { hypot(it.x - pos.x, it.y - pos.y) }
                    val ic = dc.indices.minBy { dc[it] }
                    val mids = List(4) { (pts[it] + pts[(it + 1) % 4]) / 2f }
                    val de = mids.map { hypot(it.x - pos.x, it.y - pos.y) }
                    val ie = de.indices.minBy { de[it] }
                    // corners win when both are in reach
                    active = when {
                        dc[ic] < touchR -> ic
                        de[ie] < touchR -> 4 + ie
                        else -> -1
                    }
                    dragStart = corners
                    dragTotal = Offset.Zero
                    if (active >= 0) edit()
                },
                onDragEnd = { active = -1 },
                onDragCancel = { active = -1 },
                onDrag = { change, delta ->
                    if (active >= 0) {
                        change.consume()
                        dragTotal += delta / scale
                        val next = if (active < 4) {
                            dragStart.toMutableList().also { it[active] = clampPhoto(dragStart[active] + dragTotal) }
                        } else {
                            val k = active - 4
                            val e = dragStart[(k + 1) % 4] - dragStart[k]
                            val len = e.getDistance().coerceAtLeast(1e-3f)
                            val d = (dragTotal.x * -e.y + dragTotal.y * e.x) / len
                            moveEdge(dragStart, k, d).map { clampPhoto(it) }
                        }
                        vm.corners = next
                    }
                },
            )
        }
    }
    Canvas(Modifier.fillMaxSize().onSizeChanged { box = it }.then(gestures)) {
        drawImage(
            image, srcOffset = IntOffset.Zero, srcSize = IntSize(image.width, image.height),
            dstOffset = IntOffset(ox.roundToInt(), oy.roundToInt()),
            dstSize = IntSize((pw * scale).roundToInt(), (ph * scale).roundToInt()),
        )
        if (circling) {
            if (loop.size > 1) {
                val path = Path().apply {
                    val s0 = toScreen(loop[0])
                    moveTo(s0.x, s0.y)
                    for (i in 1 until loop.size) toScreen(loop[i]).let { lineTo(it.x, it.y) }
                }
                drawPath(path, Color.White, style = Stroke(width = 6.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
                drawPath(path, primary, style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
            }
            return@Canvas
        }
        if (corners.size != 4) return@Canvas
        val pts = corners.map { toScreen(it) }
        val quad = Path().apply { moveTo(pts[0].x, pts[0].y); pts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
        drawPath(quad, Editor.Accent.copy(alpha = 0.10f))
        // a clean dashed outline over a dark rim (visible on white paper)
        drawPath(quad, Editor.Ground.copy(alpha = 0.6f), style = Stroke(width = 4.dp.toPx(), join = StrokeJoin.Round))
        drawPath(quad, primary, style = Stroke(width = 2.dp.toPx(), join = StrokeJoin.Round, pathEffect = PathEffect.dashPathEffect(floatArrayOf(8.dp.toPx(), 5.dp.toPx()))))
        // edge handles: short pills along each edge's middle
        for (k in 0 until 4) {
            val a = pts[k]
            val b = pts[(k + 1) % 4]
            val mid = (a + b) / 2f
            val deg = Math.toDegrees(atan2((b.y - a.y).toDouble(), (b.x - a.x).toDouble())).toFloat()
            val on = active == 4 + k
            val w = (if (on) 34.dp else 28.dp).toPx()
            val h = (if (on) 10.dp else 8.dp).toPx()
            rotate(deg, pivot = mid) {
                drawRect(if (on) Color.White else primary, topLeft = mid - Offset(w / 2, h / 2), size = Size(w, h))
                drawRect(Editor.Ground, topLeft = mid - Offset(w / 2, h / 2), size = Size(w, h), style = Stroke(width = 1.5.dp.toPx()))
            }
        }
        pts.forEachIndexed { i, p ->
            // solid square tabs in the highlighter; the one being dragged turns white and grows
            val r = if (i == active) handleR * 1.3f else handleR * 0.85f
            val tl = p - Offset(r, r)
            drawRect(if (i == active) Color.White else primary, topLeft = tl, size = Size(2 * r, 2 * r))
            drawRect(Editor.Ground, topLeft = tl, size = Size(2 * r, 2 * r), style = Stroke(width = 2.dp.toPx()))
        }
        if (active >= 0) {
            // magnifier loupe above the finger (below it near the top edge), 3x the display scale; an
            // edge shows its middle
            val target = if (active < 4) corners[active] else (corners[active - 4] + corners[(active - 3) % 4]) / 2f
            val p = toScreen(target)
            val lc = if (p.y - 2.2f * loupeR > 0) Offset(p.x, p.y - 2.2f * loupeR) else Offset(p.x, p.y + 2.2f * loupeR)
            val zoom = 3f * scale * pw / image.width            // screen px per display-bitmap px in the loupe
            val half = loupeR / zoom
            val dp = Offset(target.x * image.width / pw, target.y * image.height / ph)
            val circle = Path().apply { addOval(androidx.compose.ui.geometry.Rect(lc, loupeR)) }
            clipPath(circle) {
                drawRect(Color.Black, topLeft = lc - Offset(loupeR, loupeR), size = Size(2 * loupeR, 2 * loupeR))
                // source window clamped to the bitmap; the destination follows it
                val sx0 = (dp.x - half).coerceIn(0f, image.width - 1f)
                val sy0 = (dp.y - half).coerceIn(0f, image.height - 1f)
                val sx1 = (dp.x + half).coerceIn(sx0 + 1f, image.width.toFloat())
                val sy1 = (dp.y + half).coerceIn(sy0 + 1f, image.height.toFloat())
                drawImage(
                    image,
                    srcOffset = IntOffset(sx0.toInt(), sy0.toInt()),
                    srcSize = IntSize((sx1.toInt() - sx0.toInt()).coerceAtLeast(1), (sy1.toInt() - sy0.toInt()).coerceAtLeast(1)),
                    dstOffset = IntOffset((lc.x + (sx0.toInt() - dp.x) * zoom).roundToInt(), (lc.y + (sy0.toInt() - dp.y) * zoom).roundToInt()),
                    dstSize = IntSize(((sx1.toInt() - sx0.toInt()) * zoom).roundToInt(), ((sy1.toInt() - sy0.toInt()) * zoom).roundToInt()),
                )
                // the page outline inside the loupe
                val loupePts = corners.map { lc + (Offset(it.x * image.width / pw, it.y * image.height / ph) - dp) * zoom }
                val lq = Path().apply { moveTo(loupePts[0].x, loupePts[0].y); loupePts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
                drawPath(lq, primary, style = Stroke(width = 1.5.dp.toPx()))
            }
            drawCircle(Color.White, radius = loupeR, center = lc, style = Stroke(width = 3.dp.toPx()))
            val a = 10.dp.toPx()
            drawLine(primary, lc - Offset(a, 0f), lc + Offset(a, 0f), strokeWidth = 2.dp.toPx())
            drawLine(primary, lc - Offset(0f, a), lc + Offset(0f, a), strokeWidth = 2.dp.toPx())
        }
    }
}

/** Dashed outline style shared by the area editors. */
internal fun dashed(widthPx: Float, onPx: Float) = Stroke(width = widthPx, pathEffect = PathEffect.dashPathEffect(floatArrayOf(onPx, onPx)))
