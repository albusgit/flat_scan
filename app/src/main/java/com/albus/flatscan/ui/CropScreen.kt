package com.albus.flatscan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.text.BasicText
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.ui.draw.shadow
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.heightIn
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
import androidx.compose.material.icons.outlined.CropSquare
import androidx.compose.material.icons.outlined.Fullscreen
import androidx.compose.material.icons.outlined.Gesture
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.NoEdges
import com.albus.flatscan.ScanViewModel
import kotlin.math.abs
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
    // Flatten page tapped while the page is still being found: it starts once the crop is there
    var queued by remember(photo) { mutableStateOf(false) }
    LaunchedEffect(queued, vm.detecting) {
        if (!queued || vm.detecting) return@LaunchedEffect
        queued = false
        if (vm.corners.size == 4) vm.continueCrop()
    }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        Row(
            Modifier.fillMaxWidth().height(64.dp).padding(start = 4.dp, end = 12.dp, top = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { if (circling) circling = false else vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Column(Modifier.weight(1f)) {
                Text(
                    when {
                        circling -> "Circle the page"
                        vm.cropPageIndex == 1 -> "Crop the second page"
                        else -> "Crop the page"
                    },
                    style = MaterialTheme.typography.titleMedium.copy(fontSize = 17.sp, fontWeight = FontWeight.ExtraBold),
                )
                Text(
                    when {
                        vm.detecting -> "Finding the page"
                        circling -> "Draw a loop around the whole page; it is found inside it"
                        vm.cropPageIndex == 1 -> "The other page of the book spread"
                        else -> "Pull a + to bend an edge"
                    },
                    style = MaterialTheme.typography.bodySmall, color = Editor.Muted, maxLines = 1, overflow = TextOverflow.Ellipsis,
                )
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
            if (!vm.detecting && photo != null && circling) {
                Text(
                    "Draw around the page",
                    style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold),
                    modifier = Modifier.align(Alignment.BottomStart).padding(12.dp)
                        .background(Editor.Ground.copy(alpha = 0.72f), ChipShape)
                        .padding(horizontal = 12.dp, vertical = 7.dp),
                )
            }
        }
        val tools = !vm.detecting && photo != null
        Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min).padding(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CropTool(Icons.AutoMirrored.Outlined.RotateRight, "Rotate", enabled = tools && !circling) { vm.rotatePhoto() }
            CropTool(Icons.Outlined.Gesture, "Circle page", enabled = tools, on = circling) { circling = !circling }
            CropTool(Icons.Outlined.Fullscreen, "Whole photo", enabled = tools && !circling) { vm.cropWholePhoto(); noteClosed = true }
            // the points between the corners away: straight sides again
            if (vm.edgePoints.any { it.isNotEmpty() }) {
                CropTool(Icons.Outlined.CropSquare, "Reset edges", enabled = tools && !circling) { vm.edgePoints = NoEdges }
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 14.dp, bottom = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryPill(onClick = { vm.cancelCrop() }, container = Editor.Raised, border = Editor.Outline, modifier = Modifier.weight(1f)) {
                PillText("Cancel")
            }
            PrimaryPill(
                onClick = { if (vm.detecting) queued = true else vm.continueCrop() },
                enabled = !circling && photo != null && (vm.detecting || vm.corners.size == 4),
                container = Editor.Accent, content = Editor.Ground,
                modifier = Modifier.weight(1.6f),
            ) {
                PillText(if (vm.recropping) "Continue" else "Flatten page")
                Spacer(Modifier.width(8.dp))
                if (queued) CircularProgressIndicator(Modifier.size(18.dp), color = Editor.Ground, strokeWidth = 2.dp)
                else Icon(Icons.AutoMirrored.Filled.ArrowForward, null, Modifier.size(18.dp))
            }
        }
    }
}

/** A tool under the photo: an icon over its name, outlined; [on] while its mode is active. */
@Composable
private fun RowScope.CropTool(icon: ImageVector, label: String, enabled: Boolean, on: Boolean = false, onClick: () -> Unit) {
    val shape = RoundedCornerShape(10.dp)
    val fg = when {
        on -> Editor.Ground
        enabled -> Editor.Text
        else -> Editor.Text.copy(alpha = 0.4f)
    }
    Column(
        Modifier.weight(1f).fillMaxHeight().heightIn(min = 62.dp).clip(shape).background(if (on) Editor.Accent else Color.Transparent)
            .border(1.dp, if (on) Editor.Accent else Editor.Outline, shape)
            .toggleable(value = on, enabled = enabled, role = Role.Button, onValueChange = { onClick() }),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Icon(icon, null, Modifier.size(21.dp), tint = fg)
        Spacer(Modifier.height(4.dp))
        val style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold, color = fg, textAlign = TextAlign.Center)
        // two lines at most with large text, then smaller
        BasicText(
            label, style = style, maxLines = 2, modifier = Modifier.padding(horizontal = 4.dp),
            autoSize = TextAutoSize.StepBased(minFontSize = 9.sp, maxFontSize = style.fontSize),
        )
    }
}

/** A handle of the outline editor: a corner (0..3), a point between the corners, or a side's "+" (a new point). */
private sealed interface Grip {
    data class Corner(val i: Int) : Grip
    /** Point [j] of side [k] (ScanViewModel.edgePoints). */
    data class Point(val k: Int, val j: Int) : Grip
}

/** Side [k] from corner k to corner k + 1 through its points (photo px). */
private fun sidePath(corners: List<Offset>, edges: List<List<Offset>>, k: Int): List<Offset> =
    listOf(corners[k]) + edges[k] + corners[(k + 1) % 4]

/**
 * [p] moved along the side's normal [n] (unit, photo px) onto the strongest step in brightness within [reach] photo px
 * (the paper's edge against the background), on the display copy [bmp] of a [pw] px wide photo; [p] when there is none.
 */
private fun snapToEdge(bmp: android.graphics.Bitmap, pw: Int, p: Offset, n: Offset, reach: Float): Offset {
    val k = bmp.width.toFloat() / pw
    fun lum(q: Offset): Float {
        val x = (q.x * k).roundToInt().coerceIn(0, bmp.width - 1)
        val y = (q.y * k).roundToInt().coerceIn(0, bmp.height - 1)
        val c = bmp.getPixel(x, y)
        return 0.299f * ((c shr 16) and 0xFF) + 0.587f * ((c shr 8) and 0xFF) + 0.114f * (c and 0xFF)
    }
    val step = 1f / k   // one display px
    val m = (reach / step).toInt().coerceAtLeast(1)
    var best = 0f
    var at = 0
    for (i in -m..m) {
        val g = abs(lum(p + n * ((i + 1) * step)) - lum(p + n * ((i - 1) * step)))
        // the nearest of nearly equal steps
        if (g > best * 1.15f || (g >= best && abs(i) < abs(at))) { best = g; at = i }
    }
    return if (best >= 24f) p + n * (at * step) else p
}

/**
 * The page outline over the photo: the four corners (squares in the accent) and points between them (round, in
 * blue) that bend a side along a curled or torn page edge. Each side shows a "+" between its handles: pulling it
 * adds a point there, which settles on the paper's edge when let go. A tap on a point removes it. A loupe shows the
 * handle being dragged.
 */
@Composable
private fun CornerEditor(
    vm: ScanViewModel, pw: Int, ph: Int, display: android.graphics.Bitmap, circling: Boolean,
    onEdit: () -> Unit, onLoop: (List<Offset>) -> Unit,
) {
    val image = remember(display) { display.asImageBitmap() }
    var box by remember { mutableStateOf(IntSize.Zero) }
    var active by remember { mutableStateOf<Grip?>(null) }
    val edit by rememberUpdatedState(onEdit)
    val loopDone by rememberUpdatedState(onLoop)
    val loop = remember { mutableStateListOf<Offset>() }
    val primary = Editor.Accent
    val pointColor = OutlinePoint
    val density = LocalDensity.current
    val margin = with(density) { 28.dp.toPx() }
    val handleR = with(density) { 12.dp.toPx() }
    val touchR = with(density) { 40.dp.toPx() }
    val loupeR = with(density) { 64.dp.toPx() }
    // photo pixels -> screen: fit inside the box with a margin so corner handles stay reachable
    val scale = if (box.width == 0) 1f else min((box.width - 2 * margin) / pw, (box.height - 2 * margin) / ph)
    val ox = (box.width - pw * scale) / 2
    val oy = (box.height - ph * scale) / 2
    fun toScreen(p: Offset) = Offset(ox + p.x * scale, oy + p.y * scale)
    fun toPhoto(q: Offset) = Offset(((q.x - ox) / scale).coerceIn(0f, pw - 1f), ((q.y - oy) / scale).coerceIn(0f, ph - 1f))
    fun clampPhoto(p: Offset) = Offset(p.x.coerceIn(0f, pw - 1f), p.y.coerceIn(0f, ph - 1f))
    // each side's "+": the middles between its consecutive handles (side k, after handle s)
    fun pluses(): List<Triple<Int, Int, Offset>> = if (vm.corners.size != 4) emptyList() else (0 until 4).flatMap { k ->
        val v = sidePath(vm.corners, vm.edgePoints, k)
        (0 until v.size - 1).map { s -> Triple(k, s, (v[s] + v[s + 1]) / 2f) }
    }
    fun dist(p: Offset, pos: Offset) = toScreen(p).let { hypot(it.x - pos.x, it.y - pos.y) }
    // the handle under [pos], and whether it is a side's "+" (then Grip.Point is where its new point goes)
    fun gripAt(pos: Offset, withPluses: Boolean): Pair<Grip, Boolean>? {
        if (vm.corners.size != 4) return null
        var best: Pair<Grip, Boolean>? = null
        var bd = touchR
        vm.corners.forEachIndexed { i, c -> dist(c, pos).let { if (it < bd) { bd = it; best = Grip.Corner(i) to false } } }
        // a corner within reach wins over a point or a "+" near it, a point over a "+"
        if (best != null) return best
        vm.edgePoints.forEachIndexed { k, side -> side.forEachIndexed { j, c -> dist(c, pos).let { if (it < bd) { bd = it; best = Grip.Point(k, j) to false } } } }
        if (best != null || !withPluses) return best
        for ((k, s, m) in pluses()) dist(m, pos).let { if (it < bd) { bd = it; best = Grip.Point(k, s) to true } }
        return best
    }
    fun positionOf(g: Grip): Offset? = when (g) {
        is Grip.Corner -> vm.corners.getOrNull(g.i)
        is Grip.Point -> vm.edgePoints.getOrNull(g.k)?.getOrNull(g.j)
    }
    var dragStart by remember { mutableStateOf(Offset.Zero) }
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
                    val hit = gripAt(pos, withPluses = true)
                    val g = hit?.first
                    active = g
                    dragTotal = Offset.Zero
                    when {
                        g is Grip.Point && hit.second -> {
                            // a "+": a new point in the middle of that stretch of the side
                            val v = sidePath(vm.corners, vm.edgePoints, g.k)
                            val m = (v[g.j] + v[g.j + 1]) / 2f
                            vm.edgePoints = vm.edgePoints.mapIndexed { k, side -> if (k == g.k) side.toMutableList().also { it.add(g.j, m) } else side }
                            dragStart = m
                        }
                        g != null -> dragStart = positionOf(g) ?: Offset.Zero
                    }
                    if (g != null) edit()
                },
                onDragEnd = {
                    // a point settles on the paper's edge, along the side's normal
                    val g = active
                    if (g is Grip.Point) {
                        val v = sidePath(vm.corners, vm.edgePoints, g.k)
                        val e = v[g.j + 2] - v[g.j]
                        val len = e.getDistance()
                        if (len > 1f) {
                            val n = Offset(-e.y / len, e.x / len)
                            val p = vm.edgePoints[g.k][g.j]
                            val snapped = clampPhoto(snapToEdge(display, pw, p, n, 14f / scale))
                            vm.edgePoints = vm.edgePoints.mapIndexed { k, side -> if (k == g.k) side.toMutableList().also { it[g.j] = snapped } else side }
                        }
                    }
                    active = null
                },
                onDragCancel = { active = null },
                onDrag = { change, delta ->
                    val g = active ?: return@detectDragGestures
                    change.consume()
                    dragTotal += delta / scale
                    val p = clampPhoto(dragStart + dragTotal)
                    when (g) {
                        is Grip.Corner -> vm.corners = vm.corners.toMutableList().also { it[g.i] = p }
                        is Grip.Point -> vm.edgePoints = vm.edgePoints.mapIndexed { k, side -> if (k == g.k) side.toMutableList().also { it[g.j] = p } else side }
                    }
                },
            )
        }.pointerInput(pw, ph, scale) {
            detectTapGestures(onTap = { pos ->
                // a tap on a point takes it away (the side straightens there)
                val g = gripAt(pos, withPluses = false)?.first
                if (g is Grip.Point) {
                    vm.edgePoints = vm.edgePoints.mapIndexed { k, side -> if (k == g.k) side.filterIndexed { j, _ -> j != g.j } else side }
                    edit()
                }
            })
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
        if (vm.corners.size != 4) return@Canvas
        val ring = (0 until 4).flatMap { k -> sidePath(vm.corners, vm.edgePoints, k).dropLast(1) }
        val pts = ring.map { toScreen(it) }
        val outline = Path().apply { moveTo(pts[0].x, pts[0].y); pts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
        drawPath(outline, Editor.Accent.copy(alpha = 0.10f))
        // a clean dashed outline over a dark rim (visible on white paper)
        drawPath(outline, Editor.Ground.copy(alpha = 0.6f), style = Stroke(width = 4.dp.toPx(), join = StrokeJoin.Round))
        drawPath(outline, primary, style = Stroke(width = 2.dp.toPx(), join = StrokeJoin.Round, pathEffect = PathEffect.dashPathEffect(floatArrayOf(8.dp.toPx(), 5.dp.toPx()))))
        // the sides' "+": small rings with a cross, to pull out a new point
        val pr = 9.dp.toPx()
        for ((_, _, m) in pluses()) {
            val c = toScreen(m)
            drawCircle(Editor.Ground.copy(alpha = 0.75f), radius = pr, center = c)
            drawCircle(pointColor, radius = pr, center = c, style = Stroke(width = 1.5.dp.toPx()))
            val a = pr * 0.5f
            drawLine(pointColor, c - Offset(a, 0f), c + Offset(a, 0f), strokeWidth = 1.5.dp.toPx())
            drawLine(pointColor, c - Offset(0f, a), c + Offset(0f, a), strokeWidth = 1.5.dp.toPx())
        }
        vm.edgePoints.forEachIndexed { k, side ->
            side.forEachIndexed { j, p ->
                val on = active == Grip.Point(k, j)
                val r = if (on) handleR * 1.2f else handleR * 0.75f
                drawCircle(if (on) Color.White else pointColor, radius = r, center = toScreen(p))
                drawCircle(Editor.Ground, radius = r, center = toScreen(p), style = Stroke(width = 2.dp.toPx()))
            }
        }
        vm.corners.forEachIndexed { i, p ->
            // solid square tabs in the highlighter; the one being dragged turns white and grows
            val on = active == Grip.Corner(i)
            val r = if (on) handleR * 1.3f else handleR * 0.85f
            val tl = toScreen(p) - Offset(r, r)
            drawRect(if (on) Color.White else primary, topLeft = tl, size = Size(2 * r, 2 * r))
            drawRect(Editor.Ground, topLeft = tl, size = Size(2 * r, 2 * r), style = Stroke(width = 2.dp.toPx()))
        }
        val target = active?.let { positionOf(it) }
        if (target != null) {
            // magnifier loupe above the finger (below it near the top edge), 3x the display scale
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
                val loupePts = ring.map { lc + (Offset(it.x * image.width / pw, it.y * image.height / ph) - dp) * zoom }
                val lq = Path().apply { moveTo(loupePts[0].x, loupePts[0].y); loupePts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
                drawPath(lq, primary, style = Stroke(width = 1.5.dp.toPx()))
            }
            drawCircle(Color.White, radius = loupeR, center = lc, style = Stroke(width = 3.dp.toPx()))
            val a = 10.dp.toPx()
            val cross = if (active is Grip.Point) pointColor else primary
            drawLine(cross, lc - Offset(a, 0f), lc + Offset(a, 0f), strokeWidth = 2.dp.toPx())
            drawLine(cross, lc - Offset(0f, a), lc + Offset(0f, a), strokeWidth = 2.dp.toPx())
        }
    }
}

/** The outline points' colour (the corners are in the accent). */
private val OutlinePoint = Color(0xFF6CC4FF)

/** Dashed outline style shared by the area editors. */
internal fun dashed(widthPx: Float, onPx: Float) = Stroke(width = widthPx, pathEffect = PathEffect.dashPathEffect(floatArrayOf(onPx, onPx)))
