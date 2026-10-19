package com.albus.flatscan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.RotateRight
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.albus.flatscan.ScanViewModel
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.roundToInt

/** Corner adjustment: large draggable handles, a magnifier loupe while dragging, rotate. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CropScreen(vm: ScanViewModel) {
    val photo = vm.photo
    val display = vm.photoDisplay
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text("Adjust the corners") },
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            actions = {
                IconButton(onClick = { vm.rotatePhoto() }, enabled = !vm.detecting && photo != null) {
                    Icon(Icons.AutoMirrored.Outlined.RotateRight, "Rotate 90 degrees")
                }
            },
        )
        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            if (photo != null && display != null) CornerEditor(vm, photo.width, photo.height, display)
            if (vm.detecting) CircularProgressIndicator()
        }
        Button(
            onClick = { vm.process() },
            enabled = !vm.detecting && photo != null && vm.corners.size == 4,
            modifier = Modifier.fillMaxWidth().padding(24.dp).height(56.dp),
        ) { Text("Continue", style = MaterialTheme.typography.titleMedium) }
    }
}

@Composable
private fun CornerEditor(vm: ScanViewModel, pw: Int, ph: Int, display: android.graphics.Bitmap) {
    val image = remember(display) { display.asImageBitmap() }
    var box by remember { mutableStateOf(IntSize.Zero) }
    var active by remember { mutableIntStateOf(-1) }
    val corners by rememberUpdatedState(vm.corners)
    val primary = MaterialTheme.colorScheme.primary
    val margin = with(androidx.compose.ui.platform.LocalDensity.current) { 28.dp.toPx() }
    val handleR = with(androidx.compose.ui.platform.LocalDensity.current) { 14.dp.toPx() }
    val touchR = with(androidx.compose.ui.platform.LocalDensity.current) { 44.dp.toPx() }
    val loupeR = with(androidx.compose.ui.platform.LocalDensity.current) { 64.dp.toPx() }
    // photo pixels -> screen: fit inside the box with a margin so corner handles stay reachable
    val scale = if (box.width == 0) 1f else min((box.width - 2 * margin) / pw, (box.height - 2 * margin) / ph)
    val ox = (box.width - pw * scale) / 2
    val oy = (box.height - ph * scale) / 2
    fun toScreen(p: Offset) = Offset(ox + p.x * scale, oy + p.y * scale)
    var finger by remember { mutableStateOf(Offset.Zero) }
    Canvas(
        Modifier.fillMaxSize().onSizeChanged { box = it }.pointerInput(pw, ph, scale) {
            detectDragGestures(
                onDragStart = { pos ->
                    val d = corners.map { hypot(toScreen(it).x - pos.x, toScreen(it).y - pos.y) }
                    val i = d.indices.minByOrNull { d[it] } ?: -1
                    active = if (i >= 0 && d[i] < touchR) i else -1
                    finger = pos
                },
                onDragEnd = { active = -1 },
                onDragCancel = { active = -1 },
                onDrag = { change, delta ->
                    if (active >= 0) {
                        change.consume()
                        finger += delta
                        val c = corners[active] + delta / scale
                        val next = corners.toMutableList()
                        next[active] = Offset(c.x.coerceIn(0f, pw - 1f), c.y.coerceIn(0f, ph - 1f))
                        vm.corners = next
                    }
                },
            )
        },
    ) {
        drawImage(
            image, srcOffset = IntOffset.Zero, srcSize = IntSize(image.width, image.height),
            dstOffset = IntOffset(ox.roundToInt(), oy.roundToInt()),
            dstSize = IntSize((pw * scale).roundToInt(), (ph * scale).roundToInt()),
        )
        if (corners.size != 4) return@Canvas
        val pts = corners.map { toScreen(it) }
        val quad = Path().apply { moveTo(pts[0].x, pts[0].y); pts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
        drawPath(quad, primary.copy(alpha = 0.14f))
        drawPath(quad, primary, style = Stroke(width = 2.dp.toPx()))
        pts.forEachIndexed { i, p ->
            drawCircle(Color.White, radius = handleR, center = p)
            drawCircle(primary, radius = handleR, center = p, style = Stroke(width = if (i == active) 4.dp.toPx() else 2.5.dp.toPx()))
        }
        if (active >= 0) {
            // magnifier loupe above the finger (below it near the top edge), 3x the display scale
            val p = pts[active]
            val lc = if (p.y - 2.2f * loupeR > 0) Offset(p.x, p.y - 2.2f * loupeR) else Offset(p.x, p.y + 2.2f * loupeR)
            val zoom = 3f * scale * pw / image.width            // screen px per display-bitmap px in the loupe
            val half = loupeR / zoom
            val dp = Offset(corners[active].x * image.width / pw, corners[active].y * image.height / ph)
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
            drawLine(primary, lc - Offset(a, 0f), lc + Offset(a, 0f), strokeWidth = 1.5.dp.toPx())
            drawLine(primary, lc - Offset(0f, a), lc + Offset(0f, a), strokeWidth = 1.5.dp.toPx())
        }
    }
}
