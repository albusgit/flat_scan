package com.albus.flatscan.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.albus.flatscan.Accel
import com.albus.flatscan.BuildConfigInfo
import com.albus.flatscan.FilterKind
import com.albus.flatscan.NetEnv
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import androidx.compose.ui.platform.LocalContext
import com.albus.flatscan.Settings
import kotlinx.coroutines.delay
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.foundation.layout.widthIn
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import kotlin.math.min

/** The app mark: a page with a folded corner; its first accent line starts wavy and ends straight. */
@Composable
internal fun LogoMark(modifier: Modifier = Modifier) {
    val ink = MaterialTheme.colorScheme.onSurface
    val accent = MaterialTheme.colorScheme.primary
    Canvas(modifier) {
        val u = size.minDimension / 26f
        val page = Path().apply {
            moveTo(5 * u, 3 * u); lineTo(16 * u, 3 * u); lineTo(21 * u, 8 * u); lineTo(21 * u, 23 * u); lineTo(5 * u, 23 * u); close()
        }
        drawPath(page, ink, style = Stroke(width = 2 * u, join = StrokeJoin.Round))
        val wave = Path().apply {
            moveTo(8.5f * u, 12 * u)
            cubicTo(9.7f * u, 10.4f * u, 10.5f * u, 13.6f * u, 11.7f * u, 12 * u)
            cubicTo(12.9f * u, 10.4f * u, 13.7f * u, 11.6f * u, 14.3f * u, 11.6f * u)
            lineTo(17 * u, 11.6f * u)
        }
        drawPath(wave, accent, style = Stroke(width = 2 * u, cap = StrokeCap.Round, join = StrokeJoin.Round))
        drawLine(accent, Offset(8.5f * u, 17 * u), Offset(15 * u, 17 * u), strokeWidth = 2 * u, cap = StrokeCap.Round)
    }
}

/** The home headline: [lead] then [marked], which a highlighter swipes across (once ever: drawn in place after). */
@Composable
internal fun HighlightedHeadline(lead: String, marked: String) {
    val light = isPaperLight()
    val marker = if (light) Palette.Highlighter else Graphite.Highlighter
    val reduced = rememberReducedMotion()
    val settings = Settings(LocalContext.current)
    val sweep = remember { Animatable(if (settings.headlineSwiped || reduced) 1f else 0f) }
    LaunchedEffect(Unit) {
        if (sweep.value < 1f) {
            delay(250)
            sweep.animateTo(1f, tween(650, easing = FastOutSlowInEasing))
        }
        settings.headlineSwiped = true
    }
    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }
    val ink = MaterialTheme.colorScheme.onBackground
    // on graphite the marked words turn dark as the yellow arrives under them
    val markedInk = if (light) ink else lerp(ink, Graphite.Ground, sweep.value)
    val text = buildAnnotatedString {
        append(lead)
        withStyle(SpanStyle(color = markedInk)) { append(marked) }
    }
    val pad = with(LocalDensity.current) { 4.dp.toPx() }
    Text(
        text, style = MaterialTheme.typography.headlineLarge, onTextLayout = { layout = it },
        modifier = Modifier.drawBehind {
            val l = layout ?: return@drawBehind
            val end = lead.length + marked.length
            var i = lead.length
            while (i < end) {
                val line = l.getLineForOffset(i)
                val lineEnd = min(end, l.getLineEnd(line, visibleEnd = true))
                if (lineEnd <= i) break
                val left = l.getBoundingBox(i).left - pad
                val right = l.getBoundingBox(lineEnd - 1).right + pad
                val top = l.getLineTop(line)
                val bottom = l.getLineBottom(line)
                drawRect(
                    marker, topLeft = Offset(left, top + (bottom - top) * 0.06f),
                    size = Size((right - left) * sweep.value, (bottom - top) * 0.94f),
                )
                i = l.getLineEnd(line)
            }
        },
    )
}

@Composable
fun ProcessingScreen(vm: ScanViewModel) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val preview = vm.openingPreview
    if (preview != null) {
        // a stored page: its last rendering while its session is restored (no steps to report)
        val img = remember(preview) { preview.asImageBitmap() }
        Column(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Image(img, "The page", contentScale = ContentScale.Fit, modifier = Modifier.weight(1f).fillMaxWidth())
            Spacer(Modifier.height(12.dp))
            Text("Opening the page", style = MaterialTheme.typography.bodyMedium.copy(fontFamily = RecursiveMono), color = muted)
        }
        return
    }
    val photo = vm.photoDisplay
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal = 32.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        // the page peels off the photo, springs out of its curl and is ironed flat, in step with the core
        PaperUnfold(
            photo, vm.corners, vm.photo?.width ?: 0, vm.progress,
            Modifier.widthIn(max = 340.dp).fillMaxWidth().aspectRatio(0.85f),
            book = vm.processingBookPage, onFinished = { vm.unfoldLanded = true },
        )
        Spacer(Modifier.height(20.dp))
        Text(if (photo == null) "Opening the page" else "Flattening the page", style = MaterialTheme.typography.headlineSmall)
        if (vm.batch) {
            Spacer(Modifier.height(6.dp))
            Text("Photo ${vm.importIndex} of ${vm.pendingTotal}", style = MaterialTheme.typography.bodyMedium, color = muted)
        }
        Spacer(Modifier.height(14.dp))
        Row(Modifier.widthIn(max = 340.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(vm.stage, style = MaterialTheme.typography.bodyMedium.copy(fontFamily = RecursiveMono), color = muted, maxLines = 2, modifier = Modifier.weight(1f))
            // the GPU's first set-up is not the page's progress: no percentage beside it (its text says how far it is)
            if (!vm.stage.startsWith(ScanViewModel.GPU_SETUP)) {
                Text("${(vm.progress * 100).toInt()}%", style = MaterialTheme.typography.labelLarge.copy(fontFamily = RecursiveMono), color = muted)
            }
        }
    }
}

@Composable
internal fun Section(t: String) {
    val marker = if (isPaperLight()) Palette.Highlighter else Graphite.Highlighter.copy(alpha = 0.35f)
    Text(
        t.uppercase(), style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, fontSize = 13.sp, letterSpacing = 0.06.em),
        color = MaterialTheme.colorScheme.onSurface,
        modifier = Modifier.padding(top = 22.dp, bottom = 8.dp).drawBehind {
            // a highlighter stroke under the lower half of the label
            drawRect(marker, topLeft = Offset(-3.dp.toPx(), size.height * 0.45f), size = Size(size.width + 6.dp.toPx(), size.height * 0.5f))
        },
    )
}

/** A settings group: its label above a sheet of paper. */
@Composable
internal fun SettingsSheet(title: String, content: @Composable ColumnScope.() -> Unit) {
    Section(title)
    PaperSurface(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(horizontal = 14.dp, vertical = 4.dp), content = content)
    }
}

@Composable
fun MessageHost(vm: ScanViewModel) {
    val m = vm.message ?: return
    val action = vm.messageAction?.takeIf { it.message == m }
    LaunchedEffect(m, action) {
        delay(if (action != null) 5000 else 3500)
        vm.message = null
    }
    Box(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp), contentAlignment = Alignment.BottomCenter) {
        Snackbar(
            shape = MaterialTheme.shapes.small,
            action = action?.let { a ->
                {
                    TextButton(onClick = { vm.message = null; a.run() }) {
                        Text(a.label, color = MaterialTheme.colorScheme.inversePrimary, fontWeight = FontWeight.Bold)
                    }
                }
            },
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (m.startsWith("Saved")) {
                    PaperPlaneLanding(m)
                    Spacer(Modifier.width(12.dp))
                }
                Text(m)
            }
        }
    }
}

/** A paper airplane that swoops in from the left and lands beside the message. */
@Composable
private fun PaperPlaneLanding(key: Any) {
    val reduced = rememberReducedMotion()
    val t = remember(key) { Animatable(if (reduced) 1f else 0f) }
    LaunchedEffect(key) { t.animateTo(1f, tween(700, easing = FastOutSlowInEasing)) }
    val travel = with(LocalDensity.current) { 56.dp.toPx() }
    Icon(
        PaperIcons.PaperPlane, null,
        Modifier.size(22.dp).graphicsLayer {
            val k = 1f - t.value
            translationX = -travel * k
            translationY = -travel * 0.5f * kotlin.math.sin(k * Math.PI.toFloat())
            rotationZ = -20f * k
            alpha = min(1f, t.value * 2f)
        },
        tint = MaterialTheme.colorScheme.inversePrimary,
    )
}
