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

/** Shown once per launch: the highlighter swipes across the headline's last words. */
private var highlighterSwiped = false

/** The home headline: [lead] then [marked], which a highlighter swipes across (once per launch). */
@Composable
internal fun HighlightedHeadline(lead: String, marked: String) {
    val light = isPaperLight()
    val marker = if (light) Palette.Highlighter else Graphite.Highlighter
    val reduced = rememberReducedMotion()
    val sweep = remember { Animatable(if (highlighterSwiped || reduced) 1f else 0f) }
    LaunchedEffect(Unit) {
        if (sweep.value < 1f) {
            delay(250)
            sweep.animateTo(1f, tween(650, easing = FastOutSlowInEasing))
        }
        highlighterSwiped = true
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
        // the photo arrives crumpled, unfolds with the core's steps and is ironed flat
        PaperUnfold(
            photo, vm.corners, vm.photo?.width ?: 0, vm.progress,
            Modifier.widthIn(max = 340.dp).fillMaxWidth().aspectRatio(0.85f),
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
            Text("${(vm.progress * 100).toInt()}%", style = MaterialTheme.typography.labelLarge.copy(fontFamily = RecursiveMono), color = muted)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: ScanViewModel) {
    val s = vm.settings
    var filter by remember { mutableStateOf(s.defaultFilter) }
    var solid by remember { mutableStateOf(s.solidColors) }
    var autoRotate by remember { mutableStateOf(s.autoRotate) }
    var ocrAfter by remember { mutableStateOf(s.ocrAfterScan) }
    var pageFinder by remember { mutableStateOf(s.pageFinderNet) }
    var uvDoc by remember { mutableStateOf(s.uvDoc) }
    var flatNets by remember { mutableIntStateOf(s.flatNets) }
    var neuralShadows by remember { mutableStateOf(s.neuralShadows) }
    var ruleNet by remember { mutableStateOf(s.ruleNet) }
    var reviewCrops by remember { mutableStateOf(s.reviewCrops) }
    var bookRtl by remember { mutableStateOf(s.bookRightToLeft) }
    var width by remember { mutableIntStateOf(s.outputWidth) }
    var accel by remember { mutableStateOf(s.acceleration) }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text("Settings", style = MaterialTheme.typography.titleLarge) },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
        )
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Section("Default filter")
            for (f in FilterKind.forMode(com.albus.flatscan.CaptureMode.Document)) Choice(f.label, filter == f) { filter = f; s.defaultFilter = f }
            Text(
                "The filter and strength you choose on a page become the default for the next one.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            RuledDivider(Modifier.padding(vertical = 8.dp))
            SwitchRow("Solid colors", "Flatten shaded table cells to one colour", solid) { solid = it; s.solidColors = it }
            SwitchRow("Recognise text after scanning", "Hebrew and English, on the phone: search finds documents by their text", ocrAfter) { ocrAfter = it; s.ocrAfterScan = it }
            SwitchRow("Auto-rotate pages", "Turn sideways pages upright (use Turn 180° for an upside-down page)", autoRotate) { autoRotate = it; s.autoRotate = it }
            RuledDivider(Modifier.padding(vertical = 8.dp))
            Section("Finding the page")
            SwitchRow("Page finder (neural net)", "Off: the older colour-threshold detector", pageFinder) { pageFinder = it; s.pageFinderNet = it }
            SwitchRow("Neural flattening", "UVDoc and SyntheticDoc high-res; off: flatten from the page outline only", uvDoc, icon = PaperIcons.Iron) { uvDoc = it; s.uvDoc = it }
            if (uvDoc) for ((k, label) in listOf(0 to "Both nets: keep the straighter page", 1 to "UVDoc only", 2 to "SyntheticDoc high-res only")) Choice(label, flatNets == k) { flatNets = k; s.flatNets = k }
            SwitchRow("Remove shadows", "Shadow net: hand, object and leaf shadows; colours kept", neuralShadows) { neuralShadows = it; s.neuralShadows = it }
            SwitchRow("Table rules (neural net)", "Rule net: printed rules and text lines (any script); off: rules from darkness, rows from ink", ruleNet) { ruleNet = it; s.ruleNet = it }
            SwitchRow("Check every crop", "Off: only doubtful crops stop at the crop screen", reviewCrops) { reviewCrops = it; s.reviewCrops = it }
            SwitchRow("Book spreads: right page first", "For Hebrew and Arabic books; off: left page first", bookRtl) { bookRtl = it; s.bookRightToLeft = it }
            RuledDivider(Modifier.padding(vertical = 8.dp))
            Section("Output resolution")
            for ((w, label) in Settings.OUTPUT_WIDTHS) Choice(label, width == w) { width = w; s.outputWidth = w }
            RuledDivider(Modifier.padding(vertical = 8.dp))
            Section("Acceleration")
            for ((k, label) in Accel.CHOICES) Choice(label, accel == k) { accel = k; s.acceleration = k; if (k == Accel.GPU) vm.enableGpu() }
            Text(
                "Where the neural nets run. The CPU gives the reference results; Auto uses the GPU only for the nets " +
                    "whose results match the CPU closely, and any net that fails on the GPU runs on the CPU. " +
                    "${NetEnv.runtime}; GPU (full precision): ${if (NetEnv.gpuAvailable) "available" else "unavailable on this phone (${NetEnv.gpuWhy})"}.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (accel == Accel.AUTO) {
                val plan = vm.currentPlan()
                Text(
                    if (plan.note == "not benchmarked yet") "Auto has not chosen yet: the first processed page runs a one-time benchmark."
                    else "Auto (${plan.note}): ${plan.summary()}.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = { vm.runAccelBenchmark() }, enabled = !vm.selfTestRunning) { Text("Run the acceleration benchmark") }
            if (vm.batchStatus.isNotEmpty()) {
                Section("Batch mode (debug)")
                if (vm.batchRunning) LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(vm.batchStatus, style = MaterialTheme.typography.bodySmall)
            }
            RuledDivider(Modifier.padding(vertical = 8.dp))
            Section("Sync")
            val sc = vm.syncConfig
            Text(
                if (sc.configured) "Uploading to ${sc.kind.label}" + (if (sc.autoUpload) ", automatically" else ", by hand") else "Off: scans stay on this phone",
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = { vm.open(Screen.Sync) }) { Text("Sync to your server") }
            RuledDivider(Modifier.padding(vertical = 8.dp))
            StorageSection(vm)
            RuledDivider(Modifier.padding(vertical = 8.dp))
            Section("About")
            Text("Version ${BuildConfigInfo.VERSION}", style = MaterialTheme.typography.bodyMedium)
            Row(
                Modifier.fillMaxWidth().clickable(role = Role.Button) { vm.open(Screen.Licenses) }.padding(vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Open-source licences", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                Icon(Icons.Outlined.ChevronRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.height(4.dp))
            Text(vm.gpuInfo, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            RuledDivider(Modifier.padding(vertical = 8.dp))
            Section("Self-test and benchmark")
            Text(
                "Processes a bundled test page 6 times and checks the result against the reference implementation, then times the neural nets on it 6 times. Takes a few minutes.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            val ctx = androidx.compose.ui.platform.LocalContext.current
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = { vm.runSelfTest() }, enabled = !vm.selfTestRunning) { Text("Run self-test") }
                if (vm.selfTestReport.isNotEmpty()) {
                    Spacer(Modifier.size(8.dp))
                    androidx.compose.material3.TextButton(onClick = {
                        val send = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                            type = "text/plain"
                            putExtra(android.content.Intent.EXTRA_TEXT, vm.selfTestReport)
                        }
                        ctx.startActivity(android.content.Intent.createChooser(send, null))
                    }) { Text("Share report") }
                }
            }
            if (vm.selfTestRunning) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(vm.selfTestStage, style = MaterialTheme.typography.bodySmall)
            }
            if (vm.selfTestReport.isNotEmpty()) {
                Spacer(Modifier.height(12.dp))
                // printed on a till receipt with a torn edge
                val receipt = ReceiptShape()
                Text(
                    vm.selfTestReport, style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono),
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.fillMaxWidth().shadow(2.dp, receipt).background(MaterialTheme.colorScheme.surfaceContainerLowest, receipt)
                        .padding(start = 14.dp, end = 14.dp, top = 14.dp, bottom = 24.dp),
                )
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun SwitchRow(title: String, caption: String, checked: Boolean, icon: ImageVector? = null, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().toggleable(checked, role = Role.Switch, onValueChange = onChange).padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, null, Modifier.size(24.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.width(14.dp))
        }
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(caption, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = null)
    }
}

@Composable
internal fun Section(t: String) {
    Text(
        t.uppercase(), style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, fontSize = 13.sp, letterSpacing = 0.06.em),
        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 16.dp, bottom = 6.dp),
    )
}

@Composable
private fun Choice(label: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().selectable(selected, onClick = onClick, role = Role.RadioButton).padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = null)
        Spacer(Modifier.size(12.dp))
        Text(label, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
fun MessageHost(vm: ScanViewModel) {
    val m = vm.message ?: return
    LaunchedEffect(m) {
        delay(3500)
        vm.message = null
    }
    Box(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp), contentAlignment = Alignment.BottomCenter) {
        Snackbar(shape = MaterialTheme.shapes.small) {
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

/** Settings > Storage: what the scans take, and removing the original photos of old documents. */
@Composable
private fun StorageSection(vm: ScanViewModel) {
    LaunchedEffect(Unit) { vm.measureStorage() }
    var days by remember { mutableStateOf<Int?>(null) }
    Section("Storage")
    val u = vm.storage
    Text(
        if (u == null) "Measuring..." else "${com.albus.flatscan.formatBytes(u.total)} for ${u.documents} documents (${u.pages} pages); " +
            "original photos: ${com.albus.flatscan.formatBytes(u.originals)}",
        style = MaterialTheme.typography.bodyMedium,
    )
    Text(
        "Every page keeps its original photo, so a crop or the unwarp can be changed later without losing quality. " +
            "Removing the originals of old documents frees space; their pages can still be filtered, rotated and exported, but not re-cropped.",
        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    Spacer(Modifier.height(8.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for (d in listOf(30, 90, 365)) OutlinedButton(onClick = { days = d }) { Text(if (d == 365) "> 1 year" else "> $d days") }
    }
    days?.let { d ->
        AlertDialog(
            onDismissRequest = { days = null },
            title = { Text("Remove original photos?") },
            text = { Text("The original photos of documents not changed in the last ${if (d == 365) "year" else "$d days"} are deleted. Their pages stay, but can no longer be re-cropped.") },
            confirmButton = { TextButton(onClick = { vm.dropOriginals(d); days = null }) { Text("Remove") } },
            dismissButton = { TextButton(onClick = { days = null }) { Text("Cancel") } },
        )
    }
}
