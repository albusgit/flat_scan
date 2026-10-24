package com.albus.flatscan.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
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
import androidx.compose.material3.Surface
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
import androidx.compose.ui.draw.rotate
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

/** The app mark: a page with a folded corner and two accent lines. */
@Composable
private fun LogoMark(modifier: Modifier = Modifier) {
    val ink = MaterialTheme.colorScheme.onSurface
    val accent = MaterialTheme.colorScheme.primary
    Canvas(modifier) {
        val u = size.minDimension / 26f
        val page = Path().apply {
            moveTo(5 * u, 3 * u); lineTo(16 * u, 3 * u); lineTo(21 * u, 8 * u); lineTo(21 * u, 23 * u); lineTo(5 * u, 23 * u); close()
        }
        drawPath(page, ink, style = Stroke(width = 2 * u, join = StrokeJoin.Round))
        drawLine(accent, Offset(9 * u, 13 * u), Offset(17 * u, 13 * u), strokeWidth = 2 * u, cap = StrokeCap.Round)
        drawLine(accent, Offset(9 * u, 17 * u), Offset(15 * u, 17 * u), strokeWidth = 2 * u, cap = StrokeCap.Round)
    }
}

@Composable
fun ProcessingScreen(vm: ScanViewModel) {
    val photo = vm.photoDisplay
    // the bar moves smoothly between the core's progress steps
    val shown by animateFloatAsState(vm.progress, animationSpec = tween(400), label = "progress")
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (photo != null) {
            // the photo being flattened, a scan line down to the progress: it moves only when the core
            // reports a step (no endless animation competing with processing on slow phones)
            val img = remember(photo) { photo.asImageBitmap() }
            val y = shown
            val accent = MaterialTheme.colorScheme.primary
            Box(
                Modifier.heightIn(max = 300.dp).aspectRatio(photo.width.toFloat() / photo.height.coerceAtLeast(1))
                    .clip(RoundedCornerShape(16.dp)),
            ) {
                Image(img, null, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize().alpha(0.55f))
                Canvas(Modifier.fillMaxSize()) {
                    val yy = size.height * y
                    drawRect(
                        androidx.compose.ui.graphics.Brush.verticalGradient(
                            listOf(Color.Transparent, accent.copy(alpha = 0.35f)), startY = yy - 60.dp.toPx(), endY = yy,
                        ),
                        topLeft = Offset(0f, yy - 60.dp.toPx()), size = androidx.compose.ui.geometry.Size(size.width, 60.dp.toPx()),
                    )
                    drawLine(accent, Offset(0f, yy), Offset(size.width, yy), strokeWidth = 3.dp.toPx())
                }
            }
            Spacer(Modifier.height(28.dp))
        }
        Text(if (photo == null) "Opening the page" else "Flattening the page", style = MaterialTheme.typography.headlineSmall)
        if (vm.batch) {
            Spacer(Modifier.height(6.dp))
            Text("Photo ${vm.importIndex} of ${vm.pendingTotal}", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(20.dp))
        LinearProgressIndicator(
            progress = { shown },
            trackColor = MaterialTheme.colorScheme.outlineVariant, gapSize = 0.dp, drawStopIndicator = {},
            modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
        )
        Spacer(Modifier.height(14.dp))
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(vm.stage, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2, modifier = Modifier.weight(1f))
            Text("${(vm.progress * 100).toInt()}%", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
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
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            SwitchRow("Solid colors", "Flatten shaded table cells to one colour", solid) { solid = it; s.solidColors = it }
            SwitchRow("Recognise text after scanning", "Hebrew and English, on the phone: search finds documents by their text", ocrAfter) { ocrAfter = it; s.ocrAfterScan = it }
            SwitchRow("Auto-rotate pages", "Turn sideways pages upright (use Turn 180° for an upside-down page)", autoRotate) { autoRotate = it; s.autoRotate = it }
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("Finding the page")
            SwitchRow("Page finder (neural net)", "Off: the older colour-threshold detector", pageFinder) { pageFinder = it; s.pageFinderNet = it }
            SwitchRow("Neural flattening", "UVDoc and SyntheticDoc high-res; off: flatten from the page outline only", uvDoc) { uvDoc = it; s.uvDoc = it }
            if (uvDoc) for ((k, label) in listOf(0 to "Both nets: keep the straighter page", 1 to "UVDoc only", 2 to "SyntheticDoc high-res only")) Choice(label, flatNets == k) { flatNets = k; s.flatNets = k }
            SwitchRow("Remove shadows", "Shadow net: hand, object and leaf shadows; colours kept", neuralShadows) { neuralShadows = it; s.neuralShadows = it }
            SwitchRow("Table rules (neural net)", "Rule net: straighten printed rules, not text rows or curves; off: rules from darkness alone", ruleNet) { ruleNet = it; s.ruleNet = it }
            SwitchRow("Check every crop", "Off: only doubtful crops stop at the crop screen", reviewCrops) { reviewCrops = it; s.reviewCrops = it }
            SwitchRow("Book spreads: right page first", "For Hebrew and Arabic books; off: left page first", bookRtl) { bookRtl = it; s.bookRightToLeft = it }
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("Output resolution")
            for ((w, label) in Settings.OUTPUT_WIDTHS) Choice(label, width == w) { width = w; s.outputWidth = w }
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
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
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("Sync")
            val sc = vm.syncConfig
            Text(
                if (sc.configured) "Uploading to ${sc.kind.label}" + (if (sc.autoUpload) ", automatically" else ", by hand") else "Off: scans stay on this phone",
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(8.dp))
            OutlinedButton(onClick = { vm.open(Screen.Sync) }) { Text("Sync to your server") }
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            StorageSection(vm)
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("About")
            Text("Version ${BuildConfigInfo.VERSION}", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(4.dp))
            Text(vm.gpuInfo, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
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
                Spacer(Modifier.height(8.dp))
                Text(vm.selfTestReport, style = MaterialTheme.typography.bodySmall.copy(fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace))
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun SwitchRow(title: String, caption: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().toggleable(checked, role = Role.Switch, onValueChange = onChange).padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(caption, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = null)
    }
}

@Composable
private fun Section(t: String) {
    Text(
        t.uppercase(), style = MaterialTheme.typography.labelMedium.copy(fontSize = 13.sp, letterSpacing = 0.08.em),
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
        Snackbar { Text(m) }
    }
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
