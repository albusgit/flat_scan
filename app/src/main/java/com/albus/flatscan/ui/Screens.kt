package com.albus.flatscan.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import com.albus.flatscan.BuildConfigInfo
import com.albus.flatscan.FilterKind
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import com.albus.flatscan.Settings
import kotlinx.coroutines.delay

@Composable
fun HomeScreen(vm: ScanViewModel) {
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    var confirmNew by remember { mutableStateOf(false) }
    val n = vm.pages.size
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    // scrolls on short screens; otherwise "Start a new document" sits at the bottom
    BoxWithConstraints(Modifier.fillMaxSize().safeDrawingPadding()) {
        Column(
            Modifier.verticalScroll(rememberScrollState()).heightIn(min = maxHeight)
                .padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 20.dp),
        ) {
            Row(Modifier.fillMaxWidth().height(48.dp), verticalAlignment = Alignment.CenterVertically) {
                LogoMark(Modifier.size(26.dp))
                Spacer(Modifier.width(10.dp))
                Text("FlatScan", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
                IconButton(onClick = { vm.open(Screen.Settings) }) { Icon(Icons.Outlined.Settings, "Settings") }
            }
            Spacer(Modifier.height(28.dp))
            Text("Flat, clean pages\nfrom any photo.", style = MaterialTheme.typography.headlineLarge)
            Spacer(Modifier.height(10.dp))
            Text(
                "Bent, curled or shadowed: FlatScan finds the page, flattens it and whitens the paper.",
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 15.sp, lineHeight = 22.sp), color = muted,
            )
            Spacer(Modifier.height(28.dp))
            // the main action: a large accent card
            Surface(
                onClick = { vm.startCamera() },
                shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.primary, contentColor = MaterialTheme.colorScheme.onPrimary,
                modifier = Modifier.fillMaxWidth().height(168.dp),
            ) {
                Column(Modifier.fillMaxSize().padding(22.dp), verticalArrangement = Arrangement.SpaceBetween) {
                    Box(
                        Modifier.size(52.dp).background(LocalContentColor.current.copy(alpha = 0.16f), RoundedCornerShape(16.dp)),
                        contentAlignment = Alignment.Center,
                    ) { Icon(Icons.Outlined.PhotoCamera, null, Modifier.size(28.dp)) }
                    Column {
                        Text("Scan with camera", style = MaterialTheme.typography.headlineSmall)
                        Text("Hold the phone above the page", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.alpha(0.85f))
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            Surface(
                onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                shape = RoundedCornerShape(18.dp), color = MaterialTheme.colorScheme.surfaceContainerLowest,
                border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.outline),
                modifier = Modifier.fillMaxWidth().height(64.dp),
            ) {
                Row(Modifier.padding(horizontal = 18.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Outlined.PhotoLibrary, null, Modifier.size(22.dp))
                    Spacer(Modifier.width(14.dp))
                    Text("Import photos", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold), modifier = Modifier.weight(1f))
                    Text("Several at once", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp, fontWeight = FontWeight.Medium), color = muted)
                }
            }
            if (n > 0) {
                Spacer(Modifier.height(32.dp))
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "CURRENT DOCUMENT", style = MaterialTheme.typography.labelMedium.copy(fontSize = 13.sp, letterSpacing = 0.08.em),
                        color = muted, modifier = Modifier.weight(1f),
                    )
                    TextButton(onClick = { vm.open(Screen.Pages) }) { Text("Open") }
                }
                Spacer(Modifier.height(4.dp))
                DocumentCard(vm, onClick = { vm.open(Screen.Pages) })
            }
            Spacer(Modifier.weight(1f).heightIn(min = 24.dp))
            if (n > 0) {
                TextButton(
                    onClick = { confirmNew = true },
                    colors = ButtonDefaults.textButtonColors(contentColor = muted),
                    modifier = Modifier.align(Alignment.CenterHorizontally).height(44.dp),
                ) { Text("Start a new document", style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold)) }
            }
        }
    }
    if (confirmNew) {
        AlertDialog(
            onDismissRequest = { confirmNew = false },
            title = { Text("Start a new document?") },
            text = { Text(if (n == 1) "The current page will be removed from the app. Save it first if you need it." else "The current $n pages will be removed from the app. Save them first if you need them.") },
            confirmButton = { TextButton(onClick = { vm.newDocument(); confirmNew = false }) { Text("New document") } },
            dismissButton = { TextButton(onClick = { confirmNew = false }) { Text("Cancel") } },
        )
    }
}

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

/** The current document on the home screen: the first page over a second sheet, count and filter. */
@Composable
private fun DocumentCard(vm: ScanViewModel, onClick: () -> Unit) {
    val n = vm.pages.size
    val first = vm.pages.firstOrNull()
    val version = first?.let { vm.pageVersions[it] } ?: 0
    var thumb by remember(first) { mutableStateOf<android.graphics.Bitmap?>(null) }
    LaunchedEffect(first, version) { thumb = first?.let { vm.thumbnail(it) } }
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surfaceContainerLowest,
        border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.outlineVariant),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(92.dp, 96.dp)) {
                val sheet = RoundedCornerShape(4.dp)
                val paper = Color.White
                if (n > 1) {
                    Box(
                        Modifier.offset(22.dp, 2.dp).size(62.dp, 84.dp).rotate(6f)
                            .background(paper, sheet).border(1.dp, Color(0xFFE2E0DA), sheet),
                    )
                }
                Box(
                    Modifier.offset(4.dp, 8.dp).size(62.dp, 84.dp)
                        .shadow(4.dp, sheet).background(paper, sheet).border(1.dp, Color(0xFFD8D6D0), sheet).clip(sheet),
                    contentAlignment = Alignment.Center,
                ) {
                    val t = thumb
                    if (t != null) {
                        val img = remember(t) { t.asImageBitmap() }
                        Image(img, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                    } else {
                        Icon(Icons.Outlined.Description, null, tint = Color(0xFFB9B7B1))
                    }
                }
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(if (n == 1) "1 page" else "$n pages", style = MaterialTheme.typography.titleMedium)
                Text(
                    if (vm.hasSession) vm.params.filter.label else "Tap to open",
                    style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp), color = muted,
                )
            }
            Icon(Icons.Outlined.ChevronRight, null, tint = muted)
        }
    }
}

@Composable
fun ProcessingScreen(vm: ScanViewModel) {
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Processing", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(24.dp))
        LinearProgressIndicator(
            progress = { vm.progress },
            trackColor = MaterialTheme.colorScheme.outlineVariant, gapSize = 0.dp, drawStopIndicator = {},
            modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
        )
        Spacer(Modifier.height(16.dp))
        Text(vm.stage, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(vm: ScanViewModel) {
    val s = vm.settings
    var filter by remember { mutableStateOf(s.defaultFilter) }
    var solid by remember { mutableStateOf(s.solidColors) }
    var autoRotate by remember { mutableStateOf(s.autoRotate) }
    var width by remember { mutableIntStateOf(s.outputWidth) }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text("Settings", style = MaterialTheme.typography.titleLarge) },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
        )
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Section("Default filter")
            for (f in FilterKind.entries) Choice(f.label, filter == f) { filter = f; s.defaultFilter = f }
            Text(
                "The filter and strength you choose on a page become the default for the next one.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            SwitchRow("Solid colors", "Flatten shaded table cells to one colour", solid) { solid = it; s.solidColors = it }
            SwitchRow("Auto-rotate pages", "Turn sideways and upside-down pages upright", autoRotate) { autoRotate = it; s.autoRotate = it }
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("Output resolution")
            for ((w, label) in Settings.OUTPUT_WIDTHS) Choice(label, width == w) { width = w; s.outputWidth = w }
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("About")
            Text("Version ${BuildConfigInfo.VERSION}", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(4.dp))
            Text(vm.gpuInfo, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = MaterialTheme.colorScheme.outlineVariant)
            Section("Self-test and benchmark")
            Text(
                "Processes a bundled test page 6 times and checks the result against the reference implementation. Takes about a minute.",
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
