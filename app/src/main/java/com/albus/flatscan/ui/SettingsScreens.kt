package com.albus.flatscan.ui

import com.albus.flatscan.batch.BatchProcessor
import android.content.Intent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material.icons.outlined.CloudUpload
import androidx.compose.material.icons.outlined.DarkMode
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.Inventory2
import androidx.compose.material.icons.outlined.LightMode
import androidx.compose.material.icons.outlined.PhoneAndroid
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.albus.flatscan.Accel
import com.albus.flatscan.BuildConfigInfo
import com.albus.flatscan.CaptureMode
import com.albus.flatscan.FilterKind
import com.albus.flatscan.NetEnv
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import com.albus.flatscan.Settings
import com.albus.flatscan.formatBytes
import com.albus.flatscan.output.OcrPack

// Settings, laid out from the design canvas "FlatScan Settings Redesign": the everyday choices on the first screen
// (theme, scanning, searchable text, page quality), how pages are processed and where the nets run behind Advanced,
// storage on a screen of its own. Every row says what it does in plain words; the nets' names stay out of the titles.

/** Settings: the everyday choices. */
@Composable
fun SettingsScreen(vm: ScanViewModel) {
    val s = vm.settings
    var filter by remember { mutableStateOf(s.defaultFilter) }
    var autoRotate by remember { mutableStateOf(s.autoRotate) }
    var reviewCrops by remember { mutableStateOf(s.reviewCrops) }
    var bookRtl by remember { mutableStateOf(s.bookRightToLeft) }
    var ocrAfter by remember { mutableStateOf(s.ocrAfterScan) }
    var width by remember { mutableIntStateOf(s.outputWidth) }
    var workers by remember { mutableIntStateOf(s.batchWorkers) }
    val ctx = LocalContext.current
    var sheet by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(Unit) { vm.measureStorage() }
    SettingsPage("Settings", onBack = { vm.back() }) {
        SettingsGroup("Appearance") {
            Column(Modifier.padding(start = 14.dp, end = 14.dp, top = 14.dp, bottom = 16.dp)) {
                Text("Theme", style = RowTitle())
                Text("System follows your phone's dark mode setting.", style = RowCaption(), color = muted())
                Spacer(Modifier.height(10.dp))
                ThemePicker(vm.themeMode) { vm.setTheme(it) }
            }
        }
        SettingsGroup("Scanning") {
            NavRow("Default look", "For new pages. The look you pick on a page becomes the new default.", value = filter.label) { sheet = "look" }
            RowDivider()
            SwitchRow("Turn pages upright", "Sideways pages are rotated by themselves. Turn 180° on the page fixes an upside-down one.", autoRotate) {
                autoRotate = it; s.autoRotate = it
            }
            RowDivider()
            SwitchRow("Always check the crop", "Off: the crop screen opens only when the edges look doubtful.", reviewCrops) {
                reviewCrops = it; s.reviewCrops = it
            }
            RowDivider()
            NavRow(
                "Photos processed at once", "Several photos picked together are processed in the background, with their progress in a notification.",
                value = if (workers == 0) "Auto (${BatchProcessor.workers(ctx, 0)})" else "$workers",
            ) { sheet = "workers" }
            RowDivider()
            SwitchRow("Book pages: right page first", "For Hebrew and Arabic books photographed as a spread.", bookRtl) {
                bookRtl = it; s.bookRightToLeft = it
            }
        }
        SettingsGroup("Searchable text") {
            SwitchRow(
                "Read the text in scans",
                "Hebrew and English, on the phone. Search then finds documents by their words, and PDFs get selectable text.",
                ocrAfter,
            ) {
                ocrAfter = it
                s.ocrAfterScan = it
                if (it && OcrPack.state !is OcrPack.State.Ready) vm.downloadOcr()
            }
            OcrPackBox(vm)
        }
        SettingsGroup("Saving") {
            val q = Settings.OUTPUT_QUALITY.firstOrNull { it.first == width } ?: Settings.OUTPUT_QUALITY[1]
            NavRow("Page quality", q.third, value = q.second) { sheet = "quality" }
        }
        SettingsGroup("More") {
            val sc = vm.syncConfig
            NavRow("Sync", "Upload scans to your own server", value = if (sc.configured) sc.kind.label else "Off", icon = Icons.Outlined.CloudUpload) {
                vm.open(Screen.Sync)
            }
            RowDivider()
            NavRow("Storage", "Free space by removing old original photos", value = vm.storage?.let { formatBytes(it.total) }, icon = Icons.Outlined.Inventory2) {
                vm.open(Screen.Storage)
            }
            RowDivider()
            NavRow("Advanced", "How pages are flattened and cleaned, and speed. The defaults suit most pages.", icon = PaperIcons.Iron) {
                vm.open(Screen.Advanced)
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 26.dp, start = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            LogoMark(Modifier.size(32.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("FlatScan", style = MaterialTheme.typography.titleSmall)
                Text("Version ${BuildConfigInfo.VERSION}", style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono), color = muted())
            }
            TextButton(onClick = { vm.open(Screen.Licenses) }) { Text("Open-source licences") }
        }
    }
    when (sheet) {
        "look" -> LookSheet(filter, onPick = { filter = it; s.defaultFilter = it }, onDismiss = { sheet = null })
        "quality" -> ChoiceSheet(
            "Page quality", "The width pages are saved at. Finding and flattening the page always work at 1600 px.",
            Settings.OUTPUT_QUALITY.map { Choice(it.first, it.second, it.third) }, width,
            onPick = { width = it; s.outputWidth = it }, onDismiss = { sheet = null },
        )
        "workers" -> ChoiceSheet(
            "Photos processed at once", "More at once finishes a batch sooner and takes more of the phone's memory.",
            listOf(
                Choice(0, "Auto", "Two on phones with 6 GB of memory or more, else one"),
                Choice(1, "One", "The least memory"),
                Choice(2, "Two", "About a third faster than one"),
                Choice(3, "Three", "A little faster again; for phones with 8 GB or more"),
            ),
            workers, onPick = { workers = it; s.batchWorkers = it }, onDismiss = { sheet = null },
        )
    }
}

/** Settings > Advanced: how pages are flattened and cleaned, where the nets run, and the technical details. */
@Composable
fun AdvancedSettingsScreen(vm: ScanViewModel) {
    // Restore the defaults re-reads every choice from the settings
    var generation by remember { mutableIntStateOf(0) }
    var confirmReset by remember { mutableStateOf(false) }
    SettingsPage("Advanced", onBack = { vm.back() }) {
        key(generation) { AdvancedChoices(vm) }
        OutlinedButton(
            onClick = { confirmReset = true },
            modifier = Modifier.fillMaxWidth().padding(top = 26.dp).height(48.dp),
            shape = MaterialTheme.shapes.small,
        ) { Text("Restore the defaults") }
        SettingsGroup("Experiments") {
            NavRow("Depth capture", "ARCore's depth over the camera: how flat it measures a page, and frames saved to share. Needs Google Play Services for AR.") {
                vm.open(Screen.Depth)
            }
        }
        TechnicalDetails(vm)
    }
    if (confirmReset) {
        PaperDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Restore the defaults?") },
            text = { Text("Flattening, cleaning and speed go back to how the app ships. Your scans, theme and other settings stay.") },
            confirmButton = {
                TextButton(onClick = {
                    vm.settings.restoreProcessingDefaults()
                    generation++
                    confirmReset = false
                    vm.message = "Defaults restored"
                }) { Text("Restore") }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Cancel") } },
        )
    }
}

/** The flattening, cleaning and speed choices (re-read from the settings when the defaults are restored). */
@Composable
private fun AdvancedChoices(vm: ScanViewModel) {
    val s = vm.settings
    var pageFinder by remember { mutableStateOf(s.pageFinderNet) }
    var flattening by remember { mutableIntStateOf(if (s.uvDoc) s.flatNets else FLAT_OUTLINE) }
    var shadows by remember { mutableStateOf(s.neuralShadows) }
    var ruleNet by remember { mutableStateOf(s.ruleNet) }
    var solid by remember { mutableStateOf(s.solidColors) }
    var accel by remember { mutableStateOf(s.acceleration) }
    var sheet by remember { mutableStateOf(false) }
    PaperSurface(Modifier.fillMaxWidth().padding(top = 18.dp), tape = true, lifted = true) {
        Row(Modifier.padding(start = 14.dp, end = 14.dp, top = 16.dp, bottom = 14.dp)) {
            Icon(Icons.Outlined.Info, null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.width(12.dp))
            Text(
                "These change how pages are flattened and cleaned. The defaults give the best result for most pages; " +
                    "turn one off only to compare, or to work around a problem.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
    SettingsGroup("Flattening and cleaning") {
        val f = FLATTENING.first { it.key == flattening }
        NavRow("Flattening", f.caption, value = f.short) { sheet = true }
        RowDivider()
        SwitchRow("Find the page with a neural net", "Off: the older finder, by colour and edges.", pageFinder) { pageFinder = it; s.pageFinderNet = it }
        RowDivider()
        SwitchRow("Remove shadows", "Hand, phone and leaf shadows on the page; colours stay.", shadows) { shadows = it; s.neuralShadows = it }
        RowDivider()
        SwitchRow(
            "Straighten along lines",
            "A neural net finds table rules and rows of text in any script, and the page is straightened along them.", ruleNet,
        ) { ruleNet = it; s.ruleNet = it }
        RowDivider()
        SwitchRow("Even out shaded cells", "Fills shaded table cells with one flat colour.", solid) { solid = it; s.solidColors = it }
    }
    SettingsGroup("Speed") {
        Column(Modifier.selectableGroup().padding(vertical = 6.dp)) {
            for ((k, title, caption) in listOf(
                Triple(Accel.AUTO, "Auto (recommended)", "The GPU where its results match the CPU, the CPU elsewhere."),
                Triple(Accel.CPU, "CPU", "Slower; the reference results."),
                Triple(
                    Accel.GPU, "GPU",
                    if (NetEnv.gpuAvailable) "Fastest; any net that fails on it runs on the CPU." else "Not available on this phone (${NetEnv.gpuWhy}).",
                ),
            )) {
                RadioRow(title, caption, accel == k) {
                    accel = k
                    s.acceleration = k
                    if (k == Accel.GPU) vm.enableGpu()
                }
            }
        }
        if (accel == Accel.AUTO) {
            val plan = vm.currentPlan()
            Column(
                Modifier.fillMaxWidth().padding(horizontal = 14.dp)
                    .background(MaterialTheme.colorScheme.surfaceContainerLow, MaterialTheme.shapes.extraSmall).padding(horizontal = 12.dp, vertical = 10.dp),
            ) {
                Text("AUTO CHOSE", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono), color = MaterialTheme.colorScheme.primary)
                Text(
                    if (plan.note == "not benchmarked yet") "Nothing yet: the first page you scan runs a one-time speed test." else plan.summary() + ".",
                    style = RowCaption(), color = muted(),
                )
            }
        }
        OutlinedButton(
            onClick = { vm.runAccelBenchmark() }, enabled = !vm.selfTestRunning, shape = MaterialTheme.shapes.small,
            modifier = Modifier.padding(start = 14.dp, top = 12.dp, bottom = 14.dp),
        ) { Text("Run the speed test") }
    }
    if (sheet) {
        ChoiceSheet(
            "Flattening", "How a bent or curled page is made flat.",
            FLATTENING.map { Choice(it.key, it.title, it.caption) }, flattening,
            onPick = {
                flattening = it
                s.uvDoc = it != FLAT_OUTLINE
                if (it != FLAT_OUTLINE) s.flatNets = it
            },
            onDismiss = { sheet = false },
        )
    }
}

/** Advanced, at the end: the runtime and GPU, the speed test's report, and in debug builds the self-test and batch mode. */
@Composable
private fun TechnicalDetails(vm: ScanViewModel) {
    val ctx = LocalContext.current
    Section("Technical details")
    Text(
        "${NetEnv.runtime}. ${vm.gpuInfo}", style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono), color = muted(),
        modifier = Modifier.padding(horizontal = 4.dp),
    )
    if (vm.batchStatus.isNotEmpty()) {
        Spacer(Modifier.height(10.dp))
        Text("Batch mode", style = RowTitle())
        if (vm.batchRunning) LinearProgressIndicator(Modifier.fillMaxWidth().padding(vertical = 4.dp))
        Text(vm.batchStatus, style = MaterialTheme.typography.bodySmall)
    }
    // the self-test is in debug builds only (its files: src/debug/assets)
    if (vm.hasSelfTest) {
        Spacer(Modifier.height(12.dp))
        Text(
            "Self-test: processes a bundled test page 6 times and checks the result against the reference implementation, " +
                "then times the neural nets on it 6 times. Takes a few minutes.",
            style = MaterialTheme.typography.bodySmall, color = muted(),
        )
    }
    Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        if (vm.hasSelfTest) OutlinedButton(onClick = { vm.runSelfTest() }, enabled = !vm.selfTestRunning, shape = MaterialTheme.shapes.small) { Text("Run the self-test") }
        if (vm.selfTestReport.isNotEmpty()) {
            TextButton(onClick = {
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, vm.selfTestReport)
                }
                ctx.startActivity(Intent.createChooser(send, null))
            }) { Text("Share the report") }
        }
    }
    if (vm.selfTestRunning) {
        LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
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
}

/** Settings > Storage: what the scans take, and removing the original photos of old documents. */
@Composable
fun StorageScreen(vm: ScanViewModel) {
    LaunchedEffect(Unit) { vm.measureStorage() }
    var days by remember { mutableStateOf<Int?>(null) }
    val u = vm.storage
    SettingsPage("Storage", onBack = { vm.back() }) {
        SettingsGroup("On this phone") {
            Column(Modifier.padding(14.dp)) {
                if (u == null) {
                    Text("Measuring...", style = RowCaption(), color = muted())
                } else {
                    Text(formatBytes(u.total), style = MaterialTheme.typography.headlineSmall)
                    Text("${u.documents} documents, ${u.pages} pages", style = RowCaption(), color = muted())
                    Spacer(Modifier.height(10.dp))
                    Text("Original photos: ${formatBytes(u.originals)}", style = RowTitle())
                }
            }
        }
        SettingsGroup("Free up space") {
            Column(Modifier.padding(14.dp)) {
                Text(
                    "Every page keeps its original photo, so its crop or flattening can be changed later without losing quality. " +
                        "Removing the originals of old documents frees space; their pages can still be filtered, rotated and exported, but not re-cropped.",
                    style = RowCaption(), color = muted(),
                )
                Spacer(Modifier.height(12.dp))
                Text("Remove the original photos of documents not changed for", style = RowTitle())
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (d in listOf(30, 90, 365)) {
                        OutlinedButton(onClick = { days = d }, shape = MaterialTheme.shapes.small) { Text(if (d == 365) "1 year" else "$d days") }
                    }
                }
            }
        }
    }
    days?.let { d ->
        PaperDialog(
            onDismissRequest = { days = null },
            title = { Text("Remove original photos?") },
            text = { Text("The original photos of documents not changed in the last ${if (d == 365) "year" else "$d days"} are deleted. Their pages stay, but can no longer be re-cropped.") },
            confirmButton = { TextButton(onClick = { vm.dropOriginals(d); days = null }) { Text("Remove") } },
            dismissButton = { TextButton(onClick = { days = null }) { Text("Cancel") } },
        )
    }
}

// ------------------------------------------------------------------ the pieces

/** Flattening choices: the nets' key (Settings.flatNets) or [FLAT_OUTLINE] (no nets: Settings.uvDoc off). */
private const val FLAT_OUTLINE = 3

private class Flattening(val key: Int, val title: String, val short: String, val caption: String)

private val FLATTENING = listOf(
    Flattening(0, "Best of both (recommended)", "Best of both", "Two neural nets flatten the page and the straighter result is kept."),
    Flattening(1, "UVDoc only", "UVDoc", "Only the UVDoc net flattens the page."),
    Flattening(2, "SyntheticDoc only", "SyntheticDoc", "Only the SyntheticDoc high-res net flattens the page."),
    Flattening(FLAT_OUTLINE, "Outline only", "Outline only", "No neural nets: the page is flattened from its outline. For pages that are already nearly flat."),
)

/** One option of a [ChoiceSheet]. */
private class Choice<T>(val key: T, val title: String, val caption: String)

@Composable
private fun muted() = MaterialTheme.colorScheme.onSurfaceVariant

@Composable
private fun RowTitle() = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.SemiBold)

@Composable
private fun RowCaption() = MaterialTheme.typography.bodySmall

/** A settings screen: the back arrow and title over the graph-paper desk, the groups scrolling under it. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SettingsPage(title: String, onBack: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text(title, style = MaterialTheme.typography.titleLarge) },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent),
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
        )
        Column(Modifier.verticalScroll(rememberScrollState()).padding(start = 16.dp, end = 16.dp, bottom = 32.dp), content = content)
    }
}

/** A group: its highlighted label over a sheet of paper; the rows pad themselves (whole-width touch targets). */
@Composable
private fun SettingsGroup(title: String, content: @Composable ColumnScope.() -> Unit) {
    Section(title)
    PaperSurface(Modifier.fillMaxWidth()) { Column(content = content) }
}

@Composable
private fun RowDivider() = RuledDivider(Modifier.padding(horizontal = 14.dp))

/** A row that opens a picker or a screen: title, what it does, the current value and a chevron. */
@Composable
private fun NavRow(title: String, caption: String, value: String? = null, icon: ImageVector? = null, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 64.dp).clickable(role = Role.Button, onClick = onClick)
            .padding(start = 14.dp, end = 10.dp, top = 10.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Box(
                Modifier.size(36.dp).background(MaterialTheme.colorScheme.primary.copy(alpha = 0.1f), RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center,
            ) { Icon(icon, null, Modifier.size(20.dp), tint = MaterialTheme.colorScheme.primary) }
            Spacer(Modifier.width(12.dp))
        }
        Column(Modifier.weight(1f)) {
            Text(title, style = RowTitle())
            Text(caption, style = RowCaption(), color = muted())
        }
        if (!value.isNullOrEmpty()) {
            Spacer(Modifier.width(12.dp))
            Text(value, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        }
        Icon(Icons.Outlined.ChevronRight, null, tint = muted())
    }
}

@Composable
private fun SwitchRow(title: String, caption: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 64.dp).toggleable(checked, role = Role.Switch, onValueChange = onChange)
            .padding(horizontal = 14.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = RowTitle())
            Text(caption, style = RowCaption(), color = muted())
        }
        Spacer(Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = null)
    }
}

@Composable
private fun RadioRow(title: String, caption: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 56.dp).selectable(selected, role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(selected = selected, onClick = null)
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = RowTitle())
            Text(caption, style = RowCaption(), color = muted())
        }
    }
}

/** Light, Dark, System as three paper buttons; the chosen one is inked in. */
@Composable
private fun ThemePicker(mode: String, onPick: (String) -> Unit) {
    val ink = MaterialTheme.colorScheme.onSurface
    val paper = MaterialTheme.colorScheme.surfaceContainerLowest
    val shape = RoundedCornerShape(10.dp)
    Row(Modifier.fillMaxWidth().selectableGroup(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        for ((id, label, icon) in listOf(
            Triple(Settings.THEME_LIGHT, "Light", Icons.Outlined.LightMode),
            Triple(Settings.THEME_DARK, "Dark", Icons.Outlined.DarkMode),
            Triple(Settings.THEME_SYSTEM, "System", Icons.Outlined.PhoneAndroid),
        )) {
            val on = mode == id
            Column(
                Modifier.weight(1f).height(64.dp).clip(shape).background(if (on) ink else paper).border(1.5.dp, ink, shape)
                    .selectable(on, role = Role.RadioButton) { onPick(id) },
                verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Icon(icon, null, Modifier.size(20.dp), tint = if (on) paper else ink)
                Spacer(Modifier.height(4.dp))
                Text(label, style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold), color = if (on) paper else ink)
            }
        }
    }
}

/** A dashed outline (the text recognition box). */
private fun Modifier.dashedOutline(color: Color, radius: Dp) = drawBehind {
    val w = 1.5.dp.toPx()
    drawRoundRect(
        color, topLeft = Offset(w / 2, w / 2), size = Size(size.width - w, size.height - w), cornerRadius = CornerRadius(radius.toPx()),
        style = Stroke(w, pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))),
    )
}

/**
 * The text recognition pack (output/OcrPack.kt) under the switch: not in the app, a download once; its progress, its
 * size on the phone and removing it.
 */
@Composable
private fun OcrPackBox(vm: ScanViewModel) {
    val ctx = LocalContext.current
    LaunchedEffect(Unit) { OcrPack.refresh(ctx) }
    val size = vm.ocrPackBytes?.let { formatBytes(it) }
    val st = OcrPack.state
    Row(
        Modifier.fillMaxWidth().padding(start = 14.dp, end = 14.dp, bottom = 14.dp).dashedOutline(ruledColor(), 4.dp)
            .padding(start = 12.dp, end = 6.dp, top = 6.dp, bottom = 6.dp).heightIn(min = 44.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        when (st) {
            is OcrPack.State.Ready -> {
                Icon(Icons.Outlined.CheckCircle, null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(10.dp))
                Text("Installed: ${formatBytes(OcrPack.installedBytes(ctx))} on the phone.", style = RowCaption(), color = muted(), modifier = Modifier.weight(1f))
                TextButton(onClick = { vm.removeOcr() }) { Text("Remove") }
            }
            is OcrPack.State.Downloading -> Column(Modifier.weight(1f).padding(vertical = 6.dp)) {
                LinearProgressIndicator(progress = { if (st.total > 0) st.done.toFloat() / st.total else 0f }, modifier = Modifier.fillMaxWidth())
                Spacer(Modifier.height(6.dp))
                Text("Downloading: ${formatBytes(st.done)} of ${formatBytes(st.total)}", style = RowCaption().copy(fontFamily = RecursiveMono), color = muted())
            }
            else -> {
                Icon(Icons.Outlined.Download, null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(10.dp))
                Text(
                    when {
                        size == null -> "Not available for this phone's processor."
                        st is OcrPack.State.Failed -> "The download failed: ${st.message}"
                        else -> "Needs a one-time download, $size."
                    },
                    style = RowCaption(), color = if (st is OcrPack.State.Failed) MaterialTheme.colorScheme.error else muted(), modifier = Modifier.weight(1f),
                )
                if (size != null) {
                    OutlinedButton(onClick = { vm.downloadOcr() }, shape = MaterialTheme.shapes.small) {
                        Text(if (st is OcrPack.State.Failed) "Try again" else "Download")
                    }
                }
            }
        }
    }
}

/** A bottom sheet of radio choices with a line on each; picking one closes it. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> ChoiceSheet(title: String, caption: String, options: List<Choice<T>>, selected: T, onPick: (T) -> Unit, onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss, dragHandle = { TapeHandle() }, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)) {
        Column(Modifier.navigationBarsPadding().padding(bottom = 12.dp)) {
            Column(Modifier.padding(horizontal = 16.dp)) {
                Text(title, style = MaterialTheme.typography.titleLarge)
                Text(caption, style = RowCaption(), color = muted())
            }
            Spacer(Modifier.height(8.dp))
            Column(Modifier.selectableGroup()) {
                for (o in options) RadioRow(o.title, o.caption, o.key == selected) { onPick(o.key); onDismiss() }
            }
        }
    }
}

/** Default look: the document looks as small pages, each drawn the way that look leaves a page. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun LookSheet(selected: FilterKind, onPick: (FilterKind) -> Unit, onDismiss: () -> Unit) {
    val looks = FilterKind.forMode(CaptureMode.Document)
    ModalBottomSheet(onDismissRequest = onDismiss, dragHandle = { TapeHandle() }, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)) {
        Column(Modifier.navigationBarsPadding().padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
            Text("Default look", style = MaterialTheme.typography.titleLarge)
            Text("Used for new pages. You can still change the look of any page after scanning.", style = RowCaption(), color = muted())
            Spacer(Modifier.height(14.dp))
            Column(Modifier.selectableGroup(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                for (row in looks.chunked(4)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        for (f in row) LookTile(f, f == selected, Modifier.weight(1f)) { onPick(f) }
                        repeat(4 - row.size) { Spacer(Modifier.weight(1f)) }
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            OutlinedButton(onClick = onDismiss, shape = MaterialTheme.shapes.small, modifier = Modifier.align(Alignment.End)) { Text("Done") }
        }
    }
}

@Composable
private fun LookTile(f: FilterKind, on: Boolean, modifier: Modifier, onClick: () -> Unit) {
    // paper (top and bottom colour: the photo's shading where the look keeps it), ink, the blue stamp, line weight
    val (paper, ink, stamp, th) = when (f) {
        FilterKind.Original -> Look(Color(0xFFEFE6D2) to Color(0xFFD9CDB4), Color(0xFF4A4436), Color(0xFF3B5BB5), 3f)
        FilterKind.Lighten -> Look(Color(0xFFFBF7EE) to Color(0xFFF1EAD9), Color(0xFF4A4436), Color(0xFF4A68C0), 3f)
        FilterKind.AutoColor -> Look(Color.White to Color.White, Color(0xFF2A2F3D), Color(0xFF2747A8), 3f)
        FilterKind.Grayscale -> Look(Color(0xFFF7F7F7) to Color(0xFFF7F7F7), Color(0xFF3A3A3A), Color(0xFF7A7A7A), 3f)
        FilterKind.HighContrast -> Look(Color.White to Color.White, Color.Black, Color(0xFF1D3FA8), 4f)
        FilterKind.BlackWhite -> Look(Color.White to Color.White, Color.Black, Color.Black, 4f)
        else -> Look(Color.White to Color.White, Color(0xFF1B2030), Color(0xFF2747A8), 3f)
    }
    val marker = if (isPaperLight()) Palette.Highlighter else Graphite.Highlighter
    Column(
        modifier.clip(MaterialTheme.shapes.small).selectable(on, role = Role.RadioButton, onClick = onClick).padding(vertical = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            Modifier.border(2.5.dp, if (on) MaterialTheme.colorScheme.primary else Color.Transparent, RoundedCornerShape(6.dp))
                .background(if (on) marker.copy(alpha = 0.55f) else Color.Transparent, RoundedCornerShape(6.dp)).padding(5.dp),
        ) {
            Canvas(Modifier.size(56.dp, 72.dp).shadow(1.dp, RoundedCornerShape(2.dp))) {
                val u = size.width / 62f
                drawRect(Brush.verticalGradient(listOf(paper.first, paper.second)))
                drawRect(ink, Offset(8 * u, 12 * u), Size(34 * u, (th + 1) * u))
                drawRect(ink.copy(alpha = 0.8f), Offset(8 * u, 26 * u), Size(46 * u, th * u))
                drawRect(ink.copy(alpha = 0.8f), Offset(8 * u, 36 * u), Size(40 * u, th * u))
                drawRect(stamp, Offset(size.width - 26 * u, size.height - 24 * u), Size(18 * u, 14 * u), style = Stroke(2 * u))
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(
            f.label, style = MaterialTheme.typography.labelMedium.copy(fontWeight = if (on) FontWeight.Bold else FontWeight.Medium),
            maxLines = 1,
        )
    }
}

private data class Look(val paper: Pair<Color, Color>, val ink: Color, val stamp: Color, val th: Float)
