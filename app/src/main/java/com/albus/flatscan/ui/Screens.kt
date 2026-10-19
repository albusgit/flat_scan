package com.albus.flatscan.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.automirrored.outlined.NoteAdd
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Snackbar
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.albus.flatscan.BuildConfigInfo
import com.albus.flatscan.FilterKind
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import com.albus.flatscan.Settings
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(vm: ScanViewModel) {
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    var confirmNew by remember { mutableStateOf(false) }
    val n = vm.pages.size
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { },
            actions = { IconButton(onClick = { vm.open(Screen.Settings) }) { Icon(Icons.Outlined.Settings, "Settings") } },
        )
        Column(
            Modifier.weight(1f).fillMaxWidth().padding(horizontal = 32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("FlatScan", style = MaterialTheme.typography.headlineLarge)
            Spacer(Modifier.height(8.dp))
            Text(
                "Photograph a page. It is found, flattened and cleaned up.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
        Column(Modifier.fillMaxWidth().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            if (n > 0) {
                Button(onClick = { vm.open(Screen.Pages) }, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                    Icon(Icons.Outlined.Description, null)
                    Spacer(Modifier.size(12.dp))
                    Text(if (n == 1) "Continue current document (1 page)" else "Continue current document ($n pages)", style = MaterialTheme.typography.titleMedium)
                }
                Spacer(Modifier.height(12.dp))
                OutlinedButton(onClick = { confirmNew = true }, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                    Icon(Icons.AutoMirrored.Outlined.NoteAdd, null)
                    Spacer(Modifier.size(12.dp))
                    Text("New document")
                }
            } else {
                Button(onClick = { vm.startCamera() }, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                    Icon(Icons.Outlined.CameraAlt, null)
                    Spacer(Modifier.size(12.dp))
                    Text("Scan with camera", style = MaterialTheme.typography.titleMedium)
                }
                Spacer(Modifier.height(12.dp))
                OutlinedButton(
                    onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                ) {
                    Icon(Icons.Outlined.PhotoLibrary, null)
                    Spacer(Modifier.size(12.dp))
                    Text("Import from gallery")
                }
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

@Composable
fun ProcessingScreen(vm: ScanViewModel) {
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Processing", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(24.dp))
        LinearProgressIndicator(progress = { vm.progress }, modifier = Modifier.fillMaxWidth().height(6.dp))
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
            title = { Text("Settings") },
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
        )
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
            Section("Default filter")
            for (f in FilterKind.entries) Choice(f.label, filter == f) { filter = f; s.defaultFilter = f }
            Text(
                "The filter and strength you choose on a page become the default for the next one.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            HorizontalDivider(Modifier.padding(vertical = 8.dp))
            SwitchRow("Solid colors", "Flatten shaded table cells to one colour", solid) { solid = it; s.solidColors = it }
            SwitchRow("Auto-rotate pages", "Turn sideways and upside-down pages upright", autoRotate) { autoRotate = it; s.autoRotate = it }
            HorizontalDivider(Modifier.padding(vertical = 8.dp))
            Section("Output resolution")
            for ((w, label) in Settings.OUTPUT_WIDTHS) Choice(label, width == w) { width = w; s.outputWidth = w }
            HorizontalDivider(Modifier.padding(vertical = 8.dp))
            Section("About")
            Text("Version ${BuildConfigInfo.VERSION}", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(4.dp))
            Text(vm.gpuInfo, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            HorizontalDivider(Modifier.padding(vertical = 8.dp))
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
    Text(t, style = MaterialTheme.typography.titleSmall, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 12.dp, bottom = 4.dp))
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
