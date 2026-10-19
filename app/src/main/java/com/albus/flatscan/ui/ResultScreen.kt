package com.albus.flatscan.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.RotateRight
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.Collections
import androidx.compose.material.icons.outlined.Crop
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalIconToggleButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.albus.flatscan.FilterKind
import com.albus.flatscan.ImportSource
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import kotlin.math.roundToInt

private enum class Sheet { None, Save, Share }

/** TURBO-like scale of the wrinkle map legend (blue -> cyan -> green -> yellow -> red). */
private val TurboStops = listOf(
    Color(0xFF30123B), Color(0xFF4686FB), Color(0xFF1AE4B6), Color(0xFFA2FC3C), Color(0xFFFABA39), Color(0xFFE4460A), Color(0xFF7A0403),
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(vm: ScanViewModel) {
    var sheet by remember { mutableStateOf(Sheet.None) }
    var showDetails by remember { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }
    var protectMode by remember { mutableStateOf(false) }
    val exporter = rememberExportActions(vm)
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    val viewer = remember(vm.currentId, vm.rotation) { ViewerState() }
    val p = vm.params
    // protect areas only matter (and are only drawn) while Solid colors is in effect
    val protecting = protectMode && p.solid && p.filter != FilterKind.BlackWhite
    val index = vm.pages.indexOf(vm.currentId)

    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text(if (vm.pages.size > 1 && index >= 0) "Page ${index + 1} of ${vm.pages.size}" else "Result") },
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            actions = {
                IconButton(onClick = { vm.recrop() }) { Icon(Icons.Outlined.Crop, "Re-crop") }
                IconButton(onClick = { vm.open(Screen.Pages) }) { Icon(Icons.Outlined.Collections, "Pages") }
                IconButton(onClick = { sheet = Sheet.Share }, enabled = vm.page != null && !exporter.busy) { Icon(Icons.Outlined.Share, "Share") }
                Box {
                    IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreVert, "More") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(
                            text = { Text("Show wrinkle map") },
                            trailingIcon = { Checkbox(checked = vm.showWrinkles, onCheckedChange = null) },
                            onClick = { vm.setWrinkles(!vm.showWrinkles); menu = false },
                        )
                        DropdownMenuItem(text = { Text("Processing details") }, onClick = { showDetails = true; menu = false })
                    }
                }
            },
        )
        Box(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 8.dp)) {
            PageViewer(
                page = vm.page, before = vm.before, overlay = if (vm.showWrinkles) vm.overlay else null,
                pageWidth = vm.pageWidth, pageHeight = vm.pageHeight, rotation = vm.rotation,
                protect = vm.protect, protectMode = protecting, state = viewer,
                onHold = { if (it) vm.holdStarted() },
                onRect = { x0, y0, x1, y1 -> vm.addProtect(x0, y0, x1, y1) },
                modifier = Modifier.fillMaxSize(),
            )
            if (protecting) {
                Surface(
                    Modifier.align(Alignment.TopCenter).padding(8.dp),
                    shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceContainerHigh, tonalElevation = 2.dp,
                ) {
                    Row(Modifier.padding(start = 16.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text("Drag a box around each picture", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f, fill = false))
                        TextButton(onClick = { vm.clearProtect() }, enabled = vm.protect.isNotEmpty()) { Text("Clear") }
                        TextButton(onClick = { protectMode = false }) { Text("Done") }
                    }
                }
            } else if (vm.showHoldHint && vm.page != null) {
                Surface(
                    Modifier.align(Alignment.BottomCenter).padding(12.dp),
                    shape = RoundedCornerShape(50), color = MaterialTheme.colorScheme.inverseSurface.copy(alpha = 0.85f),
                ) {
                    Text(
                        "Press and hold to see the original",
                        style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.inverseOnSurface,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    )
                }
            }
            if (vm.rendering) {
                CircularProgressIndicator(Modifier.align(Alignment.TopEnd).padding(12.dp).size(22.dp), strokeWidth = 2.5.dp)
            }
        }
        if (vm.showWrinkles) WrinkleLegend(vm.overlayMaxPx, vm.overlay != null)

        if (vm.adjustOpen) {
            AdjustPanel(vm)
        } else {
            FilterPanel(vm, protecting, onProtect = { protectMode = !protecting })
        }

        Row(
            Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 4.dp, bottom = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            val ready = vm.page != null && !exporter.busy
            if (vm.pending.isNotEmpty()) {
                OutlinedButton(onClick = { sheet = Sheet.Save }, enabled = ready, modifier = Modifier.weight(1f).height(56.dp)) { Text("Save") }
                Button(onClick = { vm.nextPage() }, modifier = Modifier.weight(1.4f).height(56.dp)) {
                    Text("Next page (${vm.nextPendingIndex} of ${vm.pendingTotal})", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            } else {
                OutlinedButton(
                    onClick = {
                        if (vm.lastSource == ImportSource.Gallery) pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                        else vm.startCamera()
                    },
                    modifier = Modifier.weight(1f).height(56.dp),
                ) {
                    Icon(Icons.Outlined.Add, null)
                    Spacer(Modifier.size(6.dp))
                    Text("Add page")
                }
                Button(onClick = { sheet = Sheet.Save }, enabled = ready, modifier = Modifier.weight(1.4f).height(56.dp)) {
                    if (exporter.busy) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
                    else Text(if (vm.pages.size > 1) "Save ${vm.pages.size} pages" else "Save", style = MaterialTheme.typography.titleMedium)
                }
            }
        }
    }

    if (sheet != Sheet.None) {
        val share = sheet == Sheet.Share
        ExportSheet(share, vm.pages.size, onPick = { if (share) exporter.share(it) else exporter.save(it) }, onDismiss = { sheet = Sheet.None })
    }
    if (showDetails) {
        AlertDialog(
            onDismissRequest = { showDetails = false },
            confirmButton = { TextButton(onClick = { showDetails = false }) { Text("Close") } },
            title = { Text("Processing details") },
            text = { Text(vm.details, style = MaterialTheme.typography.bodySmall, modifier = Modifier.verticalScroll(rememberScrollState())) },
        )
    }
}

@Composable
private fun FilterPanel(vm: ScanViewModel, protectMode: Boolean, onProtect: () -> Unit) {
    val p = vm.params
    // live previews: one per filter, in the page's aspect (as rotated on screen)
    val rw = if (vm.rotation % 2 == 0) vm.pageWidth else vm.pageHeight
    val rh = if (vm.rotation % 2 == 0) vm.pageHeight else vm.pageWidth
    val thumbH = 72.dp
    val thumbW = if (rw > 0 && rh > 0) (thumbH * (rw.toFloat() / rh)).coerceIn(40.dp, 112.dp) else 52.dp
    LazyRow(
        contentPadding = PaddingValues(horizontal = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
    ) {
        items(FilterKind.entries) { f ->
            val selected = f == p.filter
            Column(
                Modifier.width(maxOf(thumbW, 64.dp)).clip(RoundedCornerShape(10.dp))
                    .selectable(selected, role = Role.RadioButton) { vm.selectFilter(f) }
                    .padding(vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                val shape = RoundedCornerShape(8.dp)
                Box(
                    Modifier.size(thumbW, thumbH).clip(shape)
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                        .border(if (selected) 2.5.dp else 1.dp, if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant, shape),
                    contentAlignment = Alignment.Center,
                ) {
                    val t = vm.thumbs[f]
                    if (t != null) PageThumb(t, vm.rotation, Modifier.fillMaxSize())
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    f.label, style = MaterialTheme.typography.labelMedium, maxLines = 1, textAlign = TextAlign.Center,
                    color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        val enabled = p.filter != FilterKind.Original
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Strength", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.width(72.dp))
            Slider(
                value = p.strength, onValueChange = { vm.setStrength(it) }, onValueChangeFinished = { vm.strengthDone() },
                enabled = enabled, modifier = Modifier.weight(1f),
            )
            Text("${(p.strength * 100).roundToInt()}%", style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.End, modifier = Modifier.width(44.dp))
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            val solidOk = p.filter != FilterKind.BlackWhite
            Row(
                Modifier.weight(1f).clip(RoundedCornerShape(8.dp)).clickable(enabled = solidOk) { vm.setSolid(!p.solid) }.padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Switch(checked = p.solid && solidOk, onCheckedChange = { vm.setSolid(it) }, enabled = solidOk)
                Spacer(Modifier.size(10.dp))
                Text("Solid colors", style = MaterialTheme.typography.bodyLarge, maxLines = 1)
            }
            if (p.solid && solidOk) {
                FilledTonalIconToggleButton(checked = protectMode, onCheckedChange = { onProtect() }) {
                    Icon(Icons.Outlined.Shield, "Protect a picture")
                }
            }
            IconButton(onClick = { vm.openAdjust(true) }) { Icon(Icons.Outlined.Tune, "Adjust") }
            IconButton(onClick = { vm.rotateResult() }) { Icon(Icons.AutoMirrored.Outlined.RotateRight, "Rotate") }
        }
    }
}

@Composable
private fun AdjustPanel(vm: ScanViewModel) {
    val p = vm.params
    Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Adjust", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            TextButton(onClick = { vm.resetAdjustments() }) { Text("Reset") }
            TextButton(onClick = { vm.openAdjust(false) }) { Text("Done") }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Background cleanup", style = MaterialTheme.typography.bodyMedium)
                Text("Removes grey specks and shading on white paper", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text("${(p.bgCleanup * 100).roundToInt()}%", style = MaterialTheme.typography.bodyMedium)
        }
        Slider(value = p.bgCleanup, onValueChange = { vm.setBgCleanup(it) })
        Text("Levels", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(bottom = 4.dp))
        LevelsEditor(vm.histogram, p.black, p.white, p.gamma, onChange = { b, w, g -> vm.setLevels(b, w, g) })
        Spacer(Modifier.height(4.dp))
    }
}

@Composable
private fun WrinkleLegend(maxPx: Float, ready: Boolean) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 4.dp)) {
        Box(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp)).background(Brush.horizontalGradient(TurboStops)))
        Row(Modifier.fillMaxWidth().padding(top = 2.dp)) {
            Text("0", style = MaterialTheme.typography.labelSmall)
            Text(
                if (ready) "how far each area was moved" else "computing...",
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center, modifier = Modifier.weight(1f),
            )
            Text(if (ready) formatPx(maxPx) else "", style = MaterialTheme.typography.labelSmall)
        }
    }
}

private fun formatPx(v: Float): String = if (v < 10f) "%.1f px".format(v) else "${v.roundToInt()} px"
