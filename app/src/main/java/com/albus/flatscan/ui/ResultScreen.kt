package com.albus.flatscan.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.outlined.RotateRight
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.automirrored.outlined.Redo
import androidx.compose.material.icons.automirrored.outlined.Undo
import androidx.compose.material.icons.outlined.AutoFixHigh
import androidx.compose.material.icons.outlined.Compare
import androidx.compose.material.icons.outlined.DoneAll
import androidx.compose.material.icons.outlined.Monitor
import androidx.compose.material.icons.outlined.Collections
import androidx.compose.material.icons.outlined.Crop
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Palette
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material.icons.outlined.SwapHoriz
import androidx.compose.material.icons.outlined.Straighten
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.material.icons.outlined.TouchApp
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.FilterKind
import com.albus.flatscan.ImportSource
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.ScreenMode
import com.albus.flatscan.Screen
import java.util.Locale
import kotlin.math.roundToInt

private enum class Sheet { None, Save, Share }

/** TURBO-like scale of the wrinkle map legend (blue -> cyan -> green -> yellow -> red). */
private val TurboStops = listOf(
    Color(0xFF30123B), Color(0xFF4686FB), Color(0xFF1AE4B6), Color(0xFFA2FC3C), Color(0xFFFABA39), Color(0xFFE4460A), Color(0xFF7A0403),
)

@Composable
fun ResultScreen(vm: ScanViewModel) = EditorTheme {
    var sheet by remember { mutableStateOf(Sheet.None) }
    var showDetails by remember { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }
    // the finger brush: Remove (add to the mask) or Keep, and its radius (page-width fraction)
    var brushAdd by remember { mutableStateOf(true) }
    var brushRadius by remember { mutableFloatStateOf(0.02f) }
    var tool by remember {
        // back from flattening again in the line editor: show the new lines
        mutableStateOf(if (vm.reopenLines) AreaMode.Lines else AreaMode.None).also { vm.reopenLines = false }
    }
    val exporter = rememberExportActions(vm)
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    val viewer = remember(vm.currentId, vm.rotation) { ViewerState() }
    // before / after split (null: off)
    var compare by remember(vm.currentId) { mutableStateOf<Float?>(null) }
    val p = vm.params
    val look = vm.look
    val colorFilter = remember(look) { look.composeFilter() }
    // the page as it will be exported (PageLook shape): same width, its own height
    val shownHeight = look.targetHw(vm.pageWidth, vm.pageHeight)?.let { (vm.pageWidth * it).roundToInt() } ?: vm.pageHeight
    // protect areas only matter (and are only drawn) while Solid colors is in effect
    val mode = when {
        tool == AreaMode.Protect && !(p.solid && p.filter != FilterKind.BlackWhite) -> AreaMode.None
        else -> tool
    }
    val index = vm.pages.indexOf(vm.currentId)
    val ready = vm.page != null && !exporter.busy
    val closeTool = {
        if (tool == AreaMode.Lines) vm.revertIgnore()
        tool = AreaMode.None
    }
    BackHandler(enabled = mode != AreaMode.None) { closeTool() }

    val topBar: @Composable () -> Unit = {
        Row(
            Modifier.fillMaxWidth().height(64.dp).padding(start = 8.dp, end = 12.dp, top = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { if (mode != AreaMode.None) closeTool() else vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Text(
                buildAnnotatedString {
                    if (vm.pages.size > 1 && index >= 0) {
                        append("Page ${index + 1} ")
                        withStyle(SpanStyle(color = Editor.Muted, fontWeight = FontWeight.SemiBold)) { append("of ${vm.pages.size}") }
                    } else {
                        append("Page")
                    }
                },
                style = MaterialTheme.typography.titleMedium, maxLines = 1,
                modifier = Modifier.padding(horizontal = 10.dp).weight(1f),
            )
            // one tap for an upside-down page: there is no automatic 180-degree turn
            IconButton(onClick = { vm.rotateResult180() }, enabled = vm.page != null) {
                Box(
                    Modifier.size(30.dp).border(1.5.dp, Editor.Inactive, CircleShape).semantics { contentDescription = "Turn 180 degrees" },
                    contentAlignment = Alignment.Center,
                ) {
                    Text("180°", style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp, fontWeight = FontWeight.ExtraBold), color = Editor.Inactive)
                }
            }
            Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreVert, "More") }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    if (vm.spreadPartner != null) {
                        DropdownMenuItem(
                            text = { Text("Swap the two book pages") },
                            leadingIcon = { Icon(Icons.Outlined.SwapHoriz, null) },
                            onClick = { vm.swapSpread(); menu = false },
                        )
                    }
                    DropdownMenuItem(
                        text = { Text("Unwarp") },
                        trailingIcon = { Checkbox(checked = !vm.perspectiveOnly, onCheckedChange = null) },
                        enabled = ready,
                        onClick = { vm.reflatten(perspective = !vm.perspectiveOnly); menu = false },
                    )
                    DropdownMenuItem(
                        text = { Text("Show wrinkle map") },
                        trailingIcon = { Checkbox(checked = vm.showWrinkles, onCheckedChange = null) },
                        onClick = { vm.setWrinkles(!vm.showWrinkles); menu = false },
                    )
                    DropdownMenuItem(text = { Text("Processing details") }, onClick = { showDetails = true; menu = false })
                    DropdownMenuItem(
                        text = { Text("Share") },
                        leadingIcon = { Icon(Icons.Outlined.Share, null) },
                        enabled = ready,
                        onClick = { sheet = Sheet.Share; menu = false },
                    )
                }
            }
            Button(
                onClick = { sheet = Sheet.Save }, enabled = ready,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Editor.Text, contentColor = Editor.Ground,
                    disabledContainerColor = Editor.Text.copy(alpha = 0.3f), disabledContentColor = Editor.Ground.copy(alpha = 0.7f),
                ),
                contentPadding = PaddingValues(horizontal = 18.dp),
                modifier = Modifier.height(40.dp),
            ) {
                if (exporter.busy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp, color = Editor.Ground)
                else Text("Save", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp))
            }
        }
    }
    val viewerBox: @Composable (Modifier) -> Unit = { boxModifier ->
        Box(boxModifier.padding(horizontal = 8.dp, vertical = 8.dp)) {
            PageViewer(
                page = vm.page, before = vm.before, overlay = if (vm.showWrinkles) vm.overlay else null,
                pageWidth = vm.pageWidth, pageHeight = shownHeight, rotation = vm.rotation,
                areas = when (mode) {
                    AreaMode.Lines -> if (vm.perspectiveOnly) emptyList() else vm.ignore
                    AreaMode.Erase -> vm.erase
                    AreaMode.Fingers -> emptyList()
                    else -> vm.protect
                },
                areaMode = mode, lines = vm.lines, state = viewer,
                onHold = { if (it) vm.holdStarted() },
                onRect = { x0, y0, x1, y1 ->
                    when (mode) {
                        AreaMode.Lines -> vm.addIgnore(x0, y0, x1, y1)
                        AreaMode.Erase -> vm.addErase(x0, y0, x1, y1)
                        else -> vm.addProtect(x0, y0, x1, y1)
                    }
                },
                onTap = { x, y, line ->
                    when {
                        mode == AreaMode.Fingers -> {}
                        mode == AreaMode.Protect -> vm.removeProtectAt(x, y)
                        mode == AreaMode.Erase -> vm.removeEraseAt(x, y)
                        line >= 0 -> vm.toggleLine(line)
                        else -> vm.removeIgnoreAt(x, y)
                    }
                },
                modifier = Modifier.fillMaxSize(),
                colorFilter = colorFilter,
                compare = if (mode == AreaMode.None) compare else null,
                onCompare = { compare = it },
                brushOverlay = if (mode == AreaMode.Fingers) vm.fingerOverlay else null,
                brushRadius = brushRadius,
                onStroke = { pts -> vm.addFingerStroke(brushAdd, brushRadius, pts) },
            )
            if (mode == AreaMode.None && vm.showHoldHint && vm.page != null && compare == null) {
                Text(
                    "Press and hold to see the original",
                    style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold), color = Color(0xFFE3E3E0),
                    modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 10.dp)
                        .background(Editor.Text.copy(alpha = 0.12f), RoundedCornerShape(16.dp))
                        .padding(horizontal = 14.dp, vertical = 7.dp),
                )
            }
            // undo / redo of every edit that does not re-run the geometry (filters, adjust, rotate, areas)
            AnimatedVisibility(
                visible = mode == AreaMode.None && (vm.history.canUndo || vm.history.canRedo),
                enter = fadeIn(), exit = fadeOut(), modifier = Modifier.align(Alignment.TopStart),
            ) {
                Row(Modifier.padding(4.dp).background(Editor.Panel.copy(alpha = 0.88f), RoundedCornerShape(24.dp))) {
                    IconButton(onClick = { vm.undo() }, enabled = vm.history.canUndo) { Icon(Icons.AutoMirrored.Outlined.Undo, "Undo") }
                    IconButton(onClick = { vm.redo() }, enabled = vm.history.canRedo) { Icon(Icons.AutoMirrored.Outlined.Redo, "Redo") }
                }
            }
            Row(Modifier.align(Alignment.TopEnd).padding(4.dp), verticalAlignment = Alignment.CenterVertically) {
                if (vm.rendering) {
                    CircularProgressIndicator(Modifier.padding(8.dp).size(22.dp), strokeWidth = 2.5.dp, color = Editor.Accent)
                }
                if (mode == AreaMode.None && vm.page != null) {
                    val on = compare != null
                    IconButton(
                        onClick = { compare = if (on) null else 0.5f },
                        modifier = Modifier.background(if (on) Editor.Accent.copy(alpha = 0.9f) else Editor.Panel.copy(alpha = 0.88f), CircleShape),
                    ) {
                        Icon(Icons.Outlined.Compare, if (on) "Stop comparing" else "Compare with the original", tint = if (on) Editor.Ground else Editor.Text)
                    }
                }
            }
            if (compare != null && mode == AreaMode.None) {
                Row(Modifier.align(Alignment.BottomCenter).padding(bottom = 10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CompareLabel("Original")
                    CompareLabel("Cleaned")
                }
            }
        }
    }
    val panel: @Composable ColumnScope.() -> Unit = {
        if (mode == AreaMode.Protect) {
            ProtectPanel(vm, onDone = { tool = AreaMode.None })
        } else if (mode == AreaMode.Lines) {
            LinesPanel(vm, onClose = closeTool)
        } else if (mode == AreaMode.Erase) {
            ErasePanel(vm, onDone = { tool = AreaMode.None })
        } else if (mode == AreaMode.Fingers) {
            FingerPanel(vm, brushAdd, { brushAdd = it }, brushRadius, { brushRadius = it }, onDone = { tool = AreaMode.None })
        } else if (vm.adjustOpen) {
            AdjustPanel(vm)
        } else {
            FilterPanel(
                vm, onProtect = { tool = AreaMode.Protect }, onErase = { tool = AreaMode.Erase }, onFingers = { tool = AreaMode.Fingers },
                onAddPage = {
                    if (vm.lastSource == ImportSource.Gallery) pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    else vm.startCamera()
                },
            )
        }
    }
    val dock: @Composable () -> Unit = {
        ToolDock(vm, linesOpen = mode == AreaMode.Lines, onLines = { if (mode == AreaMode.Lines) closeTool() else tool = AreaMode.Lines }, onTab = closeTool)
    }

    BoxWithConstraints(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))) {
        // landscape phones and tablets: the panel beside the page (below it, the page would get a sliver)
        val wide = maxWidth > maxHeight && maxWidth >= 600.dp
        if (wide) {
            Column(Modifier.fillMaxSize()) {
                topBar()
                Row(Modifier.weight(1f).fillMaxWidth()) {
                    Column(Modifier.weight(1f).fillMaxHeight()) {
                        viewerBox(Modifier.weight(1f).fillMaxWidth())
                        if (vm.showWrinkles) WrinkleLegend(vm.overlayMaxPx, vm.overlay != null)
                    }
                    Column(
                        Modifier.width(400.dp).fillMaxHeight().padding(end = 8.dp)
                            .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)).background(Editor.Panel).navigationBarsPadding(),
                    ) {
                        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) { panel() }
                        dock()
                    }
                }
            }
        } else {
            Column(Modifier.fillMaxSize()) {
                topBar()
                viewerBox(Modifier.weight(1f).fillMaxWidth())
                if (vm.showWrinkles) WrinkleLegend(vm.overlayMaxPx, vm.overlay != null)
                // the editor panel: the tool of the selected tab over the tool dock
                Column(Modifier.fillMaxWidth().clip(Editor.PanelShape).background(Editor.Panel).navigationBarsPadding()) {
                    panel()
                    dock()
                }
            }
        }
    }

    if (sheet != Sheet.None) {
        ExportSheet(vm, sheet == Sheet.Share, vm.pages.size, vm.docTitle, onRename = { vm.renameDocument(it) }, onSave = { exporter.save(it) }, onShare = { exporter.share(it) }, onDismiss = { sheet = Sheet.None })
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

/** Six tools along the bottom: three tabs (filters, adjust, unwarp lines) and three actions. */
@Composable
private fun ToolDock(vm: ScanViewModel, linesOpen: Boolean, onLines: () -> Unit, onTab: () -> Unit) {
    HorizontalDivider(color = Editor.Divider)
    Row(Modifier.fillMaxWidth().padding(start = 4.dp, end = 4.dp, top = 6.dp, bottom = 10.dp)) {
        DockItem("Filters", Icons.Outlined.Palette, Modifier.weight(1f), selected = !vm.adjustOpen && !linesOpen) { onTab(); vm.openAdjust(false) }
        DockItem("Adjust", Icons.Outlined.Tune, Modifier.weight(1f), selected = vm.adjustOpen && !linesOpen) { onTab(); vm.openAdjust(true) }
        DockItem("Lines", Icons.Outlined.Straighten, Modifier.weight(1f), selected = linesOpen) { onLines() }
        DockItem("Crop", Icons.Outlined.Crop, Modifier.weight(1f)) { vm.recrop() }
        DockItem("Rotate", Icons.AutoMirrored.Outlined.RotateRight, Modifier.weight(1f)) { vm.rotateResult() }
        DockItem("Pages", Icons.Outlined.Collections, Modifier.weight(1f), badge = vm.pages.size) { vm.open(Screen.Pages) }
    }
}

@Composable
/** A tab when [selected] is given, otherwise an action button. */
private fun DockItem(label: String, icon: ImageVector, modifier: Modifier, selected: Boolean? = null, badge: Int = 0, onClick: () -> Unit) {
    val color = if (selected == true) Editor.Accent else Editor.Inactive
    Column(
        modifier.height(58.dp).clip(RoundedCornerShape(12.dp))
            .then(if (selected != null) Modifier.selectable(selected, role = Role.Tab, onClick = onClick) else Modifier.clickable(role = Role.Button, onClick = onClick)),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        BadgedBox(
            badge = {
                if (badge > 0) Badge(containerColor = Editor.Accent, contentColor = Editor.Ground) {
                    Text("$badge", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp, fontWeight = FontWeight.ExtraBold))
                }
            },
        ) { Icon(icon, null, tint = color, modifier = Modifier.size(22.dp)) }
        Spacer(Modifier.height(4.dp))
        Text(label, style = MaterialTheme.typography.labelSmall, color = color, maxLines = 1)
    }
}

/** A 36 dp chip of the editor panel; [selected] fills it with a tint of the accent. */
@Composable
private fun EditorChip(
    text: String,
    onClick: () -> Unit,
    icon: ImageVector? = null,
    selected: Boolean = false,
    enabled: Boolean = true,
    role: Role = Role.Button,
) {
    val shape = RoundedCornerShape(18.dp)
    Row(
        Modifier.minimumInteractiveComponentSize().height(36.dp).clip(shape)
            .background(if (selected) Editor.Accent.copy(alpha = 0.18f) else Color.Transparent)
            .border(1.dp, if (selected) Editor.Accent else Editor.Outline, shape)
            .selectable(selected, enabled = enabled, role = role, onClick = onClick)
            .alpha(if (enabled) 1f else 0.4f)
            .padding(horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, null, Modifier.size(16.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp), maxLines = 1)
    }
}

@Composable
private fun FilterPanel(vm: ScanViewModel, onProtect: () -> Unit, onErase: () -> Unit, onFingers: () -> Unit, onAddPage: () -> Unit) {
    val p = vm.params
    val thumbFilter = remember(vm.look) { vm.look.composeFilter() }
    Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp), verticalAlignment = Alignment.CenterVertically) {
        val enabled = p.filter != FilterKind.Original
        Text(p.filter.label, style = MaterialTheme.typography.titleSmall, maxLines = 1, modifier = Modifier.widthIn(min = 86.dp))
        Spacer(Modifier.width(12.dp))
        Slider(
            value = p.strength, onValueChange = { vm.setStrength(it) }, onValueChangeFinished = { vm.strengthDone() },
            enabled = enabled, colors = editorSliderColors(), modifier = Modifier.weight(1f)
                .semantics { contentDescription = "Strength" },
        )
        Text(
            "${(p.strength * 100).roundToInt()}%", style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
            color = Editor.Muted, textAlign = TextAlign.End, modifier = Modifier.width(44.dp),
        )
    }
    // live previews: one per filter
    LazyRow(
        contentPadding = PaddingValues(horizontal = 20.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        modifier = Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 4.dp),
    ) {
        items(FilterKind.forMode(vm.pageMode)) { f ->
            val selected = f == p.filter
            Column(
                // wide enough for "High contrast" (66 dp cut it to "High")
                Modifier.width(80.dp).clip(RoundedCornerShape(10.dp))
                    .selectable(selected, role = Role.RadioButton) { vm.selectFilter(f) },
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                val ring = RoundedCornerShape(10.dp)
                Box(
                    Modifier.size(58.dp, 76.dp)
                        .border(if (selected) 2.5.dp else 1.dp, if (selected) Editor.Accent else Editor.Divider, ring)
                        .padding(3.dp).clip(RoundedCornerShape(6.dp)).background(Editor.Ground),
                    contentAlignment = Alignment.Center,
                ) {
                    val t = vm.thumbs[f]
                    if (t != null) PageThumb(t, vm.rotation, Modifier.fillMaxSize(), hw = vm.look.targetHw(vm.pageWidth, vm.pageHeight), colorFilter = thumbFilter)
                }
                Spacer(Modifier.height(6.dp))
                Text(
                    f.label, style = MaterialTheme.typography.labelMedium, maxLines = 1, textAlign = TextAlign.Center,
                    color = if (selected) Editor.Accent else Editor.Inactive,
                )
            }
        }
    }
    Row(
        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(start = 20.dp, end = 20.dp, top = 8.dp, bottom = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        val solidOk = p.filter != FilterKind.BlackWhite
        EditorChip(
            "Solid colors", onClick = { vm.setSolid(!p.solid) }, icon = Icons.Outlined.GridView,
            selected = p.solid && solidOk, enabled = solidOk, role = Role.Switch,
        )
        if (p.solid && solidOk) {
            val n = vm.protect.size / 4
            EditorChip(if (n > 0) "Protected pictures ($n)" else "Protect a picture", onClick = onProtect, icon = Icons.Outlined.Shield)
        }
        val ne = vm.erase.size / 4
        EditorChip(if (ne > 0) "Erased ($ne)" else "Erase", onClick = onErase, icon = Icons.Outlined.AutoFixHigh)
        if (vm.pageMode == com.albus.flatscan.CaptureMode.Document) {
            EditorChip("Fingers", onClick = onFingers, icon = Icons.Outlined.TouchApp, selected = !vm.fingersAuto || vm.fingerStrokes.isNotEmpty())
        }
        // photo of a monitor / laptop / phone: descreen (moire, pixel grid) before the filter. Auto finds only clear
        // cases (docs/results/r11/filters.md), so a tap cycles Auto -> On -> Off
        // document or whiteboard processing for this page (processed again from its photo)
        EditorChip(
            "Mode: ${vm.pageMode.label}",
            onClick = { vm.setPageMode(if (vm.pageMode == com.albus.flatscan.CaptureMode.Document) com.albus.flatscan.CaptureMode.Whiteboard else com.albus.flatscan.CaptureMode.Document) },
            icon = Icons.Outlined.Monitor, selected = vm.pageMode == com.albus.flatscan.CaptureMode.Whiteboard,
        )
        EditorChip(
            "Screen photo: ${p.screen.label}",
            onClick = { vm.setScreen(when (p.screen) { ScreenMode.Auto -> ScreenMode.On; ScreenMode.On -> ScreenMode.Off; ScreenMode.Off -> ScreenMode.Auto }) },
            icon = Icons.Outlined.Monitor, selected = p.screen == ScreenMode.On,
        )
        if (vm.pages.size > 1) {
            // the filter, strength and adjustments of this page on every page of the document
            EditorChip(
                if (vm.applyingAll) "Applying..." else "Apply to all ${vm.pages.size} pages", onClick = { vm.applyToAllPages() },
                icon = Icons.Outlined.DoneAll, enabled = !vm.applyingAll && vm.page != null,
            )
        }
        if (vm.pending.isNotEmpty()) {
            EditorChip("Next page (${vm.nextPendingIndex} of ${vm.pendingTotal})", onClick = { vm.nextPage() }, icon = Icons.AutoMirrored.Filled.ArrowForward, selected = true)
        } else {
            EditorChip("Add page", onClick = onAddPage, icon = Icons.Outlined.Add)
        }
    }
}

/**
 * Fingers: the automatic finger removal (on / off) and a brush that adds areas to it (filled with the paper
 * around them) or takes areas out (the photo comes back there). The mask is shown in orange; strokes are
 * saved with the page.
 */
@Composable
private fun FingerPanel(
    vm: ScanViewModel, add: Boolean, onAdd: (Boolean) -> Unit, radius: Float, onRadius: (Float) -> Unit, onDone: () -> Unit,
) {
    LaunchedEffect(Unit) { vm.refreshFingerOverlay() }
    DisposableEffect(Unit) { onDispose { vm.closeFingerTool() } }
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.TouchApp, null, tint = Editor.Accent, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Text("Fingers", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            Text(if (vm.fingerShare <= 0f) "none removed" else "%.1f%% of the page".format(100 * vm.fingerShare), style = MaterialTheme.typography.labelLarge, color = Editor.Muted)
        }
        Text(
            "Orange is filled with the paper around it. Paint over a finger the app missed, or switch to Keep and paint where it took too much." +
                if (vm.fingersOverText) " Text ran under a finger: that area is outlined, not guessed." else "",
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 6.dp),
        )
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Remove", onClick = { onAdd(true) }, selected = add)
            EditorChip("Keep", onClick = { onAdd(false) }, selected = !add)
            EditorChip(if (vm.fingersAuto) "Automatic: on" else "Automatic: off", onClick = { vm.setFingersAutoOn(!vm.fingersAuto) }, selected = vm.fingersAuto)
        }
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Brush", style = MaterialTheme.typography.labelLarge, color = Editor.Muted)
            Slider(value = radius, onValueChange = onRadius, valueRange = 0.005f..0.06f, modifier = Modifier.weight(1f).padding(horizontal = 10.dp))
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Undo", onClick = { vm.undoFingerStroke() }, enabled = vm.fingerStrokes.isNotEmpty())
            EditorChip("Clear", onClick = { vm.clearFingerStrokes() }, enabled = vm.fingerStrokes.isNotEmpty())
            Spacer(Modifier.weight(1f))
            Button(
                onClick = onDone, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
                contentPadding = PaddingValues(horizontal = 22.dp), modifier = Modifier.height(40.dp),
            ) { Text("Done", style = MaterialTheme.typography.titleSmall) }
        }
    }
}

/** Erase tool: fingers, stains, stamps, punch holes filled with the paper around them. */
@Composable
private fun ErasePanel(vm: ScanViewModel, onDone: () -> Unit) {
    val n = vm.erase.size / 4
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.AutoFixHigh, null, tint = Editor.Accent, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Text("Erase", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            Text(if (n == 0) "nothing erased" else if (n == 1) "1 area" else "$n areas", style = MaterialTheme.typography.labelLarge, color = Editor.Muted)
        }
        Text(
            "Drag a box over a finger, a stain, a stamp or a punch hole: it is filled with the paper around it. Tap a box to bring it back.",
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 6.dp),
        )
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Undo", onClick = { vm.undoErase() }, enabled = n > 0)
            EditorChip("Restore all", onClick = { vm.clearErase() }, enabled = n > 0)
            Spacer(Modifier.weight(1f))
            Button(
                onClick = onDone, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
                contentPadding = PaddingValues(horizontal = 22.dp), modifier = Modifier.height(40.dp),
            ) { Text("Done", style = MaterialTheme.typography.titleSmall) }
        }
    }
}

/** Protect pictures from Solid colors: boxes are drawn on the page above; this panel explains and edits them. */
@Composable
private fun ProtectPanel(vm: ScanViewModel, onDone: () -> Unit) {
    val n = vm.protect.size / 4
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Shield, null, tint = Editor.Accent, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Text("Protect pictures", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            Text(if (n == 0) "none yet" else if (n == 1) "1 box" else "$n boxes", style = MaterialTheme.typography.labelLarge, color = Editor.Muted)
        }
        Text(
            "Drag a box over each photo, logo or drawing: Solid colors leaves what is inside as it is. Tap a box to remove it.",
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 6.dp),
        )
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Undo", onClick = { vm.undoProtect() }, enabled = n > 0)
            EditorChip("Remove all", onClick = { vm.clearProtect() }, enabled = n > 0)
            Spacer(Modifier.weight(1f))
            Button(
                onClick = onDone, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
                contentPadding = PaddingValues(horizontal = 22.dp), modifier = Modifier.height(40.dp),
            ) { Text("Done", style = MaterialTheme.typography.titleSmall) }
        }
    }
}

/**
 * The lines the unwarp straightened, drawn on the page: tap one it got wrong (a logo, a curved
 * drawing, a second sheet) to ignore it, or drag a box over an area; then flatten again. The
 * unwarp can also be turned off (perspective correction only).
 */
@Composable
private fun LinesPanel(vm: ScanViewModel, onClose: () -> Unit) {
    val n = vm.ignore.size / 4
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Straighten, null, tint = Editor.Accent, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Text("Unwarp lines", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            EditorChip(
                if (vm.perspectiveOnly) "Unwarp off" else "Unwarp on", role = Role.Switch, selected = !vm.perspectiveOnly,
                onClick = { vm.reflatten(perspective = !vm.perspectiveOnly) },
            )
        }
        Text(
            when {
                vm.perspectiveOnly -> "Only the perspective is corrected. Turn the unwarp on to straighten curved lines."
                vm.lines.isEmpty() -> "No lines were found on this page: only its outline flattens it."
                else -> "Tap a line that should not be straightened (a logo, a drawing, a second sheet), or drag a box over an area. Then flatten again."
            },
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 6.dp),
        )
        if (!vm.perspectiveOnly && vm.lines.isNotEmpty()) {
            Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
                LegendDot(Color(0xFF8FB0FF), "Ruled lines")
                LegendDot(Color(0xFF5BD18B), "Text rows")
                LegendDot(Color(0xFFFF6B5E), "Ignored")
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Use all lines", onClick = { vm.clearIgnore() }, enabled = n > 0 && !vm.perspectiveOnly)
            Spacer(Modifier.weight(1f))
            if (vm.ignoreChanged) {
                TextButton(onClick = onClose) { Text("Cancel") }
                Button(
                    onClick = { vm.reflatten() }, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
                    contentPadding = PaddingValues(horizontal = 18.dp), modifier = Modifier.height(40.dp),
                ) { Text("Flatten again", style = MaterialTheme.typography.titleSmall) }
            } else {
                Button(
                    onClick = onClose, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
                    contentPadding = PaddingValues(horizontal = 22.dp), modifier = Modifier.height(40.dp),
                ) { Text("Done", style = MaterialTheme.typography.titleSmall) }
            }
        }
    }
}

@Composable
private fun LegendDot(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(width = 14.dp, height = 4.dp).clip(RoundedCornerShape(2.dp)).background(color))
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = Editor.Inactive)
    }
}

@Composable
private fun editorSliderColors() = SliderDefaults.colors(
    thumbColor = Editor.Accent, activeTrackColor = Editor.Accent, inactiveTrackColor = Editor.Outline,
    disabledThumbColor = Editor.Muted, disabledActiveTrackColor = Editor.Muted.copy(alpha = 0.5f), disabledInactiveTrackColor = Editor.Divider,
)

@Composable
private fun WrinkleLegend(maxPx: Float, ready: Boolean) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 4.dp)) {
        Box(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp)).background(Brush.horizontalGradient(TurboStops)))
        Row(Modifier.fillMaxWidth().padding(top = 2.dp)) {
            Text("0", style = MaterialTheme.typography.labelSmall)
            Text(
                if (ready) "how far each area was moved" else "computing...",
                style = MaterialTheme.typography.labelSmall, color = Editor.Muted,
                textAlign = TextAlign.Center, modifier = Modifier.weight(1f),
            )
            Text(if (ready) formatPx(maxPx) else "", style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
private fun CompareLabel(text: String) {
    Text(
        text, style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold), color = Color(0xFFE3E3E0),
        modifier = Modifier.background(Editor.Panel.copy(alpha = 0.85f), RoundedCornerShape(16.dp)).padding(horizontal = 14.dp, vertical = 7.dp),
    )
}

private fun formatPx(v: Float): String = if (v < 10f) "%.1f px".format(v) else "${v.roundToInt()} px"
