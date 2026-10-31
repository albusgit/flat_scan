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
import androidx.compose.foundation.lazy.rememberLazyListState
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
import androidx.compose.material.icons.outlined.Contrast
import androidx.compose.material.icons.outlined.DoneAll
import androidx.compose.material.icons.outlined.Collections
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Crop
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material.icons.outlined.SwapHoriz
import androidx.compose.material.icons.outlined.Tune
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
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.SizeTransform
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.FastOutLinearInEasing
import androidx.compose.ui.unit.IntSize
import androidx.compose.animation.togetherWith
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import kotlin.math.PI
import kotlin.math.sin
import androidx.compose.material.icons.outlined.TouchApp
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.foundation.text.BasicText
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.material.icons.outlined.Brush
import androidx.compose.material.icons.outlined.LayersClear
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material.icons.automirrored.outlined.RotateLeft
import androidx.compose.material.icons.outlined.Cached
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.PostAdd
import androidx.compose.material3.Switch
import androidx.compose.runtime.saveable.rememberSaveable
import com.albus.flatscan.ShadowMode
import com.albus.flatscan.CaptureMode
import com.albus.flatscan.FilterKind
import com.albus.flatscan.ImportSource
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.ScreenMode
import com.albus.flatscan.Screen
import com.albus.flatscan.countDrawnLines
import com.albus.flatscan.PageTransform
import com.albus.flatscan.paperColor
import androidx.compose.material.icons.outlined.Transform
import androidx.compose.material.icons.outlined.Draw
import androidx.compose.material3.LocalContentColor
import androidx.compose.material.icons.outlined.Restore
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.ui.layout.SubcomposeLayout
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import kotlin.math.max
import kotlin.math.min
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextOverflow
import java.util.Locale
import kotlin.math.roundToInt

private enum class Sheet { None, Save, Share }

/**
 * The five tools along the bottom, one job each (the design canvas "FlatScan editing tools and export, round 2"):
 * Filters (and the kind of page), Adjust (light and colour), Clean up (shadows, fingers, marks, solid colours), Shape
 * (flattening, AreaMode.Lines, and perspective) and Crop (Page: the outline, turns, the page's size).
 */
private enum class ResultTab { Look, Adjust, Clean, Flatten, Page }

/** What the editor panel shows: a tab's tools, or the tool open on the page. */
private enum class PanelKind { Look, Adjust, Clean, Page, Lines, Fingers, Erase, Protect, Perspective }

/** Page > Perspective's sliders, as Lightroom's Transform panel (PageTransform). */
private enum class TransformTool(val label: String, val caption: String) {
    Vertical("Vertical", "Tilts the page top to bottom: leaning sides made parallel"),
    Horizontal("Horizontal", "Tilts the page left to right: fanning text rows made parallel"),
    Rotate("Rotate", "Turns the page, up to 10 degrees each way"),
    Aspect("Aspect", "Wider or taller"),
    Scale("Scale", "Zooms the page in or out"),
    OffsetX("X offset", "Moves the page left or right"),
    OffsetY("Y offset", "Moves the page up or down"),
}

/** A flatten the iron sweeps across the page before it starts ([perspective] as in [ScanViewModel.reflatten]). */
private class IronRequest(val perspective: Boolean?)

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
    // the finger tool's smart select (a tap selects the finger) instead of the brush
    var smartSelect by remember { mutableStateOf(false) }
    // the line editor's Add line: a drag draws a line along a rule the unwarp missed
    var drawingLine by remember { mutableStateOf(false) }
    // the line editor's brush (-1 none, 0 Straighten, 1 Keep as is) and its radius (page-width fraction)
    var lineBrush by remember { mutableIntStateOf(-1) }
    var lineBrushRadius by remember { mutableFloatStateOf(0.03f) }
    // Shape > Perspective: open, drawing Guided's guides, and the corner editor's corners while it is open
    var perspectiveOpen by remember { mutableStateOf(false) }
    var guiding by remember { mutableStateOf(false) }
    var cornerQuad by remember { mutableStateOf<List<Float>?>(null) }
    var tool by remember {
        // back from flattening again in the line editor: show the new lines
        mutableStateOf(if (vm.reopenLines) AreaMode.Lines else AreaMode.None).also { vm.reopenLines = false }
    }
    // the tab it was left on when this page comes back from processing again (Remove shadows, Flatten again), else Look
    var tab by rememberSaveable {
        val kept = vm.resultTab?.takeIf { it.first == vm.currentId }?.let { k -> ResultTab.entries.firstOrNull { it.name == k.second } }
        mutableStateOf(kept ?: if (vm.adjustOpen) ResultTab.Adjust else ResultTab.Look)
    }
    LaunchedEffect(tab, vm.currentId) { vm.resultTab = vm.currentId to tab.name }
    // counts taps on the tab already open: its panel starts again at its first level (Adjust's first tool)
    var tabReset by remember { mutableIntStateOf(0) }
    val exporter = rememberExportActions(vm)
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    // a new page the way this one came in: the camera, or the photo picker
    val addPage = {
        if (vm.lastSource == ImportSource.Gallery) pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        else vm.startCamera()
    }
    // "Flatten again": the iron sweeps across the page, then the processing screen takes over
    var ironing by remember { mutableStateOf<IronRequest?>(null) }
    val sweep = remember { Animatable(0f) }
    val reduced = rememberReducedMotion()
    LaunchedEffect(ironing) {
        val r = ironing ?: return@LaunchedEffect
        sweep.snapTo(0f)
        if (!reduced) sweep.animateTo(1f, tween(750, easing = LinearEasing))
        ironing = null
        vm.reflatten(r.perspective)
    }
    val iron = { perspective: Boolean? -> if (ironing == null) ironing = IronRequest(perspective) }
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
        drawingLine = false
        lineBrush = -1
    }
    val closePerspective = {
        perspectiveOpen = false
        guiding = false
        cornerQuad = null
    }
    // the Shape tab: Flatten (the line tool) or Perspective; line edits not flattened yet stay while the two are switched
    // and are dropped when the tab is left (as the line tool's Cancel)
    val inShape = mode == AreaMode.Lines || perspectiveOpen
    val showShape = { perspective: Boolean ->
        if (perspective) {
            tool = AreaMode.None
            drawingLine = false
            lineBrush = -1
            perspectiveOpen = true
        } else {
            closePerspective()
            tool = AreaMode.Lines
        }
    }
    val leaveShape = {
        closePerspective()
        closeTool()
        vm.revertIgnore()
    }
    BackHandler(enabled = mode != AreaMode.None) { closeTool() }
    BackHandler(enabled = mode == AreaMode.None && perspectiveOpen) { if (cornerQuad != null) cornerQuad = null else leaveShape() }
    // the page's perspective as shown (none while an area tool edits the page as the core rendered it, or the corners
    // are dragged over it), and the paper it bares
    val perspective = remember(look.transform, shownHeight, vm.pageWidth) {
        if (vm.pageWidth > 0) look.transform.matrix(shownHeight.toFloat() / vm.pageWidth) else null
    }
    val shownPerspective = if (mode == AreaMode.None && cornerQuad == null) perspective else null
    val paper = remember(vm.page) { vm.page?.let { Color(paperColor(it)) } ?: Color.White }
    // Guided's guides where they are on the page shown
    val shownGuides = remember(look.transform, perspective) {
        look.transform.guides.chunked(4).filter { it.size == 4 }.map { g ->
            val m = perspective
            if (m == null) g.toFloatArray()
            else {
                val (ax, ay) = PageTransform.apply(m, g[0].toDouble(), g[1].toDouble())
                val (bx, by) = PageTransform.apply(m, g[2].toDouble(), g[3].toDouble())
                floatArrayOf(ax.toFloat(), ay.toFloat(), bx.toFloat(), by.toFloat())
            }
        }
    }

    val topBar: @Composable () -> Unit = {
        Row(
            Modifier.fillMaxWidth().height(64.dp).padding(start = 8.dp, end = 12.dp, top = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { if (mode != AreaMode.None) closeTool() else vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            // the page count opens the document's pages (reorder, select, delete, add): "1 / 3" and a chevron, in the
            // room the buttons leave (weighted: measured after them, so with large text it narrows, not Export)
            Box(Modifier.weight(1f)) {
                Row(
                    Modifier.clip(RoundedCornerShape(8.dp))
                        .clickable(role = Role.Button, onClickLabel = "Show all pages") { vm.open(Screen.Pages) }
                        .padding(start = 4.dp, end = 0.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    // "2/3" (a single page "1/1"): about 50 dp is left beside the buttons, so smaller rather than cut
                    // ("Pag"); the chevron keeps its place
                    val st = MaterialTheme.typography.titleMedium
                    BasicText(
                        if (index >= 0) "${index + 1}/${vm.pages.size}" else "1/1", style = st.copy(color = LocalContentColor.current),
                        maxLines = 1, softWrap = false, modifier = Modifier.weight(1f, fill = false),
                        autoSize = TextAutoSize.StepBased(minFontSize = 9.sp, maxFontSize = st.fontSize),
                    )
                    Icon(Icons.Outlined.ExpandMore, null, Modifier.size(16.dp), tint = Editor.Muted)
                }
            }
            // undo / redo of every edit that does not re-run the geometry (filters, adjust, rotate, areas): always
            // here, dimmed when there is nothing to undo; the area tools have their own undo in their panels
            val editing = mode == AreaMode.None
            IconButton(onClick = { vm.undo() }, enabled = editing && vm.history.canUndo, modifier = Modifier.width(40.dp)) {
                Icon(Icons.AutoMirrored.Outlined.Undo, "Undo")
            }
            IconButton(onClick = { vm.redo() }, enabled = editing && vm.history.canRedo, modifier = Modifier.width(40.dp)) {
                Icon(Icons.AutoMirrored.Outlined.Redo, "Redo")
            }
            Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreVert, "More") }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    // here rather than in the top bar, which keeps room for the page count and Export
                    if (vm.pending.isEmpty()) {
                        DropdownMenuItem(
                            text = { Text("Add a page") },
                            leadingIcon = { Icon(Icons.Outlined.PostAdd, null) },
                            onClick = { menu = false; addPage() },
                        )
                    }
                    DropdownMenuItem(
                        text = { Text("Show wrinkle map") },
                        trailingIcon = { Checkbox(checked = vm.showWrinkles, onCheckedChange = null) },
                        onClick = { vm.setWrinkles(!vm.showWrinkles); menu = false },
                    )
                    DropdownMenuItem(text = { Text("Processing details") }, onClick = { showDetails = true; menu = false })
                }
            }
            Spacer(Modifier.width(4.dp))
            // pressed into its ink shadow, as the app's other primary buttons
            InkPressable(
                onClick = { sheet = Sheet.Save }, enabled = ready, shape = ChipShape,
                container = Editor.Text, contentColor = Editor.Ground, shadow = Color.Black,
                modifier = Modifier.height(40.dp),
            ) {
                // Export opens the sheet that both saves and shares
                Row(Modifier.padding(start = 10.dp, end = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                    if (exporter.busy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp, color = Editor.Ground)
                    else Icon(PaperIcons.Export, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    // one line: the page count beside it gives way first
                    Text("Export", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), maxLines = 1, softWrap = false)
                }
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
                        mode == AreaMode.Fingers -> if (smartSelect) vm.smartSelectFinger(x, y, brushAdd)
                        mode == AreaMode.Lines && line >= 0 && vm.lines.getOrNull(5 * line)?.toInt() == 3 -> vm.removeUserLineNear(x, y)
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
                brushRadius = if (mode == AreaMode.Lines) lineBrushRadius else brushRadius,
                onStroke = { pts ->
                    when {
                        mode == AreaMode.Lines && lineBrush >= 0 -> vm.addLineStroke(lineBrush == 0, lineBrushRadius, pts)
                        mode == AreaMode.Lines -> vm.traceUserLine(pts)
                        mode == AreaMode.None && guiding -> vm.addGuide(pts[0], pts[1], pts[pts.size - 2], pts[pts.size - 1])
                        else -> vm.addFingerStroke(brushAdd, brushRadius, pts)
                    }
                },
                drawStrokes = (mode == AreaMode.Fingers && !smartSelect) || (mode == AreaMode.Lines && (drawingLine || lineBrush >= 0)) ||
                    (mode == AreaMode.None && perspectiveOpen && guiding && cornerQuad == null),
                pendingLines = if (mode == AreaMode.Lines) vm.pendingUserLines else emptyList(),
                linePaint = if (mode == AreaMode.Lines && !vm.perspectiveOnly) vm.linePaint else emptyList(),
                lineBrush = if (mode == AreaMode.Lines) lineBrush else -1,
                paintHw = if (vm.pageWidth > 0) vm.pageHeight.toFloat() / vm.pageWidth else 1f,
                transform = shownPerspective,
                paper = paper,
                grid = mode == AreaMode.None && perspectiveOpen && cornerQuad == null,
                onSwipe = if (mode == AreaMode.None && compare == null && !perspectiveOpen && vm.pages.size > 1) { step -> vm.openAdjacentPage(step) } else null,
                guides = if (mode == AreaMode.None && perspectiveOpen && cornerQuad == null && (guiding || look.transform.upright == 4)) shownGuides else null,
                cornerQuad = if (mode == AreaMode.None && perspectiveOpen) cornerQuad else null,
                onCornerQuad = { cornerQuad = it },
                fadeKey = p.filter,
            )
            // shown a few seconds on each page until the first hold: it covered the bottom of the page all the time
            var hintShown by remember(vm.currentId) { mutableStateOf(true) }
            LaunchedEffect(vm.currentId) { kotlinx.coroutines.delay(4500); hintShown = false }
            androidx.compose.animation.AnimatedVisibility(
                mode == AreaMode.None && vm.showHoldHint && hintShown && vm.page != null && compare == null,
                Modifier.align(Alignment.BottomCenter), enter = androidx.compose.animation.fadeIn(), exit = androidx.compose.animation.fadeOut(),
            ) {
                Text(
                    "Hold to see the original",
                    style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold), color = Color(0xFFE3E3E0),
                    maxLines = 1, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.padding(bottom = 10.dp)
                        // dark, as the compare button: a light pill vanished on a white page
                        .background(Editor.Panel.copy(alpha = 0.88f), ChipShape)
                        .padding(horizontal = 14.dp, vertical = 7.dp),
                )
            }
            // the nets disagree on an edge of this page: the corner screen is one tap away
            if (mode == AreaMode.None && vm.edgeDoubt && vm.page != null && compare == null) {
                Row(
                    Modifier.align(Alignment.TopStart).padding(6.dp)
                        .background(Editor.Panel.copy(alpha = 0.92f), ChipShape)
                        .clickable(onClickLabel = "Adjust the corners") { vm.recrop() }
                        .padding(start = 12.dp, end = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(Icons.Outlined.Crop, null, tint = Editor.Accent, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "Check the edges",
                        style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold), color = Color(0xFFE3E3E0),
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                    )
                    IconButton(onClick = { vm.dismissEdgeDoubt() }, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.Outlined.Close, "Dismiss", tint = Color(0xFFE3E3E0), modifier = Modifier.size(16.dp))
                    }
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
            if (ironing != null) IronSweep(sweep.value, Modifier.matchParentSize())
            if (compare != null && mode == AreaMode.None) {
                Row(Modifier.align(Alignment.BottomCenter).padding(bottom = 10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    CompareLabel("Original")
                    CompareLabel("Cleaned")
                }
            }
        }
    }
    val shownPanel = when (mode) {
        AreaMode.Protect -> PanelKind.Protect
        AreaMode.Lines -> PanelKind.Lines
        AreaMode.Erase -> PanelKind.Erase
        AreaMode.Fingers -> PanelKind.Fingers
        else -> if (perspectiveOpen) PanelKind.Perspective else when (tab) {
            ResultTab.Adjust -> PanelKind.Adjust
            ResultTab.Clean -> PanelKind.Clean
            ResultTab.Page -> PanelKind.Page
            else -> PanelKind.Look
        }
    }
    if (mode == AreaMode.Fingers) {
        LaunchedEffect(Unit) { vm.refreshFingerOverlay() }
        DisposableEffect(Unit) { onDispose { vm.closeFingerTool() } }
    }
    val panelOf: @Composable ColumnScope.(PanelKind) -> Unit = { k ->
        when (k) {
            PanelKind.Protect -> ProtectPanel(vm, onDone = { tool = AreaMode.None })
            PanelKind.Lines -> LinesPanel(
                vm, onClose = closeTool, onIron = iron, ironing = ironing != null,
                drawing = drawingLine, onDrawing = { drawingLine = it; if (it) lineBrush = -1 },
                brush = lineBrush, onBrush = { lineBrush = it; if (it >= 0) drawingLine = false },
                brushRadius = lineBrushRadius, onBrushRadius = { lineBrushRadius = it }, onPerspective = { showShape(true) },
            )
            PanelKind.Erase -> ErasePanel(vm, onDone = { tool = AreaMode.None })
            PanelKind.Fingers -> FingerPanel(
                vm, brushAdd, { brushAdd = it }, brushRadius, { brushRadius = it }, smartSelect, { smartSelect = it }, onDone = { tool = AreaMode.None },
            )
            PanelKind.Adjust -> key(tabReset) { AdjustPanel(vm) }
            PanelKind.Clean -> CleanPanel(vm, onErase = { tool = AreaMode.Erase }, onFingers = { tool = AreaMode.Fingers }, onProtect = { tool = AreaMode.Protect })
            PanelKind.Page -> PagePanel(vm)
            PanelKind.Perspective -> PerspectivePanel(
                vm, guiding = guiding, onGuiding = { guiding = it; if (it) cornerQuad = null },
                corners = cornerQuad, onCorners = { cornerQuad = it; if (it != null) guiding = false }, onDone = leaveShape,
                onFlatten = { showShape(false) },
            )
            PanelKind.Look -> FilterPanel(vm)
        }
    }
    val dock: @Composable () -> Unit = {
        ToolDock(if (inShape) ResultTab.Flatten else tab) { t ->
            if (t == ResultTab.Flatten) {
                // Shape again: back to its first level (Flatten, no brush or line drawing); the line edits stay
                if (inShape) {
                    closePerspective()
                    drawingLine = false
                    lineBrush = -1
                    tool = AreaMode.Lines
                } else {
                    if (mode != AreaMode.None) closeTool()
                    showShape(false)
                }
            } else {
                // a tool open over a tab (fingers, erase, protect) closes with it, whichever tab is picked: the same tab
                // again goes back to its first level
                if (inShape) leaveShape() else if (mode != AreaMode.None) closeTool()
                if (t == tab) tabReset++
                tab = t
                vm.openAdjust(t == ResultTab.Adjust)
            }
        }
    }

    BoxWithConstraints(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))) {
        // landscape phones and tablets: the panel beside the page (below it, the page would get a sliver)
        val wide = maxWidth > maxHeight && maxWidth >= 600.dp
        val panelCap = maxHeight * 0.5f
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
                            .clip(Editor.PanelShape).background(Editor.Panel).navigationBarsPadding(),
                    ) {
                        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) { panelOf(shownPanel) }
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
                    ToolPanel(shownPanel, cap = panelCap, panel = panelOf)
                    dock()
                }
            }
        }
    }

    if (sheet != Sheet.None) {
        ExportSheet(vm, sheet == Sheet.Share, vm.pages.size, vm.docTitle, onRename = { vm.renameDocument(it) }, onSave = { exporter.save(it) }, onShare = { exporter.share(it) }, onDismiss = { sheet = Sheet.None })
    }
    if (showDetails) {
        PaperDialog(
            onDismissRequest = { showDetails = false },
            confirmButton = { TextButton(onClick = { showDetails = false }) { Text("Close") } },
            title = { Text("Processing details") },
            text = { Text(vm.details, style = MaterialTheme.typography.bodySmall, modifier = Modifier.verticalScroll(rememberScrollState())) },
        )
    }
}

/**
 * The editor panel at the height of what it shows, as photo editors do: the page above re-fits as the panel
 * grows or shrinks (the dock stays put), so a short panel (Look, Clean up) leaves no empty space. A change of tab
 * or tool fades through (the old controls out in 90 ms, the new ones in, rising 8 dp) while the height follows on a
 * critically damped spring, which also takes a quick second tap in its stride; a change inside a panel (Levels,
 * a row that appears) animates its height too. At most [cap], then the panel scrolls. Reduced motion: no animation.
 */
@Composable
private fun ToolPanel(shown: PanelKind, cap: Dp, panel: @Composable ColumnScope.(PanelKind) -> Unit) {
    val reduced = rememberReducedMotion()
    val rise = with(LocalDensity.current) { 8.dp.roundToPx() }
    val size = spring<IntSize>(dampingRatio = Spring.DampingRatioNoBouncy, stiffness = Spring.StiffnessMediumLow)
    // with animations off the panel changes outright (AnimatedContent would draw both panels for a frame)
    if (reduced) {
        key(shown) {
            Column(Modifier.fillMaxWidth().heightIn(max = cap).verticalScroll(rememberScrollState())) { panel(shown) }
        }
        return
    }
    AnimatedContent(
        shown, label = "tool panel", contentAlignment = Alignment.TopCenter,
        modifier = Modifier.fillMaxWidth().heightIn(max = cap),
        transitionSpec = {
            (fadeIn(tween(200, delayMillis = 70, easing = LinearOutSlowInEasing)) +
                slideInVertically(tween(260, delayMillis = 70, easing = LinearOutSlowInEasing)) { rise })
                .togetherWith(fadeOut(tween(90, easing = FastOutLinearInEasing)))
                .using(SizeTransform(clip = true) { _, _ -> size })
        },
    ) { k ->
        Column(
            Modifier.fillMaxWidth().animateContentSize(size).verticalScroll(rememberScrollState()),
        ) { panel(k) }
    }
}

/** The five tool tabs along the bottom ([ResultTab]); the pages open from the page count in the top bar. */
@Composable
private fun ToolDock(selected: ResultTab, onSelect: (ResultTab) -> Unit) {
    HorizontalDivider(color = Editor.Divider)
    Row(Modifier.fillMaxWidth().selectableGroup().padding(start = 4.dp, end = 4.dp, top = 6.dp, bottom = 10.dp)) {
        for ((t, label, icon) in listOf(
            Triple(ResultTab.Look, "Filters", PaperIcons.Filters),
            Triple(ResultTab.Adjust, "Adjust", Icons.Outlined.Tune),
            Triple(ResultTab.Clean, "Clean up", PaperIcons.Sparkles),
            Triple(ResultTab.Flatten, "Shape", PaperIcons.PageShape),
            Triple(ResultTab.Page, "Crop", Icons.Outlined.Crop),
        )) {
            DockItem(label, icon, Modifier.weight(1f), selected = t == selected) { onSelect(t) }
        }
    }
}

@Composable
/** A tab when [selected] is given, otherwise an action button. */
private fun DockItem(label: String, icon: ImageVector, modifier: Modifier, selected: Boolean? = null, badge: Int = 0, onClick: () -> Unit) {
    val color = if (selected == true) Editor.Accent else Editor.Inactive
    Column(
        modifier.heightIn(min = 58.dp).clip(RoundedCornerShape(8.dp))
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
        ) {
            // the selected tab: a tint of the accent behind its icon
            val pill by animateColorAsState(if (selected == true) Editor.Accent.copy(alpha = 0.16f) else Color.Transparent, tween(160), label = "pill")
            Box(Modifier.size(56.dp, 30.dp).background(pill, CircleShape), contentAlignment = Alignment.Center) {
                Icon(icon, null, tint = color, modifier = Modifier.size(22.dp))
            }
        }
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
    val shape = ChipShape
    // dimmed through its colours (as PanelTile)
    val fg = LocalContentColor.current.let { if (enabled) it else it.copy(alpha = 0.4f) }
    val edge = (if (selected) Editor.Accent else Editor.Outline).let { if (enabled) it else it.copy(alpha = 0.5f) }
    Row(
        Modifier.minimumInteractiveComponentSize().height(36.dp).clip(shape)
            .background(if (selected) Editor.Accent.copy(alpha = 0.18f) else Color.Transparent)
            .border(1.dp, edge, shape)
            .selectable(selected, enabled = enabled, role = role, onClick = onClick)
            .padding(horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, null, Modifier.size(16.dp), tint = fg)
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp), color = fg, maxLines = 1)
    }
}

/** Look: the filter and its strength, one live preview per filter; the look on every page; the next imported photo. */
@Composable
private fun FilterPanel(vm: ScanViewModel) {
    val p = vm.params
    val thumbFilter = remember(vm.look) { vm.look.composeFilter() }
    // live previews first, one per filter: the choice this tab is for
    // opens on the chosen filter (it may be past the screen's edge)
    val strip = rememberLazyListState(initialFirstVisibleItemIndex = (FilterKind.forMode(vm.pageMode).indexOf(p.filter) - 1).coerceAtLeast(0))
    PageKindLine(vm)
    LazyRow(
        state = strip,
        contentPadding = PaddingValues(horizontal = 20.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        modifier = Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 2.dp),
    ) {
        items(FilterKind.forMode(vm.pageMode)) { f ->
            val selected = f == p.filter
            Column(
                // wide enough for "High contrast" (80 dp cut it to "High" in Recursive)
                Modifier.width(92.dp).clip(RoundedCornerShape(6.dp))
                    .selectable(selected, role = Role.RadioButton) { vm.selectFilter(f) },
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                val ring = RoundedCornerShape(6.dp)
                // the selection ring moves to the picked filter (160 ms)
                val ringWidth by animateDpAsState(if (selected) 2.5.dp else 1.dp, tween(160), label = "ring")
                val ringColor by animateColorAsState(if (selected) Editor.Accent else Editor.Divider, tween(160), label = "ring")
                Box(
                    Modifier.size(58.dp, 76.dp)
                        .border(ringWidth, ringColor, ring)
                        .padding(3.dp).clip(RoundedCornerShape(3.dp)).background(Editor.Ground),
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
    Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("Strength", style = MaterialTheme.typography.titleSmall, maxLines = 1, modifier = Modifier.width(92.dp))
        Slider(
            value = p.strength, onValueChange = { vm.setStrength(it) }, onValueChangeFinished = { vm.strengthDone() },
            enabled = p.filter != FilterKind.Original, colors = editorSliderColors(), modifier = Modifier.weight(1f)
                .semantics { contentDescription = "Strength" },
        )
        Text(
            "${(p.strength * 100).roundToInt()}%", style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
            color = Editor.Muted, textAlign = TextAlign.End, modifier = Modifier.width(44.dp),
        )
    }
    if (vm.pages.size > 1 || vm.pending.isNotEmpty()) {
        Row(
            Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // the next imported photo first: the step most often taken from here
            if (vm.pending.isNotEmpty()) {
                EditorChip("Next photo (${vm.nextPendingIndex} of ${vm.pendingTotal})", onClick = { vm.nextPage() }, icon = Icons.AutoMirrored.Filled.ArrowForward, selected = true)
            }
            if (vm.pages.size > 1) {
                // the filter, strength and adjustments of this page on every page of the document
                EditorChip(
                    if (vm.applyingAll) "Applying..." else "Use this look on all ${vm.pages.size} pages", onClick = { vm.applyToAllPages() },
                    icon = Icons.Outlined.DoneAll, enabled = !vm.applyingAll && vm.page != null,
                )
            }
        }
    } else {
        Spacer(Modifier.height(14.dp))
    }
}

/**
 * "THIS PAGE IS  Document": the kind of page as a filled-in form line above the filters it decides (a board is only
 * straightened); its menu switches the kind, which processes the page again. One line, where a row of segments took three.
 */
@Composable
private fun PageKindLine(vm: ScanViewModel) {
    var open by remember { mutableStateOf(false) }
    val kinds = listOf(
        Triple(CaptureMode.Document, "Document", "Text, forms, receipts"),
        Triple(CaptureMode.Whiteboard, "Board", "A whiteboard or a slide"),
        Triple(CaptureMode.Screen, "Screen", "A photo of a monitor or a phone"),
    )
    val icons = mapOf(CaptureMode.Document to PaperIcons.KindDocument, CaptureMode.Whiteboard to PaperIcons.KindBoard, CaptureMode.Screen to PaperIcons.KindScreen)
    val name = kinds.first { it.first == vm.pageMode }.second
    Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("THIS PAGE IS", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono), color = Editor.Muted)
        Spacer(Modifier.width(6.dp))
        Box {
            Row(
                Modifier.clip(RoundedCornerShape(4.dp)).clickable(role = Role.Button) { open = true }
                    .semantics { contentDescription = "Kind of page: $name" }
                    .padding(horizontal = 6.dp, vertical = 4.dp)
                    // written on the form's line
                    .drawBehind {
                        val y = size.height - 2.dp.toPx()
                        drawLine(Editor.Accent.copy(alpha = 0.55f), Offset(0f, y), Offset(size.width, y), 1.5.dp.toPx())
                    },
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(icons.getValue(vm.pageMode), null, Modifier.size(15.dp), tint = Editor.Accent)
                Spacer(Modifier.width(5.dp))
                Text(name, style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold), color = Editor.Accent)
                Icon(Icons.Outlined.ExpandMore, null, Modifier.size(16.dp), tint = Editor.Accent)
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                for ((m, label, what) in kinds) {
                    DropdownMenuItem(
                        text = {
                            Column {
                                Text(label, fontWeight = if (m == vm.pageMode) FontWeight.Bold else FontWeight.Normal)
                                Text(what, style = MaterialTheme.typography.bodySmall, color = Editor.Muted)
                            }
                        },
                        leadingIcon = { Icon(icons.getValue(m), null) },
                        trailingIcon = if (m == vm.pageMode) ({ Icon(Icons.Outlined.Check, null, tint = Editor.Accent) }) else null,
                        onClick = { open = false; if (m != vm.pageMode) vm.setPageMode(m) },
                    )
                }
            }
        }
    }
}

/** A row of the Clean up panel that opens a tool: icon, what it does, its state and a chevron. */
@Composable
private fun CleanRow(icon: ImageVector, title: String, caption: String, value: String, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 58.dp).clickable(role = Role.Button, onClick = onClick).padding(horizontal = 18.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        CleanIcon(icon)
        // the state beside the title, so the caption keeps the row's width
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(title, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f, fill = false))
                if (value.isNotEmpty()) Text(value, style = MaterialTheme.typography.labelMedium, color = Editor.Accent, maxLines = 1, modifier = Modifier.padding(start = 8.dp))
            }
            Text(caption, style = MaterialTheme.typography.bodySmall, color = Editor.Muted)
        }
        Icon(Icons.Outlined.ChevronRight, null, tint = Editor.Muted)
    }
}

@Composable
private fun CleanIcon(icon: ImageVector) {
    Box(Modifier.padding(end = 12.dp).size(34.dp).border(1.dp, Editor.Outline, CircleShape), contentAlignment = Alignment.Center) {
        Icon(icon, null, tint = Editor.Accent, modifier = Modifier.size(18.dp))
    }
}

/**
 * A segmented choice of the editor panel ([options]: value to label): a track with the chosen option as a solid pill
 * filling its cell. [wide]: the track takes the width it is given, the options sharing it equally.
 */
@Composable
private fun <T> EditorSegments(
    label: String, options: List<Pair<T, String>>, selected: T, modifier: Modifier = Modifier, wide: Boolean = false,
    icons: Map<T, ImageVector> = emptyMap(), onPick: (T) -> Unit,
) {
    Row(
        modifier.height(40.dp).clip(ChipShape).background(Editor.Raised).border(1.dp, Editor.Outline, ChipShape).padding(3.dp)
            .selectableGroup().semantics { contentDescription = label },
    ) {
        for ((v, text) in options) {
            val on = v == selected
            val fill by animateColorAsState(if (on) Editor.Accent else Color.Transparent, label = "segment")
            Box(
                (if (wide) Modifier.weight(1f) else Modifier).fillMaxHeight().clip(SegmentShape).background(fill)
                    .selectable(on, role = Role.RadioButton) { onPick(v) }.padding(horizontal = if (icons.isEmpty()) 14.dp else 6.dp),
                contentAlignment = Alignment.Center,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    icons[v]?.let {
                        Icon(it, null, Modifier.size(15.dp), tint = if (on) Editor.Ground else Editor.Text)
                        Spacer(Modifier.width(5.dp))
                    }
                    BasicText(
                        text, maxLines = 1, softWrap = false,
                        style = MaterialTheme.typography.labelLarge.copy(
                            fontSize = 13.sp, fontWeight = if (on) FontWeight.ExtraBold else FontWeight.Medium, color = if (on) Editor.Ground else Editor.Text,
                        ),
                        autoSize = TextAutoSize.StepBased(minFontSize = 9.sp, maxFontSize = 13.sp),
                    )
                }
            }
        }
    }
}

/** A segment inside the [ChipShape] track (3 dp in): its corners follow the track's. */
private val SegmentShape = RoundedCornerShape(5.dp)

/**
 * Clean up: one tile per job, its state under its name (as the Crop tab's tiles): Shadows and Solid colours switch on
 * and off where they are, Fingers, Erase and Protect open their tool on the page.
 */
@Composable
private fun CleanPanel(vm: ScanViewModel, onErase: () -> Unit, onFingers: () -> Unit, onProtect: () -> Unit) {
    val p = vm.params
    val doc = vm.pageMode == CaptureMode.Document
    val solidOk = p.filter != FilterKind.BlackWhite
    val ne = vm.erase.size / 4
    val np = vm.protect.size / 4
    Column(Modifier.fillMaxWidth().padding(start = 18.dp, end = 18.dp, top = 14.dp, bottom = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (doc) ShadowControls(vm)
        Row(Modifier.height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (doc) {
                CleanTile(
                    "Fingers", if (!vm.fingersAuto) "Off" else if (vm.fingerStrokes.isNotEmpty()) "Edited" else "Auto", Icons.Outlined.TouchApp,
                    Modifier.weight(1f).fillMaxHeight(), onClick = onFingers,
                )
            }
            CleanTile("Erase", if (ne == 0) "None" else if (ne == 1) "1 area" else "$ne areas", Icons.Outlined.AutoFixHigh, Modifier.weight(1f).fillMaxHeight(), onClick = onErase)
        }
        Row(Modifier.height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CleanTile(
                "Solid colours", if (!solidOk) "Not with B&W" else if (p.solid) "On" else "Off", Icons.Outlined.GridView,
                Modifier.weight(1f).fillMaxHeight(), on = p.solid && solidOk, toggle = true, enabled = solidOk,
            ) { vm.setSolid(!p.solid) }
            // the pictures Solid colours leaves alone: only with it on
            CleanTile(
                "Protect", if (np == 0) "None" else if (np == 1) "1 box" else "$np boxes", Icons.Outlined.Shield,
                Modifier.weight(1f).fillMaxHeight(), enabled = p.solid && solidOk, onClick = onProtect,
            )
        }
        // no screen switch here: on real photos of screens its descreen changed nothing visible (a pixel grid's moire
        // shows only in close-ups); it stays on Auto, and screens are a kind of page (Look)
    }
}

/**
 * Clean up > Shadows: Off, Auto or Manual (the shadows are found automatically in each); Manual adds how much of the
 * shadow is removed and how much the ink that lay in it is deepened. Off and back process the page again; the rest
 * is a render of the page as it is.
 */
@Composable
private fun ShadowControls(vm: ScanViewModel) {
    val p = vm.params
    Column(Modifier.fillMaxWidth().padding(bottom = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Contrast, null, Modifier.size(18.dp), tint = Editor.Muted)
            Spacer(Modifier.width(8.dp))
            Text("Shadows", style = MaterialTheme.typography.titleSmall, modifier = Modifier.width(80.dp))
            EditorSegments(
                "Shadows", listOf(ShadowMode.Off to "Off", ShadowMode.Auto to "Auto", ShadowMode.Manual to "Manual"), p.shadow,
                Modifier.weight(1f), wide = true,
            ) { vm.setShadowMode(it) }
        }
        if (p.shadow == ShadowMode.Manual) {
            ShadowSlider("Remove", p.shadowAmount) { vm.setShadowLevels(amount = it) }
            ShadowSlider("Contrast", p.shadowContrast) { vm.setShadowLevels(contrast = it) }
        }
    }
}

@Composable
private fun ShadowSlider(label: String, value: Float, onChange: (Float) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.labelLarge, maxLines = 1, modifier = Modifier.width(88.dp).padding(start = 26.dp))
        Slider(
            value = value, onValueChange = onChange, colors = editorSliderColors(),
            modifier = Modifier.weight(1f).semantics { contentDescription = "Shadow $label" },
        )
        Text(
            "${(value * 100).roundToInt()}%", style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold),
            color = Editor.Muted, textAlign = TextAlign.End, modifier = Modifier.width(44.dp),
        )
    }
}

/**
 * A Clean up tile: an icon over its name and its state. [toggle]: a switch, lit in the accent while [on]; otherwise it
 * opens its tool (a chevron beside the state).
 */
@Composable
private fun CleanTile(
    label: String, state: String, icon: ImageVector, modifier: Modifier, on: Boolean = false, toggle: Boolean = false, enabled: Boolean = true,
    onClick: () -> Unit,
) {
    val fg = (if (on) Editor.Accent else LocalContentColor.current).let { if (enabled) it else it.copy(alpha = 0.4f) }
    val edge = (if (on) Editor.Accent else Editor.Outline).let { if (enabled) it else it.copy(alpha = 0.5f) }
    Column(
        modifier.heightIn(min = 72.dp).clip(ChipShape)
            .background(if (on) Editor.Accent.copy(alpha = 0.16f) else Color.Transparent)
            .border(1.dp, edge, ChipShape)
            .then(
                if (toggle) Modifier.toggleable(on, enabled = enabled, role = Role.Switch) { onClick() }
                else Modifier.clickable(enabled = enabled, role = Role.Button, onClick = onClick),
            )
            .padding(horizontal = 6.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Icon(icon, null, Modifier.size(20.dp), tint = fg)
        Spacer(Modifier.height(4.dp))
        Text(label, style = MaterialTheme.typography.labelLarge, maxLines = 1, overflow = TextOverflow.Ellipsis, textAlign = TextAlign.Center, color = fg)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                state, style = MaterialTheme.typography.labelSmall, maxLines = 1, overflow = TextOverflow.Ellipsis,
                color = if (enabled) Editor.Muted else Editor.Muted.copy(alpha = 0.5f), modifier = Modifier.weight(1f, fill = false),
            )
            if (!toggle) Icon(Icons.Outlined.ChevronRight, null, Modifier.size(14.dp), tint = Editor.Muted)
        }
    }
}

/**
 * A tool tile of the editor panels (the Page panel's page actions, the Flatten panel's line tools): an icon over its
 * name; [selected] (a tool that stays on) fills it with a tint of the accent, a disabled one dims.
 */
@Composable
private fun PanelTile(
    label: String,
    icon: ImageVector,
    modifier: Modifier,
    selected: Boolean = false,
    enabled: Boolean = true,
    role: Role = Role.Button,
    onClick: () -> Unit,
) {
    // dimmed through its colours, not a layer's alpha: the outline under a layer whose alpha changed was not drawn again
    val fg = (if (selected) Editor.Accent else LocalContentColor.current).let { if (enabled) it else it.copy(alpha = 0.4f) }
    val edge = (if (selected) Editor.Accent else Editor.Outline).let { if (enabled) it else it.copy(alpha = 0.5f) }
    Column(
        modifier.heightIn(min = 64.dp).clip(ChipShape)
            .background(if (selected) Editor.Accent.copy(alpha = 0.18f) else Color.Transparent)
            .border(1.dp, edge, ChipShape)
            .selectable(selected, enabled = enabled, role = role, onClick = onClick)
            .padding(horizontal = 4.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Icon(icon, null, Modifier.size(20.dp), tint = fg)
        Spacer(Modifier.height(5.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, maxLines = 2, textAlign = TextAlign.Center, color = fg)
    }
}

/**
 * Page: crop and turns, the page's size, shadows, the two pages of a book spread.
 */
@Composable
private fun PagePanel(vm: ScanViewModel) {
    val l = vm.look
    Column(
        Modifier.fillMaxWidth().padding(start = 18.dp, end = 18.dp, top = 14.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            PanelTile("Outline", Icons.Outlined.Crop, Modifier.weight(1f).fillMaxHeight()) { vm.recrop() }
            PanelTile("Turn left", Icons.AutoMirrored.Outlined.RotateLeft, Modifier.weight(1f).fillMaxHeight()) { vm.rotateResultLeft() }
            PanelTile("Turn right", Icons.AutoMirrored.Outlined.RotateRight, Modifier.weight(1f).fillMaxHeight()) { vm.rotateResult() }
            // there is no automatic 180-degree turn: one tap for an upside-down page
            PanelTile("Turn 180°", Icons.Outlined.Cached, Modifier.weight(1f).fillMaxHeight()) { vm.rotateResult180() }
        }
        Column {
            Text("PAGE SIZE", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono), color = Editor.Muted, modifier = Modifier.padding(bottom = 6.dp))
            ShapeChooser(l) { vm.updateLook(l.copy(shape = it)) }
        }
        if (vm.spreadPartner != null) {
            EditorChip("Swap the two book pages", onClick = { vm.swapSpread() }, icon = Icons.Outlined.SwapHoriz)
        }
    }
}

/** The corner editor's corners when they are the page's own (TL, TR, BR, BL page fractions): no quad. */
private val FullQuad = listOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f)

/**
 * Page > Perspective, as Lightroom's Transform panel: Upright (Off, Auto, Level, Vertical, Full; Guided by guides drawn
 * on the page), the sliders one at a time (as the Adjust tab), Constrain crop, and the corners dragged onto the page's
 * own ([corners]: the corner editor's while it is open, [onCorners] opens, moves and closes it). All on the finished
 * page, no processing again; the top bar's undo takes it back.
 */
@Composable
private fun PerspectivePanel(
    vm: ScanViewModel, guiding: Boolean, onGuiding: (Boolean) -> Unit, corners: List<Float>?, onCorners: (List<Float>?) -> Unit, onDone: () -> Unit,
    onFlatten: () -> Unit,
) {
    val t = vm.look.transform
    var tool by rememberSaveable { mutableStateOf(TransformTool.Vertical) }
    Column(Modifier.fillMaxWidth().padding(top = 14.dp, bottom = 14.dp)) {
        ShapeSwitch(true, Modifier.padding(start = 20.dp, end = 20.dp, bottom = 6.dp)) { onFlatten() }
        Row(Modifier.padding(start = 20.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Perspective", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            TextButton(onClick = { onGuiding(false); vm.setTransform(PageTransform()) }, enabled = t != PageTransform() && corners == null) { Text("Reset") }
        }
        if (corners != null) {
            Text(
                "Drag each corner onto a corner of the page: what is between them becomes the whole page.",
                style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
            )
            Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 8.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SecondaryPill(onClick = { onCorners(null) }, container = Editor.Raised, border = Editor.Outline, modifier = Modifier.weight(1f).height(48.dp)) { PillText("Cancel") }
                PrimaryPill(
                    onClick = { vm.setTransform(t.copy(quad = corners.takeIf { it != FullQuad }, upright = -1)); onCorners(null) },
                    container = Editor.Accent, content = Editor.Ground, modifier = Modifier.weight(1.6f).height(48.dp),
                ) { PillText("Apply") }
            }
            return@Column
        }
        Text(
            "UPRIGHT", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono), color = Editor.Muted,
            modifier = Modifier.padding(start = 20.dp, top = 6.dp, bottom = 6.dp),
        )
        Row(
            Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            for ((m, label) in listOf(-1 to "Off", 3 to "Auto", 0 to "Level", 1 to "Vertical", 2 to "Full", 4 to "Guided")) {
                val on = if (m == 4) guiding || t.upright == 4 else !guiding && t.upright == m
                EditorChip(label, onClick = { onGuiding(m == 4); vm.upright(m) }, selected = on, role = Role.RadioButton)
            }
        }
        if (guiding || t.upright == 4) {
            Row(Modifier.padding(start = 20.dp, end = 8.dp, top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (t.guides.size < 8) "Draw 2 to 4 guides along lines that should be straight up or straight across."
                    else "${t.guides.size / 4} guides. Draw another, or start again.",
                    style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.weight(1f),
                )
                TextButton(onClick = { vm.clearGuides() }, enabled = t.guides.isNotEmpty()) { Text("Clear guides") }
            }
        }
        // the sliders, one at a time
        Text(
            "MANUAL", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono), color = Editor.Muted,
            modifier = Modifier.padding(start = 20.dp, top = 12.dp, bottom = 6.dp),
        )
        Row(
            Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 20.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            for (k in TransformTool.entries) EditorChip(k.label, onClick = { tool = k }, selected = tool == k, role = Role.Tab)
        }
        Box(Modifier.fillMaxWidth().heightIn(min = 76.dp).padding(start = 20.dp, end = 20.dp, top = 4.dp), contentAlignment = Alignment.CenterStart) {
            val v = when (tool) {
                TransformTool.Vertical -> t.vertical
                TransformTool.Horizontal -> t.horizontal
                TransformTool.Rotate -> t.rotate / 10f
                TransformTool.Aspect -> t.aspect
                TransformTool.Scale -> t.scale
                TransformTool.OffsetX -> t.offsetX
                TransformTool.OffsetY -> t.offsetY
            }
            Bipolar(tool.label, v, tool.caption) { x ->
                val c = vm.look.transform
                vm.setTransform(
                    when (tool) {
                        TransformTool.Vertical -> c.copy(vertical = x)
                        TransformTool.Horizontal -> c.copy(horizontal = x)
                        TransformTool.Rotate -> c.copy(rotate = 10f * x)
                        TransformTool.Aspect -> c.copy(aspect = x)
                        TransformTool.Scale -> c.copy(scale = x)
                        TransformTool.OffsetX -> c.copy(offsetX = x)
                        TransformTool.OffsetY -> c.copy(offsetY = x)
                    },
                )
            }
        }
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 20.dp).clip(ChipShape)
                .toggleable(t.constrain, role = Role.Switch) { vm.setTransform(t.copy(constrain = it)) }.padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("Constrain crop", style = MaterialTheme.typography.titleSmall)
                Text("Zooms in just enough that no blank corner shows", style = MaterialTheme.typography.bodySmall, color = Editor.Muted)
            }
            Switch(checked = t.constrain, onCheckedChange = null)
        }
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            SecondaryPill(
                onClick = { onGuiding(false); onCorners(t.quad ?: FullQuad) }, container = Editor.Raised, border = Editor.Outline,
                modifier = Modifier.weight(1f).height(48.dp),
            ) { PillText("Corners") }
            PrimaryPill(onClick = onDone, container = Editor.Accent, content = Editor.Ground, modifier = Modifier.weight(1f).height(48.dp)) { PillText("Done") }
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
    vm: ScanViewModel, add: Boolean, onAdd: (Boolean) -> Unit, radius: Float, onRadius: (Float) -> Unit,
    smart: Boolean, onSmart: (Boolean) -> Unit, onDone: () -> Unit,
) {
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.TouchApp, null, tint = Editor.Accent, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(10.dp))
            Text("Fingers", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            Text(if (vm.fingerShare <= 0f) "none removed" else "%.1f%% of the page".format(100 * vm.fingerShare), style = MaterialTheme.typography.labelLarge, color = Editor.Muted)
        }
        Text(
            when {
                smart && add -> "Tap a finger to remove it."
                smart -> "Tap an area to bring it back."
                add -> "Paint over a finger to remove it."
                else -> "Paint to bring back what was taken."
            } + if (vm.fingersOverText) " Text under a finger can't come back." else "",
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, maxLines = 2, modifier = Modifier.padding(top = 4.dp),
        )
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            EditorSegments("Tool", listOf(true to "Tap", false to "Brush"), smart, Modifier.weight(1f), wide = true, onPick = onSmart)
            EditorSegments("Remove or keep", listOf(true to "Remove", false to "Keep"), add, Modifier.weight(1f), wide = true, onPick = onAdd)
        }
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Fill", style = MaterialTheme.typography.labelLarge, color = Editor.Muted, maxLines = 1, softWrap = false)
            // the paper around it, or the inpainting net (content-aware, on the phone)
            EditorSegments("Fill with", listOf(0 to "Paper", 1 to "Smart"), vm.fingerFill, Modifier.weight(1f), wide = true) { vm.chooseFingerFill(it) }
            EditorChip("Auto", onClick = { vm.setFingersAutoOn(!vm.fingersAuto) }, selected = vm.fingersAuto, role = Role.Switch)
        }
        if (!smart) {
            Row(Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("Size", style = MaterialTheme.typography.labelLarge, color = Editor.Muted, maxLines = 1, softWrap = false)
                Slider(
                    value = radius, onValueChange = onRadius, valueRange = 0.005f..0.06f, colors = editorSliderColors(),
                    modifier = Modifier.weight(1f).padding(start = 8.dp).semantics { contentDescription = "Brush size" },
                )
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Undo", onClick = { vm.undoFingerStroke() }, enabled = vm.fingerStrokes.isNotEmpty())
            EditorChip("Clear", onClick = { vm.clearFingerStrokes() }, enabled = vm.fingerStrokes.isNotEmpty())
            Spacer(Modifier.weight(1f))
            Button(
                onClick = onDone, shape = ChipShape, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
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
            "Drag a box over a mark to erase it. Tap a box to undo it.",
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 6.dp),
        )
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Undo", onClick = { vm.undoErase() }, enabled = n > 0)
            EditorChip("Restore all", onClick = { vm.clearErase() }, enabled = n > 0)
            Spacer(Modifier.weight(1f))
            Button(
                onClick = onDone, shape = ChipShape, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
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
            "Box the photos and logos to keep their colours. Tap a box to remove it.",
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 6.dp),
        )
        Row(Modifier.fillMaxWidth().padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            EditorChip("Undo", onClick = { vm.undoProtect() }, enabled = n > 0)
            EditorChip("Remove all", onClick = { vm.clearProtect() }, enabled = n > 0)
            Spacer(Modifier.weight(1f))
            Button(
                onClick = onDone, shape = ChipShape, colors = ButtonDefaults.buttonColors(containerColor = Editor.Accent, contentColor = Editor.Ground),
                contentPadding = PaddingValues(horizontal = 22.dp), modifier = Modifier.height(40.dp),
            ) { Text("Done", style = MaterialTheme.typography.titleSmall) }
        }
    }
}

/**
 * The Flatten tool: the lines the iron straightened, drawn on the page. Tap one it got wrong (a
 * logo, a curved drawing, a second sheet) to ignore it, or drag a box over an area; paint with the
 * line brush where lines should be straightened (Straighten: only there) or left alone (Keep as is:
 * a photo, a drawing); switch the automatic lines off to straighten only what is painted or drawn;
 * then flatten again, which sends the iron across the page ([onIron]). The iron can also be lifted
 * (perspective correction only).
 */
@Composable
private fun LinesPanel(
    vm: ScanViewModel, onClose: () -> Unit, onIron: (Boolean?) -> Unit, ironing: Boolean, drawing: Boolean, onDrawing: (Boolean) -> Unit,
    brush: Int, onBrush: (Int) -> Unit, brushRadius: Float, onBrushRadius: (Float) -> Unit, onPerspective: () -> Unit,
) {
    val ignoredBoxes = vm.ignore.size / 4
    val drawn = countDrawnLines(vm.userLines)
    val ironOn = !vm.perspectiveOnly
    // a board or a screen is flat: the iron stays lifted (the kind of page is chosen in Filters)
    val ironable = vm.pageMode == CaptureMode.Document
    // the page's lines by kind (ruled, text row, table, drawn and flattened) and those ignored (their middle in a box,
    // or left out by the line brush)
    val ignore = vm.ignore.toList()
    val paint = vm.linePaint
    val hw = if (vm.pageWidth > 0) vm.pageHeight.toFloat() / vm.pageWidth else 1f
    val counts = remember(vm.lines, ignore, paint) {
        val c = IntArray(5)
        val l = vm.lines
        var i = 0
        while (5 * i + 4 < l.size) {
            val mx = (l[5 * i + 1] + l[5 * i + 3]) / 2
            val my = (l[5 * i + 2] + l[5 * i + 4]) / 2
            if (inAreas(ignore, mx, my) || leftOutByPaint(paint, mx, my, hw)) c[4]++ else c[l[5 * i].toInt().coerceIn(0, 3)]++
            i++
        }
        c
    }
    val used = counts[0] + counts[1] + counts[2] + counts[3]
    Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 14.dp, bottom = 16.dp)) {
        ShapeSwitch(false, Modifier.padding(bottom = 14.dp)) { if (!ironing) onPerspective() }
        // what the tab does and how it stands, the iron as a switch (as Remove shadows)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(if (ironOn) PaperIcons.Iron else PaperIcons.IronLifted, null, tint = Editor.Accent, modifier = Modifier.size(22.dp))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("Flatten the paper", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold))
                Text(
                    when {
                        !ironOn -> "Iron lifted: perspective only"
                        !vm.autoLines && used == 0 -> "The paper is smoothed flat"
                        used == 0 -> "No lines found on this page"
                        used == 1 -> "1 line guides the iron"
                        else -> "$used lines guide the iron"
                    },
                    style = MaterialTheme.typography.bodySmall, color = Editor.Muted,
                )
            }
            Row(
                Modifier.clip(ChipShape).toggleable(ironOn, enabled = !ironing && ironable, role = Role.Switch) { on ->
                    if (on) onIron(false) else vm.reflatten(perspective = true)
                }.padding(start = 10.dp, top = 4.dp, bottom = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Switch(checked = ironOn, onCheckedChange = null, enabled = !ironing && ironable)
            }
        }
        if (ironOn) {
            // the automatic lines: every printed line found, or only the painted and drawn ones
            Row(
                Modifier.fillMaxWidth().padding(top = 10.dp).clip(ChipShape)
                    .toggleable(vm.autoLines, enabled = !ironing, role = Role.Switch) { vm.setAutoLinesOn(it) }.padding(vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Straighten lines", style = MaterialTheme.typography.titleSmall)
                    Text(
                        if (vm.autoLines) "Every printed line and text row found" else "Only what you paint or draw",
                        style = MaterialTheme.typography.bodySmall, color = Editor.Muted,
                    )
                }
                Switch(checked = vm.autoLines, onCheckedChange = null, enabled = !ironing)
            }
            // the key to the colours on the page, with how many of each
            if (used + counts[4] + drawn > 0) {
                FlowRow(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (counts[0] > 0) LegendDot(LineColors[0], "Ruled", counts[0])
                    if (counts[1] > 0) LegendDot(LineColors[1], "Text rows", counts[1])
                    if (counts[2] > 0) LegendDot(LineColors[2], "Table", counts[2])
                    if (drawn > 0) LegendDot(DrawnLineColor, "Drawn", drawn)
                    if (counts[4] > 0) LegendDot(IgnoreColor, "Ignored", counts[4])
                }
            }
            // one line for the tool in use
            Text(
                when {
                    brush == 0 -> "Paint over the text and rules to straighten: only the lines under the paint are used."
                    brush == 1 -> "Paint over photos, drawings and logos: their edges are left as they are."
                    drawing -> "Sweep roughly along a printed line the iron missed: the line is found and followed. Tap a yellow line to remove it."
                    used == 0 -> "Paint over the text to straighten, or add a line along a printed rule."
                    else -> "Tap a line to leave it out (a logo, a second sheet), drag a box over an area, or paint."
                },
                style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 10.dp),
            )
            // the line tools, always in place (dimmed when there is nothing for them to do)
            Row(Modifier.padding(top = 12.dp).height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PanelTile(
                    "Straighten here", Icons.Outlined.Brush, Modifier.weight(1f).fillMaxHeight(), selected = brush == 0, enabled = !ironing, role = Role.Switch,
                ) { onBrush(if (brush == 0) -1 else 0) }
                PanelTile(
                    "Keep as is", Icons.Outlined.Shield, Modifier.weight(1f).fillMaxHeight(), selected = brush == 1, enabled = !ironing, role = Role.Switch,
                ) { onBrush(if (brush == 1) -1 else 1) }
                PanelTile(
                    "Add a line", Icons.Outlined.Draw, Modifier.weight(1f).fillMaxHeight(), selected = drawing, enabled = !ironing, role = Role.Switch,
                ) { onDrawing(!drawing) }
            }
            if (brush >= 0) {
                Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("Brush size", style = MaterialTheme.typography.labelLarge, color = Editor.Muted)
                    Slider(
                        value = brushRadius, onValueChange = onBrushRadius, valueRange = 0.01f..0.08f, colors = editorSliderColors(),
                        modifier = Modifier.weight(1f).padding(start = 10.dp).semantics { contentDescription = "Brush size" },
                    )
                }
            }
            Row(Modifier.padding(top = 8.dp).height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PanelTile("Undo", Icons.AutoMirrored.Outlined.Undo, Modifier.weight(1f).fillMaxHeight(), enabled = vm.lineUndoSteps > 0 && !ironing) { vm.undoLineEdit() }
                PanelTile("Clear paint", Icons.Outlined.LayersClear, Modifier.weight(1f).fillMaxHeight(), enabled = paint.isNotEmpty() && !ironing) { vm.clearLinePaint() }
                PanelTile("Use all lines", Icons.Outlined.Restore, Modifier.weight(1f).fillMaxHeight(), enabled = ignoredBoxes > 0 && !ironing) { vm.clearIgnore() }
            }
        } else {
            Text(
                if (ironable) "Only the perspective is corrected. Turn the iron on to straighten curved lines."
                else "A ${vm.pageMode.label.lowercase()} is flat: only its perspective is corrected (Perspective has more).",
                style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 10.dp),
            )
        }
        // leave, or flatten with the edits (as the crop screen's Cancel and Flatten page): "Flatten again" comes in with
        // the first edit, and until then there is only Done (a dimmed accent button read as half-enabled)
        Row(Modifier.fillMaxWidth().padding(top = 16.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            SecondaryPill(
                onClick = onClose, enabled = !ironing, container = Editor.Raised, border = Editor.Outline,
                modifier = Modifier.weight(1f).height(48.dp),
            ) { PillText(if (vm.lineEditsChanged) "Cancel" else "Done") }
            if (ironOn && (vm.lineEditsChanged || ironing)) {
                PrimaryPill(
                    onClick = { onIron(null) }, enabled = !ironing, container = Editor.Accent, content = Editor.Ground,
                    modifier = Modifier.weight(1.6f).height(48.dp),
                ) {
                    Icon(PaperIcons.Iron, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    PillText("Flatten again")
                }
            }
        }
    }
}

/** The Shape tab's two halves: Flatten (the iron and its lines) and Perspective (the finished page's Transform). */
@Composable
private fun ShapeSwitch(perspective: Boolean, modifier: Modifier = Modifier, onPick: (Boolean) -> Unit) {
    EditorSegments("Shape", listOf(false to "Flatten", true to "Perspective"), perspective, modifier.fillMaxWidth(), wide = true) {
        if (it != perspective) onPick(it)
    }
}

/** The iron crossing the page at [t] (0..1), leaving the paper behind it pressed. */
@Composable
private fun IronSweep(t: Float, modifier: Modifier) {
    Canvas(modifier) {
        val x = size.width * (t * 1.1f - 0.05f)
        drawRect(Color.White.copy(alpha = 0.10f), size = Size(x.coerceIn(0f, size.width), size.height))
        drawIron(
            Offset(x, size.height * (0.5f + sin(t * PI.toFloat() * 4f) * 0.28f)),
            size.minDimension / 260f,
        )
    }
}

@Composable
private fun LegendDot(color: Color, label: String, count: Int) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(width = 14.dp, height = 4.dp).clip(RoundedCornerShape(2.dp)).background(color))
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = Editor.Inactive)
        Spacer(Modifier.width(4.dp))
        Text("$count", style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono), color = Editor.Muted)
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
        modifier = Modifier.background(Editor.Panel.copy(alpha = 0.85f), ChipShape).padding(horizontal = 14.dp, vertical = 7.dp),
    )
}

private fun formatPx(v: Float): String = if (v < 10f) "%.1f px".format(v) else "${v.roundToInt()} px"
