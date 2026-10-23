package com.albus.flatscan.ui

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AddAPhoto
import androidx.compose.material.icons.outlined.AddPhotoAlternate
import androidx.compose.material.icons.outlined.ChevronLeft
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.DragIndicator
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.MoreHoriz
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.SaveAlt
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutLinearInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.draw.alpha
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.toOffset
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.zIndex
import com.albus.flatscan.DocumentName
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.ExportFormat
import com.albus.flatscan.ScanViewModel

private enum class PagesSheet { None, Save, Share }

/** All pages of the current document: open, delete, reorder (hold and drag), add, rename, save. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PagesScreen(vm: ScanViewModel) {
    val exporter = rememberExportActions(vm)
    var sheet by remember { mutableStateOf(PagesSheet.None) }
    var confirmDelete by remember { mutableStateOf<String?>(null) }
    // the page being crumpled away after its delete was confirmed
    var crumpling by remember { mutableStateOf<String?>(null) }
    var menu by remember { mutableStateOf(false) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    val pickPhotos = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    val n = vm.pages.size
    val ready = n > 0 && !exporter.busy
    val ctx = LocalContext.current
    var docName by remember { mutableStateOf(DocumentName.get(ctx)) }
    var renaming by remember { mutableStateOf(false) }
    val grid = rememberLazyGridState()
    // the page being dragged: the pointer in grid coordinates and where it holds the card
    var dragId by remember { mutableStateOf<String?>(null) }
    var dragPos by remember { mutableStateOf(Offset.Zero) }
    var grab by remember { mutableStateOf(Offset.Zero) }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = {
                Row(
                    Modifier.clip(RoundedCornerShape(6.dp)).clickable(role = Role.Button, onClickLabel = "Rename the document") { renaming = true }
                        .padding(end = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f, fill = false)) {
                        Text(
                            docName ?: "Document", style = MaterialTheme.typography.titleLarge.copy(fontSize = 18.sp),
                            maxLines = 1, overflow = TextOverflow.Ellipsis,
                        )
                        Text(
                            (if (n == 1) "1 page" else "$n pages") + if (n > 1) " · hold and drag to reorder" else "",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1,
                        )
                    }
                    Spacer(Modifier.width(6.dp))
                    Icon(Icons.Outlined.Edit, null, Modifier.size(16.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            },
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            actions = {
                Box {
                    IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreVert, "More") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(
                            text = { Text("Save as…") }, leadingIcon = { Icon(Icons.Outlined.SaveAlt, null) }, enabled = ready,
                            onClick = { sheet = PagesSheet.Save; menu = false },
                        )
                        DropdownMenuItem(
                            text = { Text("Rename") }, leadingIcon = { Icon(Icons.Outlined.Edit, null) },
                            onClick = { renaming = true; menu = false },
                        )
                        DropdownMenuItem(
                            text = { Text("Add page from camera") }, leadingIcon = { Icon(Icons.Outlined.AddAPhoto, null) },
                            onClick = { menu = false; vm.startCamera() },
                        )
                        DropdownMenuItem(
                            text = { Text("Add pages from gallery") }, leadingIcon = { Icon(Icons.Outlined.AddPhotoAlternate, null) },
                            onClick = { menu = false; pickPhotos() },
                        )
                    }
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
        )
        LazyVerticalGrid(
            // two columns on a phone, more on wider screens
            columns = GridCells.Adaptive(150.dp),
            state = grid,
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.weight(1f).fillMaxWidth(),
        ) {
            itemsIndexed(vm.pages, key = { _, id -> id }) { i, id ->
                val dragging = dragId == id
                PageCard(
                    vm, id, i, n,
                    onOpen = { vm.openPage(id) },
                    onDelete = { confirmDelete = id },
                    dragging = dragging,
                    crumpling = crumpling == id,
                    onCrumpled = {
                        vm.deletePage(id)
                        crumpling = null
                    },
                    modifier = Modifier
                        .zIndex(if (dragging) 1f else 0f)
                        .graphicsLayer {
                            if (dragging) {
                                // follows the finger wherever the grid has moved the item to
                                val info = grid.layoutInfo.visibleItemsInfo.firstOrNull { it.key == id }
                                if (info != null) {
                                    translationX = dragPos.x - grab.x - info.offset.x
                                    translationY = dragPos.y - grab.y - info.offset.y
                                }
                                scaleX = 1.04f
                                scaleY = 1.04f
                            }
                        }
                        .pointerInput(id) {
                            detectDragGesturesAfterLongPress(
                                onDragStart = { local ->
                                    val info = grid.layoutInfo.visibleItemsInfo.firstOrNull { it.key == id } ?: return@detectDragGesturesAfterLongPress
                                    grab = local
                                    dragPos = info.offset.toOffset() + local
                                    dragId = id
                                },
                                onDrag = { change, d ->
                                    change.consume()
                                    dragPos += d
                                    val target = grid.layoutInfo.visibleItemsInfo.firstOrNull {
                                        it.key is String && it.key != "add" && it.key != id && Rect(it.offset.toOffset(), it.size.toSize()).contains(dragPos)
                                    } ?: return@detectDragGesturesAfterLongPress
                                    val from = vm.pages.indexOf(id)
                                    val to = vm.pages.indexOf(target.key as String)
                                    if (from >= 0 && to >= 0) repeat(kotlin.math.abs(to - from)) { vm.movePage(id, if (to > from) 1 else -1) }
                                },
                                onDragEnd = { dragId = null },
                                onDragCancel = { dragId = null },
                            )
                        },
                )
            }
            if (n == 0) item(key = "empty", span = { GridItemSpan(maxLineSpan) }) { EmptyPages() }
            item(key = "add") { AddPageTile(onCamera = { vm.startCamera() }, onGallery = pickPhotos) }
        }
        RuledDivider()
        Row(
            Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryPill(onClick = { sheet = PagesSheet.Share }, enabled = ready, modifier = Modifier.weight(1f)) {
                Icon(Icons.Outlined.Share, null, Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                PillText("Share")
            }
            PrimaryPill(onClick = { exporter.save(ExportFormat.PDF) }, enabled = ready, modifier = Modifier.weight(1.5f)) {
                if (exporter.busy) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
                else PillText("Save as PDF")
            }
        }
    }
    if (sheet != PagesSheet.None) {
        val share = sheet == PagesSheet.Share
        ExportSheet(share, n, onSave = { exporter.save(it) }, onShare = { exporter.share(it) }, onDismiss = { sheet = PagesSheet.None; docName = DocumentName.get(ctx) })
    }
    if (renaming) {
        var text by remember { mutableStateOf(docName ?: "") }
        AlertDialog(
            onDismissRequest = { renaming = false },
            title = { Text("Rename the document") },
            text = {
                OutlinedTextField(
                    value = text, onValueChange = { text = it.take(80) }, singleLine = true, label = { Text("Name") },
                    placeholder = { Text(DocumentName.default()) }, keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    DocumentName.set(ctx, text)
                    docName = DocumentName.get(ctx)
                    renaming = false
                }) { Text("Rename") }
            },
            dismissButton = { TextButton(onClick = { renaming = false }) { Text("Cancel") } },
        )
    }
    confirmDelete?.let { id ->
        val number = vm.pages.indexOf(id) + 1
        AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text("Delete page $number?") },
            confirmButton = { TextButton(onClick = { crumpling = id; confirmDelete = null }) { Text("Delete") } },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Cancel") } },
        )
    }
}

private val CardHeight = 210.dp

@Composable
private fun PageCard(
    vm: ScanViewModel, id: String, index: Int, count: Int, onOpen: () -> Unit, onDelete: () -> Unit,
    dragging: Boolean = false, crumpling: Boolean = false, onCrumpled: () -> Unit = {}, modifier: Modifier = Modifier,
) {
    // deleting crumples the sheet into a ball that drops away
    val crush = remember(id) { Animatable(0f) }
    val reduced = rememberReducedMotion()
    val done by rememberUpdatedState(onCrumpled)
    LaunchedEffect(crumpling) {
        if (!crumpling) return@LaunchedEffect
        if (!reduced) crush.animateTo(1f, tween(420, easing = FastOutLinearInEasing))
        done()
    }
    val version = vm.pageVersions[id] ?: 0
    var thumb by remember(id) { mutableStateOf<Bitmap?>(null) }
    var loaded by remember(id) { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }
    LaunchedEffect(id, version) {
        vm.thumbnail(id)?.let { thumb = it }
        loaded = true
    }
    val current = id == vm.currentId
    val shape = RoundedCornerShape(4.dp)
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column(
        modifier.graphicsLayer {
            val t = crush.value
            if (t > 0f) {
                scaleX = 1f - 0.8f * t
                scaleY = 1f - 0.8f * t
                rotationZ = 35f * t
                translationY = 40.dp.toPx() * t * t
                alpha = 1f - t * t
            }
        },
    ) {
        Box(
            Modifier.fillMaxWidth().height(CardHeight)
                .shadow(if (dragging) 16.dp else 3.dp, shape, ambientColor = Palette.Ink.copy(alpha = 0.35f), spotColor = Palette.Ink.copy(alpha = 0.35f))
                .clip(shape)
                .background(Color.White)
                .then(if (current || dragging) Modifier.border(2.dp, MaterialTheme.colorScheme.primary, shape) else Modifier)
                .clickable(onClick = onOpen, onClickLabel = "Open page ${index + 1}")
                .semantics { contentDescription = "Page ${index + 1}" },
            contentAlignment = Alignment.Center,
        ) {
            val t = thumb
            if (t != null) {
                val img = remember(t) { t.asImageBitmap() }
                Image(img, null, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize().padding(6.dp))
            } else if (!loaded) {
                CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
            } else {
                Text("No preview", style = MaterialTheme.typography.bodySmall, color = muted)
            }
            Box(
                Modifier.align(Alignment.BottomStart).padding(8.dp).defaultMinSize(minWidth = 26.dp).height(24.dp)
                    .background(Palette.Ink, RoundedCornerShape(3.dp)).padding(horizontal = 7.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text("${index + 1}", style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, fontWeight = FontWeight.Bold), color = Color.White)
            }
            if (crush.value > 0f) CrumpledBall(Modifier.size(96.dp).alpha(crush.value), seed = id.hashCode())
        }
        Row(Modifier.fillMaxWidth().padding(top = 2.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                if (current) "${vm.params.filter.label} · editing" else "Page ${index + 1}",
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold),
                color = muted, maxLines = 1, modifier = Modifier.weight(1f),
            )
            Box {
                IconButton(onClick = { menu = true }) {
                    Icon(Icons.Outlined.MoreHoriz, "Page ${index + 1} actions", tint = muted)
                }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(
                        text = { Text("Move left") }, leadingIcon = { Icon(Icons.Outlined.ChevronLeft, null) }, enabled = index > 0,
                        onClick = { vm.movePage(id, -1); menu = false },
                    )
                    DropdownMenuItem(
                        text = { Text("Move right") }, leadingIcon = { Icon(Icons.Outlined.ChevronRight, null) }, enabled = index < count - 1,
                        onClick = { vm.movePage(id, 1); menu = false },
                    )
                    DropdownMenuItem(
                        text = { Text("Delete") }, leadingIcon = { Icon(Icons.Outlined.Delete, null) },
                        onClick = { menu = false; onDelete() },
                    )
                }
            }
        }
    }
}

/** No pages: the iron beside an empty sheet. */
@Composable
private fun EmptyPages() {
    Column(Modifier.fillMaxWidth().padding(vertical = 28.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Row(verticalAlignment = Alignment.Bottom) {
            PaperSurface(Modifier.size(54.dp, 72.dp), ruled = true) {}
            Spacer(Modifier.width(10.dp))
            Icon(PaperIcons.Iron, null, Modifier.size(44.dp), tint = MaterialTheme.colorScheme.onSurface)
        }
        Spacer(Modifier.height(14.dp))
        Text("Nothing to press yet.", style = MaterialTheme.typography.titleMedium)
        Text("Add a page from the camera or the gallery.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** The dashed tile after the last page; it asks where the new page comes from. */
@Composable
private fun AddPageTile(onCamera: () -> Unit, onGallery: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val accent = MaterialTheme.colorScheme.primary
    val dash = if (MaterialTheme.colorScheme.background.luminance() > 0.5f) Palette.Dashed else MaterialTheme.colorScheme.outline
    Box {
        Column(
            Modifier.fillMaxWidth().height(CardHeight).clip(RoundedCornerShape(4.dp))
                .drawBehind {
                    val w = 2.dp.toPx()
                    drawRoundRect(
                        dash, topLeft = Offset(w / 2, w / 2), size = Size(size.width - w, size.height - w),
                        cornerRadius = CornerRadius(4.dp.toPx()),
                        style = Stroke(w, pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))),
                    )
                }
                .clickable(role = Role.Button) { menu = true },
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Box(Modifier.size(48.dp).background(MaterialTheme.colorScheme.primaryContainer, RoundedCornerShape(8.dp)), contentAlignment = Alignment.Center) {
                Icon(Icons.Outlined.Add, null, tint = accent)
            }
            Spacer(Modifier.height(10.dp))
            Text("Add page", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = accent)
        }
        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
            DropdownMenuItem(
                text = { Text("Take a photo") }, leadingIcon = { Icon(Icons.Outlined.AddAPhoto, null) },
                onClick = { menu = false; onCamera() },
            )
            DropdownMenuItem(
                text = { Text("Import from gallery") }, leadingIcon = { Icon(Icons.Outlined.AddPhotoAlternate, null) },
                onClick = { menu = false; onGallery() },
            )
        }
    }
}
