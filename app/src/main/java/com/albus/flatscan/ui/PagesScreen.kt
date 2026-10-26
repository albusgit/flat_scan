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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.DriveFileMove
import androidx.compose.material.icons.automirrored.outlined.Redo
import androidx.compose.material.icons.automirrored.outlined.RotateRight
import androidx.compose.material.icons.automirrored.outlined.Undo
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AddAPhoto
import androidx.compose.material.icons.outlined.AddPhotoAlternate
import androidx.compose.material.icons.outlined.CallSplit
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.CloudUpload
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.MoreHoriz
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.PictureAsPdf
import androidx.compose.material.icons.outlined.RadioButtonUnchecked
import androidx.compose.material.icons.outlined.SaveAlt
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.SwapVert
import androidx.compose.material.icons.outlined.VerticalAlignBottom
import androidx.compose.material.icons.outlined.VerticalAlignTop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
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
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.DocumentName
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.TagEdit
import com.albus.flatscan.Tags

private enum class PagesSheet { None, Save, Share }

/** Where new pages come from (insert or add). */
private enum class Source { Camera, Photos, Pdf }

/** The Add pages sheet is open; new pages go before [before] (null: after the last page). */
private data class AddTarget(val before: String?)

/**
 * The page editor of the open document: open a page; hold and drag to reorder (the grid scrolls at its
 * edges); select pages to rotate, duplicate, move (to the start, the end or another document) or delete;
 * reverse all pages; insert pages from the camera, photos or a PDF before any page; split the document;
 * undo / redo every change of the session; rename, tag, save and share.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PagesScreen(vm: ScanViewModel) {
    val exporter = rememberExportActions(vm)
    var sheet by remember { mutableStateOf(PagesSheet.None) }
    // the pages being crumpled away (no confirmation: the message after it offers Undo)
    var crumpling by remember { mutableStateOf(setOf<String>()) }
    val delete = { ids: Set<String> -> crumpling = ids.filter { it in vm.pages }.toSet() }
    var menu by remember { mutableStateOf(false) }
    var selecting by remember { mutableStateOf(false) }
    var selected by remember { mutableStateOf(setOf<String>()) }
    var moving by remember { mutableStateOf<Set<String>?>(null) }
    var pickDocFor by remember { mutableStateOf<Set<String>?>(null) }
    var adding by remember { mutableStateOf<AddTarget?>(null) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { if (it.isEmpty()) vm.insertBefore(null) else vm.importUris(it) }
    val pickPdf = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { if (it == null) vm.insertBefore(null) else vm.importPdf(it) }
    fun add(source: Source, before: String?) {
        vm.insertBefore(before)
        when (source) {
            Source.Camera -> vm.startCamera()
            Source.Photos -> pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            Source.Pdf -> pickPdf.launch(arrayOf("application/pdf"))
        }
    }
    val n = vm.pages.size
    val ready = n > 0 && !exporter.busy
    val docName = vm.docTitle
    var renaming by remember { mutableStateOf(false) }
    var tagging by remember { mutableStateOf(false) }
    // the open document's tags and every tag of the library (the tag sheet); null while loading
    val allDocs by vm.library.documents.collectAsState(initial = null)
    val docTags = allDocs?.let { all -> Tags.parse(all.firstOrNull { it.id == vm.docId }?.tags.orEmpty()) }
    // pages that left the document (undo, move) leave the selection
    LaunchedEffect(vm.pages.toList()) { selected = selected.filter { it in vm.pages }.toSet() }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = {
                if (selecting) {
                    Text("${selected.size} selected", style = MaterialTheme.typography.titleLarge.copy(fontSize = 18.sp))
                } else Row(
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
            navigationIcon = {
                if (selecting) IconButton(onClick = { selecting = false; selected = emptySet() }) { Icon(Icons.Outlined.Close, "Done selecting") }
                else IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            },
            actions = {
                IconButton(onClick = { vm.undoPages() }, enabled = vm.canUndoPages) { Icon(Icons.AutoMirrored.Outlined.Undo, "Undo") }
                if (vm.canRedoPages) IconButton(onClick = { vm.redoPages() }) { Icon(Icons.AutoMirrored.Outlined.Redo, "Redo") }
                if (selecting) {
                    TextButton(onClick = { selected = if (selected.size == n) emptySet() else vm.pages.toSet() }) { Text(if (selected.size == n) "None" else "All") }
                } else {
                    TextButton(onClick = { selecting = true }, enabled = n > 0) { Text("Select", style = MaterialTheme.typography.labelLarge) }
                    Box {
                        IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreVert, "More") }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                            DropdownMenuItem(
                                text = { Text("Reverse page order") }, leadingIcon = { Icon(Icons.Outlined.SwapVert, null) }, enabled = n > 1,
                                onClick = { vm.reversePages(); menu = false },
                            )
                            DropdownMenuItem(
                                text = { Text("Save as…") }, leadingIcon = { Icon(Icons.Outlined.SaveAlt, null) }, enabled = ready,
                                onClick = { sheet = PagesSheet.Save; menu = false },
                            )
                            DropdownMenuItem(
                                text = { Text("Upload now") }, leadingIcon = { Icon(Icons.Outlined.CloudUpload, null) }, enabled = ready,
                                onClick = { menu = false; vm.docId?.let { vm.uploadDocuments(listOf(it)) } },
                            )
                        }
                    }
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent),
        )
        if (!selecting && docTags != null) DocumentTagsRow(docTags, onEdit = { tagging = true }, modifier = Modifier.padding(horizontal = 8.dp))
        ReorderablePageGrid(
            pages = vm.pages,
            onMove = { from, to -> vm.dragMove(from, to) },
            onDragEnd = { vm.dragEnd() },
            dragEnabled = !selecting,
            modifier = Modifier.weight(1f).fillMaxWidth(),
            footer = {
                if (n == 0) item(key = "empty", span = { GridItemSpan(maxLineSpan) }) { EmptyPages() }
                if (!selecting) item(key = "add") { AddPageTile { adding = AddTarget(null) } }
            },
        ) { id, i, dragging, m ->
            PageCard(
                vm, id, i, n,
                selecting = selecting, isSelected = id in selected, dragging = dragging, modifier = m,
                onOpen = {
                    if (selecting) selected = if (id in selected) selected - id else selected + id
                    else vm.openPage(id)
                },
                onDelete = { delete(setOf(id)) },
                crumpling = id in crumpling,
                onCrumpled = {
                    // the last of the set to finish deletes them all (one undo step)
                    if (crumpling.isNotEmpty()) {
                        val k = crumpling.size
                        vm.deletePages(crumpling)
                        crumpling = emptySet()
                        vm.showMessage(if (k == 1) "Page deleted" else "$k pages deleted", "Undo") { if (vm.canUndoPages) vm.undoPages() }
                    }
                },
                onRotate = { vm.rotatePages(setOf(id), 1) },
                onDuplicate = { vm.duplicatePages(setOf(id)) },
                onInsert = { adding = AddTarget(id) },
                onSplit = { vm.splitAt(id) },
                onMoveToDocument = { pickDocFor = setOf(id) },
            )
        }
        RuledDivider()
        if (selecting) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
                val any = selected.isNotEmpty()
                BarAction(Icons.AutoMirrored.Outlined.RotateRight, "Rotate", any) { vm.rotatePages(selected, 1) }
                BarAction(Icons.Outlined.ContentCopy, "Duplicate", any) { vm.duplicatePages(selected) }
                BarAction(Icons.AutoMirrored.Outlined.DriveFileMove, "Move", any) { moving = selected }
                BarAction(Icons.Outlined.Delete, "Delete", any) { delete(selected) }
            }
        } else Row(
            Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SecondaryPill(onClick = { sheet = PagesSheet.Share }, enabled = ready, modifier = Modifier.weight(1f)) {
                Icon(Icons.Outlined.Share, null, Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                PillText("Share")
            }
            PrimaryPill(onClick = { exporter.save(null) }, enabled = ready, modifier = Modifier.weight(1.5f)) {
                if (exporter.busy) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
                else PillText("Save")
            }
        }
    }
    adding?.let { target ->
        AddPagesSheet(
            note = target.before?.let { b -> "New pages go before page ${vm.pages.indexOf(b) + 1}." }
                ?: if (n == 0) "New pages start the document." else "New pages go after page $n. To put them somewhere else, use a page's ··· menu.",
            onPick = { adding = null; add(it, target.before) },
            onDismiss = { adding = null },
        )
    }
    if (sheet != PagesSheet.None) {
        val share = sheet == PagesSheet.Share
        ExportSheet(vm, share, n, vm.docTitle, onRename = { vm.renameDocument(it) }, onSave = { exporter.save(it) }, onShare = { exporter.share(it) }, onDismiss = { sheet = PagesSheet.None })
    }
    if (tagging) {
        TagSheet(
            subtitle = docName, edit = TagEdit(mapOf((vm.docId ?: "") to docTags.orEmpty())), known = Tags.ranked(allDocs.orEmpty()),
            onSet = { t, on -> vm.setOpenDocumentTag(t, on) }, onDismiss = { tagging = false },
        )
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
            confirmButton = { TextButton(onClick = { vm.renameDocument(text); renaming = false }) { Text("Rename") } },
            dismissButton = { TextButton(onClick = { renaming = false }) { Text("Cancel") } },
        )
    }
    moving?.let { ids ->
        AlertDialog(
            onDismissRequest = { moving = null },
            title = { Text(if (ids.size == 1) "Move the page" else "Move ${ids.size} pages") },
            text = {
                Column {
                    MoveChoice(Icons.Outlined.VerticalAlignTop, "To the start") { vm.movePagesTo(ids, 0); moving = null }
                    MoveChoice(Icons.Outlined.VerticalAlignBottom, "To the end") { vm.movePagesTo(ids, Int.MAX_VALUE); moving = null }
                    MoveChoice(Icons.AutoMirrored.Outlined.DriveFileMove, "To another document") { pickDocFor = ids; moving = null }
                    if (ids.size == 1 && vm.pages.indexOf(ids.first()) > 0) {
                        MoveChoice(Icons.Outlined.CallSplit, "Split: this page and the ones after it to a new document") { vm.splitAt(ids.first()); moving = null }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { moving = null }) { Text("Cancel") } },
        )
    }
    pickDocFor?.let { ids ->
        val docs by vm.library.documents.collectAsState(initial = emptyList())
        AlertDialog(
            onDismissRequest = { pickDocFor = null },
            title = { Text("Move to which document?") },
            text = {
                LazyColumn(Modifier.height(320.dp)) {
                    item {
                        ListItem(
                            headlineContent = { Text("New document") }, leadingContent = { Icon(Icons.Outlined.Add, null) },
                            modifier = Modifier.clickable { vm.movePagesToDocument(ids, null); pickDocFor = null; selected = emptySet() },
                        )
                    }
                    items(docs.filter { it.id != vm.docId }, key = { it.id }) { d ->
                        ListItem(
                            headlineContent = { Text(d.title, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                            supportingContent = { Text(if (d.pageCount == 1) "1 page" else "${d.pageCount} pages") },
                            modifier = Modifier.clickable { vm.movePagesToDocument(ids, d.id); pickDocFor = null; selected = emptySet() },
                        )
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { pickDocFor = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun MoveChoice(icon: ImageVector, text: String, onClick: () -> Unit) {
    ListItem(headlineContent = { Text(text) }, leadingContent = { Icon(icon, null) }, modifier = Modifier.clickable(onClick = onClick))
}

@Composable
private fun BarAction(icon: ImageVector, label: String, enabled: Boolean, onClick: () -> Unit) {
    TextButton(onClick = onClick, enabled = enabled) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(icon, null)
            Text(label, style = MaterialTheme.typography.labelMedium)
        }
    }
}

private val CardHeight = 210.dp

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PageCard(
    vm: ScanViewModel, id: String, index: Int, count: Int,
    selecting: Boolean, isSelected: Boolean, dragging: Boolean, modifier: Modifier,
    onOpen: () -> Unit, onDelete: () -> Unit, onRotate: () -> Unit, onDuplicate: () -> Unit,
    onInsert: () -> Unit, onSplit: () -> Unit, onMoveToDocument: () -> Unit,
    crumpling: Boolean = false, onCrumpled: () -> Unit = {},
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
    val accent = MaterialTheme.colorScheme.primary
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
                .then(if (current || dragging || isSelected) Modifier.border(if (isSelected) 3.dp else 2.dp, accent, shape) else Modifier)
                .clickable(onClick = onOpen, onClickLabel = if (selecting) "Select page ${index + 1}" else "Open page ${index + 1}")
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
            if (selecting) {
                Icon(
                    if (isSelected) Icons.Outlined.CheckCircle else Icons.Outlined.RadioButtonUnchecked, null,
                    tint = if (isSelected) accent else muted,
                    modifier = Modifier.align(Alignment.TopEnd).padding(8.dp).background(Color.White, CircleShape),
                )
            }
            if (crush.value > 0f) CrumpledBall(Modifier.size(96.dp).alpha(crush.value), seed = id.hashCode())
        }
        Row(Modifier.fillMaxWidth().padding(top = 2.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                if (current) "${vm.params.filter.label} · editing" else "Page ${index + 1}",
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold),
                color = muted, maxLines = 1, modifier = Modifier.weight(1f),
            )
            if (!selecting) Box {
                IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreHoriz, "Page ${index + 1} actions", tint = muted) }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(text = { Text("Rotate") }, leadingIcon = { Icon(Icons.AutoMirrored.Outlined.RotateRight, null) }, onClick = { menu = false; onRotate() })
                    DropdownMenuItem(text = { Text("Duplicate") }, leadingIcon = { Icon(Icons.Outlined.ContentCopy, null) }, onClick = { menu = false; onDuplicate() })
                    DropdownMenuItem(text = { Text("Insert pages before") }, leadingIcon = { Icon(Icons.Outlined.Add, null) }, onClick = { menu = false; onInsert() })
                    DropdownMenuItem(text = { Text("Move to another document") }, leadingIcon = { Icon(Icons.AutoMirrored.Outlined.DriveFileMove, null) }, onClick = { menu = false; onMoveToDocument() })
                    if (index > 0) {
                        DropdownMenuItem(text = { Text("Split from here") }, leadingIcon = { Icon(Icons.Outlined.CallSplit, null) }, onClick = { menu = false; onSplit() })
                    }
                    DropdownMenuItem(text = { Text("Delete") }, leadingIcon = { Icon(Icons.Outlined.Delete, null) }, onClick = { menu = false; onDelete() })
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
        Text("Add a page from the camera, photos or a PDF.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** Add pages: the camera, photos or a PDF, as three big buttons, and where the pages will go. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AddPagesSheet(note: String, onPick: (Source) -> Unit, onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss, containerColor = MaterialTheme.colorScheme.surfaceContainerLowest) {
        Column(Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, bottom = 24.dp)) {
            Text("Add pages", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(14.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SourceButton(Icons.Outlined.AddAPhoto, "Camera") { onPick(Source.Camera) }
                SourceButton(Icons.Outlined.AddPhotoAlternate, "Photos") { onPick(Source.Photos) }
                SourceButton(Icons.Outlined.PictureAsPdf, "PDF pages") { onPick(Source.Pdf) }
            }
            Spacer(Modifier.height(12.dp))
            Text(note, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun RowScope.SourceButton(icon: ImageVector, label: String, onClick: () -> Unit) {
    val shape = RoundedCornerShape(10.dp)
    Column(
        Modifier.weight(1f).height(96.dp).clip(shape).border(1.5.dp, MaterialTheme.colorScheme.onSurface, shape)
            .clickable(role = Role.Button, onClick = onClick).testTag("add_${label.substringBefore(' ').lowercase()}"),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Icon(icon, null, Modifier.size(26.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(8.dp))
        Text(label, style = MaterialTheme.typography.labelLarge)
    }
}

/** The dashed tile after the last page; it opens the Add pages sheet. */
@Composable
private fun AddPageTile(onClick: () -> Unit) {
    val accent = MaterialTheme.colorScheme.primary
    val dash = if (MaterialTheme.colorScheme.background.luminance() > 0.5f) Palette.Dashed else MaterialTheme.colorScheme.outline
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
            .clickable(role = Role.Button, onClick = onClick).testTag("add_page"),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Box(Modifier.size(48.dp).background(MaterialTheme.colorScheme.primaryContainer, RoundedCornerShape(8.dp)), contentAlignment = Alignment.Center) {
            Icon(Icons.Outlined.Add, null, tint = accent)
        }
        Spacer(Modifier.height(10.dp))
        Text("Add page", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = accent)
    }
}
