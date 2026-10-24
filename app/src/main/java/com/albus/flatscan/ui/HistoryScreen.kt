package com.albus.flatscan.ui

import android.graphics.Bitmap
import android.text.format.DateUtils
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.combinedClickable
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.CloudDone
import androidx.compose.material.icons.outlined.CloudOff
import androidx.compose.material.icons.outlined.CloudUpload
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.DeleteForever
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.RestoreFromTrash
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import com.albus.flatscan.data.DocumentSummary
import com.albus.flatscan.data.Library
import com.albus.flatscan.data.SyncStatus
import java.util.concurrent.TimeUnit

/** Documents whose title (or recognised text, when there is any) contains every word of [query]. */
fun filterDocuments(docs: List<DocumentSummary>, query: String, text: (String) -> String? = { null }): List<DocumentSummary> {
    val words = query.trim().lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
    if (words.isEmpty()) return docs
    return docs.filter { d ->
        val hay = (d.title + " " + d.tags + " " + (text(d.id) ?: "")).lowercase()
        words.all { it in hay }
    }
}

/**
 * The home screen: scan or import into a new document, and the history of documents, newest first
 * (thumbnail, page count, date, sync state). Search by title; long press selects (rename, delete).
 */
@Composable
fun HomeScreen(vm: ScanViewModel) {
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { if (it.isNotEmpty()) vm.importNew(it) }
    val docs by vm.library.documents.collectAsState(initial = null)
    var query by remember { mutableStateOf("") }
    var searching by remember { mutableStateOf(false) }
    var selected by remember { mutableStateOf(setOf<String>()) }
    var renaming by remember { mutableStateOf<DocumentSummary?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }
    val shown = remember(docs, query, vm.searchTextVersion) { filterDocuments(docs.orEmpty(), query) { vm.searchText(it) } }
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        // top bar: the app, or the selection's actions
        Row(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            if (selected.isNotEmpty()) {
                IconButton(onClick = { selected = emptySet() }) { Icon(Icons.Outlined.Close, "Clear the selection") }
                Text("${selected.size} selected", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                if (selected.size == 1) {
                    IconButton(onClick = { renaming = docs?.firstOrNull { it.id in selected } }) { Icon(Icons.Outlined.Edit, "Rename") }
                }
                IconButton(onClick = { vm.uploadDocuments(selected); selected = emptySet() }) { Icon(Icons.Outlined.CloudUpload, "Upload") }
                IconButton(onClick = { confirmDelete = true }, modifier = Modifier.testTag("delete_selected")) { Icon(Icons.Outlined.Delete, "Delete") }
            } else {
                Spacer(Modifier.width(12.dp))
                Text("FlatScan", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
                IconButton(onClick = { searching = !searching; if (!searching) query = "" }) { Icon(Icons.Outlined.Search, "Search") }
                IconButton(onClick = { vm.open(Screen.Trash) }) { Icon(Icons.Outlined.Delete, "Trash") }
                IconButton(onClick = { vm.open(Screen.Settings) }) { Icon(Icons.Outlined.Settings, "Settings") }
            }
        }
        if (searching) {
            OutlinedTextField(
                value = query, onValueChange = { query = it }, singleLine = true,
                placeholder = { Text(if (vm.hasOcrText) "Search titles and page text" else "Search titles") },
                leadingIcon = { Icon(Icons.Outlined.Search, null) },
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp).testTag("search"),
            )
        }
        LazyColumn(
            Modifier.weight(1f).fillMaxWidth().testTag("history"),
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            if (!searching && selected.isEmpty()) {
                item("mode") {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                        ModeSwitch(vm.captureMode, onPick = { vm.chooseCaptureMode(it) }, dark = false)
                    }
                }
                item("actions") {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(bottom = 8.dp)) {
                        ActionCard("Scan", "New document", Icons.Outlined.PhotoCamera, primary = true, modifier = Modifier.weight(1f)) { vm.scanNew() }
                        ActionCard("Import", "Photos, several at once", Icons.Outlined.PhotoLibrary, primary = false, modifier = Modifier.weight(1f)) {
                            pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                        }
                    }
                }
            }
            when {
                docs == null -> {}
                docs!!.isEmpty() -> item("empty") {
                    Column(Modifier.fillMaxWidth().padding(top = 48.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Flat, clean pages from any photo.", style = MaterialTheme.typography.titleLarge)
                        Spacer(Modifier.height(8.dp))
                        Text("Your scans appear here.", style = MaterialTheme.typography.bodyMedium, color = muted)
                    }
                }
                shown.isEmpty() -> item("none") {
                    Text("No document matches \"$query\"", color = muted, modifier = Modifier.padding(top = 32.dp).fillMaxWidth())
                }
                else -> items(shown, key = { it.id }) { d ->
                    DocumentRow(
                        vm, d, selected = d.id in selected, open = d.id == vm.docId,
                        onClick = {
                            if (selected.isNotEmpty()) selected = if (d.id in selected) selected - d.id else selected + d.id
                            else vm.openDocument(d.id)
                        },
                        onLongClick = { selected = selected + d.id },
                    )
                }
            }
        }
    }
    renaming?.let { d ->
        var text by remember(d.id) { mutableStateOf(d.title) }
        AlertDialog(
            onDismissRequest = { renaming = null },
            title = { Text("Rename the document") },
            text = { OutlinedTextField(value = text, onValueChange = { text = it.take(80) }, singleLine = true, label = { Text("Name") }) },
            confirmButton = {
                TextButton(onClick = { vm.renameDocument(d.id, text); renaming = null; selected = emptySet() }) { Text("Rename") }
            },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("Cancel") } },
        )
    }
    if (confirmDelete) {
        val n = selected.size
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(if (n == 1) "Delete this document?" else "Delete $n documents?") },
            text = { Text("It goes to the trash, where it is kept for ${Library.TRASH_DAYS} days.") },
            confirmButton = {
                TextButton(onClick = { vm.trashDocuments(selected); selected = emptySet(); confirmDelete = false }, modifier = Modifier.testTag("confirm_delete")) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ActionCard(title: String, caption: String, icon: androidx.compose.ui.graphics.vector.ImageVector, primary: Boolean, modifier: Modifier, onClick: () -> Unit) {
    Surface(
        onClick = onClick, shape = RoundedCornerShape(20.dp),
        color = if (primary) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerLowest,
        contentColor = if (primary) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
        border = if (primary) null else BorderStroke(1.5.dp, MaterialTheme.colorScheme.outline),
        modifier = modifier.height(112.dp),
    ) {
        Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.SpaceBetween) {
            Icon(icon, null, Modifier.size(28.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold))
                Text(caption, style = MaterialTheme.typography.bodySmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun DocumentRow(vm: ScanViewModel, d: DocumentSummary, selected: Boolean, open: Boolean, onClick: () -> Unit, onLongClick: () -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val shape = RoundedCornerShape(16.dp)
    Row(
        Modifier.fillMaxWidth().clip(shape)
            .background(if (selected) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainerLowest)
            .border(if (selected || open) 2.dp else 1.dp, if (selected || open) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant, shape)
            .combinedClickable(onClick = onClick, onLongClick = onLongClick)
            .padding(10.dp).testTag("doc_${d.id}"),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        PageThumb(vm, d.firstPage, d.modifiedAt)
        Spacer(Modifier.width(14.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(d.title, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(
                (if (d.pageCount == 1) "1 page" else "${d.pageCount} pages") + " · " + relativeDate(d.modifiedAt),
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp), color = muted,
            )
            SyncLine(d)
        }
        if (selected) Icon(Icons.Outlined.CheckCircle, "Selected", tint = MaterialTheme.colorScheme.primary)
    }
}

/** The document's sync state (feature 6); nothing when it is not synced. */
@Composable
private fun SyncLine(d: DocumentSummary) {
    val (icon, text, color) = when (d.syncStatus) {
        SyncStatus.None -> return
        SyncStatus.Pending -> Triple(Icons.Outlined.CloudUpload, "Waiting to upload", MaterialTheme.colorScheme.onSurfaceVariant)
        SyncStatus.Uploading -> Triple(Icons.Outlined.CloudUpload, "Uploading", MaterialTheme.colorScheme.primary)
        SyncStatus.Uploaded -> Triple(Icons.Outlined.CloudDone, "Uploaded", MaterialTheme.colorScheme.primary)
        SyncStatus.Failed -> Triple(Icons.Outlined.CloudOff, "Upload failed" + (d.syncError?.let { ": $it" } ?: ""), MaterialTheme.colorScheme.error)
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, null, Modifier.size(14.dp), tint = color)
        Spacer(Modifier.width(4.dp))
        Text(text, style = MaterialTheme.typography.bodySmall, color = color, maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun PageThumb(vm: ScanViewModel, page: String?, version: Long) {
    var thumb by remember(page) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(page, version, page?.let { vm.pageVersions[it] }) { thumb = page?.let { vm.thumbnail(it) } }
    val sheet = RoundedCornerShape(4.dp)
    Box(
        Modifier.size(52.dp, 70.dp).background(Color.White, sheet).border(1.dp, Color(0xFFD8D6D0), sheet).clip(sheet),
        contentAlignment = Alignment.Center,
    ) {
        val t = thumb
        if (t != null) Image(remember(t) { t.asImageBitmap() }, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        else Icon(Icons.Outlined.Description, null, tint = Color(0xFFB9B7B1))
    }
}

@Composable
private fun relativeDate(t: Long): String {
    val ctx = LocalContext.current
    return remember(t) {
        DateUtils.getRelativeDateTimeString(ctx, t, DateUtils.MINUTE_IN_MILLIS, DateUtils.WEEK_IN_MILLIS, DateUtils.FORMAT_ABBREV_ALL).toString()
    }
}

/** Deleted documents, kept [Library.TRASH_DAYS] days: restore or delete forever. */
@Composable
fun TrashScreen(vm: ScanViewModel) {
    val docs by vm.library.trash.collectAsState(initial = emptyList())
    var selected by remember { mutableStateOf(setOf<String>()) }
    var confirm by remember { mutableStateOf<Set<String>?>(null) }
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        Row(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            Text(if (selected.isEmpty()) "Trash" else "${selected.size} selected", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            if (selected.isNotEmpty()) {
                IconButton(onClick = { vm.restoreDocuments(selected); selected = emptySet() }) { Icon(Icons.Outlined.RestoreFromTrash, "Restore") }
                IconButton(onClick = { confirm = selected }) { Icon(Icons.Outlined.DeleteForever, "Delete forever") }
            } else if (docs.isNotEmpty()) {
                TextButton(onClick = { confirm = docs.map { it.id }.toSet() }) { Text("Empty") }
            }
        }
        Text(
            "Deleted documents are kept for ${Library.TRASH_DAYS} days, then removed for good.",
            style = MaterialTheme.typography.bodyMedium, color = muted, modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp),
        )
        LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (docs.isEmpty()) item { Text("The trash is empty.", color = muted, modifier = Modifier.padding(top = 24.dp)) }
            items(docs, key = { it.id }) { d ->
                val left = (Library.TRASH_DAYS - TimeUnit.MILLISECONDS.toDays(System.currentTimeMillis() - (d.deletedAt ?: 0))).coerceAtLeast(0)
                val isSel = d.id in selected
                Row(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp))
                        .background(if (isSel) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainerLowest)
                        .combinedClickableCompat { selected = if (isSel) selected - d.id else selected + d.id }
                        .padding(10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    PageThumb(vm, d.firstPage, d.modifiedAt)
                    Spacer(Modifier.width(14.dp))
                    Column(Modifier.weight(1f)) {
                        Text(d.title, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("${d.pageCount} pages · deleted for good in $left days", style = MaterialTheme.typography.bodySmall, color = muted)
                    }
                    if (isSel) Icon(Icons.Outlined.CheckCircle, "Selected", tint = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
    confirm?.let { ids ->
        AlertDialog(
            onDismissRequest = { confirm = null },
            title = { Text(if (ids.size == 1) "Delete forever?" else "Delete ${ids.size} documents forever?") },
            text = { Text("Their pages and photos are removed from the phone. This cannot be undone.") },
            confirmButton = { TextButton(onClick = { vm.deleteForever(ids); selected = selected - ids; confirm = null }) { Text("Delete forever") } },
            dismissButton = { TextButton(onClick = { confirm = null }) { Text("Cancel") } },
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
private fun Modifier.combinedClickableCompat(onClick: () -> Unit): Modifier = this.combinedClickable(onClick = onClick)
