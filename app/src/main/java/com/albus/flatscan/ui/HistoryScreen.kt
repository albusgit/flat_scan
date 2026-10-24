package com.albus.flatscan.ui

import android.graphics.Bitmap
import android.text.format.DateFormat
import android.text.format.DateUtils
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.List
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.CloudDone
import androidx.compose.material.icons.outlined.CloudOff
import androidx.compose.material.icons.outlined.CloudUpload
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.DeleteForever
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.PhotoCamera
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.RestoreFromTrash
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.Screen
import com.albus.flatscan.data.DocumentSummary
import com.albus.flatscan.data.Library
import com.albus.flatscan.data.SyncStatus
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
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
 * Where [query] was found in a document's page [text]: about a line around the first word found,
 * with every query word marked. Null when the words are not in the text (a title match).
 */
fun searchSnippet(text: String?, query: String, mark: SpanStyle): AnnotatedString? {
    if (text.isNullOrBlank()) return null
    val words = query.trim().lowercase().split(Regex("\\s+")).filter { it.isNotEmpty() }
    if (words.isEmpty()) return null
    val flat = text.replace(Regex("\\s+"), " ")
    val low = flat.lowercase()
    val at = words.map { low.indexOf(it) }.filter { it >= 0 }.minOrNull() ?: return null
    val from = (at - 36).coerceAtLeast(0).let { s -> if (s == 0) 0 else (low.indexOf(' ', s).takeIf { it in s until at } ?: s) }
    val to = (at + 84).coerceAtMost(flat.length)
    val piece = flat.substring(from, to).trim()
    val lowPiece = piece.lowercase()
    return buildAnnotatedString {
        if (from > 0) append("…")
        val base = length
        append(piece)
        if (to < flat.length) append("…")
        for (w in words) {
            var i = lowPiece.indexOf(w)
            while (i >= 0) {
                addStyle(mark, base + i, base + i + w.length)
                i = lowPiece.indexOf(w, i + w.length)
            }
        }
    }
}

/** How long ago a document was changed, for the list's groups. */
private enum class Age(val label: String) { Today("TODAY"), Week("THIS WEEK"), Earlier("EARLIER") }

private fun startOfToday(now: Long): Long = Calendar.getInstance().apply {
    timeInMillis = now
    set(Calendar.HOUR_OF_DAY, 0); set(Calendar.MINUTE, 0); set(Calendar.SECOND, 0); set(Calendar.MILLISECOND, 0)
}.timeInMillis

private fun ageOf(t: Long, now: Long): Age {
    val today = startOfToday(now)
    return when {
        t >= today -> Age.Today
        t >= today - TimeUnit.DAYS.toMillis(6) -> Age.Week
        else -> Age.Earlier
    }
}

/**
 * The home screen: carry on with the open document, find one (title or page text), the documents
 * as a ruled list grouped by age or as their first pages laid out, and Scan / Import at the bottom
 * under the thumb. Long press selects (rename, upload, trash). With nothing scanned yet it shows
 * what the app does instead.
 */
@Composable
fun HomeScreen(vm: ScanViewModel) {
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { if (it.isNotEmpty()) vm.importNew(it) }
    val importPhotos = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    val docs by vm.library.documents.collectAsState(initial = null)
    val trashed by vm.library.trash.collectAsState(initial = emptyList())
    var query by rememberSaveable { mutableStateOf("") }
    var searching by rememberSaveable { mutableStateOf(false) }
    var pagesView by remember { mutableStateOf(vm.settings.homePages) }
    var selected by remember { mutableStateOf(setOf<String>()) }
    var renaming by remember { mutableStateOf<DocumentSummary?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }
    val all = docs.orEmpty()
    val shown = remember(docs, query, vm.searchTextVersion) { filterDocuments(all, query) { vm.searchText(it) } }
    val open = all.firstOrNull { it.id == vm.docId }
    val selecting = selected.isNotEmpty()
    val empty = docs?.isEmpty() == true
    val toggle = { id: String -> selected = if (id in selected) selected - id else selected + id }
    val tap = { d: DocumentSummary -> if (selecting) toggle(d.id) else vm.openDocument(d.id) }
    val stopSearch = { searching = false; query = "" }

    BackHandler(selecting || searching) { if (selecting) selected = emptySet() else stopSearch() }

    Box(Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize().safeDrawingPadding()) {
            when {
                selecting -> SelectionTopBar(selected.size, onClear = { selected = emptySet() }, onAll = { selected = (if (searching) shown else all).map { it.id }.toSet() })
                searching -> SearchTopBar(query, hasText = vm.hasOcrText, onQuery = { query = it }, onClose = stopSearch)
                else -> AppTopBar(onSettings = { vm.open(Screen.Settings) })
            }
            LazyColumn(
                Modifier.weight(1f).fillMaxWidth().testTag("history"),
                contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 4.dp, bottom = if (searching && !selecting) 24.dp else 112.dp),
            ) {
                when {
                    docs == null -> {}
                    empty -> firstRun()
                    searching -> searchResults(vm, query, shown, selected, tap, toggle)
                    else -> {
                        if (open != null && !selecting) item("carry_on") {
                            CarryOnCard(vm, open, onAdd = { vm.startCamera() }, onOpen = { vm.openDocument(open.id) })
                        }
                        item("search_field") {
                            SearchField(vm.hasOcrText, Modifier.padding(top = 14.dp)) { searching = true; selected = emptySet() }
                        }
                        item("count") {
                            CountRow(all.size, all.sumOf { it.pageCount }, pagesView) { pagesView = it; vm.settings.homePages = it }
                        }
                        if (pagesView) laidOutPages(vm, all, selected, tap, toggle)
                        else ruledList(vm, all, selected, tap, toggle)
                        item("trash") {
                            TrashLine(trashed.size, last = !pagesView) { vm.open(Screen.Trash) }
                        }
                    }
                }
            }
        }
        when {
            selecting -> SelectionActions(
                Modifier.align(Alignment.BottomCenter), canRename = selected.size == 1,
                onRename = { renaming = all.firstOrNull { it.id in selected } },
                onUpload = { vm.uploadDocuments(selected); selected = emptySet() },
                onTrash = { confirmDelete = true },
            )
            !searching && docs != null -> {
                if (empty) StartHere(Modifier.align(Alignment.BottomEnd).navigationBarsPadding().padding(end = 52.dp, bottom = 96.dp))
                HomeDock(Modifier.align(Alignment.BottomCenter), onImport = importPhotos, onScan = { vm.scanNew() })
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
            title = { Text(if (n == 1) "Move this document to the trash?" else "Move $n documents to the trash?") },
            text = { Text("The trash keeps it for ${Library.TRASH_DAYS} days.") },
            confirmButton = {
                TextButton(onClick = { vm.trashDocuments(selected); selected = emptySet(); confirmDelete = false }, modifier = Modifier.testTag("confirm_delete")) { Text("Move to trash") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
    }
}

// ---------------------------------------------------------------- top bars

@Composable
private fun AppTopBar(onSettings: () -> Unit) {
    Row(Modifier.fillMaxWidth().height(60.dp).padding(start = 18.dp, end = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        LogoMark(Modifier.size(26.dp))
        Spacer(Modifier.width(10.dp))
        Text("FlatScan", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
        IconButton(onClick = onSettings) { Icon(Icons.Outlined.Settings, "Settings") }
    }
}

@Composable
private fun SelectionTopBar(count: Int, onClear: () -> Unit, onAll: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(60.dp).background(MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.6f)).padding(horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onClear) { Icon(Icons.Outlined.Close, "Clear the selection") }
        Text("$count selected", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
        TextButton(onClick = onAll) { Text("Select all") }
    }
}

@Composable
private fun SearchTopBar(query: String, hasText: Boolean, onQuery: (String) -> Unit, onClose: () -> Unit) {
    val focus = remember { FocusRequester() }
    LaunchedEffect(Unit) { runCatching { focus.requestFocus() } }
    Row(Modifier.fillMaxWidth().height(64.dp).padding(start = 4.dp, end = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onClose) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Stop searching") }
        OutlinedTextField(
            value = query, onValueChange = onQuery, singleLine = true,
            placeholder = { Text(if (hasText) "Search titles and page text" else "Search titles") },
            trailingIcon = { if (query.isNotEmpty()) IconButton(onClick = { onQuery("") }) { Icon(Icons.Outlined.Close, "Clear the search") } },
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
            shape = RoundedCornerShape(10.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerLowest, unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerLowest,
            ),
            modifier = Modifier.weight(1f).focusRequester(focus).testTag("search"),
        )
    }
}

// ---------------------------------------------------------------- browsing

/** The document still open: add a page to it or open it, without hunting for it in the list. */
@Composable
private fun CarryOnCard(vm: ScanViewModel, d: DocumentSummary, onAdd: () -> Unit, onOpen: () -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    PaperSurface(Modifier.fillMaxWidth().padding(top = 12.dp).testTag("carry_on"), tape = true) {
        Row(Modifier.padding(start = 12.dp, end = 14.dp, top = 16.dp, bottom = 14.dp), verticalAlignment = Alignment.CenterVertically) {
            PageThumb(vm, d.firstPage, d.modifiedAt, d.pageCount, width = 62.dp, tilt = true)
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text("CARRY ON WITH", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono, letterSpacing = 0.1.em), color = muted)
                Text(d.title, style = MaterialTheme.typography.titleLarge.copy(fontSize = 19.sp), maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(
                    pageCount(d.pageCount) + " · " + relativeDate(d.modifiedAt),
                    style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono), color = muted, maxLines = 1, overflow = TextOverflow.Ellipsis,
                )
                Row(Modifier.padding(top = 10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    InkPressable(
                        onClick = onAdd, shape = RoundedCornerShape(9.dp), depth = 2.dp,
                        container = MaterialTheme.colorScheme.surfaceContainerLowest, contentColor = MaterialTheme.colorScheme.onSurface,
                        border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.onSurface), shadow = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.height(42.dp).testTag("carry_on_add"),
                    ) {
                        Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Outlined.Add, null, Modifier.size(17.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Add a page", style = MaterialTheme.typography.labelLarge)
                        }
                    }
                    InkPressable(
                        onClick = onOpen, shape = RoundedCornerShape(9.dp), depth = 2.dp,
                        container = MaterialTheme.colorScheme.primary, contentColor = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier.height(42.dp),
                    ) { Text("Open", style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(horizontal = 18.dp)) }
                }
            }
        }
    }
}

/** Looks like a search field; tapping it opens the search bar (and the keyboard). */
@Composable
private fun SearchField(hasText: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val shape = RoundedCornerShape(10.dp)
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Row(
        modifier.fillMaxWidth().height(48.dp).clip(shape).background(MaterialTheme.colorScheme.surfaceContainerLowest)
            .border(1.dp, MaterialTheme.colorScheme.outline, shape).clickable(onClickLabel = "Search", onClick = onClick)
            .padding(horizontal = 12.dp).testTag("search_open"),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Outlined.Search, null, Modifier.size(19.dp), tint = muted)
        Spacer(Modifier.width(10.dp))
        Text(if (hasText) "Search titles and page text" else "Search titles", style = MaterialTheme.typography.bodyMedium, color = muted)
    }
}

/** "6 DOCUMENTS · 25 PAGES" and the list / pages switch. */
@Composable
private fun CountRow(docs: Int, pages: Int, pagesView: Boolean, onView: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(top = 18.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(
            "${if (docs == 1) "1 DOCUMENT" else "$docs DOCUMENTS"} · ${if (pages == 1) "1 PAGE" else "$pages PAGES"}",
            style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, letterSpacing = 0.08.em),
            color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f),
        )
        Row(Modifier.clip(RoundedCornerShape(10.dp)).background(MaterialTheme.colorScheme.surfaceVariant).padding(3.dp)) {
            ViewChoice(Icons.AutoMirrored.Outlined.List, "List", !pagesView) { onView(false) }
            ViewChoice(Icons.Outlined.GridView, "Pages", pagesView) { onView(true) }
        }
    }
}

@Composable
private fun ViewChoice(icon: ImageVector, label: String, on: Boolean, onClick: () -> Unit) {
    val shape = RoundedCornerShape(7.dp)
    Box(
        Modifier.size(42.dp, 34.dp)
            .then(if (on) Modifier.shadow(1.dp, shape).background(MaterialTheme.colorScheme.surfaceContainerLowest, shape) else Modifier)
            .clip(shape).clickable(role = Role.RadioButton, onClickLabel = label, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, label, Modifier.size(19.dp), tint = if (on) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant) }
}

/**
 * The documents on one ruled sheet: a red margin with the time (today), the day (this week) or the
 * date, grouped under highlighted labels. Each row is its own list item; together they draw one sheet.
 */
private fun LazyListScope.ruledList(
    vm: ScanViewModel, docs: List<DocumentSummary>, selected: Set<String>,
    onTap: (DocumentSummary) -> Unit, onLong: (String) -> Unit,
) {
    val now = System.currentTimeMillis()
    var first = true
    for ((age, group) in docs.groupBy { ageOf(it.modifiedAt, now) }.toSortedMap()) {
        item("age_${age.name}") { SheetStrip(top = first) { AgeLabel(age.label) } }
        first = false
        items(group, key = { it.id }) { d ->
            SheetStrip(rule = true) {
                LedgerRow(vm, d, now, selected = d.id in selected, selecting = selected.isNotEmpty(), open = d.id == vm.docId, onClick = { onTap(d) }, onLongClick = { onLong(d.id) })
            }
        }
    }
}

/** One strip of the ruled sheet: paper, its side edges, the red margin, a rule along the top. */
@Composable
private fun SheetStrip(top: Boolean = false, bottom: Boolean = false, rule: Boolean = false, content: @Composable BoxScope.() -> Unit) {
    val light = isPaperLight()
    val edge = if (light) Palette.OutlineSoft else MaterialTheme.colorScheme.outlineVariant
    val margin = MaterialTheme.colorScheme.error.copy(alpha = if (light) 0.45f else 0.35f)
    val ruleColor = if (light) Color(0xFFDCE4F0) else MaterialTheme.colorScheme.outlineVariant
    val shape = RoundedCornerShape(topStart = if (top) 4.dp else 0.dp, topEnd = if (top) 4.dp else 0.dp, bottomStart = if (bottom) 4.dp else 0.dp, bottomEnd = if (bottom) 4.dp else 0.dp)
    Box(
        Modifier.fillMaxWidth().clip(shape).background(MaterialTheme.colorScheme.surfaceContainerLowest).drawBehind {
            val w = 1.dp.toPx()
            drawLine(margin, Offset(60.dp.toPx(), 0f), Offset(60.dp.toPx(), size.height), strokeWidth = w)
            if (rule) drawLine(ruleColor, Offset(0f, 0f), Offset(size.width, 0f), strokeWidth = w)
            drawLine(edge, Offset(w / 2, 0f), Offset(w / 2, size.height), strokeWidth = w)
            drawLine(edge, Offset(size.width - w / 2, 0f), Offset(size.width - w / 2, size.height), strokeWidth = w)
            if (top) drawLine(edge, Offset(0f, w / 2), Offset(size.width, w / 2), strokeWidth = w)
            if (bottom) drawLine(edge, Offset(0f, size.height - w / 2), Offset(size.width, size.height - w / 2), strokeWidth = w)
        },
        content = content,
    )
}

@Composable
private fun AgeLabel(label: String) {
    val marker = if (isPaperLight()) Palette.Highlighter.copy(alpha = 0.9f) else Graphite.Highlighter.copy(alpha = 0.35f)
    Text(
        label, style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono, fontWeight = FontWeight.Bold, letterSpacing = 0.12.em),
        modifier = Modifier.padding(start = 72.dp, top = 14.dp, bottom = 6.dp).drawBehind {
            drawRect(marker, Offset(-3.dp.toPx(), size.height * 0.5f), androidx.compose.ui.geometry.Size(size.width + 6.dp.toPx(), size.height * 0.45f))
        },
    )
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun LedgerRow(
    vm: ScanViewModel, d: DocumentSummary, now: Long, selected: Boolean, selecting: Boolean, open: Boolean,
    onClick: () -> Unit, onLongClick: () -> Unit,
) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Row(
        Modifier.fillMaxWidth().height(68.dp)
            .background(if (selected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f) else Color.Transparent)
            .combinedClickable(onClick = onClick, onLongClick = onLongClick).padding(end = 14.dp).testTag("doc_${d.id}"),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            marginLabel(d.modifiedAt, now), style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono, fontSize = 11.sp),
            color = muted, textAlign = TextAlign.Center, maxLines = 1, modifier = Modifier.width(60.dp),
        )
        Spacer(Modifier.width(12.dp))
        PageThumb(vm, d.firstPage, d.modifiedAt, d.pageCount, width = 36.dp)
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(1.dp)) {
            Text(d.title, style = MaterialTheme.typography.titleSmall.copy(fontSize = 15.sp, fontWeight = FontWeight.Bold), maxLines = 1, overflow = TextOverflow.Ellipsis)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(pageCount(d.pageCount), style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono), color = muted, maxLines = 1)
                SyncLine(d, Modifier.padding(start = 8.dp))
            }
        }
        when {
            selecting -> CheckDot(selected)
            open -> Text(
                "Open", style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold), color = Palette.Ink,
                modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(Palette.Highlighter).padding(horizontal = 7.dp, vertical = 3.dp),
            )
        }
    }
}

@Composable
private fun CheckDot(on: Boolean) {
    val c = MaterialTheme.colorScheme
    Box(
        Modifier.size(24.dp).clip(CircleShape).background(if (on) c.primary else c.surfaceContainerLowest)
            .border(2.dp, if (on) c.primary else c.onSurfaceVariant, CircleShape),
        contentAlignment = Alignment.Center,
    ) { if (on) Icon(Icons.Outlined.Check, "Selected", Modifier.size(15.dp), tint = c.onPrimary) }
}

/** The documents as their first pages laid on the desk, two across, each a little askew. */
private fun LazyListScope.laidOutPages(
    vm: ScanViewModel, docs: List<DocumentSummary>, selected: Set<String>,
    onTap: (DocumentSummary) -> Unit, onLong: (String) -> Unit,
) {
    val now = System.currentTimeMillis()
    val tilts = floatArrayOf(-1.2f, 1f, 0.6f, -0.8f, 1.3f, -0.5f)
    docs.chunked(2).forEachIndexed { row, pair ->
        item("pages_${pair.first().id}") {
            Row(Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 10.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                pair.forEachIndexed { k, d ->
                    PageTile(
                        vm, d, now, tilts[(row * 2 + k) % tilts.size], selected = d.id in selected, selecting = selected.isNotEmpty(), open = d.id == vm.docId,
                        onClick = { onTap(d) }, onLongClick = { onLong(d.id) }, modifier = Modifier.weight(1f),
                    )
                }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PageTile(
    vm: ScanViewModel, d: DocumentSummary, now: Long, tilt: Float, selected: Boolean, selecting: Boolean, open: Boolean,
    onClick: () -> Unit, onLongClick: () -> Unit, modifier: Modifier,
) {
    var thumb by remember(d.firstPage) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(d.firstPage, d.modifiedAt, d.firstPage?.let { vm.pageVersions[it] }) { thumb = d.firstPage?.let { vm.thumbnail(it) } }
    val sheet = RoundedCornerShape(2.dp)
    val behind = minOf(d.pageCount - 1, 2).coerceAtLeast(0)
    Column(modifier.combinedClickable(onClick = onClick, onLongClick = onLongClick).testTag("doc_${d.id}")) {
        Box(Modifier.fillMaxWidth().height(186.dp)) {
            for (k in behind downTo 1) {
                Box(
                    Modifier.padding(start = (12 + 5 * k).dp, end = (12 - 5 * k).coerceAtLeast(0).dp, top = (8 - 2 * k).dp, bottom = 10.dp).fillMaxSize()
                        .rotate(tilt + 3.5f * k).shadow(1.dp, sheet).background(Color.White, sheet).border(1.dp, Palette.Outline, sheet),
                )
            }
            Box(
                Modifier.padding(start = 10.dp, end = 14.dp, top = 8.dp, bottom = 10.dp).fillMaxSize().rotate(tilt)
                    .then(if (selected) Modifier.border(3.dp, MaterialTheme.colorScheme.primary, sheet).padding(4.dp) else Modifier)
                    .shadow(3.dp, sheet).background(Color.White, sheet).border(1.dp, Palette.Outline, sheet).clip(sheet),
                contentAlignment = Alignment.Center,
            ) {
                val t = thumb
                if (t != null) Image(remember(t) { t.asImageBitmap() }, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                else Icon(Icons.Outlined.Description, null, tint = Color(0xFFB9B7B1))
            }
            if (open && !selecting) Text(
                "OPEN", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono, fontWeight = FontWeight.ExtraBold, letterSpacing = 0.1.em),
                color = Palette.Ink, modifier = Modifier.offset(2.dp, 8.dp).rotate(-8f).background(Palette.Highlighter.copy(alpha = 0.9f)).padding(horizontal = 10.dp, vertical = 2.dp),
            )
            if (d.pageCount > 1) Text(
                "${d.pageCount}", style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono, fontWeight = FontWeight.Bold), color = Color.White,
                modifier = Modifier.align(Alignment.BottomEnd).padding(end = 8.dp, bottom = 4.dp).clip(RoundedCornerShape(11.dp)).background(Palette.Ink).padding(horizontal = 7.dp, vertical = 3.dp),
            )
            if (selecting) Box(Modifier.align(Alignment.TopEnd).padding(top = 14.dp, end = 18.dp)) { CheckDot(selected) }
        }
        Text(d.title, style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold), maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(horizontal = 4.dp))
        Text(
            listOf(pageCount(d.pageCount), marginLabel(d.modifiedAt, now).lowercase().replaceFirstChar { it.titlecase() }).joinToString(" · "),
            style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono, fontSize = 11.sp),
            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, modifier = Modifier.padding(horizontal = 4.dp),
        )
    }
}

/** The foot of the list (the last strip of the ruled sheet): the trash and how long it keeps things. */
@Composable
private fun TrashLine(count: Int, last: Boolean, onClick: () -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val body: @Composable () -> Unit = {
        Row(Modifier.fillMaxWidth().height(54.dp).clickable(onClickLabel = "Open the trash", onClick = onClick), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.width(60.dp), contentAlignment = Alignment.Center) { Icon(Icons.Outlined.Delete, null, Modifier.size(19.dp), tint = muted) }
            Spacer(Modifier.width(12.dp))
            Text("Trash", style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold), color = muted)
            Spacer(Modifier.width(8.dp))
            Text(
                (if (count == 0) "empty" else if (count == 1) "1 document" else "$count documents") + ", kept ${Library.TRASH_DAYS} days",
                style = MaterialTheme.typography.bodySmall, color = muted, maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
        }
    }
    if (last) SheetStrip(bottom = true, rule = true) { body() }
    else Box(Modifier.padding(top = 4.dp)) { body() }
}

// ---------------------------------------------------------------- search

private fun LazyListScope.searchResults(
    vm: ScanViewModel, query: String, shown: List<DocumentSummary>, selected: Set<String>,
    onTap: (DocumentSummary) -> Unit, onLong: (String) -> Unit,
) {
    if (query.isBlank()) {
        item("search_hint") {
            Text(
                if (vm.hasOcrText) "Type a word from a title or from the pages." else "Type a word from a title.",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 16.dp),
            )
        }
        return
    }
    item("search_count") {
        Text(
            when (shown.size) { 0 -> "NOTHING FOUND"; 1 -> "1 DOCUMENT"; else -> "${shown.size} DOCUMENTS" },
            style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, letterSpacing = 0.08.em),
            color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 14.dp, bottom = 10.dp),
        )
    }
    items(shown, key = { it.id }) { d ->
        SearchResultRow(vm, d, query, selected = d.id in selected, onClick = { onTap(d) }, onLongClick = { onLong(d.id) })
    }
    item("search_note") {
        Text(
            "Page text is searched in documents scanned with Searchable text on.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 6.dp),
        )
    }
}

@Composable
private fun SearchResultRow(vm: ScanViewModel, d: DocumentSummary, query: String, selected: Boolean, onClick: () -> Unit, onLongClick: () -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val mark = SpanStyle(
        color = MaterialTheme.colorScheme.onSurface, fontWeight = FontWeight.Bold,
        background = if (isPaperLight()) Palette.Highlighter.copy(alpha = 0.85f) else Graphite.Highlighter.copy(alpha = 0.35f),
    )
    val snippet = remember(d.id, query, vm.searchTextVersion) { searchSnippet(vm.searchText(d.id), query, mark) }
    PaperRow(selected, selected, Modifier.padding(bottom = 10.dp).testTag("doc_${d.id}"), onClick, onLongClick) {
        PageThumb(vm, d.firstPage, d.modifiedAt, d.pageCount, width = 44.dp)
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(d.title, style = MaterialTheme.typography.titleSmall.copy(fontSize = 15.sp, fontWeight = FontWeight.Bold), maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (snippet != null) Text(snippet, style = MaterialTheme.typography.bodySmall.copy(fontSize = 13.sp, lineHeight = 19.sp), color = muted, maxLines = 3, overflow = TextOverflow.Ellipsis)
            Text(
                pageCount(d.pageCount) + " · " + relativeDate(d.modifiedAt),
                style = MaterialTheme.typography.labelSmall.copy(fontFamily = RecursiveMono), color = muted, maxLines = 1,
            )
        }
    }
}

// ---------------------------------------------------------------- nothing scanned yet

@OptIn(ExperimentalLayoutApi::class)
private fun LazyListScope.firstRun() {
    item("headline") {
        Column(Modifier.padding(start = 4.dp, end = 4.dp, top = 14.dp)) {
            HighlightedHeadline("Flat, clean pages\nfrom ", "any photo.")
            Spacer(Modifier.height(12.dp))
            Text(
                "Bent, curled or in shadow: FlatScan finds the page, irons it flat and whitens the paper, all on your phone.",
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 15.sp, lineHeight = 22.sp), color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    item("how") {
        PaperSurface(Modifier.fillMaxWidth().padding(top = 26.dp), tape = true) {
            Column(Modifier.padding(start = 12.dp, end = 12.dp, top = 18.dp, bottom = 12.dp)) {
                HowItWorks(
                    Modifier.fillMaxWidth().height(112.dp),
                    ink = MaterialTheme.colorScheme.onSurface, paper = MaterialTheme.colorScheme.surfaceContainerLowest,
                    accent = MaterialTheme.colorScheme.primary, marker = if (isPaperLight()) Palette.Highlighter else Graphite.Highlighter,
                )
                Row(Modifier.fillMaxWidth().padding(top = 6.dp)) {
                    for (t in listOf("1 · Snap it", "2 · Iron it", "3 · Flat, clean")) {
                        Text(
                            t, style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono), color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center, modifier = Modifier.weight(1f),
                        )
                    }
                }
            }
        }
    }
    item("good_at") {
        Column(Modifier.padding(top = 22.dp)) {
            Text(
                "GOOD AT", style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, letterSpacing = 0.1.em),
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(bottom = 8.dp),
            )
            FlowRow(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                for (t in listOf("Curled book pages", "Crumpled receipts", "Forms and tables", "Whiteboards", "Pages in shadow")) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Outlined.Check, null, Modifier.size(17.dp), tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(5.dp))
                        Text(t, style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold))
                    }
                }
            }
        }
    }
}

/** First run: a handwritten note and an arrow down to the Scan button. */
@Composable
private fun StartHere(modifier: Modifier) {
    val pen = MaterialTheme.colorScheme.primary
    Row(modifier.rotate(-4f), verticalAlignment = Alignment.Bottom) {
        Text(
            "Start here", style = MaterialTheme.typography.titleMedium.copy(fontFamily = RecursiveDisplay), color = pen,
            modifier = Modifier.padding(bottom = 18.dp, end = 4.dp),
        )
        androidx.compose.foundation.Canvas(Modifier.size(40.dp, 46.dp)) {
            val u = size.width / 44f
            val stroke = androidx.compose.ui.graphics.drawscope.Stroke(2.4f * u, cap = androidx.compose.ui.graphics.StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round)
            val p = androidx.compose.ui.graphics.Path().apply { moveTo(4 * u, 6 * u); cubicTo(24 * u, 4 * u, 38 * u, 16 * u, 34 * u, 42 * u) }
            drawPath(p, pen, style = stroke)
            val head = androidx.compose.ui.graphics.Path().apply { moveTo(26 * u, 36 * u); lineTo(34 * u, 44 * u); lineTo(40 * u, 34 * u) }
            drawPath(head, pen, style = stroke)
        }
    }
}

// ---------------------------------------------------------------- the bottom

/** Scan and Import, always under the thumb; the list fades out behind them. */
@Composable
private fun HomeDock(modifier: Modifier, onImport: () -> Unit, onScan: () -> Unit) {
    val desk = MaterialTheme.colorScheme.background
    Row(
        modifier.fillMaxWidth().background(Brush.verticalGradient(0f to desk.copy(alpha = 0f), 0.4f to desk))
            .navigationBarsPadding().padding(start = 16.dp, end = 16.dp, top = 28.dp, bottom = 14.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.Bottom,
    ) {
        PaperSurface(onClick = onImport, modifier = Modifier.weight(1f).padding(bottom = 4.dp).height(60.dp).testTag("import")) {
            Box(Modifier.fillMaxWidth().padding(top = 9.dp).height(1.dp).background(MaterialTheme.colorScheme.error.copy(alpha = 0.5f)))
            Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.PhotoLibrary, null, Modifier.size(21.dp))
                Spacer(Modifier.width(9.dp))
                Text("Import", style = MaterialTheme.typography.titleMedium)
            }
        }
        InkPressable(
            onClick = onScan, shape = RoundedCornerShape(12.dp), depth = 4.dp,
            container = MaterialTheme.colorScheme.primary, contentColor = MaterialTheme.colorScheme.onPrimary,
            modifier = Modifier.weight(1.75f).height(64.dp).testTag("scan"),
        ) {
            IronOnSheet(Modifier.align(Alignment.CenterEnd).offset(x = 6.dp, y = 8.dp).size(78.dp), color = LocalContentColor.current.copy(alpha = 0.24f))
            Row(Modifier.fillMaxSize().padding(horizontal = 18.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.PhotoCamera, null, Modifier.size(25.dp))
                Spacer(Modifier.width(10.dp))
                Text("Scan", style = MaterialTheme.typography.titleLarge)
            }
        }
    }
}

/** What a selection can do, labelled, where the dock was. */
@Composable
private fun SelectionActions(modifier: Modifier, canRename: Boolean, onRename: () -> Unit, onUpload: () -> Unit, onTrash: () -> Unit) {
    Row(
        modifier.fillMaxWidth().shadow(8.dp).background(MaterialTheme.colorScheme.surfaceContainerLowest)
            .navigationBarsPadding().padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        SelectionAction(Icons.Outlined.Edit, "Rename", enabled = canRename, onClick = onRename)
        SelectionAction(Icons.Outlined.CloudUpload, "Upload", onClick = onUpload)
        SelectionAction(Icons.Outlined.Delete, "Trash", tint = MaterialTheme.colorScheme.error, tag = "delete_selected", onClick = onTrash)
    }
}

@Composable
private fun RowScope.SelectionAction(
    icon: ImageVector, label: String, enabled: Boolean = true, tint: Color = MaterialTheme.colorScheme.onSurface, tag: String? = null, onClick: () -> Unit,
) {
    val c = if (enabled) tint else tint.copy(alpha = 0.35f)
    Column(
        Modifier.weight(1f).height(60.dp).clip(RoundedCornerShape(10.dp)).clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .then(if (tag != null) Modifier.testTag(tag) else Modifier),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Icon(icon, null, Modifier.size(22.dp), tint = c)
        Spacer(Modifier.height(4.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = c)
    }
}

// ---------------------------------------------------------------- shared pieces

private fun pageCount(n: Int) = if (n == 1) "1 page" else "$n pages"

/** The list's margin: the time today, YEST., the weekday this week, else the date (with the year when it is not this one). */
@Composable
private fun marginLabel(t: Long, now: Long): String {
    val ctx = LocalContext.current
    return remember(t, now / 60_000) {
        val today = startOfToday(now)
        val loc = Locale.getDefault()
        val then = Calendar.getInstance().apply { timeInMillis = t }
        val thisYear = Calendar.getInstance().apply { timeInMillis = now }.get(Calendar.YEAR) == then.get(Calendar.YEAR)
        when {
            t >= today -> DateFormat.getTimeFormat(ctx).format(Date(t))
            t >= today - TimeUnit.DAYS.toMillis(1) -> "YEST."
            t >= today - TimeUnit.DAYS.toMillis(6) -> SimpleDateFormat("EEE", loc).format(Date(t)).uppercase(loc)
            thisYear -> SimpleDateFormat("d MMM", loc).format(Date(t)).uppercase(loc)
            else -> SimpleDateFormat("MMM yyyy", loc).format(Date(t)).uppercase(loc)
        }
    }
}

/** A history entry on a sheet of paper; selected or open, it is outlined in the accent. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PaperRow(selected: Boolean, outlined: Boolean, modifier: Modifier, onClick: () -> Unit, onLongClick: (() -> Unit)?, content: @Composable RowScope.() -> Unit) {
    val shape = MaterialTheme.shapes.extraSmall
    PaperSurface(
        modifier.fillMaxWidth().then(if (outlined) Modifier.border(2.dp, MaterialTheme.colorScheme.primary, shape) else Modifier),
        color = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceContainerLowest,
    ) {
        Row(
            Modifier.fillMaxWidth().combinedClickable(onClick = onClick, onLongClick = onLongClick).padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) { content() }
    }
}

/** The document's sync state (feature 6); nothing when it is not synced. */
@Composable
private fun SyncLine(d: DocumentSummary, modifier: Modifier = Modifier) {
    val (icon, text, color) = when (d.syncStatus) {
        SyncStatus.None -> return
        SyncStatus.Pending -> Triple(Icons.Outlined.CloudUpload, "Waiting to upload", MaterialTheme.colorScheme.onSurfaceVariant)
        SyncStatus.Uploading -> Triple(Icons.Outlined.CloudUpload, "Uploading", MaterialTheme.colorScheme.primary)
        SyncStatus.Uploaded -> Triple(Icons.Outlined.CloudDone, "Uploaded", MaterialTheme.colorScheme.primary)
        SyncStatus.Failed -> Triple(Icons.Outlined.CloudOff, "Upload failed" + (d.syncError?.let { ": $it" } ?: ""), MaterialTheme.colorScheme.error)
    }
    Row(modifier, verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, null, Modifier.size(14.dp), tint = color)
        Spacer(Modifier.width(4.dp))
        Text(text, style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold), color = color, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

/**
 * The first page over the sheets behind it (up to two), squared up like a stack on a desk; [width]
 * sets the page (A4-ish proportions), [tilt] turns the front page a little and fans the rest.
 */
@Composable
private fun PageThumb(vm: ScanViewModel, page: String?, version: Long, pages: Int, width: Dp = 52.dp, tilt: Boolean = false) {
    var thumb by remember(page) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(page, version, page?.let { vm.pageVersions[it] }) { thumb = page?.let { vm.thumbnail(it) } }
    val sheet = RoundedCornerShape(2.dp)
    val edge = Palette.Outline
    val behind = minOf(pages - 1, 2).coerceAtLeast(0)
    val h = width * 1.35f
    val step = width / 9f
    Box(Modifier.size(width + step * behind + (if (tilt) 6.dp else 0.dp), h + 4.dp)) {
        for (k in behind downTo 1) {
            Box(
                Modifier.offset(step * k, (4 - 2 * k).dp).size(width, h).rotate(if (tilt) 4f * k else 0f)
                    .background(Color.White, sheet).border(1.dp, edge, sheet),
            )
        }
        Box(
            Modifier.offset(0.dp, 4.dp).size(width, h).rotate(if (tilt) -2f else 0f).shadow(3.dp, sheet).background(Color.White, sheet).border(1.dp, edge, sheet).clip(sheet),
            contentAlignment = Alignment.Center,
        ) {
            val t = thumb
            if (t != null) Image(remember(t) { t.asImageBitmap() }, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            else Icon(Icons.Outlined.Description, null, Modifier.size(width / 2), tint = Color(0xFFB9B7B1))
        }
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
            if (docs.isEmpty()) item {
                Row(Modifier.fillMaxWidth().padding(top = 24.dp), verticalAlignment = Alignment.CenterVertically) {
                    CrumpledBall(Modifier.size(48.dp), paper = MaterialTheme.colorScheme.surfaceContainerLowest, ink = MaterialTheme.colorScheme.onSurface)
                    Spacer(Modifier.width(14.dp))
                    Text("The trash is empty.", style = MaterialTheme.typography.titleMedium)
                }
            }
            items(docs, key = { it.id }) { d ->
                val left = (Library.TRASH_DAYS - TimeUnit.MILLISECONDS.toDays(System.currentTimeMillis() - (d.deletedAt ?: 0))).coerceAtLeast(0)
                val isSel = d.id in selected
                PaperRow(isSel, isSel, Modifier, onClick = { selected = if (isSel) selected - d.id else selected + d.id }, onLongClick = null) {
                    PageThumb(vm, d.firstPage, d.modifiedAt, d.pageCount)
                    Spacer(Modifier.width(14.dp))
                    Column(Modifier.weight(1f)) {
                        Text(d.title, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(
                            (if (d.pageCount == 1) "1 page" else "${d.pageCount} pages") + " · deleted for good in $left " + (if (left == 1L) "day" else "days"),
                            style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono), color = muted,
                        )
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
