package com.albus.flatscan.ui

import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.Folder
import androidx.compose.material.icons.outlined.GridOn
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.SaveAlt
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.ColorMode
import com.albus.flatscan.ExportPrefs
import com.albus.flatscan.Exporter
import com.albus.flatscan.FileNameTemplate
import com.albus.flatscan.OutputFormat
import com.albus.flatscan.OutputOptions
import com.albus.flatscan.SizePreset
import com.albus.flatscan.PdfMargin
import com.albus.flatscan.PdfPageSize
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.formatBytes
import com.albus.flatscan.output.OcrPack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Save / share of the whole document; hold it at screen level (the launchers must outlive the sheet). */
class ExportActions(
    /** Saves with options [o] (null: the remembered ones): asks where, or into the default folder. */
    val save: (OutputOptions?) -> Unit,
    val share: (OutputOptions?) -> Unit,
    private val busyState: () -> Boolean,
    private val stageState: () -> String,
) {
    val busy: Boolean get() = busyState()
    /** What a running export is doing ("Recognising text, page 2 of 5"). */
    val stage: String get() = stageState()
}

@Composable
fun rememberExportActions(vm: ScanViewModel): ExportActions {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }
    var stage by remember { mutableStateOf("") }
    // the export waiting for the system picker's answer
    var pendingPrepared by remember { mutableStateOf<Exporter.Prepared?>(null) }
    var pendingOptions by remember { mutableStateOf<OutputOptions?>(null) }

    fun launchExport(work: suspend () -> String?) {
        if (busy) return
        busy = true
        scope.launch {
            val msg = try {
                work()
            } catch (e: Exception) {
                Log.e(ScanViewModel.TAG, "export failed", e)
                "Could not save: ${e.message ?: e.javaClass.simpleName}"
            }
            busy = false
            stage = ""
            if (msg != null) vm.message = msg
        }
    }

    suspend fun prepared(o: OutputOptions): Exporter.Prepared? =
        vm.prepareExport(o) { s, _ -> scope.launch { stage = s } }?.takeIf { it.pages.isNotEmpty() }

    val savePdf = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { uri: Uri? ->
        val o = pendingOptions
        if (uri != null && o != null) launchExport {
            val p = pendingPrepared ?: prepared(o) ?: return@launchExport "Nothing to save"
            withContext(Dispatchers.IO) { ctx.contentResolver.openOutputStream(uri, "wt")?.use { vm.exporter.writePdf(p, it) } ?: error("cannot open the file") }
            if (p.pages.size == 1) "Saved" else "Saved ${p.pages.size} pages"
        }
        pendingPrepared = null
    }
    val saveFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree: Uri? ->
        val o = pendingOptions
        if (tree != null && o != null) launchExport {
            val p = pendingPrepared ?: prepared(o) ?: return@launchExport "Nothing to save"
            withContext(Dispatchers.IO) { vm.exporter.saveToFolder(p, tree) }
        }
        pendingPrepared = null
    }
    return remember(vm) {
        ExportActions(
            save = { given ->
                val o = given ?: ExportPrefs.output(ctx)
                pendingOptions = o
                val folder = o.folder?.let(Uri::parse)
                when {
                    !o.askWhere && folder != null -> launchExport {
                        val p = prepared(o) ?: return@launchExport "Nothing to save"
                        withContext(Dispatchers.IO) { vm.exporter.saveToFolder(p, folder) } + " to the export folder"
                    }
                    o.format == OutputFormat.Pdf -> savePdf.launch(vm.exportFileName(o) + ".pdf")
                    else -> saveFolder.launch(null)
                }
            },
            share = { given ->
                val o = given ?: ExportPrefs.output(ctx)
                launchExport {
                    val p = prepared(o) ?: return@launchExport "Nothing to share"
                    val intent = withContext(Dispatchers.IO) { vm.exporter.shareIntent(p) }
                    ctx.startActivity(intent)
                    null
                }
            },
            busyState = { busy },
            stageState = { stage },
        )
    }
}


/**
 * Bottom sheet for saving or sharing all [pageCount] pages, laid out from the design canvas "FlatScan editing tools and
 * export, round 2": the name; what you get (the format as three cards, colour, size with the size the export will
 * have, searchable text); paper, margins, the file name and where to save folded under More options; and Share and
 * Save side by side at the bottom, out of the scroll ([share]: the sheet was opened to share, so Share is the filled
 * one). The choices are remembered.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ExportSheet(
    vm: ScanViewModel, share: Boolean, pageCount: Int, title: String?, onRename: (String) -> Unit,
    onSave: (OutputOptions) -> Unit, onShare: (OutputOptions) -> Unit, onDismiss: () -> Unit,
) {
    val ctx = LocalContext.current
    var name by remember { mutableStateOf(title ?: "") }
    val placeholder = remember { com.albus.flatscan.DocumentName.default() }
    var o by remember { mutableStateOf(ExportPrefs.output(ctx)) }
    var more by remember { mutableStateOf(false) }
    var estimate by remember { mutableStateOf<String?>(null) }
    val state = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val pickFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
        if (uri != null) {
            ctx.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            o = o.copy(folder = uri.toString(), askWhere = false)
        }
    }
    // the size the export will have: the pages encoded with these options (OCR left out: it adds only text)
    LaunchedEffect(o.format, o.color, o.size, o.pageSize, o.margin) {
        estimate = null
        kotlinx.coroutines.delay(300)
        val p = vm.prepareExport(o.copy(searchable = false)) { _, _ -> }
        estimate = p?.let { "about " + formatBytes(it.totalBytes) }
    }
    fun go(sharing: Boolean) {
        if (name != (title ?: "")) onRename(name)
        ExportPrefs.setOutput(ctx, o)
        onDismiss()
        if (sharing) onShare(o) else onSave(o)
    }
    val pdf = o.format == OutputFormat.Pdf || o.format == OutputFormat.PdfAndJpeg
    val toFolder = !o.askWhere && o.folder != null
    val folderName = o.folder?.let { Uri.parse(it).lastPathSegment?.substringAfterLast(':') ?: "chosen folder" }
    val fileName = vm.exportFileName(o) + if (pdf) ".pdf" else ""
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = state) {
        Column(Modifier.navigationBarsPadding()) {
            Column(
                Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()).padding(start = 20.dp, end = 20.dp, bottom = 12.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                OutlinedTextField(
                    value = name, onValueChange = { name = it.take(80) },
                    label = { Text("Name") }, placeholder = { Text(placeholder) }, singleLine = true,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                    modifier = Modifier.fillMaxWidth(),
                )
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SheetLabel("Format")
                    Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min).selectableGroup(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FormatCard("PDF", "One file", PaperIcons.KindDocument, pdf, Modifier.weight(1f).fillMaxHeight()) {
                            if (!pdf) o = o.copy(format = OutputFormat.Pdf)
                        }
                        FormatCard("JPEG", "A picture per page", Icons.Outlined.Image, o.format == OutputFormat.Jpeg, Modifier.weight(1f).fillMaxHeight()) {
                            o = o.copy(format = OutputFormat.Jpeg)
                        }
                        FormatCard("PNG", "Lossless pictures", Icons.Outlined.GridOn, o.format == OutputFormat.Png, Modifier.weight(1f).fillMaxHeight()) {
                            o = o.copy(format = OutputFormat.Png)
                        }
                    }
                    if (pdf) {
                        Row(
                            Modifier.clip(RoundedCornerShape(8.dp))
                                .toggleable(o.format == OutputFormat.PdfAndJpeg, role = Role.Checkbox) {
                                    o = o.copy(format = if (it) OutputFormat.PdfAndJpeg else OutputFormat.Pdf)
                                }.padding(end = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Checkbox(checked = o.format == OutputFormat.PdfAndJpeg, onCheckedChange = null, modifier = Modifier.minimumInteractiveComponentSize())
                            Text("Also save each page as a JPEG", style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SheetLabel("Colour")
                    SheetSegments(
                        "Colour", listOf(ColorMode.Color to "Colour", ColorMode.Grayscale to "Grey", ColorMode.BlackWhite to "B&W"), o.color,
                    ) { o = o.copy(color = it) }
                }
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        SheetLabel("Size", Modifier.weight(1f))
                        Text(
                            "$pageCount ${if (pageCount == 1) "page" else "pages"}, " + (estimate ?: "working it out..."),
                            style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono, fontWeight = FontWeight.Medium),
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    SheetSegments("Size", SizePreset.entries.map { it to it.label }, o.size) { o = o.copy(size = it) }
                    if (o.color == ColorMode.BlackWhite && o.format != OutputFormat.Pdf) {
                        Text("Black and white pictures are saved as PNG.", style = MaterialTheme.typography.bodySmall, color = muted)
                    }
                }
                if (pdf) SearchableRow(vm, o.searchable) { o = o.copy(searchable = it) }
                Column {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    Row(
                        Modifier.fillMaxWidth().heightIn(min = 60.dp).clickable(role = Role.Button, onClickLabel = if (more) "Hide options" else "Show options") { more = !more },
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("More options", style = MaterialTheme.typography.titleSmall)
                            Text(
                                listOfNotNull(
                                    if (pdf) (if (o.pageSize == PdfPageSize.Fit) "Page size" else o.pageSize.label) else null,
                                    if (pdf && o.pageSize != PdfPageSize.Fit) "${o.margin.label.lowercase()} margins" else null,
                                    fileName,
                                ).joinToString(" · "),
                                style = MaterialTheme.typography.bodySmall, color = muted, maxLines = 1,
                            )
                        }
                        Icon(Icons.Outlined.ExpandMore, null, Modifier.rotate(if (more) 180f else 0f), tint = muted)
                    }
                    if (more) {
                        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                            if (pdf) {
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    SheetLabel("Paper")
                                    SheetSegments(
                                        "Paper", listOf(PdfPageSize.A4 to "A4", PdfPageSize.Letter to "Letter", PdfPageSize.Fit to "Fit the page"), o.pageSize,
                                    ) { o = o.copy(pageSize = it) }
                                }
                                if (o.pageSize != PdfPageSize.Fit) {
                                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                        SheetLabel("Margins")
                                        SheetSegments("Margins", PdfMargin.entries.map { it to it.label }, o.margin) { o = o.copy(margin = it) }
                                    }
                                }
                            }
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedTextField(
                                    value = o.template, onValueChange = { o = o.copy(template = it.take(80)) }, singleLine = true,
                                    label = { Text("File name") }, modifier = Modifier.fillMaxWidth(),
                                )
                                // the template's fields, added with a tap rather than typed in braces
                                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    for (f in listOf("title", "date", "time", "pages", "n")) {
                                        SheetChip("+ $f", false) { o = o.copy(template = (if (o.template.isBlank()) "{$f}" else o.template.trimEnd() + "_{$f}").take(80)) }
                                    }
                                }
                                Text(fileName, style = MaterialTheme.typography.bodySmall, color = muted)
                            }
                            if (o.template.contains("{date}")) {
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    SheetLabel("Date")
                                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                        for (d in FileNameTemplate.DATE_FORMATS) SheetChip(d, d == o.dateFormat) { o = o.copy(dateFormat = d) }
                                    }
                                }
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Outlined.Folder, null, Modifier.size(22.dp), tint = MaterialTheme.colorScheme.primary)
                                Spacer(Modifier.width(12.dp))
                                Column(Modifier.weight(1f)) {
                                    Text("Save to", style = MaterialTheme.typography.titleSmall)
                                    Text(if (toFolder) folderName ?: "" else "Ask each time", style = MaterialTheme.typography.bodySmall, color = muted)
                                }
                                if (toFolder) TextButton(onClick = { o = o.copy(askWhere = true) }) { Text("Ask each time") }
                                TextButton(onClick = { pickFolder.launch(null) }) { Text(if (o.folder == null) "Choose a folder" else "Change") }
                            }
                        }
                    }
                }
            }
            // the two actions, always in view
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                val shape = RoundedCornerShape(26.dp)
                val shareContent: @Composable () -> Unit = {
                    Icon(Icons.Outlined.Share, null, Modifier.size(20.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Share", style = MaterialTheme.typography.titleMedium)
                }
                val saveContent: @Composable () -> Unit = {
                    Icon(Icons.Outlined.SaveAlt, null, Modifier.size(20.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(if (toFolder) "Save to folder" else "Save", style = MaterialTheme.typography.titleMedium, maxLines = 1)
                }
                val m = Modifier.weight(1f).height(52.dp)
                if (share) {
                    Button(onClick = { go(true) }, modifier = m, shape = shape) { shareContent() }
                    OutlinedButton(onClick = { go(false) }, modifier = m, shape = shape) { saveContent() }
                } else {
                    OutlinedButton(onClick = { go(true) }, modifier = m, shape = shape) { shareContent() }
                    Button(onClick = { go(false) }, modifier = m, shape = shape) { saveContent() }
                }
            }
        }
    }
}

/** Searchable text (PDF): a download the first time (output/OcrPack.kt), so its state is the caption. */
@Composable
private fun SearchableRow(vm: ScanViewModel, on: Boolean, onChange: (Boolean) -> Unit) {
    val ocr = OcrPack.state
    val size = vm.ocrPackBytes?.let { formatBytes(it) }
    val ready = ocr is OcrPack.State.Ready
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).toggleable(on && ready, enabled = size != null, role = Role.Switch) {
            if (it && !ready) vm.downloadOcr()
            onChange(it)
        },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text("Searchable text", style = MaterialTheme.typography.titleSmall)
            Text(
                when {
                    ocr is OcrPack.State.Ready -> "Text you can select and search in the PDF (Hebrew and English)"
                    ocr is OcrPack.State.Downloading -> "Downloading text recognition: ${formatBytes(ocr.done)} of ${formatBytes(ocr.total)}"
                    ocr is OcrPack.State.Failed -> "The download failed: ${ocr.message}"
                    size == null -> "Not available for this phone's processor"
                    else -> "Text you can select and search; a one-time $size download"
                },
                style = MaterialTheme.typography.bodySmall,
                color = if (ocr is OcrPack.State.Failed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Switch(checked = on && ready, onCheckedChange = null, enabled = size != null)
    }
}

@Composable
private fun SheetLabel(text: String, modifier: Modifier = Modifier) =
    Text(text, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = modifier)

/** A format: its icon, name and what you get; the chosen one outlined and tinted in the accent. */
@Composable
private fun FormatCard(title: String, caption: String, icon: ImageVector, selected: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val accent = MaterialTheme.colorScheme.primary
    val shape = RoundedCornerShape(12.dp)
    val fg = if (selected) accent else MaterialTheme.colorScheme.onSurface
    Column(
        modifier.heightIn(min = 92.dp).clip(shape)
            .background(if (selected) accent.copy(alpha = 0.12f) else MaterialTheme.colorScheme.surfaceContainerHigh)
            .border(if (selected) 2.dp else 1.dp, if (selected) accent else MaterialTheme.colorScheme.outline, shape)
            .selectable(selected, role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = 6.dp, vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Icon(icon, null, Modifier.size(26.dp), tint = fg)
        Spacer(Modifier.height(4.dp))
        Text(title, style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold), color = fg)
        Text(caption, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Normal), color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
    }
}

/** A choice of a few short options as one track, the chosen one filled with the accent. */
@Composable
private fun <T> SheetSegments(label: String, options: List<Pair<T, String>>, selected: T, onPick: (T) -> Unit) {
    val shape = RoundedCornerShape(12.dp)
    Row(
        Modifier.fillMaxWidth().height(44.dp).clip(shape).background(MaterialTheme.colorScheme.surfaceContainerHighest)
            .border(1.dp, MaterialTheme.colorScheme.outline, shape).padding(3.dp)
            .selectableGroup().semantics { contentDescription = label },
    ) {
        for ((v, text) in options) {
            val on = v == selected
            val fill by animateColorAsState(if (on) MaterialTheme.colorScheme.primary else Color.Transparent, label = "segment")
            Box(
                Modifier.weight(1f).fillMaxHeight().clip(RoundedCornerShape(9.dp)).background(fill)
                    .selectable(on, role = Role.RadioButton) { onPick(v) },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text, style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontWeight = if (on) FontWeight.ExtraBold else FontWeight.Medium),
                    color = if (on) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface, maxLines = 1,
                )
            }
        }
    }
}

@Composable
private fun SheetChip(text: String, selected: Boolean, onClick: () -> Unit) {
    val shape = ChipShape
    Box(
        Modifier.minimumInteractiveComponentSize().height(34.dp).clip(shape)
            .background(if (selected) MaterialTheme.colorScheme.primary.copy(alpha = 0.16f) else Color.Transparent)
            .border(1.dp, if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline, shape)
            .selectable(selected, role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp),
        contentAlignment = Alignment.Center,
    ) { Text(text, style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, fontFamily = RecursiveMono, fontWeight = FontWeight.Medium)) }
}
