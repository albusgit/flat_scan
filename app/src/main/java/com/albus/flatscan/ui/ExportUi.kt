package com.albus.flatscan.ui

import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ExpandLess
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.PictureAsPdf
import androidx.compose.material.icons.outlined.SaveAlt
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Button
import androidx.compose.material3.ListItem
import androidx.compose.material3.Switch
import androidx.compose.material3.TextButton
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.ImeAction
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
 * Bottom sheet for saving or sharing all [pageCount] pages: Save / Share, the document name, the format
 * (PDF, JPEG, PNG, PDF + JPEG), colour (colour, grayscale, black and white), size preset with the size the
 * export will have, paper and margins, searchable text, the file name template and where to save. The
 * choices are remembered.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExportSheet(
    vm: ScanViewModel, share: Boolean, pageCount: Int, title: String?, onRename: (String) -> Unit,
    onSave: (OutputOptions) -> Unit, onShare: (OutputOptions) -> Unit, onDismiss: () -> Unit,
) {
    val ctx = LocalContext.current
    var sharing by remember { mutableStateOf(share) }
    var name by remember { mutableStateOf(title ?: "") }
    val placeholder = remember { com.albus.flatscan.DocumentName.default() }
    var o by remember { mutableStateOf(ExportPrefs.output(ctx)) }
    var options by remember { mutableStateOf(false) }
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
        estimate = p?.let { "About " + com.albus.flatscan.formatBytes(it.totalBytes) }
    }
    fun go() {
        if (name != (title ?: "")) onRename(name)
        ExportPrefs.setOutput(ctx, o)
        onDismiss()
        if (sharing) onShare(o) else onSave(o)
    }
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = state) {
        Column(Modifier.navigationBarsPadding().verticalScroll(rememberScrollState()).padding(bottom = 16.dp)) {
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 4.dp)) {
                SegmentedButton(
                    selected = !sharing, onClick = { sharing = false }, shape = SegmentedButtonDefaults.itemShape(0, 2),
                    icon = { SegmentedButtonDefaults.Icon(!sharing) { Icon(Icons.Outlined.SaveAlt, null, Modifier.size(18.dp)) } },
                ) { Text("Save to device") }
                SegmentedButton(
                    selected = sharing, onClick = { sharing = true }, shape = SegmentedButtonDefaults.itemShape(1, 2),
                    icon = { SegmentedButtonDefaults.Icon(sharing) { Icon(Icons.Outlined.Share, null, Modifier.size(18.dp)) } },
                ) { Text("Share") }
            }
            OutlinedTextField(
                value = name, onValueChange = { name = it.take(80) },
                label = { Text("Document name") }, placeholder = { Text(placeholder) }, singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp),
            )
            OptionRow("Format", OutputFormat.entries, o.format, { it.label }) { o = o.copy(format = it) }
            OptionRow("Colour", ColorMode.entries, o.color, { it.label }) { o = o.copy(color = it) }
            OptionRow("Size", SizePreset.entries, o.size, { it.label }) { o = o.copy(size = it) }
            Text(
                (estimate ?: "Working out the size...") + " for $pageCount ${if (pageCount == 1) "page" else "pages"}" +
                    (if (o.color == ColorMode.BlackWhite && o.format != OutputFormat.Pdf) "; black and white pictures are PNG" else ""),
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 24.dp, end = 24.dp, bottom = 8.dp),
            )
            if (o.format != OutputFormat.Jpeg && o.format != OutputFormat.Png) {
                // text recognition is a download (output/OcrPack.kt): switching this on fetches it
                val ocr = com.albus.flatscan.output.OcrPack.state
                val size = vm.ocrPackBytes?.let { com.albus.flatscan.formatBytes(it) }
                ListItem(
                    headlineContent = { Text("Searchable text") },
                    supportingContent = {
                        Text(
                            when {
                                ocr is com.albus.flatscan.output.OcrPack.State.Ready -> "Recognise the text (Hebrew and English) and put it invisibly over the page"
                                ocr is com.albus.flatscan.output.OcrPack.State.Downloading ->
                                    "Downloading text recognition: ${com.albus.flatscan.formatBytes(ocr.done)} of ${com.albus.flatscan.formatBytes(ocr.total)}"
                                ocr is com.albus.flatscan.output.OcrPack.State.Failed -> "The text recognition download failed: ${ocr.message}"
                                size == null -> "Text recognition is not available for this phone's processor"
                                else -> "Recognise the text (Hebrew and English): needs text recognition, a $size download"
                            },
                        )
                    },
                    trailingContent = {
                        Switch(
                            checked = o.searchable && ocr is com.albus.flatscan.output.OcrPack.State.Ready, enabled = size != null,
                            onCheckedChange = {
                                if (it && ocr !is com.albus.flatscan.output.OcrPack.State.Ready) vm.downloadOcr()
                                o = o.copy(searchable = it)
                            },
                        )
                    },
                    colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                )
            }
            Row(
                Modifier.fillMaxWidth().clickable(role = Role.Button) { options = !options }.padding(horizontal = 24.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Paper, file names, where to save", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                Icon(if (options) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore, if (options) "Hide options" else "Show options")
            }
            if (options) {
                OptionRow("Paper", PdfPageSize.entries, o.pageSize, { it.label }) { o = o.copy(pageSize = it) }
                if (o.pageSize != PdfPageSize.Fit) OptionRow("Margins", PdfMargin.entries, o.margin, { it.label }) { o = o.copy(margin = it) }
                OutlinedTextField(
                    value = o.template, onValueChange = { o = o.copy(template = it.take(80)) }, singleLine = true,
                    label = { Text("File name template") }, supportingText = { Text("{title} {date} {time} {pages} {n}") },
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 4.dp),
                )
                OptionRow("Date format", FileNameTemplate.DATE_FORMATS, o.dateFormat, { it }) { o = o.copy(dateFormat = it) }
                Text(
                    "File name: " + vm.exportFileName(o) + if (o.format == OutputFormat.Jpeg || o.format == OutputFormat.Png) "" else ".pdf",
                    style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(horizontal = 24.dp),
                )
                ListItem(
                    headlineContent = { Text("Always ask where to save") },
                    supportingContent = { Text(if (o.askWhere || o.folder == null) "The system file picker opens every time" else "Saved into the export folder") },
                    trailingContent = { Switch(checked = o.askWhere || o.folder == null, onCheckedChange = { a -> if (!a && o.folder == null) pickFolder.launch(null) else o = o.copy(askWhere = a) }) },
                    colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                )
                TextButton(onClick = { pickFolder.launch(null) }, modifier = Modifier.padding(horizontal = 12.dp)) {
                    Text(if (o.folder == null) "Choose an export folder" else "Export folder: " + (Uri.parse(o.folder).lastPathSegment?.substringAfterLast(':') ?: "chosen") + " (change)")
                }
            }
            Button(onClick = { go() }, modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp).height(52.dp)) {
                Text(if (sharing) "Share" else if (!o.askWhere && o.folder != null) "Save to the export folder" else "Save")
            }
        }
    }
}

@Composable
private fun <T> OptionRow(label: String, values: List<T>, selected: T, text: (T) -> String, onPick: (T) -> Unit) {
    Column(Modifier.padding(start = 24.dp, end = 8.dp, bottom = 6.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (v in values) {
                val on = v == selected
                val shape = ChipShape
                Box(
                    Modifier.minimumInteractiveComponentSize().height(36.dp).clip(shape)
                        .background(if (on) MaterialTheme.colorScheme.primary.copy(alpha = 0.16f) else Color.Transparent)
                        .border(1.dp, if (on) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline, shape)
                        .selectable(on, role = Role.RadioButton) { onPick(v) }
                        .padding(horizontal = 14.dp),
                    contentAlignment = Alignment.Center,
                ) { Text(text(v), style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp)) }
            }
        }
    }
}

/** A format icon on a rounded accent tile. */
@Composable
private fun FormatIcon(icon: ImageVector) {
    Box(
        Modifier.size(44.dp).background(MaterialTheme.colorScheme.primary.copy(alpha = 0.14f), RoundedCornerShape(8.dp)),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, null, tint = MaterialTheme.colorScheme.primary) }
}
