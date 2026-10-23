package com.albus.flatscan.ui

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
import androidx.compose.material3.ListItem
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
import com.albus.flatscan.ExportFormat
import com.albus.flatscan.ExportPrefs
import com.albus.flatscan.Exporter
import com.albus.flatscan.PdfImageSize
import com.albus.flatscan.PdfMargin
import com.albus.flatscan.PdfOptions
import com.albus.flatscan.PdfPageSize
import com.albus.flatscan.ScanViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Save / share of the whole document; hold it at screen level (the launchers must outlive the sheet). */
class ExportActions(
    val save: (ExportFormat) -> Unit,
    val share: (ExportFormat) -> Unit,
    private val busyState: () -> Boolean,
) {
    val busy: Boolean get() = busyState()
}

@Composable
fun rememberExportActions(vm: ScanViewModel): ExportActions {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }
    var name by remember { mutableStateOf(vm.exportName()) }

    fun launchExport(work: suspend () -> String?) {
        if (busy) return
        busy = true
        scope.launch {
            val msg = try {
                work()
            } catch (e: Exception) {
                Log.e(ScanViewModel.TAG, "export failed", e)
                "Could not save"
            }
            busy = false
            if (msg != null) vm.message = msg
        }
    }

    fun pagesOrNull(files: List<java.io.File>): List<java.io.File>? = files.ifEmpty { null }

    val savePdf = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument(ExportFormat.PDF.mime)) { uri: Uri? ->
        if (uri != null) launchExport {
            val files = pagesOrNull(vm.exportPages()) ?: return@launchExport "Nothing to save"
            val o = ExportPrefs.pdf(ctx)
            withContext(Dispatchers.IO) { Exporter.savePdf(ctx, files, uri, o) }
            if (files.size == 1) "Saved" else "Saved ${files.size} pages"
        }
    }
    val saveJpg = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument(ExportFormat.JPG.mime)) { uri: Uri? ->
        if (uri != null) launchExport {
            val files = pagesOrNull(vm.exportPages()) ?: return@launchExport "Nothing to save"
            withContext(Dispatchers.IO) { Exporter.saveJpg(ctx, files[0], uri) }
            "Saved"
        }
    }
    val saveJpgFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { tree: Uri? ->
        if (tree != null) launchExport {
            val files = pagesOrNull(vm.exportPages()) ?: return@launchExport "Nothing to save"
            val n = withContext(Dispatchers.IO) { Exporter.saveJpgsToFolder(ctx, files, tree, name) }
            "Saved $n pages to \"$name\""
        }
    }
    return remember(vm) {
        ExportActions(
            save = { f ->
                name = vm.exportName()
                when {
                    f == ExportFormat.PDF -> savePdf.launch("$name.pdf")
                    vm.pages.size <= 1 -> saveJpg.launch("$name.jpg")
                    else -> saveJpgFolder.launch(null)
                }
            },
            share = { f ->
                launchExport {
                    val files = pagesOrNull(vm.exportPages()) ?: return@launchExport "Nothing to share"
                    val o = ExportPrefs.pdf(ctx)
                    val intent = withContext(Dispatchers.IO) { Exporter.shareIntent(ctx, files, f, vm.exportName(), o) }
                    ctx.startActivity(intent)
                    null
                }
            },
            busyState = { busy },
        )
    }
}

/**
 * Bottom sheet for saving or sharing all [pageCount] pages: Save / Share, the file name (the
 * document's name), PDF or JPG, and the PDF options (paper, margins, file size), which are remembered.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExportSheet(
    share: Boolean, pageCount: Int, title: String?, onRename: (String) -> Unit,
    onSave: (ExportFormat) -> Unit, onShare: (ExportFormat) -> Unit, onDismiss: () -> Unit,
) {
    val ctx = LocalContext.current
    var sharing by remember { mutableStateOf(share) }
    var name by remember { mutableStateOf(title ?: "") }
    val placeholder = remember { com.albus.flatscan.DocumentName.default() }
    var pdf by remember { mutableStateOf(ExportPrefs.pdf(ctx)) }
    var options by remember { mutableStateOf(false) }
    val state = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    fun pick(f: ExportFormat) {
        if (name != (title ?: "")) onRename(name)
        ExportPrefs.setPdf(ctx, pdf)
        onDismiss()
        if (sharing) onShare(f) else onSave(f)
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
                label = { Text("File name") }, placeholder = { Text(placeholder) }, singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp),
            )
            val many = pageCount > 1
            ListItem(
                headlineContent = { Text("PDF document", style = MaterialTheme.typography.titleMedium) },
                supportingContent = { Text((if (many) "All $pageCount pages in one PDF, " else "One page, ") + pdfSummary(pdf)) },
                leadingContent = { FormatIcon(Icons.Outlined.PictureAsPdf) },
                colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                modifier = Modifier.clickable { pick(ExportFormat.PDF) },
            )
            ListItem(
                headlineContent = { Text(if (many) "JPG images" else "JPG image", style = MaterialTheme.typography.titleMedium) },
                supportingContent = {
                    Text(
                        when {
                            !many -> "The page as a picture"
                            sharing -> "One picture per page"
                            else -> "One picture per page, into a folder you choose"
                        },
                    )
                },
                leadingContent = { FormatIcon(Icons.Outlined.Image) },
                colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                modifier = Modifier.clickable { pick(ExportFormat.JPG) },
            )
            Row(
                Modifier.fillMaxWidth().clickable(role = Role.Button) { options = !options }.padding(horizontal = 24.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("PDF options", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                Icon(if (options) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore, if (options) "Hide PDF options" else "Show PDF options")
            }
            if (options) {
                OptionRow("Paper", PdfPageSize.entries, pdf.pageSize, { it.label }) { pdf = pdf.copy(pageSize = it) }
                if (pdf.pageSize != PdfPageSize.Fit) OptionRow("Margins", PdfMargin.entries, pdf.margin, { it.label }) { pdf = pdf.copy(margin = it) }
                OptionRow("Images", PdfImageSize.entries, pdf.imageSize, { it.label }) { pdf = pdf.copy(imageSize = it) }
            }
        }
    }
}

private fun pdfSummary(o: PdfOptions): String = buildString {
    append(if (o.pageSize == PdfPageSize.Fit) "pages the shape of the scans" else o.pageSize.label)
    if (o.pageSize != PdfPageSize.Fit && o.margin != PdfMargin.None) append(", ${o.margin.label.lowercase()} margins")
    if (o.imageSize == PdfImageSize.Compact) append(", smaller file")
}

@Composable
private fun <T> OptionRow(label: String, values: List<T>, selected: T, text: (T) -> String, onPick: (T) -> Unit) {
    Column(Modifier.padding(start = 24.dp, end = 8.dp, bottom = 6.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (v in values) {
                val on = v == selected
                val shape = RoundedCornerShape(18.dp)
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
        Modifier.size(44.dp).background(MaterialTheme.colorScheme.primary.copy(alpha = 0.14f), RoundedCornerShape(14.dp)),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, null, tint = MaterialTheme.colorScheme.primary) }
}
