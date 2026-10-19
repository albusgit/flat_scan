package com.albus.flatscan.ui

import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.PictureAsPdf
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.albus.flatscan.ExportFormat
import com.albus.flatscan.Exporter
import com.albus.flatscan.ScanViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Save / share of the whole document; hold it at screen level (the launchers must outlive the sheet). */
class ExportActions(
    val save: (ExportFormat) -> Unit,
    val share: (ExportFormat) -> Unit,
    private val busyState: () -> Boolean,
) {
    val busy: Boolean get() = busyState()
}

private fun baseName() = "Scan " + SimpleDateFormat("yyyy-MM-dd HH.mm", Locale.US).format(Date())

@Composable
fun rememberExportActions(vm: ScanViewModel): ExportActions {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }
    var name by remember { mutableStateOf(baseName()) }

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
            withContext(Dispatchers.IO) { Exporter.savePdf(ctx, files, uri) }
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
                name = baseName()
                when {
                    f == ExportFormat.PDF -> savePdf.launch("$name.pdf")
                    vm.pages.size <= 1 -> saveJpg.launch("$name.jpg")
                    else -> saveJpgFolder.launch(null)
                }
            },
            share = { f ->
                launchExport {
                    val files = pagesOrNull(vm.exportPages()) ?: return@launchExport "Nothing to share"
                    val intent = withContext(Dispatchers.IO) { Exporter.shareIntent(ctx, files, f, baseName()) }
                    ctx.startActivity(intent)
                    null
                }
            },
            busyState = { busy },
        )
    }
}

/** Bottom sheet: PDF or JPG, for saving or sharing all [pageCount] pages. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExportSheet(share: Boolean, pageCount: Int, onPick: (ExportFormat) -> Unit, onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.navigationBarsPadding().padding(bottom = 16.dp)) {
            Text(if (share) "Share as" else "Save as", style = MaterialTheme.typography.titleLarge, modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp))
            val many = pageCount > 1
            ListItem(
                headlineContent = { Text("PDF document", style = MaterialTheme.typography.titleMedium) },
                supportingContent = { Text(if (many) "All $pageCount pages in one PDF, A4" else "One A4 page") },
                leadingContent = { FormatIcon(Icons.Outlined.PictureAsPdf) },
                colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                modifier = Modifier.clickable { onDismiss(); onPick(ExportFormat.PDF) },
            )
            ListItem(
                headlineContent = { Text(if (many) "JPG images" else "JPG image", style = MaterialTheme.typography.titleMedium) },
                supportingContent = {
                    Text(
                        when {
                            !many -> "The page as a picture"
                            share -> "One picture per page"
                            else -> "One picture per page, into a folder you choose"
                        },
                    )
                },
                leadingContent = { FormatIcon(Icons.Outlined.Image) },
                colors = ListItemDefaults.colors(containerColor = Color.Transparent),
                modifier = Modifier.clickable { onDismiss(); onPick(ExportFormat.JPG) },
            )
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
