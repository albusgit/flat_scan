package com.albus.flatscan.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.RotateRight
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.PictureAsPdf
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
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

private enum class Sheet { None, Save, Share }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(vm: ScanViewModel) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var sheet by remember { mutableStateOf(Sheet.None) }
    var showDetails by remember { mutableStateOf(false) }
    fun export(format: ExportFormat, uri: android.net.Uri?) {
        val bmp = vm.output() ?: return
        if (uri == null) return
        scope.launch {
            val ok = withContext(Dispatchers.IO) { runCatching { Exporter.save(ctx, bmp, format, uri) }.isSuccess }
            vm.message = if (ok) "Saved" else "Could not save"
        }
    }
    val saveJpg = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument(ExportFormat.JPG.mime)) { export(ExportFormat.JPG, it) }
    val savePdf = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument(ExportFormat.PDF.mime)) { export(ExportFormat.PDF, it) }
    val baseName = remember { "Scan " + SimpleDateFormat("yyyy-MM-dd HH.mm", Locale.US).format(Date()) }

    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text("Result") },
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back to corners") } },
            actions = {
                IconButton(onClick = { showDetails = true }) { Icon(Icons.Outlined.Info, "Processing details") }
                IconButton(onClick = { sheet = Sheet.Share }, enabled = vm.result != null) { Icon(Icons.Outlined.Share, "Share") }
            },
        )
        Box(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 16.dp), contentAlignment = Alignment.Center) {
            val shown = if (vm.showBefore) vm.before else vm.result
            if (shown != null) {
                val img = remember(shown) { shown.asImageBitmap() }
                Image(img, contentDescription = if (vm.showBefore) "Original photo" else "Cleaned page", contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
            }
            if (vm.rendering) CircularProgressIndicator()
        }
        Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                SegmentedButton(selected = vm.showBefore, onClick = { vm.showBefore = true }, shape = SegmentedButtonDefaults.itemShape(0, 2)) { Text("Before") }
                SegmentedButton(selected = !vm.showBefore, onClick = { vm.showBefore = false }, shape = SegmentedButtonDefaults.itemShape(1, 2)) { Text("After") }
            }
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                SegmentedButton(selected = !vm.grayscale, onClick = { vm.changeGray(false) }, shape = SegmentedButtonDefaults.itemShape(0, 2)) { Text("Auto color") }
                SegmentedButton(selected = vm.grayscale, onClick = { vm.changeGray(true) }, shape = SegmentedButtonDefaults.itemShape(1, 2)) { Text("Grayscale") }
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Row(Modifier.weight(1f).clickable { vm.changeSolid(!vm.solid) }, verticalAlignment = Alignment.CenterVertically) {
                    Switch(checked = vm.solid, onCheckedChange = { vm.changeSolid(it) })
                    Spacer(Modifier.size(12.dp))
                    Text("Solid colors", style = MaterialTheme.typography.bodyLarge)
                }
                TextButton(onClick = { vm.rotateResult() }) {
                    Icon(Icons.AutoMirrored.Outlined.RotateRight, null)
                    Spacer(Modifier.size(6.dp))
                    Text("Rotate")
                }
            }
        }
        Button(
            onClick = { sheet = Sheet.Save },
            enabled = vm.result != null && !vm.rendering,
            modifier = Modifier.fillMaxWidth().padding(start = 24.dp, end = 24.dp, bottom = 16.dp).height(56.dp),
        ) { Text("Save", style = MaterialTheme.typography.titleMedium) }
    }

    if (sheet != Sheet.None) {
        ModalBottomSheet(onDismissRequest = { sheet = Sheet.None }) {
            Column(Modifier.navigationBarsPadding().padding(bottom = 16.dp)) {
                val share = sheet == Sheet.Share
                Text(if (share) "Share as" else "Save as", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp))
                ListItem(
                    headlineContent = { Text("JPG image") },
                    supportingContent = { Text("The page as a picture") },
                    leadingContent = { Icon(Icons.Outlined.Image, null) },
                    modifier = Modifier.clickable {
                        sheet = Sheet.None
                        if (share) vm.output()?.let { Exporter.share(ctx, it, ExportFormat.JPG) } else saveJpg.launch("$baseName.jpg")
                    },
                )
                ListItem(
                    headlineContent = { Text("PDF document") },
                    supportingContent = { Text("One A4 page") },
                    leadingContent = { Icon(Icons.Outlined.PictureAsPdf, null) },
                    modifier = Modifier.clickable {
                        sheet = Sheet.None
                        if (share) vm.output()?.let { Exporter.share(ctx, it, ExportFormat.PDF) } else savePdf.launch("$baseName.pdf")
                    },
                )
            }
        }
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
