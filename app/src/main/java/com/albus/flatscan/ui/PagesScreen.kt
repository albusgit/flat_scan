package com.albus.flatscan.ui

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.AddAPhoto
import androidx.compose.material.icons.outlined.AddPhotoAlternate
import androidx.compose.material.icons.outlined.ChevronLeft
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.albus.flatscan.ScanViewModel

private enum class PagesSheet { None, Save, Share }

/** All pages of the current document: open, delete, reorder, add, save. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PagesScreen(vm: ScanViewModel) {
    val exporter = rememberExportActions(vm)
    var sheet by remember { mutableStateOf(PagesSheet.None) }
    var confirmDelete by remember { mutableStateOf<String?>(null) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    val n = vm.pages.size
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text(if (n == 1) "1 page" else "$n pages") },
            navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
            actions = {
                IconButton(onClick = { vm.startCamera() }) { Icon(Icons.Outlined.AddAPhoto, "Add page from camera") }
                IconButton(onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) {
                    Icon(Icons.Outlined.AddPhotoAlternate, "Add pages from gallery")
                }
                IconButton(onClick = { sheet = PagesSheet.Share }, enabled = n > 0 && !exporter.busy) { Icon(Icons.Outlined.Share, "Share") }
            },
        )
        LazyVerticalGrid(
            columns = GridCells.Adaptive(150.dp),
            contentPadding = PaddingValues(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.weight(1f).fillMaxWidth(),
        ) {
            itemsIndexed(vm.pages, key = { _, id -> id }) { i, id ->
                PageCard(
                    vm, id, i, n,
                    onOpen = { vm.openPage(id) },
                    onDelete = { confirmDelete = id },
                )
            }
        }
        Button(
            onClick = { sheet = PagesSheet.Save },
            enabled = n > 0 && !exporter.busy,
            modifier = Modifier.fillMaxWidth().padding(start = 24.dp, end = 24.dp, bottom = 16.dp).height(56.dp),
        ) {
            if (exporter.busy) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onPrimary)
            else Text(if (n > 1) "Save $n pages" else "Save", style = MaterialTheme.typography.titleMedium)
        }
    }
    if (sheet != PagesSheet.None) {
        val share = sheet == PagesSheet.Share
        ExportSheet(share, n, onPick = { if (share) exporter.share(it) else exporter.save(it) }, onDismiss = { sheet = PagesSheet.None })
    }
    confirmDelete?.let { id ->
        val number = vm.pages.indexOf(id) + 1
        AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text("Delete page $number?") },
            confirmButton = { TextButton(onClick = { vm.deletePage(id); confirmDelete = null }) { Text("Delete") } },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Cancel") } },
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PageCard(vm: ScanViewModel, id: String, index: Int, count: Int, onOpen: () -> Unit, onDelete: () -> Unit) {
    val version = vm.pageVersions[id] ?: 0
    var thumb by remember(id) { mutableStateOf<Bitmap?>(null) }
    var loaded by remember(id) { mutableStateOf(false) }
    LaunchedEffect(id, version) {
        vm.thumbnail(id)?.let { thumb = it }
        loaded = true
    }
    val current = id == vm.currentId
    val shape = RoundedCornerShape(10.dp)
    Column {
        Box(
            Modifier.fillMaxWidth().aspectRatio(0.707f).clip(shape)
                .background(MaterialTheme.colorScheme.surfaceVariant)
                .border(if (current) 2.5.dp else 1.dp, if (current) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant, shape)
                .combinedClickable(onClick = onOpen, onLongClick = onDelete, onLongClickLabel = "Delete page")
                .semantics { contentDescription = "Page ${index + 1}" },
            contentAlignment = Alignment.Center,
        ) {
            val t = thumb
            if (t != null) {
                val img = remember(t) { t.asImageBitmap() }
                Image(img, null, contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize().padding(4.dp))
            } else if (!loaded) {
                CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
            } else {
                Text("No preview", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text(
                "${index + 1}",
                style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onPrimary,
                modifier = Modifier.align(Alignment.BottomStart).padding(8.dp)
                    .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(50)).padding(horizontal = 8.dp, vertical = 2.dp),
            )
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            IconButton(onClick = { vm.movePage(id, -1) }, enabled = index > 0) { Icon(Icons.Outlined.ChevronLeft, "Move left") }
            IconButton(onClick = onDelete) { Icon(Icons.Outlined.Delete, "Delete page") }
            IconButton(onClick = { vm.movePage(id, 1) }, enabled = index < count - 1) { Icon(Icons.Outlined.ChevronRight, "Move right") }
        }
    }
}
