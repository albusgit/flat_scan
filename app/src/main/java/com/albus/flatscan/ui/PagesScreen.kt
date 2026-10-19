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
import androidx.compose.foundation.combinedClickable
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
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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

/** All pages of the current document: open, delete, reorder, add, save. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PagesScreen(vm: ScanViewModel) {
    val exporter = rememberExportActions(vm)
    var sheet by remember { mutableStateOf(PagesSheet.None) }
    var confirmDelete by remember { mutableStateOf<String?>(null) }
    var menu by remember { mutableStateOf(false) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    val pickPhotos = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    val n = vm.pages.size
    val ready = n > 0 && !exporter.busy
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = {
                Column {
                    Text("Document", style = MaterialTheme.typography.titleLarge.copy(fontSize = 18.sp))
                    Text(if (n == 1) "1 page" else "$n pages", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
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
            columns = GridCells.Fixed(2),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.weight(1f).fillMaxWidth(),
        ) {
            itemsIndexed(vm.pages, key = { _, id -> id }) { i, id ->
                PageCard(
                    vm, id, i, n,
                    onOpen = { vm.openPage(id) },
                    onDelete = { confirmDelete = id },
                )
            }
            item(key = "add") { AddPageTile(onCamera = { vm.startCamera() }, onGallery = pickPhotos) }
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
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

private val CardHeight = 210.dp

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PageCard(vm: ScanViewModel, id: String, index: Int, count: Int, onOpen: () -> Unit, onDelete: () -> Unit) {
    val version = vm.pageVersions[id] ?: 0
    var thumb by remember(id) { mutableStateOf<Bitmap?>(null) }
    var loaded by remember(id) { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }
    LaunchedEffect(id, version) {
        vm.thumbnail(id)?.let { thumb = it }
        loaded = true
    }
    val current = id == vm.currentId
    val shape = RoundedCornerShape(14.dp)
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column {
        Box(
            Modifier.fillMaxWidth().height(CardHeight)
                .shadow(8.dp, shape, ambientColor = Palette.Ink.copy(alpha = 0.3f), spotColor = Palette.Ink.copy(alpha = 0.3f))
                .clip(shape)
                .background(Color.White)
                .then(if (current) Modifier.border(2.dp, MaterialTheme.colorScheme.primary, shape) else Modifier)
                .combinedClickable(onClick = onOpen, onLongClick = { menu = true }, onLongClickLabel = "Page actions")
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
                Modifier.align(Alignment.BottomStart).padding(8.dp).defaultMinSize(minWidth = 26.dp).height(26.dp)
                    .background(Palette.Ink, RoundedCornerShape(13.dp)).padding(horizontal = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text("${index + 1}", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.ExtraBold), color = Color.White)
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 2.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                if (current) "${vm.params.filter.label} · editing" else "Page ${index + 1}",
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold),
                color = muted, maxLines = 1, modifier = Modifier.weight(1f),
            )
            Box {
                IconButton(onClick = { menu = true }, modifier = Modifier.size(36.dp)) {
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

/** The dashed tile after the last page; it asks where the new page comes from. */
@Composable
private fun AddPageTile(onCamera: () -> Unit, onGallery: () -> Unit) {
    var menu by remember { mutableStateOf(false) }
    val accent = MaterialTheme.colorScheme.primary
    val dash = if (MaterialTheme.colorScheme.background.luminance() > 0.5f) Palette.Dashed else MaterialTheme.colorScheme.outline
    Box {
        Column(
            Modifier.fillMaxWidth().height(CardHeight).clip(RoundedCornerShape(14.dp))
                .drawBehind {
                    val w = 2.dp.toPx()
                    drawRoundRect(
                        dash, topLeft = Offset(w / 2, w / 2), size = Size(size.width - w, size.height - w),
                        cornerRadius = CornerRadius(14.dp.toPx()),
                        style = Stroke(w, pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx()))),
                    )
                }
                .clickable(role = Role.Button) { menu = true },
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Box(Modifier.size(48.dp).background(MaterialTheme.colorScheme.primaryContainer, CircleShape), contentAlignment = Alignment.Center) {
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
