package com.albus.flatscan.ui

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.albus.flatscan.FileNameTemplate
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.sync.Http
import com.albus.flatscan.sync.SyncKind
import com.albus.flatscan.sync.UntrustedCertificateException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Settings > Sync to your server: the target (WebDAV, paperless-ngx, a folder), its address and
 * credentials, Test connection (a real upload and delete), automatic upload and its conditions, and the
 * file name template (shared with export).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SyncScreen(vm: ScanViewModel) {
    val c = vm.syncConfig
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var kind by remember { mutableStateOf(c.kind) }
    var url by remember { mutableStateOf(c.url) }
    var user by remember { mutableStateOf(c.user) }
    var secret by remember { mutableStateOf(c.secret) }
    var folder by remember { mutableStateOf(c.folderUri) }
    var tags by remember { mutableStateOf(c.paperlessTags) }
    var corr by remember { mutableStateOf(c.paperlessCorrespondent) }
    var auto by remember { mutableStateOf(c.autoUpload) }
    var wifi by remember { mutableStateOf(c.wifiOnly) }
    var charging by remember { mutableStateOf(c.chargingOnly) }
    var template by remember { mutableStateOf(c.nameTemplate) }
    var dateFmt by remember { mutableStateOf(c.dateFormat) }
    var testing by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<Pair<Boolean, String>?>(null) }
    var trust by remember { mutableStateOf<UntrustedCertificateException?>(null) }
    val pickFolder = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri: Uri? ->
        if (uri != null) {
            ctx.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            folder = uri.toString()
            c.folderUri = folder
        }
    }

    fun save() {
        c.kind = kind; c.url = url; c.user = user; c.secret = secret
        c.paperlessTags = tags; c.paperlessCorrespondent = corr
        c.autoUpload = auto; c.wifiOnly = wifi; c.chargingOnly = charging
        c.nameTemplate = template; c.dateFormat = dateFmt
    }

    fun test() {
        save()
        testing = true
        result = null
        scope.launch {
            val r = withContext(Dispatchers.IO) { runCatching { c.uploader()?.test() ?: error("Fill in the settings first") } }
            testing = false
            r.onSuccess { result = true to it }.onFailure { e ->
                val u = Http.untrusted(e)
                if (u != null) trust = u else result = false to (e.message ?: e.javaClass.simpleName)
            }
        }
    }

    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = { Text("Sync to your server", style = MaterialTheme.typography.titleLarge) },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent),
            navigationIcon = { IconButton(onClick = { save(); vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
        )
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                "Finished documents are uploaded as PDF to a server or folder you own. Free, no account, no FlatScan cloud: " +
                    "scans never leave the phone unless you set this up.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            SettingsSheet("Where to") {
                for (k in SyncKind.entries) RadioRow(k.label, kind == k) { kind = k; result = null }
                when (kind) {
                    SyncKind.WebDav -> {
                        OutlinedTextField(url, { url = it }, label = { Text("Folder URL") }, singleLine = true, modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("https://cloud.example.com/remote.php/dav/files/me/Scans/") },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri))
                        OutlinedTextField(user, { user = it }, label = { Text("User name") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(secret, { secret = it }, label = { Text("Password or app password") }, singleLine = true, modifier = Modifier.fillMaxWidth(),
                            visualTransformation = PasswordVisualTransformation(), keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password))
                        Hint("Nextcloud / ownCloud: Settings > WebDAV shows the address; use an app password.")
                    }
                    SyncKind.Paperless -> {
                        OutlinedTextField(url, { url = it }, label = { Text("Server address") }, singleLine = true, modifier = Modifier.fillMaxWidth(),
                            placeholder = { Text("https://paperless.example.com") }, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri))
                        OutlinedTextField(secret, { secret = it }, label = { Text("API token") }, singleLine = true, modifier = Modifier.fillMaxWidth(),
                            visualTransformation = PasswordVisualTransformation())
                        OutlinedTextField(tags, { tags = it }, label = { Text("Tags for every upload (comma separated)") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(corr, { corr = it }, label = { Text("Correspondent (optional)") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                        Hint("The token is in paperless under your profile (My Profile > API Auth Token). The document's own tags are sent too.")
                    }
                    SyncKind.Folder -> {
                        OutlinedButton(onClick = { pickFolder.launch(null) }) { Text(if (folder == null) "Pick a folder" else "Pick another folder") }
                        folder?.let { Hint(Uri.parse(it).lastPathSegment ?: it) }
                        Hint("Works with Syncthing, cloud-folder apps and SMB apps that offer a folder: no network access by FlatScan.")
                    }
                    SyncKind.Off -> {}
                }
                if (kind == SyncKind.WebDav || kind == SyncKind.Paperless) {
                    val norm = Http.normalizeUrl(url)
                    if (norm != null && norm.startsWith("http://")) Text(
                        "Plain http: the password and the documents travel unencrypted. Use https if the server has it.",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error,
                    )
                }
                if (kind != SyncKind.Off) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        OutlinedButton(onClick = { test() }, enabled = !testing) { Text("Test connection") }
                        if (testing) {
                            Spacer(Modifier.size(12.dp))
                            CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
                        }
                    }
                    Hint("The test uploads a small file and deletes it again, with your credentials, as a real upload would.")
                    result?.let { (ok, msg) ->
                        Text((if (ok) "Works: " else "Failed: ") + msg, color = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodyMedium)
                    }
                }
                Spacer(Modifier.height(10.dp))
            }
            if (kind != SyncKind.Off) {
                SettingsSheet("Uploads") {
                    SwitchLine("Upload automatically", "When you go back from a document you scanned or changed", auto) { auto = it }
                    if (kind != SyncKind.Folder) SwitchLine("Only on Wi-Fi", "Wait for an unmetered network", wifi) { wifi = it }
                    SwitchLine("Only while charging", "Wait until the phone is charging", charging) { charging = it }
                    Hint("Without automatic upload, upload a document from the history: long press, then Upload.")
                    Spacer(Modifier.height(10.dp))
                }
            }
            SettingsSheet("File names") {
                OutlinedTextField(template, { template = it }, label = { Text("Template") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                Hint("{title}, {date}, {time}, {pages}. A name already taken gets (2), (3)... nothing is overwritten.")
                for (f in FileNameTemplate.DATE_FORMATS) RadioRow(f, dateFmt == f) { dateFmt = f }
                Text(
                    "Example: " + FileNameTemplate.render(template, vm.docTitle ?: "Electricity bill", System.currentTimeMillis(), 3, dateFmt) + ".pdf",
                    style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono),
                    modifier = Modifier.padding(bottom = 10.dp),
                )
            }
            Spacer(Modifier.height(24.dp))
        }
    }
    trust?.let { u ->
        AlertDialog(
            onDismissRequest = { trust = null },
            title = { Text("Trust this certificate?") },
            text = {
                Column {
                    Text("${u.host} presented a certificate that is not signed by a known authority (self-signed?). Trust it only if the fingerprint matches your server's.")
                    Spacer(Modifier.height(8.dp))
                    Text("SHA-256:", style = MaterialTheme.typography.labelMedium)
                    Text(u.sha256, style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono))
                    Spacer(Modifier.height(4.dp))
                    Text(u.subject, style = MaterialTheme.typography.bodySmall)
                }
            },
            confirmButton = { TextButton(onClick = { c.pin(u.host, u.sha256); trust = null; test() }) { Text("Trust and test again") } },
            dismissButton = { TextButton(onClick = { trust = null; result = false to "Certificate not trusted" }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun Hint(t: String) = Text(t, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

@Composable
private fun RadioRow(label: String, selected: Boolean, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        androidx.compose.material3.RadioButton(selected = selected, onClick = onClick)
        Text(label, modifier = Modifier.padding(start = 4.dp))
    }
}

@Composable
private fun SwitchLine(title: String, caption: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(caption, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        androidx.compose.material3.Switch(checked = checked, onCheckedChange = onChange)
    }
}
