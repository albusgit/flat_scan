package com.albus.flatscan.ui

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.albus.flatscan.ScanViewModel
import com.albus.flatscan.depth.DepthCapture
import com.google.ar.core.ArCoreApk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The depth test (Settings > Advanced > Depth capture, experimental): ARCore's depth over the camera, how far it
 * strays from a flat plane in the middle of the view, and frames saved for desktop analysis (depth/DepthCapture.kt).
 * ARCore is optional: a phone without it, or without Google Play Services for AR, says so here and nowhere else.
 */
@Composable
fun DepthScreen(vm: ScanViewModel) {
    val ctx = LocalContext.current
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val request = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    var status by remember { mutableStateOf("Checking ARCore…") }
    var ready by remember { mutableStateOf(false) }
    var installAsked by remember { mutableStateOf(false) }
    // the availability is checked again on returning from the Play Store
    var resumes by remember { mutableIntStateOf(0) }
    val owner = LocalLifecycleOwner.current
    DisposableEffect(owner) {
        val obs = LifecycleEventObserver { _, e -> if (e == Lifecycle.Event.ON_RESUME) resumes++ }
        owner.lifecycle.addObserver(obs)
        onDispose { owner.lifecycle.removeObserver(obs) }
    }
    LaunchedEffect(granted, resumes) {
        if (ready) return@LaunchedEffect
        if (!granted) {
            status = "The depth test needs the camera."
            return@LaunchedEffect
        }
        val core = ArCoreApk.getInstance()
        var a = core.checkAvailability(ctx)
        while (a.isTransient) {
            delay(200)
            a = core.checkAvailability(ctx)
        }
        status = when {
            a == ArCoreApk.Availability.SUPPORTED_INSTALLED -> {
                ready = true
                ""
            }
            a.isSupported -> {
                val act = ctx.findActivity()
                if (act == null) "Could not open the Play Store." else try {
                    when (core.requestInstall(act, !installAsked)) {
                        ArCoreApk.InstallStatus.INSTALLED -> {
                            ready = true
                            ""
                        }
                        else -> {
                            installAsked = true
                            "Install Google Play Services for AR from the Play Store, then come back."
                        }
                    }
                } catch (e: Exception) {
                    "Google Play Services for AR could not be installed: ${e.javaClass.simpleName}"
                }
            }
            a.isUnknown -> "Could not check ARCore on this phone (${Build.MANUFACTURER} ${Build.MODEL}): $a. " +
                "It needs Google Play Services for AR, which emulators and some phones lack."
            else -> "ARCore does not support this phone (${Build.MANUFACTURER} ${Build.MODEL})."
        }
    }
    if (ready) {
        DepthLive(onBack = { vm.back() })
        return
    }
    Column(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
        IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
        Text("Depth capture", style = MaterialTheme.typography.titleLarge, modifier = Modifier.padding(vertical = 12.dp))
        Text(status, style = MaterialTheme.typography.bodyLarge)
        if (!granted) Button(onClick = { request.launch(Manifest.permission.CAMERA) }, modifier = Modifier.padding(top = 16.dp)) { Text("Allow the camera") }
    }
}

@Composable
private fun DepthLive(onBack: () -> Unit) {
    val ctx = LocalContext.current
    val cap = remember { DepthCapture(ctx.applicationContext) }
    var live by remember { mutableStateOf(DepthCapture.Live()) }
    var error by remember { mutableStateOf<String?>(null) }
    var showDepth by remember { mutableStateOf(true) }
    val scope = rememberCoroutineScope()
    val view = remember {
        GLSurfaceView(ctx).apply {
            preserveEGLContextOnPause = true
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            setRenderer(cap.renderer)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
    }
    val owner = LocalLifecycleOwner.current
    DisposableEffect(owner) {
        cap.onLive = { live = it }
        error = cap.open()
        // a new observer is brought up to the current state, so this also resumes on first show
        val obs = LifecycleEventObserver { _, e ->
            when (e) {
                Lifecycle.Event.ON_RESUME -> {
                    try {
                        cap.resume()
                    } catch (ex: Exception) {
                        error = "The camera is busy: ${ex.javaClass.simpleName}"
                    }
                    view.onResume()
                }
                Lifecycle.Event.ON_PAUSE -> {
                    view.onPause()
                    cap.pause()
                }
                else -> {}
            }
        }
        owner.lifecycle.addObserver(obs)
        onDispose {
            owner.lifecycle.removeObserver(obs)
            view.onPause()
            cap.pause()
            cap.close()
        }
    }
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { view }, modifier = Modifier.fillMaxSize())
        IconButton(onClick = onBack, modifier = Modifier.safeDrawingPadding().padding(4.dp)) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = Color.White)
        }
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Color.Black.copy(alpha = 0.7f)).safeDrawingPadding().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            val white = Color.White
            val small = MaterialTheme.typography.bodySmall
            error?.let { Text(it, color = Color(0xFFFF8A80), style = MaterialTheme.typography.bodyMedium) }
            Text(cap.info, color = white.copy(alpha = 0.8f), style = small)
            Text(
                listOf(live.tracking, live.depthSize.takeIf { it.isNotEmpty() }?.let { "depth $it" }).filterNotNull().filter { it.isNotEmpty() }.joinToString(" · "),
                color = white, style = small,
            )
            if (live.distanceMm > 0) Text(
                "Middle of the view: %.0f mm away. Off a flat plane: %.1f mm RMS (smoothed depth)".format(live.distanceMm, live.smoothRmsMm) +
                    (if (live.rawPoints > 0) ", %.1f mm (raw, %d points)".format(live.rawRmsMm, live.rawPoints) else ""),
                color = white, style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                "Lay a page on a table and move the phone slowly until tracking starts. On a flat sheet the number above is " +
                    "the depth's noise; on a curled page it is noise plus the curl. Save 3 frames from slightly different angles.",
                color = white.copy(alpha = 0.7f), style = small,
            )
            if (live.note.isNotEmpty()) Text(live.note, color = Color(0xFFFFE082), style = small)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = showDepth, onCheckedChange = { showDepth = it; cap.showDepth = it })
                Spacer(Modifier.width(8.dp))
                Text("Depth colours", color = white, style = small)
                Spacer(Modifier.weight(1f))
                Button(onClick = { cap.requestSave() }) { Text("Save frame (${live.saved})") }
                Spacer(Modifier.width(8.dp))
                OutlinedButton(enabled = live.saved > 0, onClick = {
                    scope.launch {
                        val zip = withContext(Dispatchers.IO) { cap.zipSaved() } ?: return@launch
                        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", zip)
                        val send = Intent(Intent.ACTION_SEND).apply {
                            type = "application/zip"
                            putExtra(Intent.EXTRA_STREAM, uri)
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        ctx.startActivity(Intent.createChooser(send, null))
                    }
                }) { Text("Share", color = white) }
            }
        }
    }
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}
