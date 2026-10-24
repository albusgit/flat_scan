package com.albus.flatscan.ui

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.Collections
import androidx.compose.material.icons.outlined.FlashOff
import androidx.compose.material.icons.outlined.FlashOn
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.ui.draw.rotate
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.albus.flatscan.ScanViewModel
import java.io.File
import kotlin.math.max

@Composable
fun CameraScreen(vm: ScanViewModel) = EditorTheme {
    val ctx = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val request = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) request.launch(Manifest.permission.CAMERA) }
    if (!granted) {
        Column(
            Modifier.fillMaxSize().safeDrawingPadding().padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Camera access is needed to scan a page.", style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center)
            PrimaryPill(
                onClick = { request.launch(Manifest.permission.CAMERA) }, container = Editor.Accent, content = Editor.Ground,
                modifier = Modifier.padding(top = 20.dp),
            ) { PillText("Allow camera") }
        }
    } else {
        CameraPreview(vm)
    }
}

@OptIn(ExperimentalCamera2Interop::class)
@Composable
private fun CameraPreview(vm: ScanViewModel) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val capture = remember {
        ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build()
    }
    var busy by remember { mutableStateOf(false) }
    var focalMmPerLongSide by remember { mutableFloatStateOf(0f) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var torch by remember { mutableStateOf(false) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { vm.importUris(it) }
    val previewView = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    DisposableEffect(Unit) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            try {
                provider.unbindAll()
                val cam = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                camera = cam
                // focal length per pixel of the long side, from the camera's own data (EXIF is preferred if present)
                val info = Camera2CameraInfo.from(cam.cameraInfo)
                val f = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
                val size = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                if (f != null && size != null) focalMmPerLongSide = f / max(size.width, size.height)
            } catch (e: Exception) {
                Log.e("FlatScan", "camera bind failed", e)
                vm.message = "Camera unavailable"
            }
        }, ContextCompat.getMainExecutor(ctx))
        onDispose { runCatching { future.get().unbindAll() } }
    }
    Box(Modifier.fillMaxSize().background(Editor.Ground)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        Row(Modifier.fillMaxWidth().safeDrawingPadding().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = Editor.Text) }
            Box(Modifier.weight(1f))
            // document or whiteboard / projector screen (remembered; imports follow it)
            ModeSwitch(vm.captureMode, onPick = { vm.chooseCaptureMode(it) }, dark = true)
            Box(Modifier.weight(1f))
            // the torch for dim rooms (a flash would glare on glossy paper)
            val hasTorch = camera?.cameraInfo?.hasFlashUnit() == true
            if (hasTorch) {
                IconButton(onClick = {
                    torch = !torch
                    camera?.cameraControl?.enableTorch(torch)
                }) { Icon(if (torch) Icons.Outlined.FlashOn else Icons.Outlined.FlashOff, if (torch) "Torch off" else "Torch on", tint = Editor.Text) }
            }
        }
        // gallery import on the left, the document's pages on the right of the shutter
        Row(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().safeDrawingPadding().padding(bottom = 40.dp, start = 28.dp, end = 28.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(
                onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !busy, modifier = Modifier.size(56.dp).background(Editor.Panel.copy(alpha = 0.7f), CircleShape),
            ) { Icon(Icons.Outlined.PhotoLibrary, "Import from gallery", tint = Editor.Text) }
            Box(Modifier.weight(1f))
            Box(Modifier.size(80.dp))
            Box(Modifier.weight(1f))
            val n = vm.pages.size
            if (n > 0) {
                IconButton(
                    onClick = { vm.open(com.albus.flatscan.Screen.Pages) }, enabled = !busy,
                    modifier = Modifier.size(56.dp).background(Editor.Panel.copy(alpha = 0.7f), CircleShape),
                ) {
                    BadgedBox(badge = { Badge(containerColor = Editor.Accent, contentColor = Editor.Ground) { Text("$n") } }) {
                        Icon(Icons.Outlined.Collections, if (n == 1) "The document: 1 page" else "The document: $n pages", tint = Editor.Text)
                    }
                }
            } else {
                Box(Modifier.size(56.dp))
            }
        }
        Box(
            Modifier.align(Alignment.BottomCenter).safeDrawingPadding().padding(bottom = 28.dp).size(80.dp)
                .border(4.dp, Color.White, CircleShape).padding(8.dp).background(Color.White, CircleShape)
                .semantics { contentDescription = "Take photo" }
                .clickable(enabled = !busy) {
                    busy = true
                    ctx.cacheDir.listFiles { f -> f.name.startsWith("capture_") }?.forEach { it.delete() }
                    val file = File(ctx.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
                    capture.takePicture(
                        ImageCapture.OutputFileOptions.Builder(file).build(),
                        ContextCompat.getMainExecutor(ctx),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                vm.loadUri(Uri.fromFile(file), focalMmPerLongSide)
                            }

                            override fun onError(exception: ImageCaptureException) {
                                busy = false
                                vm.message = "Capture failed"
                            }
                        },
                    )
                },
            contentAlignment = Alignment.Center,
        ) {
            if (busy) {
                // a tiny crumpled ball tumbling while the photo is taken
                val spin = rememberInfiniteTransition(label = "capture")
                val turn by spin.animateFloat(0f, 360f, infiniteRepeatable(tween(1400, easing = LinearEasing)), label = "turn")
                CrumpledBall(Modifier.size(40.dp).rotate(turn), seed = 5)
            }
        }
    }
}

/** Document / Whiteboard: what the next scans are (camera screen, home). */
@Composable
fun ModeSwitch(mode: com.albus.flatscan.CaptureMode, onPick: (com.albus.flatscan.CaptureMode) -> Unit, dark: Boolean, modifier: Modifier = Modifier) {
    val fg = if (dark) Editor.Text else androidx.compose.material3.MaterialTheme.colorScheme.onSurface
    val accent = if (dark) Editor.Accent else androidx.compose.material3.MaterialTheme.colorScheme.primary
    Row(
        modifier.background(if (dark) Editor.Panel.copy(alpha = 0.7f) else androidx.compose.material3.MaterialTheme.colorScheme.surfaceContainerHigh, RoundedCornerShape(10.dp))
            .then(if (dark) Modifier else Modifier.border(1.dp, androidx.compose.material3.MaterialTheme.colorScheme.outline, RoundedCornerShape(10.dp)))
            .padding(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        for (m in com.albus.flatscan.CaptureMode.entries) {
            val on = m == mode
            Box(
                Modifier.clip(ChipShape).background(if (on) accent else Color.Transparent)
                    .clickable { onPick(m) }.padding(horizontal = 14.dp, vertical = 8.dp)
                    .semantics { contentDescription = m.label + if (on) " mode, selected" else " mode" },
            ) {
                Text(m.label, color = if (on) (if (dark) Editor.Ground else androidx.compose.material3.MaterialTheme.colorScheme.onPrimary) else fg,
                    style = androidx.compose.material3.MaterialTheme.typography.labelLarge)
            }
        }
    }
}
