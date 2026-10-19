package com.albus.flatscan.ui

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
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
                onClick = { request.launch(Manifest.permission.CAMERA) }, container = Palette.Accent, content = Color.White,
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
    val previewView = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    DisposableEffect(Unit) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            try {
                provider.unbindAll()
                val camera = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                // focal length per pixel of the long side, from the camera's own data (EXIF is preferred if present)
                val info = Camera2CameraInfo.from(camera.cameraInfo)
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
        IconButton(onClick = { vm.back() }, modifier = Modifier.safeDrawingPadding().padding(8.dp)) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back", tint = Editor.Text)
        }
        Box(
            Modifier.align(Alignment.BottomCenter).safeDrawingPadding().padding(bottom = 32.dp).size(80.dp)
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
            if (busy) CircularProgressIndicator(Modifier.size(32.dp), strokeWidth = 3.dp, color = Palette.Accent)
        }
    }
}
