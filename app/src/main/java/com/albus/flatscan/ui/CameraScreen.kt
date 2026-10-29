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
import android.os.SystemClock
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import com.albus.flatscan.CameraGuide
import com.albus.flatscan.GuideState
import com.albus.flatscan.NativeLib
import com.albus.flatscan.PreviewFrame
import java.util.concurrent.Executors
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material.icons.outlined.Close
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.albus.flatscan.CaptureMode
import com.albus.flatscan.Screen
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
import androidx.compose.material.icons.outlined.FlashOff
import androidx.compose.material.icons.outlined.FlashOn
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.draw.alpha
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import androidx.compose.ui.draw.rotate
import androidx.compose.material3.Icon
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
    // camera guidance: small frames (about 320 x 240, 4:3 like the preview and the photo) on their own thread
    val analysis = remember { guideAnalysis() }
    val guideExec = remember { Executors.newSingleThreadExecutor { r -> Thread(r, "flatscan-guide") } }
    val guide = remember { CameraGuide() }
    var guideState by remember { mutableStateOf(GuideState()) }
    var guiding by remember { mutableStateOf(false) }   // the analysis is bound (some cameras take only preview + photo)
    val findPage = vm.captureMode != CaptureMode.Whiteboard && vm.settings.pageFinderNet
    val findPageNow by rememberUpdatedState(findPage)
    LaunchedEffect(findPage) {
        guide.reset()
        guideState = guide.state
    }
    DisposableEffect(Unit) {
        onDispose {
            analysis.clearAnalyzer()
            guideExec.shutdown()
        }
    }
    // the analyzer runs while the shutter is free: a photo being taken gets the CPU to itself
    DisposableEffect(guiding, busy) {
        if (guiding && !busy) {
            val main = ContextCompat.getMainExecutor(ctx)
            var last = 0L
            analysis.setAnalyzer(guideExec) { image ->
                image.use {
                    val now = SystemClock.elapsedRealtime()
                    if (now - last < GUIDE_INTERVAL_MS) return@use
                    last = now
                    val searched = findPageNow
                    val plane = it.planes[0]
                    val a = runCatching {
                        NativeLib.previewPage(plane.buffer, it.width, it.height, plane.rowStride, it.imageInfo.rotationDegrees, searched)
                    }.onFailure { e -> Log.e("FlatScan", "camera guidance failed", e) }.getOrNull()
                    val frame = a?.let { PreviewFrame.from(it, searched) } ?: return@use
                    main.execute { if (frame.searched == findPageNow) guideState = guide.update(frame) }
                }
            }
        }
        onDispose { analysis.clearAnalyzer() }
    }
    DisposableEffect(Unit) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().setResolutionSelector(ratio43()).build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            try {
                provider.unbindAll()
                // preview, photo and the guidance's frames; a camera that cannot stream all three gets the first two
                val cam = try {
                    provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture, analysis).also { guiding = true }
                } catch (e: Exception) {
                    Log.w("FlatScan", "camera bind with guidance failed: without it", e)
                    provider.unbindAll()
                    guiding = false
                    provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                }
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
    // the shutter's white flash (85 %, fading over 320 ms)
    val flash = remember { Animatable(0f) }
    val reduced = rememberReducedMotion()
    val scope = rememberCoroutineScope()
    val shoot = {
        busy = true
        if (!reduced) scope.launch {
            flash.snapTo(0.85f)
            flash.animateTo(0f, tween(320, easing = LinearOutSlowInEasing))
        }
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
    }
    Box(Modifier.fillMaxSize().background(Editor.Ground)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        val frame = guideState.frame
        val quad = guideState.quad
        if (quad != null && frame != null && !busy) GuideQuad(quad, frame.width, frame.height, guideState.ready)
        if (flash.value > 0f) Box(Modifier.fillMaxSize().graphicsLayer { alpha = flash.value }.background(Color.White))
        // top: close, the guidance's hint, the torch
        Row(Modifier.fillMaxWidth().safeDrawingPadding().padding(horizontal = 10.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            RoundButton(onClick = { vm.back() }) { Icon(Icons.Outlined.Close, "Close the camera", tint = Editor.Text) }
            Box(Modifier.weight(1f), contentAlignment = Alignment.Center) {
                val ready = guideState.ready
                val hasTorch = camera?.cameraInfo?.hasFlashUnit() == true
                Text(
                    CameraGuide.text(guideState.hint, vm.captureMode == CaptureMode.Whiteboard, hasTorch && !torch),
                    style = MaterialTheme.typography.labelLarge, color = if (ready) Editor.Ground else Editor.Text,
                    modifier = Modifier.clip(RoundedCornerShape(17.dp)).background(if (ready) Editor.Accent else Editor.Ground.copy(alpha = 0.6f))
                        .semantics { liveRegion = LiveRegionMode.Polite }.testTag("cameraHint")
                        .padding(horizontal = 14.dp, vertical = 8.dp),
                )
            }
            // the torch for dim rooms (a flash would glare on glossy paper)
            if (camera?.cameraInfo?.hasFlashUnit() == true) {
                RoundButton(onClick = {
                    torch = !torch
                    camera?.cameraControl?.enableTorch(torch)
                }, on = torch) { Icon(if (torch) Icons.Outlined.FlashOn else Icons.Outlined.FlashOff, if (torch) "Torch off" else "Torch on", tint = if (torch) Editor.Ground else Editor.Text) }
            } else {
                Box(Modifier.size(48.dp))
            }
        }
        // bottom: what is being scanned, then photos / shutter / the document so far
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .background(Brush.verticalGradient(0f to Color.Transparent, 0.3f to Editor.Ground.copy(alpha = 0.85f)))
                .safeDrawingPadding().padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = 22.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            // document or whiteboard / projector screen (remembered; imports follow it)
            ModePills(vm.captureMode) { vm.chooseCaptureMode(it) }
            Spacer(Modifier.height(18.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
                    LabelledButton("Photos", enabled = !busy, onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) {
                        Box(Modifier.size(52.dp).background(Editor.Panel.copy(alpha = 0.8f), CircleShape), contentAlignment = Alignment.Center) {
                            Icon(Icons.Outlined.PhotoLibrary, null, tint = Editor.Text)
                        }
                    }
                }
                // a capture asked for before CameraX has bound the camera never completes (the screen would wait
                // for ever): the shutter waits for the camera, dimmed
                val ready = camera != null
                // pressed in while the photo is taken
                val sink by animateFloatAsState(if (busy) 0.78f else 1f, if (reduced) snap() else tween(140), label = "shutter")
                Box(
                    Modifier.size(80.dp).alpha(if (ready) 1f else 0.45f).clip(CircleShape).border(4.dp, Color.White, CircleShape)
                        .semantics { contentDescription = "Take photo" }.testTag("shutter")
                        .clickable(enabled = ready && !busy) { shoot() },
                    contentAlignment = Alignment.Center,
                ) {
                    Box(Modifier.size(64.dp).graphicsLayer { scaleX = sink; scaleY = sink }.background(Color.White, CircleShape))
                }
                Box(Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
                    val n = vm.pages.size
                    if (n > 0) {
                        LabelledButton(
                            "Done", enabled = !busy, description = if (n == 1) "Done: the document, 1 page" else "Done: the document, $n pages",
                            onClick = { vm.open(Screen.Pages) },
                        ) { PagesStack(vm, vm.pages.last(), n) }
                    }
                }
            }
        }
    }
}

/** At most this often a frame goes to the guidance (about 10 a second: two CPU cores busy for less of the time). */
private const val GUIDE_INTERVAL_MS = 100L

/** 4:3, the photo's shape, so the preview and the guidance's frames show the same field of view. */
private fun ratio43(): ResolutionSelector =
    ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()

/** The guidance's frames: RGBA, about 320 x 240, only the latest kept (a slow frame never builds a queue). */
private fun guideAnalysis(): ImageAnalysis = ImageAnalysis.Builder()
    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .setResolutionSelector(
        ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy(Size(320, 240), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
            .build(),
    )
    .build()

/** The page the guidance found, outlined on the preview (frame fractions [quad] of a [frameW] x [frameH] frame); the accent when Ready. */
@Composable
private fun GuideQuad(quad: FloatArray, frameW: Int, frameH: Int, ready: Boolean) {
    val line = if (ready) Editor.Accent else Color.White.copy(alpha = 0.85f)
    Canvas(Modifier.fillMaxSize()) {
        val p = CameraGuide.toView(quad, frameW, frameH, size.width, size.height)
        val path = Path().apply {
            moveTo(p[0], p[1])
            for (k in 1 until 4) lineTo(p[2 * k], p[2 * k + 1])
            close()
        }
        drawPath(path, line.copy(alpha = if (ready) 0.18f else 0.1f))
        drawPath(path, line, style = Stroke(width = 3.dp.toPx(), join = StrokeJoin.Round))
    }
}

/** A 48 dp round button on the camera picture; [on] fills it with the accent. */
@Composable
private fun RoundButton(onClick: () -> Unit, on: Boolean = false, content: @Composable () -> Unit) {
    Box(
        Modifier.size(48.dp).clip(CircleShape).background(if (on) Editor.Accent else Editor.Ground.copy(alpha = 0.55f)).clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { content() }
}

/** A round control with its name under it. */
@Composable
private fun LabelledButton(label: String, enabled: Boolean, onClick: () -> Unit, description: String? = null, icon: @Composable () -> Unit) {
    Column(
        Modifier.clip(RoundedCornerShape(12.dp)).clickable(enabled = enabled, role = Role.Button, onClick = onClick).padding(4.dp)
            .then(if (description != null) Modifier.semantics(mergeDescendants = true) { contentDescription = description } else Modifier),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        icon()
        Spacer(Modifier.height(6.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = Editor.Text)
    }
}

/** The pages scanned so far: the last one on a second sheet, with the count. */
@Composable
private fun PagesStack(vm: ScanViewModel, last: String, n: Int) {
    var thumb by remember(last) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(last, vm.pageVersions[last]) { thumb = vm.thumbnail(last) }
    val sheet = RoundedCornerShape(2.dp)
    Box(Modifier.size(54.dp)) {
        if (n > 1) Box(Modifier.offset(14.dp, 4.dp).size(34.dp, 44.dp).rotate(7f).background(Color(0xFFE8ECF2), sheet))
        Box(Modifier.offset(8.dp, 7.dp).size(34.dp, 44.dp).shadow(4.dp, sheet).background(Color.White, sheet).clip(sheet)) {
            thumb?.let { Image(remember(it) { it.asImageBitmap() }, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
        }
        Text(
            "$n", style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.ExtraBold), color = Editor.Ground, textAlign = TextAlign.Center,
            modifier = Modifier.align(Alignment.TopEnd).size(22.dp).background(Editor.Accent, CircleShape).padding(top = 2.dp),
        )
    }
}

/** Document / Whiteboard / Screen, as pills over the camera picture. */
@Composable
private fun ModePills(mode: CaptureMode, onPick: (CaptureMode) -> Unit) {
    Row(Modifier.selectableGroup(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        for (m in CaptureMode.entries) {
            val on = m == mode
            Text(
                m.label.uppercase(),
                style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp, letterSpacing = 0.04.em, fontWeight = if (on) FontWeight.ExtraBold else FontWeight.SemiBold),
                color = if (on) Editor.Ground else Editor.Text,
                modifier = Modifier.clip(RoundedCornerShape(18.dp)).background(if (on) Editor.Accent else Editor.Panel.copy(alpha = 0.7f))
                    .selectable(selected = on, role = Role.RadioButton) { onPick(m) }
                    .padding(horizontal = 16.dp, vertical = 9.dp),
            )
        }
    }
}
