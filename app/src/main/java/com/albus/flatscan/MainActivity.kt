package com.albus.flatscan

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.content.pm.ApplicationInfo
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.SizeTransform
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import com.albus.flatscan.ui.CameraScreen
import com.albus.flatscan.ui.graphPaper
import com.albus.flatscan.ui.rememberReducedMotion
import com.albus.flatscan.ui.CropScreen
import com.albus.flatscan.ui.DepthScreen
import com.albus.flatscan.ui.FlatScanTheme
import com.albus.flatscan.ui.HomeScreen
import com.albus.flatscan.ui.MessageHost
import com.albus.flatscan.ui.PagesScreen
import com.albus.flatscan.ui.LicensesScreen
import com.albus.flatscan.ui.ProcessingScreen
import com.albus.flatscan.ui.ResultScreen
import com.albus.flatscan.ui.AdvancedSettingsScreen
import com.albus.flatscan.ui.SettingsScreen
import com.albus.flatscan.ui.StorageScreen
import com.albus.flatscan.ui.TrashScreen
import com.albus.flatscan.ui.SyncScreen

class MainActivity : ComponentActivity() {
    private val vm: ScanViewModel by viewModels()

    /** Android 13+: the batch's progress notification needs the permission; asked once, when a first batch starts. */
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FlatScanTheme(vm.themeMode) {
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    Box(Modifier.fillMaxSize().graphPaper()) { App(vm) }
                    LaunchedEffect(vm.askNotifications) {
                        if (vm.askNotifications) {
                            vm.askNotifications = false
                            askNotificationsOnce()
                        }
                    }
                }
            }
        }
        if (savedInstanceState == null) {
            openFromNotification(intent)
            startBatch(intent)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        openFromNotification(intent)
        startBatch(intent)
    }

    /** The batch's notification: its document on the pages screen. */
    private fun openFromNotification(i: Intent?) {
        val doc = i?.getStringExtra(EXTRA_OPEN_DOCUMENT) ?: return
        i.removeExtra(EXTRA_OPEN_DOCUMENT)
        vm.openDocument(doc)
    }

    /** Asks for the notification permission the first time a batch starts (not again once answered). */
    private fun askNotificationsOnce() {
        if (Build.VERSION.SDK_INT < 33) return
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) return
        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        if (prefs.getBoolean("notifications_asked", false)) return
        prefs.edit().putBoolean("notifications_asked", true).apply()
        notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    companion object {
        /** The batch notification's document id ([com.albus.flatscan.batch.BatchService]). */
        const val EXTRA_OPEN_DOCUMENT = "com.albus.flatscan.OPEN_DOCUMENT"
    }

    /**
     * Debug builds: the batch mode (ScanViewModel.runBatch), e.g.
     * `adb shell am start -S -n com.albus.flatscan/.MainActivity --es batch s35 --es accel cpu` (-S: a fresh
     * start; to a running app the intent only brings it to the front); extras: batch
     * (the folder under files/batch), out, accel (auto, cpu, gpu), shade (--ez), ext (.jpg / .png),
     * static_models (--ez, no effect now), rules (--ez: the rule net; default Settings > Table rules), only
     * (comma-separated names). `--ez selftest true` instead: the self-test (one run), its report written to
     * files/batch_out/selftest_report.txt (external files). `--es bench gpu`: Auto's benchmark with those
     * candidates even where the GPU is unavailable (the emulator: the CPU fallback), its report in
     * files/batch_out/bench_report.txt (`--es bench gpu,gpu16,cpu16`: the fp16 backends too, measured only: Auto's stored
     * choice is not replaced). `--ei threads N` with any of these: the CPU nets on N threads. `--es open_test SET`: the photos of files/batch/SET imported as pages,
     * then each stored page opened in turn, the times in files/batch_out/open_report.txt (ScanViewModel.runOpenTest).
     * `--es ocrPackBase URL`: the text recognition pack from URL instead of the GitHub release (a local server
     * through `adb reverse`; empty: the release again), with any other extra. `--es ocr_test PHOTO`: the pack
     * downloaded if needed and files/batch/PHOTO recognised, report in files/batch_out/ocr_test.txt.
     */
    private fun startBatch(i: Intent?) {
        if (i == null || applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        i.getStringExtra("ocrPackBase")?.let { com.albus.flatscan.output.OcrPack.setDebugBase(this, it) }
        i.getStringExtra("ocr_test")?.let { photo ->
            vm.runOcrTest(photo, java.io.File(getExternalFilesDir(null), "batch_out/ocr_test.txt"))
            return
        }
        if (i.getBooleanExtra("selftest", false)) {
            vm.runSelfTest(1, java.io.File(getExternalFilesDir(null), "batch_out/selftest_report.txt"))
            vm.open(Screen.Settings)
            return
        }
        // `--ei batch_workers N` with any extra: photos a background batch processes at once (0: the phone's choice)
        if (i.hasExtra("batch_workers")) vm.settings.batchWorkers = i.getIntExtra("batch_workers", 0)
        // `--ez no_ahead true` with any extra: the flattening nets do not start ahead of processing (A/B checks)
        if (i.hasExtra("no_ahead")) vm.netsAheadOff = i.getBooleanExtra("no_ahead", false)
        // `--ei threads N` with any extra: the CPU nets on N threads (0: the default) from the next net configuration
        if (i.hasExtra("threads")) vm.netThreads = i.getIntExtra("threads", 0)
        // `--ez gpu_exhaustive true` with any extra: the LiteRT GPU tunes its kernels exhaustively (to time the trade-off)
        if (i.hasExtra("gpu_exhaustive")) NativeLib.setGpuTuning(i.getBooleanExtra("gpu_exhaustive", false))
        // `--ez gpu_warm true`: the GPU sessions' setup ms per net into files/batch_out/gpu_warm.txt (ScanViewModel.debugWarmGpu)
        if (i.getBooleanExtra("gpu_warm", false)) {
            vm.debugWarmGpu(java.io.File(getExternalFilesDir(null), "batch_out/gpu_warm.txt"))
            return
        }
        i.getStringExtra("open_test")?.let { set ->
            vm.runOpenTest(set, java.io.File(getExternalFilesDir(null), "batch_out/open_report.txt"))
            return
        }
        i.getStringExtra("bench")?.let { cand ->
            vm.runAccelBenchmark(cand, java.io.File(getExternalFilesDir(null), "batch_out/bench_report.txt"))
            vm.open(Screen.Settings)
            return
        }
        val set = i.getStringExtra("batch") ?: return
        vm.runBatch(
            set, i.getStringExtra("out"), i.getStringExtra("accel"), if (i.hasExtra("shade")) i.getBooleanExtra("shade", true) else null,
            i.getStringExtra("ext"), i.getBooleanExtra("static_models", false), i.getStringExtra("only"),
            if (i.hasExtra("rules")) i.getBooleanExtra("rules", true) else null,
        )
        vm.open(Screen.Settings)
    }

    override fun onStop() {
        // keep the page being edited on disk: the process may be killed in the background
        vm.persist()
        super.onStop()
    }
}

/** How deep a screen sits under Home: a deeper screen slides in from the right. */
private fun depth(s: Screen) = when (s) {
    Screen.Home -> 0
    Screen.Settings, Screen.Trash, Screen.Pages, Screen.Camera -> 1
    Screen.Licenses, Screen.Sync, Screen.Advanced, Screen.Storage, Screen.Crop -> 2
    Screen.Processing, Screen.Depth -> 3
    Screen.Result -> 4
}

@Composable
private fun App(vm: ScanViewModel) {
    BackHandler(enabled = vm.screen != Screen.Home && vm.screen != Screen.Processing) { vm.back() }
    // the screen stays on through the self-test, the acceleration benchmark and the batch mode
    val view = LocalView.current
    val busy = vm.selfTestRunning || vm.batchRunning
    DisposableEffect(busy) {
        view.keepScreenOn = busy
        onDispose { view.keepScreenOn = false }
    }
    val reduced = rememberReducedMotion()
    Box(Modifier.fillMaxSize()) {
        // with animations off the screen changes outright: AnimatedContent, even with no transition, draws the old
        // and the new screen over each other for a frame
        if (reduced) ScreenContent(vm, vm.screen)
        // screens slide like sheets: deeper screens come in from the right, going back slides them out; the
        // flattened page grows into the result
        else AnimatedContent(
            vm.screen,
            transitionSpec = {
                val fwd = depth(targetState) >= depth(initialState)
                when {
                    initialState == Screen.Processing && targetState == Screen.Result ->
                        (fadeIn(tween(280, 60)) + scaleIn(tween(360, easing = FastOutSlowInEasing), initialScale = 0.94f)) togetherWith fadeOut(tween(200))
                    else -> (slideInHorizontally(tween(320, easing = FastOutSlowInEasing)) { w -> if (fwd) w / 3 else -w / 3 } + fadeIn(tween(240))) togetherWith
                        (slideOutHorizontally(tween(320, easing = FastOutSlowInEasing)) { w -> if (fwd) -w / 6 else w / 3 } + fadeOut(tween(200)))
                }.using(SizeTransform(clip = false))
            },
            label = "screen",
        ) { screen -> ScreenContent(vm, screen) }
        MessageHost(vm)
    }
}

@Composable
private fun ScreenContent(vm: ScanViewModel, screen: Screen) {
    when (screen) {
        Screen.Home -> HomeScreen(vm)
        Screen.Camera -> CameraScreen(vm)
        Screen.Crop -> CropScreen(vm)
        Screen.Processing -> ProcessingScreen(vm)
        Screen.Result -> ResultScreen(vm)
        Screen.Pages -> PagesScreen(vm)
        Screen.Settings -> SettingsScreen(vm)
        Screen.Licenses -> LicensesScreen(vm)
        Screen.Trash -> TrashScreen(vm)
        Screen.Sync -> SyncScreen(vm)
        Screen.Advanced -> AdvancedSettingsScreen(vm)
        Screen.Storage -> StorageScreen(vm)
        Screen.Depth -> DepthScreen(vm)
    }
}
