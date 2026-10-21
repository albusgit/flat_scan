package com.albus.flatscan

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import com.albus.flatscan.ui.CameraScreen
import com.albus.flatscan.ui.CropScreen
import com.albus.flatscan.ui.FlatScanTheme
import com.albus.flatscan.ui.HomeScreen
import com.albus.flatscan.ui.MessageHost
import com.albus.flatscan.ui.PagesScreen
import com.albus.flatscan.ui.ProcessingScreen
import com.albus.flatscan.ui.ResultScreen
import com.albus.flatscan.ui.SettingsScreen

class MainActivity : ComponentActivity() {
    private val vm: ScanViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FlatScanTheme {
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    App(vm)
                }
            }
        }
        if (savedInstanceState == null) startBatch(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        startBatch(intent)
    }

    /**
     * Debug builds: the batch mode (ScanViewModel.runBatch), e.g.
     * `adb shell am start -S -n com.albus.flatscan/.MainActivity --es batch s35 --es accel cpu` (-S: a fresh
     * start; to a running app the intent only brings it to the front); extras: batch
     * (the folder under files/batch), out, accel (auto, cpu, npu, nnapi), shade (--ez), ext (.jpg / .png),
     * static_models (--ez), only (comma-separated names), qnn_packs (--ez: fetch the NPU packs on any phone,
     * to check their delivery on an emulator).
     */
    private fun startBatch(i: Intent?) {
        val set = i?.getStringExtra("batch") ?: return
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        if (i.getBooleanExtra("qnn_packs", false)) QnnPacks.force = true
        vm.runBatch(
            set, i.getStringExtra("out"), i.getStringExtra("accel"), if (i.hasExtra("shade")) i.getBooleanExtra("shade", true) else null,
            i.getStringExtra("ext"), i.getBooleanExtra("static_models", false), i.getStringExtra("only"),
        )
        vm.open(Screen.Settings)
    }

    override fun onStop() {
        // keep the page being edited on disk: the process may be killed in the background
        vm.persist()
        super.onStop()
    }
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
    Box(Modifier.fillMaxSize()) {
        when (vm.screen) {
            Screen.Home -> HomeScreen(vm)
            Screen.Camera -> CameraScreen(vm)
            Screen.Crop -> CropScreen(vm)
            Screen.Processing -> ProcessingScreen(vm)
            Screen.Result -> ResultScreen(vm)
            Screen.Pages -> PagesScreen(vm)
            Screen.Settings -> SettingsScreen(vm)
        }
        MessageHost(vm)
    }
}
