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
import com.albus.flatscan.ui.TrashScreen
import com.albus.flatscan.ui.SyncScreen

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
     * (the folder under files/batch), out, accel (auto, cpu, gpu), shade (--ez), ext (.jpg / .png),
     * static_models (--ez, no effect now), rules (--ez: the rule net; default Settings > Table rules), only
     * (comma-separated names). `--ez selftest true` instead: the self-test (one run), its report written to
     * files/batch_out/selftest_report.txt (external files).
     */
    private fun startBatch(i: Intent?) {
        if (i == null || applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        if (i.getBooleanExtra("selftest", false)) {
            vm.runSelfTest(1, java.io.File(getExternalFilesDir(null), "batch_out/selftest_report.txt"))
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
            Screen.Trash -> TrashScreen(vm)
            Screen.Sync -> SyncScreen(vm)
        }
        MessageHost(vm)
    }
}
