package com.albus.flatscan

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
import androidx.compose.ui.Modifier
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
