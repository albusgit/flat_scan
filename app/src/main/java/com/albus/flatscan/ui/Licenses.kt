package com.albus.flatscan.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.ScanViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * A third-party component in the release APK. [files] are its licence texts in assets/licenses/,
 * shown in this order under [notice] (the attribution or copyright lines).
 */
class OssComponent(
    val name: String,
    val version: String,
    val copyright: String,
    val license: String,
    val url: String,
    val files: List<String>,
    val notice: String = copyright,
)

private const val APACHE = "Apache-2.0.txt"

/**
 * Every third-party component in the release APK (README "Licences" lists the same). Versions:
 * app/build.gradle.kts, build.gradle.kts, tools/opencv_min/build_opencv_min.sh (the release's slim OpenCV,
 * which follows the Maven AAR's build: oneTBB, libjpeg-turbo, libpng, cpufeatures, Carotene and KleidiCV HALs;
 * zlib is Android's system libz, not in the APK). Our own models (ShadeNet, rule net) are not listed.
 */
// TODO page finder: origin/licence not documented (pagefinder.tflite ships in the APK; list it once its origin,
//  licence and training data are known, see README "Licences")
val OssComponents = listOf(
    OssComponent(
        "UVDoc model", "weights of the authors' repository",
        "Copyright (c) 2023 Tanguy MAGNE", "MIT", "https://github.com/tanguymagne/UVDoc",
        listOf("UVDoc-MIT.txt"),
        notice = "UVDoc: Neural Grid-based Document Unwarping. Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung. " +
            "SIGGRAPH Asia 2023.",
    ),
    OssComponent(
        "SyntheticDoc high-res model", "weights of the authors' repository",
        "Copyright (c) 2026 Tanguy MAGNE", "MIT", "https://github.com/tanguymagne/SyntheticDoc",
        listOf("SyntheticDoc-MIT.txt"),
        notice = "SyntheticDoc: A Large Synthetic Dataset for Document Unwarping and Illumination Correction. " +
            "Daniel Woortmann, Tanguy Magne, Olga Sorkine-Hornung. ECCV 2026. Trained on the SyntheticDoc dataset (below).",
    ),
    OssComponent(
        "SyntheticDoc dataset", "2026 (training data of the SyntheticDoc high-res model; not in the app)",
        "Daniel Woortmann, Tanguy Magne, Olga Sorkine-Hornung; ETH Zurich", "CC BY 4.0",
        "https://doi.org/10.3929/ethz-c-000801994", listOf("CC-BY-4.0.txt"),
        notice = "\"SyntheticDoc: A Large Synthetic Dataset for Document Unwarping and Illumination Correction\" by " +
            "Daniel Woortmann, Tanguy Magne and Olga Sorkine-Hornung, ETH Zurich, 2026. DOI 10.3929/ethz-c-000801994. " +
            "Licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The model was trained on it; " +
            "no images of the dataset are in the app.",
    ),
    OssComponent(
        "LiteRT (runtime and GPU delegate)", "1.4.0", "Copyright The TensorFlow Authors", "Apache-2.0 (with a Caffe BSD notice)",
        "https://github.com/google-ai-edge/LiteRT", listOf("LiteRT-LICENSE.txt"),
    ),
    OssComponent(
        "OpenCV", "4.11.0", "Copyright (C) 2000-2022 Intel Corporation, OpenCV Foundation and others (full list below)",
        "Apache-2.0", "https://github.com/opencv/opencv", listOf("OpenCV-COPYRIGHT.txt", APACHE),
        notice = "",
    ),
    OssComponent(
        "oneTBB (in OpenCV)", "2021.11.0", "Copyright (c) 2005-2023 Intel Corporation", "Apache-2.0",
        "https://github.com/oneapi-src/oneTBB", listOf(APACHE),
    ),
    OssComponent(
        "libjpeg-turbo (in OpenCV)", "3.0.3", "Copyright (C) 2009-2023 D. R. Commander and others; the Independent JPEG Group",
        "IJG License, BSD-3-Clause, zlib License", "https://libjpeg-turbo.org",
        listOf("libjpeg-turbo-LICENSE.txt", "libjpeg-turbo-README-ijg.txt"),
        notice = "This software is based in part on the work of the Independent JPEG Group.",
    ),
    OssComponent(
        "libpng (in OpenCV)", "1.6.43", "Copyright (c) 1995-2024 The PNG Reference Library Authors; (c) 2018-2024 Cosmin Truta",
        "PNG Reference Library License v2", "http://www.libpng.org/pub/png/libpng.html", listOf("libpng-LICENSE.txt"),
    ),
    OssComponent(
        "Carotene HAL (in OpenCV)", "0.0.1", "Copyright (C) 2014, NVIDIA Corporation", "BSD-3-Clause",
        "https://github.com/opencv/opencv/tree/4.11.0/3rdparty/carotene", listOf("Carotene-BSD-3.txt"),
    ),
    OssComponent(
        "KleidiCV HAL (in OpenCV)", "0.3.0", "Copyright 2023 - 2024 Arm Limited and/or its affiliates", "Apache-2.0",
        "https://gitlab.arm.com/kleidi/kleidicv", listOf(APACHE),
    ),
    OssComponent(
        "cpufeatures (in OpenCV)", "from OpenCV 4.11.0", "Copyright (C) 2016 The Android Open Source Project", "Apache-2.0",
        "https://github.com/opencv/opencv/tree/4.11.0/3rdparty/cpufeatures", listOf("cpufeatures-LICENSE.txt", APACHE),
        notice = "",
    ),
    OssComponent(
        "AndroidX and Kotlin", "Compose BOM 2025.05.00 (UI, Material3, Material icons), Activity 1.10.1, Core 1.16.0, " +
            "Lifecycle 2.9.0, CameraX 1.4.2, ExifInterface 1.4.1; Kotlin standard library 2.1.21",
        "Copyright The Android Open Source Project; Copyright JetBrains s.r.o. and Kotlin Programming Language contributors",
        "Apache-2.0", "https://developer.android.com/jetpack/androidx", listOf(APACHE),
    ),
    OssComponent(
        "LLVM libc++ (Android NDK r27c)", "LLVM 18 (clang r522817c)", "Copyright the LLVM Project contributors",
        "Apache-2.0 with LLVM Exceptions", "https://libcxx.llvm.org", listOf("libcxx-LICENSE.txt"),
    ),
    OssComponent(
        "Recursive font", "1.085", "Copyright 2020 The Recursive Project Authors", "SIL Open Font License 1.1",
        "https://github.com/arrowtype/recursive", listOf("Recursive-OFL.txt"),
        notice = "",
    ),
)

/** Settings > Open-source licences: the components; tapping one shows its full licence text. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LicensesScreen(vm: ScanViewModel) {
    var open by rememberSaveable { mutableStateOf<String?>(null) }
    val selected = OssComponents.firstOrNull { it.name == open }
    BackHandler(enabled = selected != null) { open = null }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopAppBar(
            title = {
                Text(selected?.name ?: "Open-source licences", style = MaterialTheme.typography.titleLarge, maxLines = 1, overflow = TextOverflow.Ellipsis)
            },
            colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            navigationIcon = {
                IconButton(onClick = { if (selected != null) open = null else vm.back() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") }
            },
        )
        if (selected != null) LicenseText(selected) else LicenseList { open = it.name }
    }
}

@Composable
private fun LicenseList(onOpen: (OssComponent) -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    LazyColumn(Modifier.fillMaxSize()) {
        item {
            Text(
                "FlatScan is built with these open-source components. Tap one for its licence.",
                style = MaterialTheme.typography.bodyMedium, color = muted,
                modifier = Modifier.padding(start = 16.dp, end = 16.dp, bottom = 8.dp),
            )
        }
        items(OssComponents, key = { it.name }) { c ->
            Row(
                Modifier.fillMaxWidth().clickable(role = Role.Button, onClickLabel = "Show the licence") { onOpen(c) }
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(c.name, style = MaterialTheme.typography.titleMedium)
                    Text(c.version, style = MaterialTheme.typography.bodySmall, color = muted)
                    Text(c.copyright, style = MaterialTheme.typography.bodySmall, color = muted)
                    Text(
                        c.license, style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono),
                        color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 2.dp),
                    )
                    Text(c.url, style = MaterialTheme.typography.bodySmall, color = muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                Icon(Icons.Outlined.ChevronRight, null, tint = muted)
            }
            RuledDivider(Modifier.padding(horizontal = 16.dp))
        }
    }
}

@Composable
private fun LicenseText(c: OssComponent) {
    val ctx = LocalContext.current
    val text by produceState<String?>(null, c) {
        value = withContext(Dispatchers.IO) {
            c.files.joinToString("\n\n") { f ->
                runCatching { ctx.assets.open("licenses/$f").bufferedReader().use { it.readText() } }.getOrElse { "($f is missing)" }
            }
        }
    }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp)) {
        Text("${c.license} · ${c.version}", style = MaterialTheme.typography.labelMedium.copy(fontFamily = RecursiveMono), color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(4.dp))
        Text(c.url, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        if (c.notice.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Text(c.notice, style = MaterialTheme.typography.bodyMedium)
        }
        if (c.notice != c.copyright) {
            Spacer(Modifier.height(6.dp))
            Text(c.copyright, style = MaterialTheme.typography.bodySmall)
        }
        Spacer(Modifier.height(12.dp))
        SelectionContainer {
            Text(
                text ?: "", style = MaterialTheme.typography.bodySmall.copy(fontFamily = RecursiveMono, fontSize = 11.sp, lineHeight = 15.sp),
            )
        }
        Spacer(Modifier.height(24.dp))
    }
}
