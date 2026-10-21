plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// ONNX Runtime for the nets, linked into libflatscan through its C/C++ API (core/src/infer.cpp). The AARs
// ship no prefab, so unpackOrt extracts their headers and libraries into build/ort/<flavour> (headers/,
// jni/<ABI>/, the layout core/CMakeLists.txt's FS_ORT_ROOT reads), which also feeds jniLibs:
//  * arm64-v8a: onnxruntime-android-qnn (the QNN execution provider; it has no NNAPI provider) and the
//    QNN HTP libraries of com.qualcomm.qti:qnn-runtime (HTP v68-v79, the system library and the on-device
//    graph preparer; the DSP and GPU backends are left out), whose licence and notice file go to
//    assets/licenses/ (the licence lets them be redistributed only inside an application, with the notices);
//  * every other ABI (the x86_64 emulator): onnxruntime-android, the same ORT version (NNAPI, XNNPACK).
// -Port=plain: onnxruntime-android on arm64-v8a too (NNAPI instead of QNN, no Qualcomm libraries).
//
// Packaging (-Ppacks, default true with -Port=qnn): the base keeps ONNX Runtime and the original models;
// the QNN libraries and the fixed-shape models only the NPU runs go into on-demand asset packs (qnn_htp:
// libQnnHtp, libQnnHtpPrepare, libQnnSystem and the static models; qnn_v68 .. qnn_v79: one Hexagon
// generation's skeleton + stub each). Play then delivers the base split for the phone's ABI and screen,
// and the app fetches the packs only on a Snapdragon that uses the NPU, and only its own generation
// (QnnPacks.kt). `bundleDebug` / `bundleRelease` build the app bundle with the packs; an APK built with
// `assembleDebug` has no NPU libraries (the NPU then falls back to the CPU). -Ppacks=false: the old single
// APK with every library in lib/ (compressed, extracted at install for the DSP).
val ortVersion = "1.22.0"
val qnnVersion = "2.33.0"
val ortFlavour = (project.findProperty("ort") as String? ?: "qnn").also { require(it == "qnn" || it == "plain") { "-Port=qnn|plain" } }
val ortDir = layout.buildDirectory.dir("ort/$ortFlavour")
val ortPlain: Configuration by configurations.creating { isTransitive = false }
val ortQnn: Configuration by configurations.creating { isTransitive = false }
// -PqnnHtp=V73,V75: only these HTP generations (default all the AAR has: V68 Snapdragon 888 .. V79 8 Elite)
val qnnHtp = (project.findProperty("qnnHtp") as String? ?: "V68,V69,V73,V75,V79").split(",").map { it.trim().uppercase() }
val qnnCommon = listOf("libQnnHtp.so", "libQnnHtpPrepare.so", "libQnnSystem.so")
val qnnLibs = qnnCommon + qnnHtp.flatMap { listOf("libQnnHtp${it}Skel.so", "libQnnHtp${it}Stub.so") }
val usePacks = ortFlavour == "qnn" && (project.findProperty("packs") as String? ?: "true").toBoolean()
// the fixed-shape models (tools/accel/prepare_models.py, tools/shadow_net/export.py --static): accelerators only
val npuModels = listOf("uvdoc_static.onnx", "syntheticdoc_hr_static.onnx", "shadenet_512x352.onnx")
val qnnPacks = listOf("qnn_htp") + qnnHtp.map { "qnn_${it.lowercase()}" }
val unpackOrt by tasks.registering(Sync::class) {
    into(ortDir)
    inputs.property("packs", usePacks)   // the include patterns below depend on it
    val qnn = ortFlavour == "qnn"
    from({ zipTree(ortPlain.singleFile) }) {
        include("headers/**")
        include("jni/*/libonnxruntime.so")
        if (qnn) exclude("jni/arm64-v8a/**")
    }
    if (qnn) {
        from({ zipTree(ortQnn.files.single { it.name.startsWith("onnxruntime-android-qnn") }) }) {
            include("jni/arm64-v8a/libonnxruntime.so")
        }
        from({ zipTree(ortQnn.files.single { it.name.startsWith("qnn-runtime") }) }) {
            if (!usePacks) include(qnnLibs.map { "jni/arm64-v8a/$it" })
            include("LICENSE.pdf", "NOTICE.txt")
            eachFile { if (!path.startsWith("jni/")) path = "assets/licenses/qnn-runtime-$qnnVersion-$name" }
        }
    }
    includeEmptyDirs = false
}
// the asset packs' contents (their src/main/assets, git-ignored): assets/qnn/lib/ (the common libraries),
// assets/qnn/models/ (the static models), assets/qnn/<v>/ (one generation's skeleton and stub)
val syncQnnPacks by tasks.registering {
    val packDirs = qnnPacks.map { rootProject.file("$it/src/main/assets") }
    outputs.dirs(packDirs)
    doLast {
        val qnnAar = ortQnn.files.single { it.name.startsWith("qnn-runtime") }
        packDirs.forEach { project.delete(it) }
        project.copy {
            from(zipTree(qnnAar)) { include(qnnCommon.map { "jni/arm64-v8a/$it" }); eachFile { path = "qnn/lib/$name" } }
            from(file("src/main/assets/models")) { include(npuModels); eachFile { path = "qnn/models/$name" } }
            includeEmptyDirs = false
            into(packDirs[0])
        }
        qnnHtp.forEachIndexed { i, v ->
            project.copy {
                from(zipTree(qnnAar)) { include("jni/arm64-v8a/libQnnHtp${v}Skel.so", "jni/arm64-v8a/libQnnHtp${v}Stub.so"); eachFile { path = "qnn/${v.lowercase()}/$name" } }
                includeEmptyDirs = false
                into(packDirs[i + 1])
            }
        }
    }
}
if (usePacks) rootProject.subprojects { if (name in qnnPacks) tasks.configureEach { dependsOn(syncQnnPacks) } }
// CMake configures against the unpacked package; the merge tasks package it
tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("buildCMake") || name.startsWith("generateJsonModel") ||
        (name.startsWith("merge") && (name.endsWith("JniLibFolders") || name.endsWith("Assets")))) dependsOn(unpackOrt)
}

android {
    namespace = "com.albus.flatscan"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.albus.flatscan"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        // phones: arm64-v8a; add x86_64 for an emulator with -Pabis=arm64-v8a,x86_64
        val abis = (project.findProperty("abis") as String? ?: "arm64-v8a").split(",")
        ndk { abiFilters += abis }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared", "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                // ONNX Runtime for the nets (core/src/infer.cpp), unpacked by unpackOrt; the execution
                // providers are looked up at run time, so each ABI uses those its library has
                arguments += listOf("-DFS_ORT_ROOT=" + ortDir.get().asFile.invariantSeparatorsPath,
                    "-DFS_ORT_QNN=ON", "-DFS_ORT_NNAPI=ON", "-DFS_ORT_XNNPACK=ON")
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
            }
        }
    }

    buildTypes {
        debug {
            // the native core is numeric code; keep it optimised in debug builds too
            externalNativeBuild { cmake { arguments += listOf("-DCMAKE_BUILD_TYPE=Release") } }
        }
        release {
            isMinifyEnabled = false
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    androidResources {
        noCompress += "onnx"
        // with the packs the static models are in qnn_htp, not in the base
        if (usePacks) ignoreAssetsPatterns += npuModels.map { "!$it" }
    }
    if (usePacks) assetPacks += qnnPacks.map { ":$it" }
    bundle {
        abi { enableSplit = true }
        density { enableSplit = true }
        language { enableSplit = true }
    }
    sourceSets["main"].jniLibs.srcDir(ortDir.map { it.dir("jni") })
    sourceSets["main"].assets.srcDir(ortDir.map { it.dir("assets") })
    buildFeatures {
        compose = true
        prefab = true
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // -Ppacks=false: compressed native libraries, extracted at install (the DSP loads the HTP skeleton from
    // the native library directory). With the packs the skeleton comes from a pack's files, so the libraries
    // stay uncompressed in the APK and are not copied out at install (smaller on the phone; Play compresses
    // the download).
    packaging {
        jniLibs {
            // -PcompressLibs=true: compressed anyway (a smaller APK file to download directly, without the NPU)
            useLegacyPackaging = !usePacks || (project.findProperty("compressLibs") as String?).toBoolean()
            // OpenCV's AAR bundles an older, 4 KB-aligned libc++_shared; the NDK's own copy
            // (built by externalNativeBuild) is 16 KB-aligned and ABI-compatible
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

dependencies {
    implementation("org.opencv:opencv:4.11.0")
    // the on-demand QNN packs (QnnPacks.kt)
    implementation("com.google.android.play:asset-delivery:2.3.0")
    ortPlain("com.microsoft.onnxruntime:onnxruntime-android:$ortVersion@aar")
    ortQnn("com.microsoft.onnxruntime:onnxruntime-android-qnn:$ortVersion@aar")
    ortQnn("com.qualcomm.qti:qnn-runtime:$qnnVersion@aar")
    implementation(platform("androidx.compose:compose-bom:2025.05.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.0")
    implementation("androidx.exifinterface:exifinterface:1.4.1")
    implementation("androidx.camera:camera-core:1.4.2")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
}
