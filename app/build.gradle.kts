import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val localProps = Properties().apply { rootProject.file("local.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) } }

// The nets run on LiteRT (the TensorFlow Lite runtime) in the C++ core (core/src/infer.cpp): its CPU kernels are the
// reference backend and the fallback, its GPU delegate the accelerator. infer.cpp dlopens the two libraries by name
// (no link-time dependency), so syncLitert copies them out of the LiteRT AARs (they also hold a Java API the app does
// not use) into jniLibs (build/litert/jni/<ABI>/: libtensorflowlite_jni.so, the runtime with its CPU kernels and
// XNNPACK; libtensorflowlite_gpu_jni.so, the GPU delegate, loaded only when a net asks for the GPU), and the .tflite
// models into the APK's assets/models/ (stored uncompressed: the core reads them in place). ONNX Runtime and the
// .onnx models are no longer in the app (Round 11, docs/results/r11/runtime.md); the desktop builds
// (tools/desktop_check) keep them, and the .onnx files stay in app/src/main/assets/models/ for those builds only.
// -PlitertModels=DIR: the .tflite files from DIR instead of app/src/main/litert/models (same file names: pagefinder,
// uvdoc_static, syntheticdoc_hr_static, shadenet (dynamic H x W, the CPU's), shadenet_512x352 (the GPU's)).
val litertVersion = "1.4.0"
val litertAars: Configuration by configurations.creating { isTransitive = false }
val litertLibs = listOf("libtensorflowlite_jni.so", "libtensorflowlite_gpu_jni.so")
val litertModels = file(project.findProperty("litertModels") as String? ?: "src/main/litert/models")
val litertDir = layout.buildDirectory.dir("litert")
// phones: arm64-v8a; add x86_64 for an emulator with -Pabis=arm64-v8a,x86_64
val abis = (project.findProperty("abis") as String? ?: "arm64-v8a").split(",").map { it.trim() }
val syncLitert by tasks.registering(Sync::class) {
    into(litertDir)
    inputs.property("abis", abis)
    litertAars.files.forEach { aar ->
        from({ zipTree(aar) }) {
            include(abis.flatMap { abi -> litertLibs.map { "jni/$abi/$it" } })
        }
    }
    from(litertModels) {
        include("*.tflite")
        into("assets/models")
    }
    includeEmptyDirs = false
    doLast {
        for (abi in abis) {
            val got = File(litertDir.get().asFile, "jni/$abi").list().orEmpty().toSet()
            require(got == litertLibs.toSet()) { "LiteRT $litertVersion AARs, $abi: found ${got.sorted()}, expected $litertLibs" }
        }
        val models = File(litertDir.get().asFile, "assets/models").list().orEmpty().filter { it.endsWith(".tflite") }
        require(models.isNotEmpty()) { "no .tflite models in $litertModels" }
    }
}
tasks.configureEach {
    if (name.startsWith("merge") && (name.endsWith("JniLibFolders") || name.endsWith("Assets") || name.endsWith("NativeLibs")) ||
        name.contains("lint", ignoreCase = true) || name.startsWith("generate") && name.endsWith("Assets")) dependsOn(syncLitert)
}

// OpenCV 4.11.0 for the core. opencv.min.dir in local.properties (or -PopencvMin=DIR): the slim static build of
// tools/opencv_min/build_opencv_min.sh (core imgproc imgcodecs calib3d photo + features2d flann, the AAR's HALs,
// dispatch and TBB), linked into libflatscan; otherwise (or -PopencvMin=aar) the org.opencv:opencv AAR's
// libopencv_java4.so (16 modules) through prefab.
val opencvMin: String? = (project.findProperty("opencvMin") as String? ?: localProps.getProperty("opencv.min.dir"))
    ?.takeIf { it.isNotBlank() && it != "aar" }
    ?.also { require(file("$it/sdk/native/jni/OpenCVConfig.cmake").isFile) { "opencvMin: no sdk/native/jni/OpenCVConfig.cmake in $it" } }

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
        ndk { abiFilters += abis }
        externalNativeBuild {
            cmake {
                // libc++: linked into libflatscan with the slim OpenCV (nothing else in the APK needs libc++_shared.so: the
                // LiteRT and CameraX libraries carry their own); the AAR's libopencv_java4.so needs the shared one
                arguments += listOf("-DANDROID_STL=" + (if (opencvMin != null) "c++_static" else "c++_shared"), "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                if (opencvMin != null) arguments += "-DFS_OPENCV_MIN=" + file(opencvMin).invariantSeparatorsPath
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
            }
        }
    }

    // R8 code and resource shrinking (proguard-rules.pro): the unshrunk dex was 61 MB (19 MB compressed), mostly
    // Compose and its full icon set. Release builds always; debug with -Pminify=true (to check the rules with the
    // debug batch mode). Release is signed with the debug key for sideloading; Play needs an upload key.
    val minifyDebug = (project.findProperty("minify") as String?).toBoolean()
    buildTypes {
        debug {
            // the native core is numeric code; keep it optimised in debug builds too
            externalNativeBuild { cmake { arguments += listOf("-DCMAKE_BUILD_TYPE=Release") } }
            isMinifyEnabled = minifyDebug
            isShrinkResources = minifyDebug
            if (minifyDebug) proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        release {
            // the same native build as debug (CMake Release, not AGP's RelWithDebInfo): the same numbers
            externalNativeBuild { cmake { arguments += listOf("-DCMAKE_BUILD_TYPE=Release") } }
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    androidResources {
        // the core reads the models in place from the APK (AAsset_getBuffer)
        noCompress += "tflite"
        // the .onnx models are the desktop builds' (ONNX Runtime): not in the app
        ignoreAssetsPatterns += "!*.onnx"
    }
    bundle {
        abi { enableSplit = true }
        density { enableSplit = true }
        language { enableSplit = true }
    }
    sourceSets["main"].jniLibs.srcDir(litertDir.map { it.dir("jni") })
    sourceSets["main"].assets.srcDir(litertDir.map { it.dir("assets") })
    buildFeatures {
        compose = true
        prefab = opencvMin == null
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        jniLibs {
            // uncompressed in the APK (loaded in place, not copied out at install: less space on the phone; Play
            // compresses the download); -PcompressLibs=true: compressed (a smaller APK file to download directly)
            useLegacyPackaging = (project.findProperty("compressLibs") as String?).toBoolean()
            // OpenCV's AAR bundles an older, 4 KB-aligned libc++_shared; the NDK's own copy
            // (built by externalNativeBuild) is 16 KB-aligned and ABI-compatible
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

dependencies {
    if (opencvMin == null) implementation("org.opencv:opencv:4.11.0")
    // the nets' runtime, copied out by syncLitert (not linked, no Java API used)
    litertAars("com.google.ai.edge.litert:litert:$litertVersion@aar")
    litertAars("com.google.ai.edge.litert:litert-gpu:$litertVersion@aar")
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
