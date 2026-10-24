import java.security.MessageDigest
import java.util.Properties
import java.util.zip.ZipFile

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
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
// uvdoc_static, syntheticdoc_hr_static, shadenet (512 x 352) and rulenet (1152 x 800): one file per net; the core rewrites
// the last two for each input size, core/src/tflite_shape.h).
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

// Text recognition (output/Ocr.kt, OcrPack.kt) is a download, not part of the APK: Tesseract4Android's native
// libraries and the heb + eng tessdata_fast models (ocr/tessdata/) go into one zip per ABI,
// build/ocrpack/packs/flatscan-ocr-<ocrPackVersion>-<ABI>.zip (lib/*.so, tessdata/*.traineddata; reproducible: no
// timestamps, fixed order), to be published as assets of the GitHub release tagged ocr-pack-<ocrPackVersion>
// (ocrPackBase). The APK carries each zip's SHA-256 and size (assets/ocr_pack.json) and leaves the libraries out
// (packaging.jniLibs.excludes); Tesseract4Android's Java classes stay in it. A new pack (other libraries or
// models): raise ocrPackVersion and publish its zips under the new tag.
val ocrPackVersion = 1
val ocrPackBase = "https://github.com/albusgit/flat_scan/releases/download/ocr-pack-$ocrPackVersion/"
val ocrLibs = listOf("libjpeg.so", "libpngx.so", "libleptonica.so", "libtesseract.so")
val tessAar: Configuration by configurations.creating { isTransitive = false }
val ocrPackDir = layout.buildDirectory.dir("ocrpack")
val ocrPacks = abis.map { abi ->
    tasks.register<Zip>("ocrPack_" + abi.replace("-", "_")) {
        archiveFileName.set("flatscan-ocr-$ocrPackVersion-$abi.zip")
        destinationDirectory.set(ocrPackDir.map { it.dir("packs") })
        isPreserveFileTimestamps = false
        isReproducibleFileOrder = true
        from({ tessAar.files.map { zipTree(it) } }) {
            include(ocrLibs.map { "jni/$abi/$it" })
            eachFile { path = "lib/$name" }
        }
        from("ocr/tessdata") {
            include("*.traineddata")
            into("tessdata")
        }
        includeEmptyDirs = false
    }
}
val ocrPackInfo by tasks.registering {
    val packs = ocrPacks.map { t -> t.flatMap { it.archiveFile } }
    inputs.files(packs)
    val out = ocrPackDir.map { it.file("assets/ocr_pack.json") }
    outputs.file(out)
    val version = ocrPackVersion
    val base = ocrPackBase
    val names = abis
    doLast {
        val entries = names.zip(packs).joinToString(",\n") { (abi, p) ->
            val f = p.get().asFile
            ZipFile(f).use { z ->
                val got = z.entries().toList().map { it.name }.toSet()
                for (lib in ocrLibs) require("lib/$lib" in got) { "${f.name}: no lib/$lib (Tesseract4Android AAR, $abi)" }
                require(got.any { it.startsWith("tessdata/") }) { "${f.name}: no tessdata" }
            }
            val sha = MessageDigest.getInstance("SHA-256").digest(f.readBytes()).joinToString("") { "%02x".format(it) }
            "    \"$abi\": {\"file\": \"${f.name}\", \"sha256\": \"$sha\", \"size\": ${f.length()}}"
        }
        out.get().asFile.apply { parentFile.mkdirs() }.writeText("{\"version\": $version, \"base\": \"$base\", \"packs\": {\n$entries\n}}\n")
    }
}
tasks.register("ocrPacks") {
    group = "build"
    description = "The text recognition packs to publish (build/ocrpack/packs) and their pins (assets/ocr_pack.json)"
    dependsOn(ocrPackInfo)
}
tasks.configureEach {
    if (name.startsWith("merge") && (name.endsWith("JniLibFolders") || name.endsWith("Assets") || name.endsWith("NativeLibs")) ||
        name.contains("lint", ignoreCase = true) || name.startsWith("generate") && name.endsWith("Assets")) dependsOn(syncLitert)
    if (name.startsWith("merge") && name.endsWith("Assets") || name.contains("lint", ignoreCase = true) ||
        name.startsWith("generate") && name.endsWith("Assets")) dependsOn(ocrPackInfo)
}

// OpenCV 4.11.0 for the core. opencv.min.dir in local.properties (or -PopencvMin=DIR): the slim static build of
// tools/opencv_min/build_opencv_min.sh (core imgproc imgcodecs calib3d photo + features2d flann, the AAR's HALs,
// dispatch and TBB), linked into libflatscan; otherwise (or -PopencvMin=aar) the org.opencv:opencv AAR's
// libopencv_java4.so (16 modules) through prefab.
val opencvMin: String? = (project.findProperty("opencvMin") as String? ?: localProps.getProperty("opencv.min.dir"))
    ?.takeIf { it.isNotBlank() && it != "aar" }
    ?.also { require(file("$it/sdk/native/jni/OpenCVConfig.cmake").isFile) { "opencvMin: no sdk/native/jni/OpenCVConfig.cmake in $it" } }
// A release without the slim build would ship the AAR's libraries without a word (native libraries 16.3 MB
// instead of 5.9 MB): it stops unless -PopencvMin=aar asks for the AAR.
val opencvAarAsked = (project.findProperty("opencvMin") as String?) == "aar"
tasks.configureEach {
    if (opencvMin == null && !opencvAarAsked && name.startsWith("pre") && name.endsWith("ReleaseBuild"))
        doFirst { throw GradleException("release build without the slim OpenCV: set opencv.min.dir in local.properties (tools/opencv_min/build_opencv_min.sh) or pass -PopencvMin=aar") }
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
    sourceSets["main"].assets.srcDir(ocrPackDir.map { it.dir("assets") })
    buildFeatures {
        compose = true
        prefab = opencvMin == null
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    testOptions {
        // JVM tests run on Robolectric (Room, Compose UI, real bitmap decoding with native graphics)
        unitTests.isIncludeAndroidResources = true
        unitTests.all { it.maxHeapSize = "3g" }
    }
    // the Room schemas as assets of debug builds: the migration tests (Robolectric) read them from there
    sourceSets["debug"].assets.srcDir("$projectDir/schemas")
    packaging {
        jniLibs {
            // uncompressed in the APK (loaded in place, not copied out at install: less space on the phone; Play
            // compresses the download); -PcompressLibs=true: compressed (a smaller APK file to download directly)
            useLegacyPackaging = (project.findProperty("compressLibs") as String?).toBoolean()
            // OpenCV's AAR bundles an older, 4 KB-aligned libc++_shared; the NDK's own copy
            // (built by externalNativeBuild) is 16 KB-aligned and ABI-compatible
            pickFirsts += "**/libc++_shared.so"
            // Tesseract4Android's native libraries: in the text recognition pack, not the APK (ocrPacks)
            excludes += ocrLibs.map { "**/$it" }
        }
    }
}

ksp {
    // the Room schema of every version is kept in app/schemas (migration tests read them)
    arg("room.schemaLocation", "$projectDir/schemas")
}

// Robolectric's Android runtime jar, fetched by Gradle (which retries; Robolectric's own download does not)
// and handed to the tests offline
val robolectricRuntime: Configuration by configurations.creating { isTransitive = false }
val robolectricJars = layout.buildDirectory.dir("robolectric-jars")
val syncRobolectric by tasks.registering(Sync::class) {
    from(robolectricRuntime)
    into(robolectricJars)
}
tasks.withType<Test>().configureEach {
    dependsOn(syncRobolectric)
    // the sync integration tests (tools/sync_it/run.sh) find their servers through these; unset: skipped
    for (k in listOf("FS_WEBDAV_URL", "FS_WEBDAV_USER", "FS_WEBDAV_PASS", "FS_WEBDAV_ROOT", "FS_PAPERLESS_URL", "FS_PAPERLESS_TOKEN")) {
        System.getenv(k)?.let { environment(k, it) }
    }
    systemProperty("robolectric.offline", "true")
    systemProperty("robolectric.dependency.dir", robolectricJars.get().asFile.absolutePath)
}

val roomVersion = "2.7.2"
dependencies {
    // sync (feature 6): HTTP for WebDAV / paperless-ngx, queued uploads, encrypted credentials
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("androidx.work:work-runtime-ktx:2.10.5")
    implementation("androidx.security:security-crypto:1.1.0")
    testImplementation("androidx.work:work-testing:2.10.5")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    // OCR (feature 3): Tesseract 5 for Android, Apache-2.0: its Java classes in the APK, its native libraries and the
    // heb + eng tessdata_fast models (ocr/tessdata) in the downloaded text recognition pack (ocrPacks)
    implementation("com.github.adaptech-cz.Tesseract4Android:tesseract4android:4.8.0")
    tessAar("com.github.adaptech-cz.Tesseract4Android:tesseract4android:4.8.0@aar")
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.17")
    robolectricRuntime("org.robolectric:android-all-instrumented:15-robolectric-13954326-i7")
    testImplementation("androidx.test:core-ktx:1.6.1")
    testImplementation("androidx.test.ext:junit-ktx:1.2.1")
    testImplementation("androidx.room:room-testing:$roomVersion")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
    testImplementation(platform("androidx.compose:compose-bom:2025.05.00"))
    testImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
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
