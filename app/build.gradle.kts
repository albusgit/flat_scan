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
val ortVersion = "1.22.0"
val qnnVersion = "2.33.0"
val ortFlavour = (project.findProperty("ort") as String? ?: "qnn").also { require(it == "qnn" || it == "plain") { "-Port=qnn|plain" } }
val ortDir = layout.buildDirectory.dir("ort/$ortFlavour")
val ortPlain: Configuration by configurations.creating { isTransitive = false }
val ortQnn: Configuration by configurations.creating { isTransitive = false }
// -PqnnHtp=V73,V75: only these HTP generations (default all the AAR has: V68 Snapdragon 888 .. V79 8 Elite)
val qnnHtp = (project.findProperty("qnnHtp") as String? ?: "V68,V69,V73,V75,V79").split(",").map { it.trim().uppercase() }
val qnnLibs = listOf("libQnnHtp.so", "libQnnHtpPrepare.so", "libQnnSystem.so") +
    qnnHtp.flatMap { listOf("libQnnHtp${it}Skel.so", "libQnnHtp${it}Stub.so") }
val unpackOrt by tasks.registering(Sync::class) {
    into(ortDir)
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
            include(qnnLibs.map { "jni/arm64-v8a/$it" })
            include("LICENSE.pdf", "NOTICE.txt")
            eachFile { if (!path.startsWith("jni/")) path = "assets/licenses/qnn-runtime-$qnnVersion-$name" }
        }
    }
    includeEmptyDirs = false
}
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
    // compressed native libraries: OpenCV alone is ~22 MB per ABI uncompressed
    packaging {
        jniLibs {
            useLegacyPackaging = true
            // OpenCV's AAR bundles an older, 4 KB-aligned libc++_shared; the NDK's own copy
            // (built by externalNativeBuild) is 16 KB-aligned and ABI-compatible
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

dependencies {
    implementation("org.opencv:opencv:4.11.0")
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
