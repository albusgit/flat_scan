plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
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
