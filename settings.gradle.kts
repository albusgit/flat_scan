pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "FlatScan"
include(":app")
// on-demand asset packs with the Qualcomm NPU libraries (app/build.gradle.kts, QnnPacks.kt)
include(":qnn_htp", ":qnn_v68", ":qnn_v69", ":qnn_v73", ":qnn_v75", ":qnn_v79")
// on-demand asset pack with the LiteRT GPU runtime and its models (app/build.gradle.kts, Packs.kt)
include(":litert_gpu")
