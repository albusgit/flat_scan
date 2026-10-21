// On-demand asset pack (Play Asset Delivery): the QNN HTP libraries every Snapdragon needs and the fixed-shape models the NPU runs.
// Its assets are copied in by :app:syncQnnPacks (app/build.gradle.kts) from the QNN AAR and app/src/main/assets/models;
// the app fetches it only on a Snapdragon phone that uses the NPU (QnnPacks.kt).
plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName.set("qnn_htp")
    dynamicDelivery {
        deliveryType.set("on-demand")
    }
}
