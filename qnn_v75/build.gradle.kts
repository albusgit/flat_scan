// On-demand asset pack (Play Asset Delivery): the HTP v75 skeleton (DSP side) and stub (CPU side) libraries, for one Hexagon generation.
// Its assets are copied in by :app:syncQnnPacks (app/build.gradle.kts) from the QNN AAR and app/src/main/assets/models;
// the app fetches it only on a Snapdragon phone that uses the NPU (QnnPacks.kt).
plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName.set("qnn_v75")
    dynamicDelivery {
        deliveryType.set("on-demand")
    }
}
