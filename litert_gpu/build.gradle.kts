// On-demand asset pack (Play Asset Delivery): the LiteRT (TensorFlow Lite) runtime with its GPU delegate and the models the GPU runs.
// Its assets are copied in by :app:syncLitertPack (app/build.gradle.kts) from the LiteRT AARs and app/src/main/litert/models;
// the app fetches it on any phone when Settings > Acceleration is GPU or Auto (LiteRtPacks in Packs.kt).
plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName.set("litert_gpu")
    dynamicDelivery {
        deliveryType.set("on-demand")
    }
}
