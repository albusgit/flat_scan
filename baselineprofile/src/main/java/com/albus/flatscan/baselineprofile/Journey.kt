package com.albus.flatscan.baselineprofile

import androidx.benchmark.macro.MacrobenchmarkScope
import androidx.test.uiautomator.By
import androidx.test.uiautomator.BySelector
import androidx.test.uiautomator.Until

const val PACKAGE = "com.albus.flatscan"

private fun MacrobenchmarkScope.tap(sel: BySelector, timeoutMs: Long = 20_000) {
    val o = device.wait(Until.findObject(sel), timeoutMs) ?: error("not on screen: $sel")
    o.click()
}

/**
 * Home -> camera -> photo -> crop -> flattening -> result (its tool tabs and the save sheet) -> home -> Settings:
 * the code a first scan runs. The camera permission is granted first (a new install would ask).
 */
fun MacrobenchmarkScope.firstScan() {
    device.executeShellCommand("pm grant $PACKAGE android.permission.CAMERA")
    pressHome()
    startActivityAndWait()
    tap(By.text("Scan"))
    // a tap before CameraX is ready never completes (the emulator retries its camera check for about 6 s: the
    // capture stays pending and the camera screen open)
    tap(By.desc("Take photo").also { device.wait(Until.hasObject(it), 20_000); Thread.sleep(10_000) })
    // the crop screen, or straight on to the flattening when the crop is trusted (a new install's first photo
    // runs uncompiled: up to a minute or two on the emulator)
    // "Flatten page" (its pill) is disabled while the detection runs
    val crop = By.clickable(true).enabled(true).hasDescendant(By.text("Flatten page"))
    val flattening = By.desc("The page being flattened")
    val result = By.desc("Compare with the original")
    var waited = 0L
    while (!device.hasObject(crop) && !device.hasObject(flattening) && !device.hasObject(result) && waited < 120_000) {
        Thread.sleep(500)
        waited += 500
    }
    if (device.hasObject(crop)) tap(crop)
    // the flattening: about 25 s on the emulator's CPU
    device.wait(Until.hasObject(result), 180_000) || error("no result screen")
    for (t in listOf("Adjust", "Look", "Clean up")) tap(By.text(t))
    tap(By.text("Save"))
    device.wait(Until.hasObject(By.text("Save to device")), 10_000)
    device.pressBack()   // the save sheet
    device.waitForIdle()
    device.pressBack()   // home
    tap(By.desc("Settings"))
    device.waitForIdle()
    device.pressBack()
}
