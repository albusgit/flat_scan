package com.albus.flatscan

import android.app.Application
import androidx.work.Configuration

/**
 * The application: WorkManager is initialised on demand (its first [androidx.work.WorkManager.getInstance], from
 * text recognition or sync), not at every start by androidx.startup (the manifest removes its initializer).
 */
class FlatScanApp : Application(), Configuration.Provider {
    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder().build()
}
