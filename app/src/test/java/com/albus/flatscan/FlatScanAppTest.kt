package com.albus.flatscan

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.work.WorkManager
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** WorkManager without its startup initializer: the first getInstance initialises it from [FlatScanApp]'s configuration. */
@RunWith(AndroidJUnit4::class)
class FlatScanAppTest {
    @Test
    fun workManagerStartsOnDemand() {
        val ctx: Context = ApplicationProvider.getApplicationContext()
        assertTrue(ctx is FlatScanApp)
        WorkManager.getInstance(ctx)
        assertTrue(WorkManager.isInitialized())
    }
}
