package com.albus.flatscan

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.util.Log
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Native work that can take the whole process down (the LiteRT GPU delegate on a GPU it was never run on, a
 * driver bug) cannot be caught. Instead the work is bracketed: [begin] writes what is about to run to disk
 * (synchronously), [end] clears it. A marker found at the next start ([check]) means the process died inside
 * that work. Then:
 *  * if the GPU took part, the GPU is turned off for this phone ([gpuDisabled]): Auto and the GPU setting use
 *    the CPU, so the same crash cannot repeat at every start;
 *  * a report (what ran, Android's exit reason for the last process, a Java stack trace if there was one) is
 *    kept for the self-test area and the start message ([lastReport]).
 */
object CrashGuard {
    private const val TAG = "FlatScan"
    private const val PREFS = "crash_guard"

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /**
     * [what]: shown in the report; [gpu]: the GPU delegate may run inside. Returns the marker it replaces
     * (guarded work inside guarded work: the self-test runs Auto's benchmark), for [end] to put back.
     */
    fun begin(ctx: Context, what: String, gpu: Boolean): Pair<String, Boolean>? {
        val p = prefs(ctx)
        val outer = p.getString("running", null)?.let { it to p.getBoolean("running_gpu", false) }
        p.edit().putString("running", what).putBoolean("running_gpu", gpu || outer?.second == true).commit()
        return outer
    }

    fun end(ctx: Context, outer: Pair<String, Boolean>? = null) {
        val e = prefs(ctx).edit()
        if (outer != null) e.putString("running", outer.first).putBoolean("running_gpu", outer.second)
        else e.remove("running").remove("running_gpu")
        e.commit()
    }

    /** Runs [block] between [begin] and [end] (also ended when it throws: a Java exception is not a crash). */
    inline fun <T> guard(ctx: Context, what: String, gpu: Boolean, block: () -> T): T {
        val outer = begin(ctx, what, gpu)
        try {
            return block()
        } finally {
            end(ctx, outer)
        }
    }

    /** The GPU crashed the app on this phone: the nets stay on the CPU. */
    fun gpuDisabled(ctx: Context): Boolean = prefs(ctx).getBoolean("gpu_disabled", false)

    fun enableGpu(ctx: Context) {
        prefs(ctx).edit().putBoolean("gpu_disabled", false).apply()
    }

    /** The report of the last crash inside guarded work, or "" (cleared by [clearReport]). */
    fun lastReport(ctx: Context): String = prefs(ctx).getString("report", "") ?: ""

    fun clearReport(ctx: Context) {
        prefs(ctx).edit().remove("report").apply()
    }

    /** Java crashes anywhere: their stack trace goes into the next report. Call once per process. */
    private var installed = false

    fun installHandler(ctx: Context) {
        if (installed) return
        installed = true
        val app = ctx.applicationContext
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            runCatching {
                prefs(app).edit().putString("java_trace", "${t.name}: ${Log.getStackTraceString(e)}".take(6000)).commit()
            }
            previous?.uncaughtException(t, e)
        }
    }

    /**
     * At start: turns a marker left by a dead process into a report (and turns the GPU off if it took part).
     * Returns the new report, or null when the last process did not die in guarded work.
     */
    fun check(ctx: Context): String? {
        val p = prefs(ctx)
        val what = p.getString("running", null) ?: return null.also { p.edit().remove("java_trace").apply() }
        val gpu = p.getBoolean("running_gpu", false)
        val exit = exitInfo(ctx)
        p.edit().remove("running").remove("running_gpu").commit()
        // stopped by the user (swiped away, force stop) while it ran: not a crash
        if (exit != null && exit.first in USER_STOPS) return null
        // the system freed memory while the app was in the background: not the GPU's fault
        val gpuFault = gpu && (exit == null || exit.first !in MEMORY_KILLS)
        val sb = StringBuilder()
        sb.append("FlatScan closed unexpectedly during: ").append(what).append('\n')
        if (gpuFault) sb.append("The GPU was in use, so the nets now run on the CPU on this phone (Settings > Acceleration: choose GPU, or run the acceleration benchmark, to try the GPU again).\n")
        sb.append("Device: ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}), ${NetEnv.soc}\n")
        exit?.let { sb.append(it.second).append('\n') }
        p.getString("java_trace", null)?.let { sb.append("Java exception:\n").append(it).append('\n') }
        val report = sb.toString()
        Log.e(TAG, report)
        p.edit().remove("java_trace").putString("report", report).apply { if (gpuFault) putBoolean("gpu_disabled", true) }.commit()
        return report
    }

    private val USER_STOPS = setOf(ApplicationExitInfo.REASON_USER_REQUESTED, ApplicationExitInfo.REASON_USER_STOPPED)
    private val MEMORY_KILLS = setOf(ApplicationExitInfo.REASON_LOW_MEMORY)

    /** Android's record of how the last process ended (API 30+). */
    private fun exitInfo(ctx: Context): Pair<Int, String>? {
        if (Build.VERSION.SDK_INT < 30) return null
        return runCatching {
            val am = ctx.getSystemService(ActivityManager::class.java)
            val e = am.getHistoricalProcessExitReasons(ctx.packageName, 0, 1).firstOrNull() ?: return null
            val reason = when (e.reason) {
                ApplicationExitInfo.REASON_CRASH_NATIVE -> "native crash"
                ApplicationExitInfo.REASON_CRASH -> "Java crash"
                ApplicationExitInfo.REASON_LOW_MEMORY -> "out of memory (killed by the system)"
                ApplicationExitInfo.REASON_ANR -> "not responding (ANR)"
                ApplicationExitInfo.REASON_SIGNALED -> "killed by signal ${e.status}"
                ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "excessive resource use"
                ApplicationExitInfo.REASON_USER_REQUESTED, ApplicationExitInfo.REASON_USER_STOPPED -> "stopped by the user"
                else -> "reason ${e.reason}"
            }
            val at = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.ROOT).format(Date(e.timestamp))
            // on screen or not: the low-memory killer takes background apps first
            val where = when {
                e.importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND -> "on screen"
                e.importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_PERCEPTIBLE -> "visible, not in front (importance ${e.importance})"
                else -> "in the background (importance ${e.importance})"
            }
            e.reason to ("Exit: $reason at $at, the app $where" + (e.description?.takeIf { it.isNotEmpty() }?.let { " ($it)" } ?: "") +
                ", memory at exit: " + (if (e.pss > 0) "PSS ${e.pss / 1024} MB, " else "") + "RSS ${e.rss / 1024} MB")
        }.getOrNull()
    }
}
