package com.albus.flatscan.batch

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import com.albus.flatscan.MainActivity
import com.albus.flatscan.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * The batch's foreground service: it keeps the process alive while [BatchProcessor] works (the app left or swiped
 * away) and shows how far it is in a notification, "3 of 12 photos" with a bar, the time left once a photo has
 * finished, and Stop. At the end that notification gives way to "12 pages ready", which opens the document.
 */
class BatchService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var started = false
    private var startedAt = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val batch = BatchProcessor.get(this)
        if (intent?.action == ACTION_STOP) {
            batch.stop()
            return START_NOT_STICKY
        }
        channels(this)
        if (!started) {
            started = true
            startedAt = SystemClock.elapsedRealtime()
            val type = when {
                Build.VERSION.SDK_INT >= 35 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
                Build.VERSION.SDK_INT >= 29 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                else -> 0
            }
            if (Build.VERSION.SDK_INT >= 29) startForeground(ID_PROGRESS, progress(batch.state.value), type)
            else startForeground(ID_PROGRESS, progress(batch.state.value))
            scope.launch {
                batch.state.collectLatest { s ->
                    if (s.running) {
                        notifySafely(ID_PROGRESS, progress(s))
                    } else {
                        ServiceCompat.stopForeground(this@BatchService, ServiceCompat.STOP_FOREGROUND_REMOVE)
                        if (s.total > 0) notifySafely(ID_DONE, done(s))
                        stopSelf()
                    }
                }
            }
        }
        return START_NOT_STICKY
    }

    /** Android 15's limit on a media-processing service (6 h): the batch stops where it is. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        BatchProcessor.get(this).stop()
        stopSelf()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun openApp(doc: String?): PendingIntent {
        val i = Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        if (doc != null) i.putExtra(MainActivity.EXTRA_OPEN_DOCUMENT, doc)
        return PendingIntent.getActivity(this, doc?.hashCode() ?: 0, i, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }

    private fun progress(s: BatchState): Notification {
        val stop = PendingIntent.getService(this, 1, Intent(this, BatchService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE)
        // the time left from the pace so far, once a photo has finished
        val elapsed = SystemClock.elapsedRealtime() - startedAt
        val doneShare = s.fraction
        val left = if (s.finished > 0 && doneShare > 0f) (elapsed * (1f - doneShare) / doneShare / 1000f).roundToInt() else -1
        val text = buildString {
            append("${(s.finished + 1).coerceAtMost(s.total)} of ${s.total} photos")
            if (s.failed > 0) append(" · ${s.failed} failed")
            if (left >= 0) append(" · ").append(if (left < 60) "less than a minute left" else "about ${(left + 30) / 60} min left")
        }
        return NotificationCompat.Builder(this, CHANNEL_PROGRESS)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("Processing pages" + if (s.title.isNotEmpty()) " · ${s.title}" else "")
            .setContentText(text)
            .setSubText(if (s.workers > 1) "${s.workers} at a time" else null)
            .setProgress(1000, (doneShare * 1000).roundToInt(), s.finished == 0 && doneShare == 0f)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setContentIntent(openApp(s.items.lastOrNull()?.doc))
            .addAction(0, "Stop", stop)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun done(s: BatchState): Notification {
        val pages = s.items.sumOf { it.pages.size }
        val title = when {
            pages == 0 -> "No pages were made"
            pages == 1 -> "1 page ready"
            else -> "$pages pages ready"
        }
        val text = buildString {
            if (s.title.isNotEmpty()) append(s.title)
            if (s.failed > 0) append(if (isEmpty()) "" else " · ").append(if (s.failed == 1) "1 photo could not be processed" else "${s.failed} photos could not be processed")
            if (s.stopped) append(if (isEmpty()) "" else " · ").append("stopped")
        }
        return NotificationCompat.Builder(this, CHANNEL_DONE)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setAutoCancel(true)
            .setContentIntent(openApp(s.items.lastOrNull { it.pages.isNotEmpty() }?.doc ?: s.items.lastOrNull()?.doc))
            .build()
    }

    /** Without the notification permission (Android 13+) the batch still runs; only nothing is shown. */
    private fun notifySafely(id: Int, n: Notification) {
        val nm = NotificationManagerCompat.from(this)
        if (Build.VERSION.SDK_INT >= 33 && !nm.areNotificationsEnabled()) return
        runCatching { nm.notify(id, n) }
    }

    companion object {
        private const val CHANNEL_PROGRESS = "batch"
        private const val CHANNEL_DONE = "batch_done"
        private const val ID_PROGRESS = 41
        private const val ID_DONE = 42
        private const val ACTION_STOP = "com.albus.flatscan.batch.STOP"

        private fun channels(ctx: Context) {
            val nm = ctx.getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(NotificationChannel(CHANNEL_PROGRESS, "Processing pages", NotificationManager.IMPORTANCE_LOW).apply {
                description = "How far a batch of photos is"
                setShowBadge(false)
            })
            nm.createNotificationChannel(NotificationChannel(CHANNEL_DONE, "Pages ready", NotificationManager.IMPORTANCE_DEFAULT).apply {
                description = "A batch of photos is done"
            })
        }
    }
}
