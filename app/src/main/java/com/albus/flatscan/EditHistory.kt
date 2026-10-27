package com.albus.flatscan

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest

/** What undo / redo on the result screen restores: every edit that does not re-run the geometry. */
data class EditState(
    val params: RenderParams,
    val look: PageLook,
    val rotation: Int,
    val protect: List<Float>,
    val erase: List<Float>,
    val fingersAuto: Boolean = true,
    val fingerStrokes: List<Float> = emptyList(),
    val fingerFill: Int = 0,
)

/**
 * Undo / redo of one page's edits. [record] is given every settled state (after a debounce, see
 * ScanViewModel.trackEdits); a state equal to the current one is ignored, so restoring a state does
 * not record it again. [reset] starts a new page's history.
 */
class EditHistory(private val limit: Int = 60) {
    private val past = ArrayDeque<EditState>()
    private val future = ArrayDeque<EditState>()
    private var current: EditState? = null
    // observable sizes for the buttons
    private var version by mutableIntStateOf(0)

    val canUndo: Boolean get() { version; return past.isNotEmpty() }
    val canRedo: Boolean get() { version; return future.isNotEmpty() }

    fun reset(state: EditState?) {
        past.clear()
        future.clear()
        current = state
        version++
    }

    fun record(state: EditState) {
        val c = current
        if (c == state) return
        if (c != null) {
            past.addLast(c)
            while (past.size > limit) past.removeFirst()
        }
        future.clear()
        current = state
        version++
    }

    /** The state to go back to (it becomes the current one), or null. */
    fun undo(): EditState? {
        val prev = past.removeLastOrNull() ?: return null
        current?.let { future.addLast(it) }
        current = prev
        version++
        return prev
    }

    fun redo(): EditState? {
        val next = future.removeLastOrNull() ?: return null
        current?.let { past.addLast(it) }
        current = next
        version++
        return next
    }
}

/**
 * Feeds [history] the current page's settled edits: a new page (another [ScanViewModel.currentId])
 * starts a new history; a change is recorded once it has not changed for 450 ms (a slider drag is
 * one step).
 */
suspend fun trackEdits(vm: ScanViewModel, history: EditHistory) {
    var page: String? = null
    snapshotFlow { vm.currentId to vm.editState() }.collectLatest { (id, s) ->
        if (id != page) {
            page = id
            history.reset(if (id != null) s else null)
            return@collectLatest
        }
        if (id == null) return@collectLatest
        delay(450)
        history.record(s)
    }
}

/** Look choices remembered for new pages (their own preferences file, beside Settings). */
object LookPrefs {
    private fun prefs(ctx: Context) = ctx.getSharedPreferences("look", Context.MODE_PRIVATE)

    /** A new page's look: the defaults with the page shape chosen last. */
    fun newPageLook(ctx: Context): PageLook {
        val name = prefs(ctx).getString("shape", null)
        return PageLook(shape = PageShape.entries.firstOrNull { it.name == name } ?: PageShape.Auto)
    }

    fun rememberShape(ctx: Context, shape: PageShape) {
        prefs(ctx).edit().putString("shape", shape.name).apply()
    }
}
