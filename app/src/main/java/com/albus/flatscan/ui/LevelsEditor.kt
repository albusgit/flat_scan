package com.albus.flatscan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

private const val MIN_GAP = 0.05f

/** Position of the midtone handle between black and white for [gamma] (out = in^(1/gamma)). */
private fun midPos(black: Float, white: Float, gamma: Float) = black + (white - black) * 0.5f.pow(gamma)

/**
 * Levels editor: the luminance [histogram] (256 bins) as a filled curve with draggable
 * black-point, midtone (gamma) and white-point handles under it.
 */
@Composable
fun LevelsEditor(
    histogram: IntArray?,
    black: Float,
    white: Float,
    gamma: Float,
    onChange: (black: Float, white: Float, gamma: Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val fill = MaterialTheme.colorScheme.onSurfaceVariant
    val track = MaterialTheme.colorScheme.outlineVariant
    val dim = MaterialTheme.colorScheme.surface
    val accent = MaterialTheme.colorScheme.primary
    val b by rememberUpdatedState(black)
    val w by rememberUpdatedState(white)
    val g by rememberUpdatedState(gamma)
    val change by rememberUpdatedState(onChange)
    var active by remember { mutableIntStateOf(-1) }     // 0 black, 1 midtone, 2 white
    // a robust top for the curve: the paper-white spike would flatten everything else
    val curve = remember(histogram) {
        histogram?.takeIf { it.size >= 2 }?.let { h ->
            val sorted = h.sortedDescending()
            val top = sqrt(sorted[minOf(3, sorted.size - 1)].coerceAtLeast(1).toFloat())
            FloatArray(h.size) { (sqrt(h[it].toFloat()) / top).coerceAtMost(1f) }
        }
    }
    Canvas(
        modifier.fillMaxWidth().height(132.dp)
            .semantics { contentDescription = "Levels: black point, midtones and white point" }
            .pointerInput(Unit) {
                val m = 14.dp.toPx()
                val reach = 40.dp.toPx()
                fun xOf(v: Float) = m + v * (size.width - 2 * m)
                fun vOf(x: Float) = ((x - m) / (size.width - 2 * m)).coerceIn(0f, 1f)
                detectDragGestures(
                    onDragStart = { pos ->
                        val xs = listOf(xOf(b), xOf(midPos(b, w, g)), xOf(w))
                        val d = xs.map { abs(it - pos.x) }
                        val i = d.indices.minBy { d[it] }
                        active = if (d[i] < reach) i else -1
                    },
                    onDragEnd = { active = -1 },
                    onDragCancel = { active = -1 },
                    onDrag = { ch, _ ->
                        if (active < 0) return@detectDragGestures
                        ch.consume()
                        val v = vOf(ch.position.x)
                        when (active) {
                            0 -> change(v.coerceIn(0f, minOf(0.5f, w - MIN_GAP)), w, g)
                            2 -> change(b, v.coerceIn(maxOf(0.5f, b + MIN_GAP), 1f), g)
                            1 -> {
                                val t = ((v - b) / (w - b)).coerceIn(0.5f.pow(3f), 0.5f.pow(0.3f))
                                change(b, w, (ln(t) / ln(0.5f)).coerceIn(0.3f, 3f))
                            }
                        }
                    },
                )
            },
    ) {
        val m = 14.dp.toPx()
        val handleH = 30.dp.toPx()
        val histH = size.height - handleH - 6.dp.toPx()
        val x0 = m
        val x1 = size.width - m
        fun xOf(v: Float) = x0 + v * (x1 - x0)
        // histogram
        drawRect(track.copy(alpha = 0.35f), Offset(x0, 0f), Size(x1 - x0, histH))
        if (curve != null) {
            val path = Path().apply {
                moveTo(x0, histH)
                val n = curve.size
                for (i in 0 until n) lineTo(xOf(i / (n - 1f)), histH - curve[i] * histH * 0.95f)
                lineTo(x1, histH)
                close()
            }
            drawPath(path, fill.copy(alpha = 0.55f))
        }
        // clipped ranges are dimmed
        drawRect(dim.copy(alpha = 0.6f), Offset(x0, 0f), Size(xOf(black) - x0, histH))
        drawRect(dim.copy(alpha = 0.6f), Offset(xOf(white), 0f), Size(x1 - xOf(white), histH))
        val mid = midPos(black, white, gamma)
        for (v in listOf(black, mid, white)) drawLine(accent.copy(alpha = 0.5f), Offset(xOf(v), 0f), Offset(xOf(v), histH), 1.dp.toPx())
        // handle track and handles
        val ty = histH + 6.dp.toPx()
        drawLine(track, Offset(x0, ty), Offset(x1, ty), 2.dp.toPx())
        handle(xOf(black), ty, Color(0xFF111111), accent, active == 0)
        handle(xOf(mid), ty, Color(0xFF8A8A8A), accent, active == 1)
        handle(xOf(white), ty, Color.White, accent, active == 2)
    }
}

/** An upward triangle with its tip on the track. */
private fun DrawScope.handle(x: Float, y: Float, color: Color, accent: Color, active: Boolean) {
    val hw = 11.dp.toPx()
    val hh = 20.dp.toPx()
    val p = Path().apply {
        moveTo(x, y)
        lineTo(x + hw, y + hh)
        lineTo(x - hw, y + hh)
        close()
    }
    drawPath(p, color)
    drawPath(p, if (active) accent else Color(0xFF6B6B6B), style = Stroke(if (active) 2.5.dp.toPx() else 1.dp.toPx()))
}
