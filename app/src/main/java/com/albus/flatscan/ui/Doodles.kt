package com.albus.flatscan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * The iron of the Flatten tool, drawn [s] times its 60-unit size with the middle of its soleplate
 * at [at] (facing right, or left with [flip]): white body, ink outline, ballpoint-blue soleplate, a highlighter dial and a soft shadow.
 */
fun DrawScope.drawIron(at: Offset, s: Float, alpha: Float = 1f, flip: Boolean = false) {
    translate(at.x, at.y) {
        scale(if (flip) -s else s, s, pivot = Offset.Zero) {
            val ink = Palette.Ink.copy(alpha = alpha)
            val line = Stroke(width = 2.4f, cap = StrokeCap.Round, join = StrokeJoin.Round)
            drawOval(Color(0xFF141928).copy(alpha = 0.22f * alpha), topLeft = Offset(-28f, 8f), size = Size(60f, 10f))
            val body = Path().apply {
                moveTo(-28f, 10f); lineTo(28f, 10f); lineTo(18f, -6f)
                cubicTo(14f, -12f, 9f, -14f, 2f, -14f); lineTo(-28f, -14f); close()
            }
            drawPath(body, Color.White.copy(alpha = alpha))
            drawPath(body, ink, style = line)
            drawRect(Palette.Accent.copy(alpha = alpha), topLeft = Offset(-28f, 4f), size = Size(56f, 6f))
            drawRect(ink, topLeft = Offset(-28f, 4f), size = Size(56f, 6f), style = line)
            val handle = Path().apply {
                moveTo(-18f, -14f); cubicTo(-18f, -26f, -12f, -28f, -4f, -28f); lineTo(10f, -28f); lineTo(10f, -14f)
            }
            drawPath(handle, ink, style = line)
            drawCircle(Palette.Highlighter.copy(alpha = alpha), radius = 3f, center = Offset(-8f, -4f))
        }
    }
}

/**
 * A crumpled paper ball of radius [r] at [center]: an uneven outline, faceted shade on its
 * lower right and a few creases. [seed] picks the crumple.
 */
fun DrawScope.drawCrumpledBall(center: Offset, r: Float, seed: Int, paper: Color, ink: Color) {
    val rnd = Random(seed)
    val n = 12
    val pts = List(n) { i ->
        val a = i * 2 * PI / n + (rnd.nextFloat() - 0.5f) * 0.35
        val rr = r * (0.82f + rnd.nextFloat() * 0.22f)
        center + Offset((cos(a) * rr).toFloat(), (sin(a) * rr).toFloat())
    }
    val inner = List(5) { center + Offset((rnd.nextFloat() - 0.5f) * r * 0.9f, (rnd.nextFloat() - 0.5f) * r * 0.9f) }
    val outline = Path().apply { moveTo(pts[0].x, pts[0].y); pts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
    drawPath(outline, paper)
    // facets: triangles between the rim and the inner points, shaded by which way they face
    for (i in 0 until n) {
        val a = pts[i]
        val b = pts[(i + 1) % n]
        val c = inner[i % inner.size]
        val mid = (a + b + c) / 3f - center
        val facing = (mid.x * 0.6f + mid.y * 0.8f) / r       // towards the lower right: away from the light
        val k = (facing * 0.5f + (rnd.nextFloat() - 0.5f) * 0.25f).coerceIn(-0.3f, 0.45f)
        val tri = Path().apply { moveTo(a.x, a.y); lineTo(b.x, b.y); lineTo(c.x, c.y); close() }
        if (k > 0f) drawPath(tri, ink.copy(alpha = k * 0.35f)) else drawPath(tri, Color.White.copy(alpha = -k * 0.6f))
    }
    val crease = r * 0.045f
    val creaseInk = ink.copy(alpha = 0.35f)
    for (i in 0 until n step 2) drawLine(creaseInk, pts[i], inner[(i / 2) % inner.size], strokeWidth = crease, cap = StrokeCap.Round)
    drawLine(creaseInk, inner[0], inner[1], strokeWidth = crease, cap = StrokeCap.Round)
    drawLine(creaseInk, inner[2], inner[3], strokeWidth = crease, cap = StrokeCap.Round)
    drawPath(outline, ink, style = Stroke(width = r * 0.07f, join = StrokeJoin.Round))
}

/** A crumpled paper ball filling the smaller side of [modifier]'s size. */
@Composable
fun CrumpledBall(modifier: Modifier = Modifier, seed: Int = 7, paper: Color = Color.White, ink: Color = Palette.Ink) {
    Canvas(modifier) { drawCrumpledBall(center, size.minDimension / 2f * 0.92f, seed, paper, ink) }
}
