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

/**
 * Line art for the camera card: a sheet with ruled lines under an iron, in [color] (the card's
 * content colour, faint).
 */
@Composable
fun IronOnSheet(modifier: Modifier = Modifier, color: Color) {
    Canvas(modifier) {
        val u = size.minDimension / 100f
        val line = Stroke(width = 3f * u, cap = StrokeCap.Round, join = StrokeJoin.Round)
        // the sheet, tilted a little, its first line wavy and the rest straight
        val sheet = Path().apply {
            moveTo(18 * u, 30 * u); lineTo(70 * u, 22 * u); lineTo(80 * u, 88 * u); lineTo(28 * u, 96 * u); close()
        }
        drawPath(sheet, color, style = line)
        val wave = Path().apply {
            moveTo(27 * u, 42 * u)
            cubicTo(33 * u, 34 * u, 37 * u, 46 * u, 43 * u, 38 * u)
            cubicTo(49 * u, 30 * u, 53 * u, 40 * u, 66 * u, 33 * u)
        }
        drawPath(wave, color, style = line)
        for (k in 0..2) {
            val y = 54f + 12f * k
            drawLine(color, Offset((30 + 1.5f * k) * u, (y + 0.0f) * u), Offset((70 + 1.5f * k) * u, (y - 6f) * u), strokeWidth = 3f * u, cap = StrokeCap.Round)
        }
        // the iron on top, pressing the lower right
        val iron = Path().apply {
            moveTo(46 * u, 80 * u); lineTo(98 * u, 72 * u); lineTo(90 * u, 58 * u)
            cubicTo(87 * u, 53 * u, 82 * u, 51 * u, 76 * u, 52 * u); lineTo(48 * u, 56 * u); close()
        }
        drawPath(iron, color.copy(alpha = color.alpha * 0.25f))
        drawPath(iron, color, style = line)
        val handle = Path().apply {
            moveTo(56 * u, 55 * u); cubicTo(55 * u, 44 * u, 60 * u, 41 * u, 67 * u, 40 * u); lineTo(80 * u, 38 * u); lineTo(81 * u, 51 * u)
        }
        drawPath(handle, color, style = line)
    }
}

/**
 * The empty home screen's three steps: a crumpled ball, the iron pressing a half-flat sheet, a
 * flat clean page, joined by dashed arrows.
 */
@Composable
fun HowItWorks(modifier: Modifier = Modifier, ink: Color, paper: Color, accent: Color, marker: Color) {
    Canvas(modifier) {
        val w = size.width / 3f
        val h = size.height
        val r = minOf(w, h) * 0.3f
        // 1: the crumpled photo
        drawCrumpledBall(Offset(w * 0.5f, h * 0.5f), r, 9, paper, ink)
        // 2: a sheet with a couple of creases, the iron on it
        val c2 = Offset(w * 1.5f, h * 0.52f)
        val sw = r * 1.5f
        val sh = r * 1.9f
        drawRect(paper, Offset(c2.x - sw / 2, c2.y - sh / 2), Size(sw, sh))
        drawRect(ink, Offset(c2.x - sw / 2, c2.y - sh / 2), Size(sw, sh), style = Stroke(r * 0.06f))
        drawLine(ink.copy(alpha = 0.35f), Offset(c2.x - sw / 2, c2.y - sh * 0.1f), Offset(c2.x + sw * 0.1f, c2.y + sh / 2), strokeWidth = r * 0.04f)
        drawLine(ink.copy(alpha = 0.35f), Offset(c2.x - sw * 0.2f, c2.y - sh / 2), Offset(c2.x + sw / 2, c2.y + sh * 0.05f), strokeWidth = r * 0.04f)
        drawIron(Offset(c2.x + sw * 0.1f, c2.y - sh * 0.05f), r / 34f)
        // 3: the flat page: ruled lines, the first highlighted
        val c3 = Offset(w * 2.5f, h * 0.5f)
        val tl = Offset(c3.x - sw / 2, c3.y - sh / 2)
        drawRect(paper, tl, Size(sw, sh))
        drawRect(ink, tl, Size(sw, sh), style = Stroke(r * 0.06f))
        drawRect(marker, Offset(tl.x + sw * 0.14f, tl.y + sh * 0.18f), Size(sw * 0.6f, sh * 0.1f))
        for (k in 0..3) {
            val y = tl.y + sh * (0.24f + 0.17f * k)
            drawLine(accent, Offset(tl.x + sw * 0.16f, y), Offset(tl.x + sw * (if (k == 3) 0.6f else 0.84f), y), strokeWidth = r * 0.07f, cap = StrokeCap.Round)
        }
        // dashed arrows between the steps
        val dash = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(r * 0.12f, r * 0.1f))
        for (k in 1..2) {
            val x0 = w * k - w * 0.14f
            val x1 = w * k + w * 0.1f
            val y = h * 0.5f
            drawLine(ink.copy(alpha = 0.55f), Offset(x0, y), Offset(x1, y), strokeWidth = r * 0.05f, pathEffect = dash)
            drawLine(ink.copy(alpha = 0.55f), Offset(x1, y), Offset(x1 - r * 0.14f, y - r * 0.12f), strokeWidth = r * 0.05f, cap = StrokeCap.Round)
            drawLine(ink.copy(alpha = 0.55f), Offset(x1, y), Offset(x1 - r * 0.14f, y + r * 0.12f), strokeWidth = r * 0.05f, cap = StrokeCap.Round)
        }
    }
}
