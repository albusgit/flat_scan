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
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.drawscope.rotate
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
fun DrawScope.drawIron(at: Offset, s: Float, alpha: Float = 1f, flip: Boolean = false, soleplate: Color = Palette.Accent, outline: Float = 2.4f) {
    translate(at.x, at.y) {
        scale(if (flip) -s else s, s, pivot = Offset.Zero) {
            val ink = Palette.Ink.copy(alpha = alpha)
            val line = Stroke(width = outline, cap = StrokeCap.Round, join = StrokeJoin.Round)
            drawOval(Color(0xFF141928).copy(alpha = 0.22f * alpha), topLeft = Offset(-28f, 8f), size = Size(60f, 10f))
            val body = Path().apply {
                moveTo(-28f, 10f); lineTo(28f, 10f); lineTo(18f, -6f)
                cubicTo(14f, -12f, 9f, -14f, 2f, -14f); lineTo(-28f, -14f); close()
            }
            drawPath(body, Color.White.copy(alpha = alpha))
            drawPath(body, ink, style = line)
            drawRect(soleplate.copy(alpha = alpha), topLeft = Offset(-28f, 4f), size = Size(56f, 6f))
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
 * The first-launch picture: the iron crossing a ruled page, the lines behind it straight and pressed flat, the
 * lines ahead of it still wavy and creased, steam rising behind it and its cord off to the side. Drawn in a
 * 350 x 312 frame scaled to the width; [accent] is the iron's soleplate, [pencil] the steam and the cord,
 * [shadow] the page's and the iron's shadow on the desk.
 */
@Composable
fun IronHero(modifier: Modifier, accent: Color, pencil: Color, shadow: Color) {
    Canvas(modifier) {
        val k = size.width / 350f
        fun o(x: Float, y: Float) = Offset(x * k, y * k)
        val rule = Color(0xFF9FB5F2).copy(alpha = 0.6f)
        val text = Color(0xFF3B3F4A)
        // a wave across the part ahead of the iron: [a] up and down over three humps from x0 to 340
        fun wave(x0: Float, y: Float, a: Float, x1: Float = 340f) = Path().apply {
            val step = (x1 - x0) / 3f
            moveTo(x0 * k, y * k)
            for (i in 0 until 3) {
                val sx = x0 + step * i
                quadraticTo((sx + step / 2) * k, (y + if (i % 2 == 0) -a else a) * k, (sx + step) * k, y * k)
            }
        }
        rotate(-2.5f, pivot = o(175f, 156f)) {
            drawRect(shadow, topLeft = o(26f, 18f), size = Size(304f * k, 286f * k))
            drawRect(Color(0xFFFBFBF8), topLeft = o(18f, 10f), size = Size(304f * k, 286f * k))
            // behind the iron: flat
            clipRect(right = 172f * k) {
                for (i in 0 until 10) { val y = 50f + 24f * i; drawLine(rule, o(18f, y), o(322f, y), strokeWidth = k) }
                drawLine(Palette.RedPen.copy(alpha = 0.5f), o(52f, 10f), o(52f, 296f), strokeWidth = k)
                for ((y, w) in listOf(40f to 130f, 64f to 120f, 88f to 130f, 112f to 84f, 160f to 130f, 184f to 126f, 208f to 130f, 232f to 100f)) {
                    drawRoundRect(text, topLeft = o(62f, y), size = Size(w * k, 5f * k), cornerRadius = androidx.compose.ui.geometry.CornerRadius(2.5f * k))
                }
            }
            // ahead of it: still creased
            clipRect(left = 172f * k) {
                drawRect(Color(0xFFECEDEF), topLeft = o(172f, 10f), size = Size(150f * k, 286f * k))
                for (i in 0 until 10) drawPath(wave(160f, 50f + 24f * i, if (i % 2 == 0) 7f else -7f), rule, style = Stroke(width = k))
                for ((n, y) in listOf(41f, 65f, 89f, 161f, 185f, 209f, 233f).withIndex()) {
                    drawPath(wave(172f, y, if (n % 2 == 0) 6f else -6f, if (n % 3 == 2) 270f else 306f), text.copy(alpha = 0.8f), style = Stroke(width = 5f * k, cap = StrokeCap.Round))
                }
                val crease = Color.Black.copy(alpha = 0.07f)
                drawLine(crease, o(214f, 16f), o(252f, 292f), strokeWidth = 12f * k, cap = StrokeCap.Round)
                drawLine(crease, o(282f, 20f), o(262f, 290f), strokeWidth = 12f * k, cap = StrokeCap.Round)
                drawLine(crease, o(176f, 120f), o(322f, 176f), strokeWidth = 12f * k, cap = StrokeCap.Round)
            }
        }
        // steam behind the iron
        for ((c, r) in listOf(o(104f, 96f) to 8f, o(84f, 74f) to 11f, o(60f, 58f) to 7f, o(118f, 70f) to 6f)) {
            drawCircle(Color.White.copy(alpha = 0.10f), r * k, c)
            drawCircle(pencil, r * k, c, style = Stroke(width = 1.5f * k))
        }
        drawOval(shadow.copy(alpha = shadow.alpha * 0.65f), topLeft = o(114f, 187f), size = Size(172f * k, 26f * k))
        drawIron(o(190f, 160f), 2.9f * k, soleplate = accent, outline = 1.1f)
        // the cord, off the top
        drawPath(
            Path().apply { moveTo(220f * k, 78f * k); cubicTo(250f * k, 40f * k, 300f * k, 50f * k, 350f * k, 10f * k) },
            pencil, style = Stroke(width = 2f * k, cap = StrokeCap.Round),
        )
    }
}
