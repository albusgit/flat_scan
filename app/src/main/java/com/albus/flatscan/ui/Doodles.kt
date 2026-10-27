package com.albus.flatscan.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.pow
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
 * The first-launch picture: an iron crossing a ruled page. Behind it the page lies flat and crisp (straight rules, a
 * highlighted heading, a pressed sheen, a little steam rising off it); ahead of it the paper is still creased: its
 * edge waves, a corner is folded up, soft folds shade it and the rules and text ripple, more the further from the
 * iron. Drawn in a 350 x 312 frame scaled to the width; [accent] is the soleplate, [pencil] the steam and the cord,
 * [shadow] the shadows on the desk.
 */
@Composable
fun IronHero(modifier: Modifier, accent: Color, pencil: Color, shadow: Color) {
    Canvas(modifier) {
        if (size.width < 1f || size.height < 1f) return@Canvas
        val seam = 182f                                     // how far the iron has got
        // how far a point of a line ahead of the iron is out of line (none behind it)
        fun ripple(x: Float, phase: Float): Float {
            if (x <= seam) return 0f
            val t = ((x - seam) / 150f).coerceIn(0f, 1f)
            return 7f * t.pow(0.8f) * sin((x - seam) * 0.075f + phase)
        }
        fun line(x0: Float, x1: Float, y: Float, phase: Float) = Path().apply {
            var x = x0
            moveTo(x, y + ripple(x, phase))
            while (x < x1) { x = min(x1, x + 4f); lineTo(x, y + ripple(x, phase)) }
        }
        // the creased part ahead of the iron (its edge waves, its top corner lifts), and the whole sheet
        fun Path.creasedEdge() {
            quadraticTo(205f, 15f, 232f, 19f); quadraticTo(262f, 24f, 290f, 17f); quadraticTo(312f, 11f, 328f, 6f)
            quadraticTo(322f, 40f, 333f, 72f); quadraticTo(338f, 110f, 322f, 150f); quadraticTo(312f, 190f, 328f, 228f)
            quadraticTo(337f, 262f, 320f, 292f)
            quadraticTo(290f, 300f, 258f, 293f); quadraticTo(222f, 286f, seam, 292f)
        }
        val ahead = Path().apply { moveTo(seam, 22f); creasedEdge(); close() }
        val sheet = Path().apply { moveTo(24f, 22f); lineTo(seam, 22f); creasedEdge(); lineTo(24f, 292f); close() }
        val ink = Color(0xFF3B3F4A)
        val rule = Color(0xFF9FB5F2).copy(alpha = 0.65f)
        scale(size.width / 350f, pivot = Offset.Zero) {
            rotate(-3f, pivot = Offset(175f, 156f)) {
                // the page's shadow on the desk, soft in three steps
                for ((d, f) in listOf(3f to 0.5f, 6f to 0.3f, 10f to 0.15f)) translate(d * 0.6f, d) { drawPath(sheet, shadow.copy(alpha = shadow.alpha * f)) }
                drawPath(sheet, Color(0xFFFCFCFA))
                // ahead of the iron: the paper still greyed by its creases, folds shading it (the shapes filled with the
                // gradients: a gradient inside a path clip crashed the emulator's renderer)
                drawPath(ahead, Brush.horizontalGradient(0f to Color(0x00F0F1F4), 1f to Color(0xFFF0F1F4), startX = seam, endX = seam + 22f))
                drawPath(
                    ahead,
                    Brush.horizontalGradient(
                        0f to Color.Transparent, 0.1f to Color.Black.copy(alpha = 0.07f), 0.22f to Color.White.copy(alpha = 0.55f),
                        0.38f to Color.Black.copy(alpha = 0.1f), 0.5f to Color.White.copy(alpha = 0.4f), 0.66f to Color.Black.copy(alpha = 0.09f),
                        0.8f to Color.White.copy(alpha = 0.45f), 0.92f to Color.Black.copy(alpha = 0.06f), 1f to Color.Transparent,
                        startX = seam, endX = 340f,
                    ),
                )
                // behind it: pressed, a sheen fading off behind the iron (the flat part is a plain rectangle)
                drawRect(
                    Brush.horizontalGradient(0f to Color.Transparent, 1f to Color.White.copy(alpha = 0.6f), startX = 90f, endX = seam),
                    topLeft = Offset(90f, 22f), size = Size(seam - 90f, 270f),
                )
                clipPath(sheet) {
                    for ((a, b) in listOf(Offset(232f, 19f) to Offset(250f, 293f), Offset(290f, 17f) to Offset(272f, 293f))) {
                        drawLine(Color.Black.copy(alpha = 0.08f), a, b, strokeWidth = 1.2f)
                        drawLine(Color.White.copy(alpha = 0.75f), a + Offset(1.5f, 0f), b + Offset(1.5f, 0f), strokeWidth = 1.2f)
                    }
                    drawRect(Palette.Highlighter.copy(alpha = 0.85f), topLeft = Offset(58f, 38f), size = Size(98f, 12f))
                    for (i in 0 until 11) drawPath(line(24f, 336f, 56f + 22f * i, i * 0.5f), rule, style = Stroke(width = 1f))
                    drawLine(Palette.RedPen.copy(alpha = 0.5f), Offset(52f, 0f), Offset(52f, 320f), strokeWidth = 1f)
                    val text = Stroke(width = 3.6f, cap = StrokeCap.Round)
                    drawPath(line(64f, 148f, 44f, 0f), ink, style = text)
                    for ((i, end) in listOf(1 to 300f, 2 to 290f, 3 to 214f, 5 to 306f, 6 to 296f, 7 to 250f, 9 to 300f, 10 to 186f)) {
                        drawPath(line(64f, end, 47f + 22f * i, i * 0.5f), ink.copy(alpha = 0.8f), style = text)
                    }
                }
                // the corner ahead, folded up
                val fold = Path().apply { moveTo(298f, 15f); lineTo(328f, 6f); lineTo(321f, 38f); close() }
                translate(2f, 3f) { drawPath(fold, Color.Black.copy(alpha = 0.08f)) }
                drawPath(fold, Color(0xFFDADDE3))
                drawPath(fold, Color.Black.copy(alpha = 0.12f), style = Stroke(width = 0.8f))
            }
            // steam rising off the paper just pressed, behind the iron
            for ((x, y) in listOf(90f to 186f, 108f to 180f)) {
                val wisp = Path().apply { moveTo(x, y); cubicTo(x - 12f, y - 14f, x + 10f, y - 28f, x - 2f, y - 42f); cubicTo(x - 10f, y - 52f, x + 4f, y - 60f, x - 4f, y - 70f) }
                drawPath(wisp, Brush.verticalGradient(listOf(pencil.copy(alpha = 0f), pencil.copy(alpha = 0.6f)), startY = y - 70f, endY = y), style = Stroke(width = 3.4f, cap = StrokeCap.Round))
            }
            // the iron's shadow, its cord trailing from the heel off the page, the iron
            drawOval(
                Brush.radialGradient(listOf(shadow.copy(alpha = shadow.alpha * 0.9f), Color.Transparent), center = Offset(200f, 196f), radius = 80f),
                topLeft = Offset(118f, 184f), size = Size(168f, 24f),
            )
            drawPath(
                Path().apply { moveTo(120f, 166f); cubicTo(96f, 160f, 84f, 206f, 52f, 214f); cubicTo(24f, 220f, -4f, 250f, -60f, 262f) },
                pencil, style = Stroke(width = 2.8f, cap = StrokeCap.Round),
            )
            translate(196f, 168f) { scale(2.6f, pivot = Offset.Zero) { heroIron(accent) } }
        }
    }
}

/** The first-launch picture's iron, 66 units long with its soleplate along y = 8..13 and its nose at +x. */
private fun DrawScope.heroIron(accent: Color) {
    val ink = Palette.Ink
    val body = Path().apply {
        moveTo(-30f, 8f); lineTo(-30f, -6f); quadraticTo(-30f, -12f, -24f, -12f); lineTo(4f, -12f); cubicTo(18f, -12f, 30f, -2f, 35f, 8f); close()
    }
    val sole = Path().apply {
        moveTo(-31f, 8f); lineTo(35f, 8f); quadraticTo(37.5f, 11f, 33f, 13f); lineTo(-29f, 13f); quadraticTo(-31f, 13f, -31f, 11f); close()
    }
    // the handle, behind the body
    val handle = Path().apply { moveTo(-20f, -11f); cubicTo(-20f, -24f, -14f, -25f, -6f, -25f); lineTo(10f, -25f); quadraticTo(14f, -25f, 14f, -20f); lineTo(14f, -11f) }
    drawPath(handle, ink, style = Stroke(width = 3.2f, cap = StrokeCap.Round, join = StrokeJoin.Round))
    translate(0f, -0.6f) { drawPath(handle, Color(0xFF4A5165), style = Stroke(width = 1f, cap = StrokeCap.Round, join = StrokeJoin.Round)) }
    drawPath(body, Brush.verticalGradient(listOf(Color.White, Color(0xFFE3E7EE)), startY = -12f, endY = 8f))
    drawPath(Path().apply { moveTo(-25f, -9f); lineTo(3f, -9f) }, Color.White, style = Stroke(width = 1.4f, cap = StrokeCap.Round))
    drawPath(sole, accent)
    drawLine(Color.White.copy(alpha = 0.35f), Offset(-28f, 9.6f), Offset(31f, 9.6f), strokeWidth = 0.8f)
    val edge = Stroke(width = 0.8f, join = StrokeJoin.Round)
    drawPath(body, ink, style = edge)
    drawPath(sole, ink, style = edge)
    drawCircle(Palette.Highlighter, radius = 3.6f, center = Offset(-10f, -2.5f))
    drawCircle(ink, radius = 3.6f, center = Offset(-10f, -2.5f), style = Stroke(width = 0.7f))
    drawLine(ink, Offset(-10f, -2.5f), Offset(-10f, -5.4f), strokeWidth = 0.7f, cap = StrokeCap.Round)
}
