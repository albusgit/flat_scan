package com.albus.flatscan.ui

import android.graphics.Bitmap
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

private const val GX = 12
private const val GY = 16
private const val TEX_MAX = 640

/** The core's progress steps (Screens: stage names); the animation flows towards the next one between reports. */
private val STEPS = floatArrayOf(0.02f, 0.05f, 0.08f, 0.14f, 0.2f, 0.3f, 0.85f, 0.9f, 0.92f, 0.95f, 1f)

/**
 * The processing animation, drawn every frame while the page is processed: [photo] (the page cut
 * from it with [corners], photo pixels of a [photoWidth]-wide photo) drops in as a crumpled ball
 * that tumbles and breathes, unfolds as [progress] rises (between the core's reports it keeps
 * flowing towards the next step; in any case it is open within ~3 s), and from then on an iron glides over the
 * sheet row by row, steam behind it, pressing out the creases it passes. At the end the sheet is
 * pressed flat and whitens. With system animations off, a still crumpled frame cross-fades to the
 * flat page.
 */
@Composable
fun PaperUnfold(
    photo: Bitmap?,
    corners: List<Offset>,
    photoWidth: Int,
    progress: Float,
    modifier: Modifier = Modifier,
) {
    val reduced = rememberReducedMotion()
    val texture by produceState<Bitmap?>(null, photo) {
        value = withContext(Dispatchers.Default) { runCatching { pageTexture(photo, corners, photoWidth) }.getOrNull() }
    }
    val tex = texture
    val sim = remember(tex) { tex?.let { Unfold(seedOf(it)) } }
    val target by rememberUpdatedState(progress)
    var frame by remember { mutableLongStateOf(0L) }
    LaunchedEffect(sim, reduced) {
        val s = sim ?: return@LaunchedEffect
        var last = -1L
        while (true) {
            withFrameNanos { now ->
                val dt = if (last < 0) 0f else ((now - last) / 1e9f).coerceAtMost(0.05f)
                last = now
                s.step(dt, target, reduced)
                frame = now
            }
        }
    }
    val light = isPaperLight()
    val accent = if (light) Palette.Accent else Graphite.Accent
    val ink = MaterialTheme.colorScheme.onSurface
    val paint = remember { Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG) }
    Canvas(
        modifier.semantics {
            contentDescription = "The page being flattened"
            progressBarRangeInfo = ProgressBarRangeInfo(progress.coerceIn(0f, 1f), 0f..1f)
        },
    ) {
        frame                                              // redraw on every frame of the simulation
        if (tex == null || sim == null) return@Canvas
        if (reduced) {
            val p = sim.shown
            sim.draw(this, tex, paint, accent, ink, still = 0.08f, alpha = 1f - p)
            sim.draw(this, tex, paint, accent, ink, still = 1f, alpha = p)
        } else {
            sim.draw(this, tex, paint, accent, ink)
        }
    }
}

/** The page region of [photo] (bounding box of [corners]), at most [TEX_MAX] px; a ruled blank sheet without a photo. */
private fun pageTexture(photo: Bitmap?, corners: List<Offset>, photoWidth: Int): Bitmap {
    if (photo == null || photo.isRecycled) return blankSheet()
    val k = if (photoWidth > 0) photo.width.toFloat() / photoWidth else 1f
    var x0 = 0; var y0 = 0; var x1 = photo.width; var y1 = photo.height
    if (corners.size == 4) {
        x0 = (corners.minOf { it.x } * k).toInt().coerceIn(0, photo.width - 1)
        y0 = (corners.minOf { it.y } * k).toInt().coerceIn(0, photo.height - 1)
        x1 = (corners.maxOf { it.x } * k).toInt().coerceIn(x0 + 1, photo.width)
        y1 = (corners.maxOf { it.y } * k).toInt().coerceIn(y0 + 1, photo.height)
        if (x1 - x0 < 16 || y1 - y0 < 16) { x0 = 0; y0 = 0; x1 = photo.width; y1 = photo.height }
    }
    val w = x1 - x0
    val h = y1 - y0
    val s = min(1f, TEX_MAX.toFloat() / max(w, h))
    val cut = Bitmap.createBitmap(photo, x0, y0, w, h)
    val out = if (s < 1f) Bitmap.createScaledBitmap(cut, (w * s).toInt().coerceAtLeast(1), (h * s).toInt().coerceAtLeast(1), true) else cut
    if (out !== cut && cut !== photo) cut.recycle()
    return if (out === photo) photo.copy(Bitmap.Config.ARGB_8888, false) else out
}

/** A plain ruled sheet, for a page that opens without its photo. */
private fun blankSheet(): Bitmap {
    val b = Bitmap.createBitmap(300, 400, Bitmap.Config.ARGB_8888)
    val c = android.graphics.Canvas(b)
    c.drawColor(android.graphics.Color.WHITE)
    val p = Paint().apply { color = 0xFFC3D3EC.toInt(); strokeWidth = 2f }
    var y = 44f
    while (y < 400f) { c.drawLine(0f, y, 300f, y, p); y += 22f }
    p.color = 0xFFD9473A.toInt()
    c.drawLine(40f, 0f, 40f, 400f, p)
    return b
}

/** A seed from a few pixels, so each page crumples its own way (and the same way every time). */
private fun seedOf(b: Bitmap): Int {
    var h = b.width * 31 + b.height
    for (i in 0 until 16) h = h * 31 + b.getPixel((i * 7919) % b.width, (i * 104729) % b.height)
    return h
}

/** Fast first: the big folds let go as soon as the core starts. */
private fun easeOut(t: Float) = 1 - (1 - t).pow(3)
private fun easeInOut(t: Float) = if (t < 0.5f) 4 * t * t * t else 1 - (-2 * t + 2).pow(3) / 2
private fun clamp01(x: Float) = x.coerceIn(0f, 1f)

/** One steam puff behind the iron: where it left the iron (page fractions) and how old it is. */
private class Puff(val u: Float, val v: Float, var age: Float, val drift: Float)

/**
 * One page's crumple and the state of its animation: the shown progress, the drop, the iron's
 * path and the creases it has pressed.
 */
private class Unfold(seed: Int) {
    private val n = (GX + 1) * (GY + 1)
    private val u = FloatArray(n)
    private val v = FloatArray(n)
    private val cx = FloatArray(n)
    private val cy = FloatArray(n)
    private val z = FloatArray(n)
    private val wz = FloatArray(n)
    private val phase = FloatArray(n)
    private val delay = FloatArray(n)
    private val pressed = FloatArray(n)
    private val verts = FloatArray(2 * n)
    private val zs = FloatArray(n)
    private val puffs = ArrayList<Puff>()

    var shown = 0f
        private set
    private var time = 0f
    private var reported = 0f
    private var reportedAt = 0f
    private var drop = 0f
    private var dropV = 0f
    private var ironT = 0f
    private var ironU = -0.2f
    private var ironV = 0.2f
    private var ironLeft = false
    private var ironAlpha = 0f
    private var puffClock = 0f

    init {
        val r = Random(seed)
        val turn = (r.nextFloat() - 0.5f) * 1.4f
        for (j in 0..GY) for (i in 0..GX) {
            val k = j * (GX + 1) + i
            val uu = i.toFloat() / GX * 2 - 1
            val vv = j.toFloat() / GY * 2 - 1
            val dist = min(1f, hypot(uu, vv) / 1.414f)
            val rad = dist.pow(0.55f) * 0.3f
            val a = atan2(vv, uu) + turn + (r.nextFloat() - 0.5f) * 0.5f
            u[k] = i.toFloat() / GX
            v[k] = j.toFloat() / GY
            cx[k] = cos(a) * rad + (r.nextFloat() - 0.5f) * 0.1f
            cy[k] = sin(a) * rad * 1.1f + (r.nextFloat() - 0.5f) * 0.1f
            z[k] = (r.nextFloat() - 0.5f) * 2
            wz[k] = r.nextFloat() - 0.5f
            phase[k] = r.nextFloat() * 2 * PI.toFloat()
            delay[k] = dist * 0.55f + r.nextFloat() * 0.12f
        }
    }

    /** Advances the animation by [dt] s; [target] is the core's last reported progress. */
    fun step(dt: Float, target: Float, reduced: Boolean) {
        time += dt
        if (target > reported) {
            reported = target
            reportedAt = time
        }
        if (reduced) {
            shown = reported
            drop = 1f
            return
        }
        // between reports the page keeps moving towards the next step (80% of the way, slower in the long solve)
        val next = STEPS.firstOrNull { it > reported + 1e-4f } ?: 1f
        val tau = if (reported >= 0.3f && reported < 0.85f) 10f else 2.5f
        val goal = reported + (next - reported) * 0.8f * (1 - exp(-(time - reportedAt) / tau))
        shown += (max(goal, shown) - shown) * (1 - exp(-dt * 4f))
        // the unwrapping plays out in the first seconds whatever the core reports (a slow first step, the
        // page finder or the surface fit on a slow phone, left a ball on screen): open by ~3 s, then the
        // iron works the sheet while the core finishes
        val byTime = 0.02f + 0.27f * easeInOut(clamp01((time - 0.5f) / 2.4f))
        if (shown < byTime) shown = byTime
        if (reported >= 1f && shown > 0.995f) shown = 1f
        // the drop: a damped spring that lands with a small bounce
        dropV += ((1 - drop) * 110f - dropV * 9f) * dt
        drop += dropV * dt
        // the iron glides over the sheet in rows (boustrophedon), from the solve stage until the page is flat
        val active = shown in 0.26f..0.94f
        ironAlpha = (ironAlpha + (if (active) dt else -dt) * 4f).coerceIn(0f, 1f)
        if (active || ironAlpha > 0f) {
            ironT += dt
            val pass = 1.6f
            val rows = floatArrayOf(0.2f, 0.42f, 0.64f, 0.84f, 0.53f, 0.31f)
            val idx = floor(ironT / pass).toInt()
            val f = (ironT - idx * pass) / pass
            val row = rows[idx % rows.size]
            val prev = rows[(idx + rows.size - 1) % rows.size]
            ironLeft = idx % 2 == 1
            // steady along the row, easing only into the turns
            val along = -0.06f + 1.12f * (0.2f * easeInOut(f) + 0.8f * f)
            ironU = if (ironLeft) 1f - along else along
            ironV = prev + (row - prev) * easeInOut(clamp01(f / 0.18f))
            if (active) {
                for (k in 0 until n) {
                    val du = abs(u[k] - ironU)
                    val dv = abs(v[k] - ironV)
                    if (du < 0.15f && dv < 0.15f) pressed[k] = min(1f, pressed[k] + dt * 6f * (1 - du / 0.15f))
                }
                puffClock += dt
                if (puffClock > 0.16f) {
                    puffClock = 0f
                    puffs += Puff(ironU + (if (ironLeft) 0.12f else -0.12f), ironV, 0f, (Random.nextFloat() - 0.5f))
                }
            }
        }
        for (pf in puffs) pf.age += dt
        puffs.removeAll { it.age > 1.2f }
    }

    /**
     * Draws the page; [still] (with system animations off) draws that progress without motion, at [alpha].
     */
    fun draw(scope: DrawScope, tex: Bitmap, paint: Paint, accent: Color, ink: Color, still: Float = -1f, alpha: Float = 1f) = with(scope) {
        if (alpha <= 0.01f) return@with
        val p = if (still >= 0f) still else shown
        val live = if (still >= 0f) 0f else 1f
        val aspect = tex.width.toFloat() / tex.height
        val pw = min(size.width * 0.86f, size.height * 0.84f * aspect)
        val ph = pw / aspect
        val unit = pw / 230f
        val open = clamp01((p - 0.02f) / 0.3f)
        val ball = 1 - open
        // the ball tumbles, bobs and breathes while it waits to open
        val angle = live * ball * (0.28f * sin(time * 1.15f) + 0.1f * sin(time * 2.7f))
        val breathe = 1 + live * ball * 0.04f * sin(time * 3.2f)
        val ox = size.width / 2
        val oy = size.height / 2 - (1 - (if (still >= 0f) 1f else drop)) * size.height * 0.6f +
            live * ball * sin(time * 2.3f) * pw * 0.025f
        val ca = cos(angle)
        val sa = sin(angle)
        val flatten = clamp01((p - 0.86f) / 0.08f)           // the final press: every crease out
        val relax = clamp01((p - 0.2f) / 0.6f)
        // the opening runs from 2% to 30%: the middle first, the outer folds follow, every fold moving all along
        val q = clamp01((p - 0.02f) / 0.28f)
        for (k in 0 until n) {
            val t = easeOut(clamp01((q - delay[k] * 0.5f) / (1 - delay[k] * 0.5f)))
            val fx = (u[k] - 0.5f) * pw
            val fy = (v[k] - 0.5f) * ph
            val rx = cx[k] * pw * 1.35f * breathe
            val ry = cy[k] * pw * 1.35f * breathe
            val bx = rx * ca - ry * sa
            val by = rx * sa + ry * ca
            // creases: relax a little by themselves, the iron presses them out; they stir while the paper is loose
            val wr = (1 - 0.5f * t) * (1 - 0.35f * relax) * (1 - pressed[k]) * (1 - flatten)
            val stir = 1 + live * 0.3f * sin(time * 1.4f + phase[k])
            verts[2 * k] = ox + bx + (fx - bx) * t + wz[k] * 7 * wr * stir * unit
            verts[2 * k + 1] = oy + by + (fy - by) * t + wz[k] * 5 * wr * stir * unit
            zs[k] = (z[k] * (1 - t) * 40 + wz[k] * 18 * wr * stir) * unit
        }
        val flat = clamp01((p - 0.3f) / 0.55f)
        // the shadow on the desk: small under the ball, wide under the sheet
        val sw = pw * (0.28f + 0.27f * open)
        val sh = ph * (0.06f + 0.2f * open)
        val sc = Offset(ox, size.height / 2 + ph * (0.16f + 0.32f * open))
        val shadowA = 0.24f * alpha * (0.4f + 0.6f * (if (still >= 0f) 1f else drop))
        scale(1f, sh / sw, pivot = sc) {
            drawCircle(Brush.radialGradient(listOf(Color(0xFF141928).copy(alpha = shadowA), Color.Transparent), center = sc, radius = sw * 1.2f), radius = sw * 1.2f, center = sc)
        }
        paint.alpha = (alpha * 255).toInt()
        drawIntoCanvas { it.nativeCanvas.drawBitmapMesh(tex, GX, GY, verts, 0, null, 0, paint) }
        shade(alpha)
        val gridA = sin(flat * PI).toFloat() * 0.22f * alpha
        if (gridA > 0.01f) grid(accent.copy(alpha = gridA))
        // the paper whitens at the end (shadow and colour cast removal); the sheet is flat by then
        val white = clamp01((p - 0.92f) / 0.07f)
        if (white > 0f) drawRect(Color.White.copy(alpha = 0.3f * white * alpha), Offset(ox - pw / 2, oy - ph / 2), Size(pw, ph))
        if (still < 0f && ironAlpha > 0f) {
            val left = ox - pw / 2
            val top = oy - ph / 2
            for (pf in puffs) {
                val a = (1 - pf.age / 1.2f) * 0.3f * ironAlpha * alpha
                val c = Offset(left + pf.u * pw + pf.drift * pf.age * 14.dp.toPx(), top + pf.v * ph - 14.dp.toPx() - pf.age * 34.dp.toPx())
                drawCircle(ink.copy(alpha = a), radius = (3f + pf.age * 6f).dp.toPx() * unit.coerceAtMost(1.4f), center = c, style = Stroke(1.5.dp.toPx()))
            }
            // the iron lifts a little where it turns at the end of a row
            val lift = (1 - min(1f, min(abs(ironU + 0.06f), abs(ironU - 1.06f)) / 0.1f)) * 6.dp.toPx()
            drawIron(Offset(left + ironU * pw, top + ironV * ph - lift), unit * 1.35f, ironAlpha * alpha, flip = ironLeft)
        }
    }

    /** Crease light and shade: each triangle lit from the upper left, batched by strength into a few paths. */
    private fun DrawScope.shade(alpha: Float) {
        val dark = arrayOfNulls<Path>(12)
        val light = arrayOfNulls<Path>(8)
        fun tri(a: Int, b: Int, c: Int) {
            val ux = verts[2 * b] - verts[2 * a]; val uy = verts[2 * b + 1] - verts[2 * a + 1]; val uz = zs[b] - zs[a]
            val vx = verts[2 * c] - verts[2 * a]; val vy = verts[2 * c + 1] - verts[2 * a + 1]; val vz = zs[c] - zs[a]
            var nx = uy * vz - uz * vy
            var ny = uz * vx - ux * vz
            var nz = ux * vy - uy * vx
            val l = sqrt(nx * nx + ny * ny + nz * nz).takeIf { it > 1e-6f } ?: return
            nx /= l; ny /= l; nz /= l
            if (nz < 0) { nx = -nx; ny = -ny; nz = -nz }
            // a flat facet lights to about 0.7: shade only the difference
            val k = -0.45f * nx - 0.55f * ny + 0.7f * nz - 0.7f
            if (abs(k) < 0.01f) return
            val bucket = if (k < 0) min(11, (min(0.55f, -k * 1.3f) / 0.05f).toInt()) else min(7, (min(0.35f, k * 1.6f) / 0.05f).toInt())
            if (bucket == 0) return
            val arr = if (k < 0) dark else light
            val path = arr[bucket] ?: Path().also { arr[bucket] = it }
            path.moveTo(verts[2 * a], verts[2 * a + 1])
            path.lineTo(verts[2 * b], verts[2 * b + 1])
            path.lineTo(verts[2 * c], verts[2 * c + 1])
            path.close()
        }
        for (j in 0 until GY) for (i in 0 until GX) {
            val a = j * (GX + 1) + i
            val b = a + 1
            val c = a + GX + 1
            val d = c + 1
            tri(a, b, d)
            tri(a, d, c)
        }
        dark.forEachIndexed { i, path -> if (path != null) drawPath(path, Color(0xFF1E190F).copy(alpha = (i + 0.5f) * 0.05f * alpha)) }
        light.forEachIndexed { i, path -> if (path != null) drawPath(path, Color.White.copy(alpha = (i + 0.5f) * 0.05f * alpha)) }
    }

    /** A faint dashed grid through the mesh: the warp map being solved. */
    private fun DrawScope.grid(color: Color) {
        val path = Path()
        for (j in 1 until GY step 2) {
            for (i in 0..GX) {
                val k = j * (GX + 1) + i
                if (i == 0) path.moveTo(verts[2 * k], verts[2 * k + 1]) else path.lineTo(verts[2 * k], verts[2 * k + 1])
            }
        }
        for (i in 1 until GX step 2) {
            for (j in 0..GY) {
                val k = j * (GX + 1) + i
                if (j == 0) path.moveTo(verts[2 * k], verts[2 * k + 1]) else path.lineTo(verts[2 * k], verts[2 * k + 1])
            }
        }
        drawPath(path, color, style = Stroke(width = 1.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(3.dp.toPx(), 4.dp.toPx()))))
    }
}
