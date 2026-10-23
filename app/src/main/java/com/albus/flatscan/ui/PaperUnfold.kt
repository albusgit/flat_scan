package com.albus.flatscan.ui

import android.graphics.Bitmap
import android.graphics.Paint
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
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

/**
 * The processing animation: [photo] (the page cut from it with [corners], photo pixels of a
 * [photoWidth]-wide photo) starts as a crumpled ball, unfolds as [progress] rises and an iron
 * presses out the last creases at 85-92%. It only moves when the progress changes (in the long
 * 30-85% stage it creeps about 70% of the way over 6 s, then holds), so nothing animates
 * endlessly while the core works. With system animations off, a still crumpled frame cross-fades
 * to the flat page.
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
    val mesh = remember(tex) { tex?.let { Crumple(seedOf(it)) } }
    val shown = remember(tex) { Animatable(0f) }
    val drop = remember(tex) { Animatable(if (reduced) 1f else 0f) }
    LaunchedEffect(tex) { if (tex != null && !reduced) drop.animateTo(1f, spring(dampingRatio = 0.55f, stiffness = Spring.StiffnessLow)) }
    LaunchedEffect(tex, progress) {
        val target = max(progress, shown.value)            // never runs backwards
        if (reduced) {
            shown.snapTo(target)
            return@LaunchedEffect
        }
        // the iron's stretch moves slower so the sweep reads as a sweep
        shown.animateTo(target, tween(if (target in 0.85f..0.93f) 800 else 450, easing = FastOutSlowInEasing))
        if (target >= 0.3f && target < 0.85f) {
            shown.animateTo(target + (0.85f - target) * 0.7f, tween(6000, easing = LinearOutSlowInEasing))
        }
    }
    val accent = if (isPaperLight()) Palette.Accent else Graphite.Accent
    val paint = remember { Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG) }
    Canvas(
        modifier.semantics {
            contentDescription = "The page being flattened"
            progressBarRangeInfo = ProgressBarRangeInfo(progress.coerceIn(0f, 1f), 0f..1f)
        },
    ) {
        if (tex == null || mesh == null) return@Canvas
        if (reduced) {
            val p = shown.value
            mesh.draw(this, tex, paint, 0.08f, 1f, accent, alpha = 1f - p)
            mesh.draw(this, tex, paint, 1f, 1f, accent, alpha = p)
        } else {
            mesh.draw(this, tex, paint, shown.value, drop.value, accent)
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

private fun ease(t: Float) = if (t < 0.5f) 4 * t * t * t else 1 - (-2 * t + 2).pow(3) / 2
/** Fast first: the big folds let go as soon as the core starts. */
private fun easeOut(t: Float) = 1 - (1 - t).pow(3)
private fun clamp01(x: Float) = x.coerceIn(0f, 1f)

/** The crumple of one page: where each mesh vertex sits in the ball and how high it is. */
private class Crumple(seed: Int) {
    private val n = (GX + 1) * (GY + 1)
    private val u = FloatArray(n)
    private val v = FloatArray(n)
    private val cx = FloatArray(n)
    private val cy = FloatArray(n)
    private val z = FloatArray(n)
    private val wz = FloatArray(n)
    private val delay = FloatArray(n)
    private val verts = FloatArray(2 * n)
    private val zs = FloatArray(n)

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
            delay[k] = dist * 0.55f + r.nextFloat() * 0.12f
        }
    }

    /** Draws the page at progress [p] (0..1), [drop] (0..1) of the way down from above. */
    fun draw(scope: DrawScope, tex: Bitmap, paint: Paint, p: Float, drop: Float, accent: Color, alpha: Float = 1f) = with(scope) {
        if (alpha <= 0.01f) return@with
        val aspect = tex.width.toFloat() / tex.height
        val pw = min(size.width * 0.86f, size.height * 0.84f * aspect)
        val ph = pw / aspect
        val ox = size.width / 2
        val oy = size.height / 2 - (1 - drop) * size.height * 0.45f
        val unit = pw / 230f
        val sweep = clamp01((p - 0.85f) / 0.07f)              // the iron's way across the sheet
        for (k in 0 until n) {
            val t = easeOut(clamp01((p - 0.02f - delay[k] * 0.3f) / (0.36f - delay[k] * 0.3f)))
            val fx = (u[k] - 0.5f) * pw
            val fy = (v[k] - 0.5f) * ph
            val bx = cx[k] * pw * 1.15f
            val by = cy[k] * pw * 1.15f
            // creases relax with the long stage but only the iron presses them out
            var wr = (1 - 0.7f * clamp01((p - 0.3f) / 0.52f)) * (1 - t * 0.5f)
            wr *= clamp01((u[k] - sweep * 1.15f + 0.08f) / 0.1f)
            verts[2 * k] = ox + bx + (fx - bx) * t + wz[k] * 6 * wr * unit
            verts[2 * k + 1] = oy + by + (fy - by) * t + wz[k] * 4 * wr * unit
            zs[k] = (z[k] * (1 - t) * 40 + wz[k] * 16 * wr) * unit
        }
        val flat = clamp01((p - 0.3f) / 0.55f)
        // the shadow on the desk widens as the sheet opens
        val sw = pw * (0.25f + 0.3f * flat)
        val sh = ph * (0.07f + 0.2f * flat)
        val sc = Offset(ox, oy + ph * (0.18f + 0.3f * flat))
        scale(1f, sh / sw, pivot = sc) {
            drawCircle(
                Brush.radialGradient(listOf(Color(0xFF141928).copy(alpha = 0.22f * alpha), Color.Transparent), center = sc, radius = sw * 1.2f),
                radius = sw * 1.2f, center = sc,
            )
        }
        paint.alpha = (alpha * 255).toInt()
        drawIntoCanvas { it.nativeCanvas.drawBitmapMesh(tex, GX, GY, verts, 0, null, 0, paint) }
        shade(alpha)
        // the paper whitens at the end (shadow and colour cast removal); the sheet is flat by then
        val white = clamp01((p - 0.92f) / 0.07f)
        if (white > 0f) drawRect(Color.White.copy(alpha = 0.3f * white * alpha), Offset(ox - pw / 2, oy - ph / 2), androidx.compose.ui.geometry.Size(pw, ph))
        val gridA = sin(flat * PI).toFloat() * 0.35f * alpha
        if (gridA > 0.01f) grid(accent.copy(alpha = gridA))
        if (p > 0.845f && p < 0.93f) {
            val a = clamp01((p - 0.845f) / 0.01f) * clamp01((0.93f - p) / 0.01f) * alpha
            val left = ox - pw / 2
            val top = oy - ph / 2
            drawIron(
                Offset(left + pw * (sweep * 1.1f - 0.02f), top + ph * (0.5f + sin(sweep * PI * 5).toFloat() * 0.28f)),
                pw / 230f, a,
            )
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
