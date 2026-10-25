package com.albus.flatscan.ui

import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.DashPathEffect
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
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
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalDensity
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
import kotlin.math.roundToInt
import kotlin.math.sign
import kotlin.math.sin
import kotlin.random.Random

// The processing animation of the design "FlatScan Flattening Animation" (take 3): the page, ironed flat. Its
// numbers (beats, report points, mesh, light, the iron's rows) are the design's; the page and the photo are the real
// ones. Two differences: the resampling blur is left out (it needs a RuntimeShader; the steam lifting off carries
// that beat), and a book page gets the gutter's curl and shadow but no facing page.

private const val NX = 40
private const val NY = 56
private const val NV = (NX + 1) * (NY + 1)
private const val TEX_MAX = 900
private const val F = 3.2f                         // the camera's distance (perspective)
private const val U0 = -0.1f                       // the iron's path runs a little past both edges ...
private const val U1 = 1.1f
private const val HW = 0.13f                       // ... and HW is half its width
private const val RULE = 0.8f                      // of each row: the glide; the rest is the turn to the next
private const val ROW_TIME = 0.9f                  // s per row, turn included
private const val FRAME_ROT = -0.03f               // the photo lies a little askew on the desk

/** The core's report points (pipeline.cpp); between reports the progress creeps towards the next. */
private val STEPS = floatArrayOf(0.02f, 0.05f, 0.08f, 0.14f, 0.2f, 0.3f, 0.85f, 0.9f, 0.92f, 0.95f, 1f)

/** Each beat (from, to, natural seconds): it plays at that length, up to 3x faster to catch up, 6x once the core is done. */
private val BEATS = arrayOf(
    floatArrayOf(0f, 0.02f, 0.35f), floatArrayOf(0.02f, 0.05f, 0.5f), floatArrayOf(0.05f, 0.2f, 1.4f), floatArrayOf(0.2f, 0.3f, 0.8f),
    floatArrayOf(0.85f, 0.9f, 0.5f), floatArrayOf(0.9f, 0.92f, 0.6f), floatArrayOf(0.92f, 0.95f, 0.5f), floatArrayOf(0.95f, 1f, 0.8f),
)

/**
 * The processing animation, drawn every frame while the page is processed. The photo drops onto the desk and corner
 * brackets snap onto the page's corners; a corner peels up and the page pops off the photo; it floats while a pencil
 * mesh drapes over it; the curl springs out, overshoots a little and settles, the creases staying; an iron glides over
 * the page row by row, puffing steam, and presses the creases out behind it (on a slow phone it goes back over its
 * work in lazy figures of eight until the core reports 85 %, then finishes the last row in about half a second); the
 * iron hops off and the page gives a springy thump flat; the last steam lifts off top to bottom; a band of light wipes
 * off the shadow and the yellow cast; the sheet lands with a puff and crop marks snap round it.
 *
 * Each beat has a natural length and plays at it when the core is slow; when the core runs ahead the beats compress
 * (to a third, a sixth once the core is done), and the animation never runs ahead of the core's reports ([progress]).
 * [photo] is the photo, [corners] the page's corners in it (photo pixels of a [photoWidth]-wide photo, TL TR BR BL);
 * [book]: a book page (the gutter's curl and shadow). [onFinished] is called once, when the landing has played. With
 * system animations off: the core's progress as it is, without the iron, the steam or the light band.
 */
@Composable
fun PaperUnfold(
    photo: Bitmap?,
    corners: List<Offset>,
    photoWidth: Int,
    progress: Float,
    modifier: Modifier = Modifier,
    book: Boolean = false,
    onFinished: () -> Unit = {},
) {
    val reduced = rememberReducedMotion()
    val texture by produceState<Bitmap?>(null, photo) {
        value = withContext(Dispatchers.Default) { runCatching { pageTexture(photo, corners, photoWidth) }.getOrNull() }
    }
    val tex = texture
    val sim = remember(tex, book) { tex?.let { Flatten(seedOf(it), it.height.toFloat() / it.width, book) } }
    val target by rememberUpdatedState(progress)
    val finished by rememberUpdatedState(onFinished)
    var frame by remember { mutableLongStateOf(0L) }
    LaunchedEffect(sim, reduced) {
        val s = sim ?: return@LaunchedEffect
        var last = -1L
        var told = false
        while (true) {
            withFrameNanos { now ->
                val dt = if (last < 0) 0f else ((now - last) / 1e9f).coerceAtMost(0.05f)
                last = now
                s.step(dt, target, reduced)
                frame = now
            }
            if (!told && s.landed) {
                told = true
                finished()
            }
        }
    }
    val light = isPaperLight()
    val accent = if (light) Palette.Accent else Graphite.Accent
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val drop = with(LocalDensity.current) { 46.dp.toPx() }
    Canvas(
        modifier
            .graphicsLayer {
                // the drop onto the desk at the start
                frame
                val t = sim?.dropT ?: 1f
                if (!reduced && t < 1f) {
                    translationY = -(1 - backOut(t)) * drop
                    rotationZ = -(1 - t) * 3f
                } else {
                    translationY = 0f
                    rotationZ = 0f
                }
            }
            .semantics {
                contentDescription = "The page being flattened"
                progressBarRangeInfo = ProgressBarRangeInfo(progress.coerceIn(0f, 1f), 0f..1f)
            },
    ) {
        frame                                              // redraw on every frame of the animation
        if (tex == null || sim == null) return@Canvas
        drawIntoCanvas { sim.draw(it.nativeCanvas, size.width, size.height, tex, photo, corners, photoWidth, accent.toArgb(), muted.toArgb(), reduced) }
    }
}

/**
 * The page as it lies in [photo]: the quadrilateral of [corners] straightened into a rectangle of its own proportions
 * (at most [TEX_MAX] px), so the page that peels off is the real one; a plain ruled sheet without a photo.
 */
private fun pageTexture(photo: Bitmap?, corners: List<Offset>, photoWidth: Int): Bitmap {
    if (photo == null || photo.isRecycled) return blankSheet()
    val k = if (photoWidth > 0) photo.width.toFloat() / photoWidth else 1f
    val q = if (corners.size == 4) corners.map { Offset(it.x * k, it.y * k) }
    else listOf(Offset(0f, 0f), Offset(photo.width.toFloat(), 0f), Offset(photo.width.toFloat(), photo.height.toFloat()), Offset(0f, photo.height.toFloat()))
    val w0 = ((q[1] - q[0]).getDistance() + (q[2] - q[3]).getDistance()) / 2
    val h0 = ((q[3] - q[0]).getDistance() + (q[2] - q[1]).getDistance()) / 2
    if (w0 < 16f || h0 < 16f) return blankSheet()
    val s = min(1f, TEX_MAX / max(w0, h0))
    val w = (w0 * s).roundToInt().coerceAtLeast(1)
    val h = (h0 * s).roundToInt().coerceAtLeast(1)
    val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    val m = Matrix()
    val src = floatArrayOf(q[0].x, q[0].y, q[1].x, q[1].y, q[2].x, q[2].y, q[3].x, q[3].y)
    val dst = floatArrayOf(0f, 0f, w.toFloat(), 0f, w.toFloat(), h.toFloat(), 0f, h.toFloat())
    if (!m.setPolyToPoly(src, 0, dst, 0, 4)) return blankSheet()
    val c = android.graphics.Canvas(out)
    c.drawColor(android.graphics.Color.WHITE)
    c.drawBitmap(photo, m, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
    return out
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

/** A seed from a few pixels, so each page creases its own way (and the same way every time). */
private fun seedOf(b: Bitmap): Int {
    var h = b.width * 31 + b.height
    for (i in 0 until 16) h = h * 31 + b.getPixel((i * 7919) % b.width, (i * 104729) % b.height)
    return h
}

private fun clamp(x: Float, a: Float = 0f, b: Float = 1f) = min(b, max(a, x))
private fun sm(a: Float, b: Float, x: Float): Float { val t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t) }
private fun ease(t: Float) = if (t < 0.5f) 4 * t * t * t else 1 - (-2 * t + 2).pow(3) / 2
private fun backOut(t: Float): Float { val c1 = 1.70158f; val c3 = c1 + 1; return 1 + c3 * (t - 1).pow(3) + c1 * (t - 1).pow(2) }
private fun mix(a: Float, b: Float, t: Float) = a + (b - a) * t

/** One steam puff: where it left the iron (page fractions), how old it is, which way it drifts. */
private class Puff(val u: Float, val v: Float, var age: Float, val drift: Float)

/** Where the iron is and how it faces: u, v (page fractions), lift (px of the design), face (-1..1: the flip). */
private class IronAt(val u: Float, val v: Float, val lift: Float, val face: Float)

/**
 * One page's animation: its mesh (the design's 41 x 57 points, its creases, curl and light), the iron's rows, and the
 * pacing that follows the core's reports.
 */
private class Flatten(seed: Int, private val ph: Float, private val book: Boolean) {
    private val gutter = if (book) 1f else 0f
    // the iron's rows: L bands down the page; each mesh row lies between two of them
    private val l = clamp(7 * ph / (4f / 3f), 4f, 10f).roundToInt()
    private val bandV = FloatArray(l) { (it + 0.5f) / l }
    private val rowA = IntArray(NY + 1)
    private val rowB = IntArray(NY + 1)
    private val rowT = FloatArray(NY + 1)
    private val creaseZ = FloatArray(NV)
    private val sx = FloatArray(NV)
    private val sy = FloatArray(NV)
    private val p3 = FloatArray(NV * 3)
    private val verts = FloatArray(NV * 2)
    private val colors = IntArray(NV)
    private val spark = Array(14) { i -> val r = Random(100 + i); floatArrayOf(0.06f + 0.88f * r.nextFloat(), 0.04f + 0.92f * r.nextFloat(), 0.6f + 0.6f * r.nextFloat()) }
    private val rnd = Random(seed)

    init {
        for (j in 0..NY) {
            val v = j.toFloat() / NY
            when {
                v <= bandV[0] -> { rowA[j] = 0; rowB[j] = 0; rowT[j] = 0f }
                v >= bandV[l - 1] -> { rowA[j] = l - 1; rowB[j] = l - 1; rowT[j] = 0f }
                else -> {
                    val a = min(l - 2, floor(v * l - 0.5f).toInt())
                    rowA[j] = a; rowB[j] = a + 1; rowT[j] = sm(0.25f, 0.75f, (v - bandV[a]) * l)
                }
            }
        }
        // creases: tent-shaped ridges along random segments, each page creased its own way
        val r = Random(seed)
        val amp = if (book) 0.6f else 1f
        class Crease(val u: Float, val v: Float, val ca: Float, val sa: Float, val len: Float, val w: Float, val h: Float)
        val cr = List(12) { i ->
            val big = i < 6
            val a = r.nextFloat() * PI.toFloat()
            Crease(
                r.nextFloat(), r.nextFloat() * ph, cos(a), sin(a), (if (big) 0.7f else 0.3f) + 0.4f * r.nextFloat(),
                (if (big) 0.065f else 0.04f) + 0.02f * r.nextFloat(), (if (r.nextFloat() < 0.5f) -1f else 1f) * (if (big) 0.024f else 0.013f) * amp,
            )
        }
        for (j in 0..NY) for (i in 0..NX) {
            val x = i.toFloat() / NX
            val y = j.toFloat() / NY * ph
            var z = 0f
            for (c in cr) {
                val dx = x - c.u
                val dy = y - c.v
                val along = dx * c.ca + dy * c.sa
                val across = -dx * c.sa + dy * c.ca
                z += c.h * max(0f, 1 - abs(across) / c.w) * (1 - sm(c.len * 0.3f, c.len * 0.5f, abs(along)))
            }
            creaseZ[j * (NX + 1) + i] = z
        }
    }

    // ---- the pacing (the design's step()) ----
    private var t = 0f
    private var reported = 0f
    private var reportedAt = 0f
    private var next = 0
    private var shown = 0f
    private var vp = 0f
    private var inTrace = false
    private var penPos = 0f
    private var penA = 0f
    private var penExit = 0f
    private var mode = 0                               // 0: along the rows, 1: going back over its work, 2: back to its row
    private var tt = 0f
    private var touchFrom: IronAt? = null
    private var backT = 0f
    private var backFrom: IronAt? = null
    private var rushSpeed = 0f
    private val puffs = ArrayList<Puff>()
    private var puffClock = 0f
    private var time = 0f
    private var live = true

    /** The drop onto the desk (0..1, the first 0.55 s). */
    val dropT: Float get() = clamp(t / 0.55f)
    /** The landing has played (the core done, the animation at its end). */
    val landed: Boolean get() = vp >= 1f && reported >= 1f

    private fun baseRate(p: Float): Float {
        for (b in BEATS) if (p >= b[0] - 1e-6f && p < b[1] - 1e-6f) return (b[1] - b[0]) / b[2]
        return 1f
    }

    private fun dirOf(b: Int) = if (b % 2 == 1) -1 else 1
    private class Pen(val k: Int, val f: Float, val tf: Float)
    private fun penAt(p: Float): Pen {
        var k = floor(p).toInt()
        var f = p - k
        if (k >= l) { k = l - 1; f = 1f }
        return Pen(k, f, min(1f, f / RULE))
    }
    private fun ironU(k: Int, tf: Float) = if (dirOf(k) > 0) mix(U0, U1, tf) else mix(U1, U0, tf)

    /** How pressed row [i] is at [u]: flat where the iron has been, creased ahead of it. */
    private fun bandS(i: Int, u: Float, pen: Pen, done: Boolean): Float {
        if (done || i < pen.k) return 1f
        if (i > pen.k) return 0f
        if (pen.tf >= 1f) return 1f
        val d = dirOf(i)
        val front = ironU(i, pen.tf) + d * HW * 0.6f
        return if (d > 0) 1 - sm(front - 0.07f, front + 0.02f, u) else sm(front - 0.02f, front + 0.07f, u)
    }

    /** The iron on its row: gliding across, then turning (lifting and flipping) down to the next row. */
    private fun bandIron(p: Float): IronAt {
        val pen = penAt(p)
        val d = dirOf(pen.k)
        if (pen.f <= RULE || pen.k >= l - 1) return IronAt(ironU(pen.k, pen.tf), bandV[pen.k], 0f, d.toFloat())
        val h = ease((pen.f - RULE) / (1 - RULE))
        return IronAt(if (d > 0) U1 else U0, mix(bandV[pen.k], bandV[pen.k + 1], h), sin(PI.toFloat() * h) * 10, d * clamp(cos(PI.toFloat() * h) * 3, -1f, 1f))
    }

    /** While a slow core solves, the iron goes back over its work in a lazy figure of eight. */
    private fun loopIron(tt: Float): IronAt {
        val w = 0.8f
        val top = bandV[0]
        val bot = bandV[max(0, l - 2)]
        return IronAt(0.5f + 0.3f * sin(w * tt), (top + bot) / 2 + (bot - top) * 0.42f * sin(2 * w * tt), 2 + 1.5f * sin(tt * 5), clamp(cos(w * tt) * 4, -1f, 1f))
    }

    private fun iron(): IronAt {
        if (mode == 1) {
            val lp = loopIron(tt)
            val k = clamp(tt / 0.7f)
            val b = ease(k)
            val f = touchFrom ?: lp
            return IronAt(mix(f.u, lp.u, b), mix(f.v, lp.v, b), sin(PI.toFloat() * k) * 12 + lp.lift * b, mix(f.face, lp.face, b))
        }
        if (mode == 2) {
            val k = ease(clamp(backT / 0.3f))
            val f = backFrom ?: bandIron(penPos)
            val to = bandIron(penPos)
            return IronAt(mix(f.u, to.u, k), mix(f.v, to.v, k), sin(PI.toFloat() * k) * 12, mix(f.face, to.face, k))
        }
        return bandIron(penPos)
    }

    private fun burst(n: Int) {
        val iv = iron()
        repeat(n) { puffs += Puff(iv.u + (rnd.nextFloat() - 0.5f) * 0.14f, iv.v, rnd.nextFloat() * 0.2f, (rnd.nextFloat() - 0.5f) * 2) }
    }

    /** Advances the animation by [dt] s; [target] is the core's last reported progress. */
    fun step(dt: Float, target: Float, reduced: Boolean) {
        live = !reduced
        t += dt
        time += dt
        if (target > reported) {
            reported = target
            reportedAt = t
        }
        if (reduced) {
            shown = reported
            vp = reported
            penPos = if (reported >= 0.85f) l.toFloat() else 0f
            penA = 0f
            inTrace = reported >= 0.3f
        } else {
            // between reports the progress keeps creeping towards the next step
            val nx = STEPS.firstOrNull { it > reported + 1e-4f } ?: 1f
            val tau = if (reported >= 0.3f && reported < 0.85f) 10f else 2.5f
            val goal = reported + (nx - reported) * 0.8f * (1 - exp(-(t - reportedAt) / tau))
            shown += (max(goal, shown) - shown) * (1 - exp(-dt * (if (reported >= 1f) 12f else 4f)))
            if (reported >= 1f && shown > 0.995f) shown = 1f
            val penDone = penPos >= l - 1e-4f
            // the beats: natural speed, faster when behind, never past the core
            val doneCore = reported >= 1f
            val squeeze = if (doneCore) 6f else 3f
            val pull = if (doneCore) 5f else 2.5f
            fun advance(cap: Float) {
                val r = baseRate(vp)
                vp = min(min(shown, cap), vp + dt * min(r * squeeze, max(r, (shown - vp) * pull)))
            }
            if (!inTrace) {
                advance(0.3f)
                if (vp >= 0.3f - 1e-6f && shown >= 0.3f) { vp = 0.3f; inTrace = true }
            } else if (!penDone) {
                vp = 0.3f
            } else {
                if (vp < 0.85f) vp = 0.85f
                advance(1f)
                if (reported >= 1f && shown >= 1f && vp > 0.9995f) vp = 1f
            }
            if (inTrace && !penDone) {
                val rush = reported >= 0.85f
                when (mode) {
                    0 -> {
                        // at 85 % the rest goes in one quick pass of about half a second (less once the core is done)
                        if (rush) rushSpeed = max(rushSpeed, max(1.6f, (l - penPos) / (if (reported >= 1f) 0.3f else 0.6f)))
                        val before = floor(penPos).toInt()
                        val cap = if (rush) l.toFloat() else l - 1f
                        penPos = min(cap, penPos + dt * (if (rush) rushSpeed else 1 / ROW_TIME))
                        if (floor(penPos).toInt() > before && penPos < l) burst(4)
                        if (!rush && penPos >= cap - 1e-6f) { touchFrom = bandIron(penPos); tt = 0f; mode = 1 }
                    }
                    1 -> {
                        tt += dt
                        if (rush) { backFrom = iron(); backT = 0f; mode = 2 }
                    }
                    else -> {
                        backT += dt
                        if (backT >= 0.3f) mode = 0
                    }
                }
                // steam behind the iron
                puffClock += dt
                if (puffClock > (if (mode == 1) 0.2f else 0.11f)) {
                    puffClock = 0f
                    val iv = iron()
                    val f = sign(iv.face).takeIf { it != 0f } ?: 1f
                    puffs += Puff(iv.u - f * 0.12f, iv.v, 0f, rnd.nextFloat() - 0.5f)
                }
            }
            if (penPos >= l - 1e-4f) penExit = clamp(penExit + dt * 3)
            penA = clamp(penA + (if (inTrace && penExit < 1f) dt else -dt) * 5)
        }
        for (p in puffs) p.age += dt
        puffs.removeAll { it.age >= 1.1f }
    }

    // ---- the mesh (the design's computeMesh) ----
    private var w = 300f
    private var h = 350f
    private var ppu = 1f
    private var cxr = 1f; private var sxr = 0f; private var cyr = 1f; private var syr = 0f; private var czr = 1f; private var szr = 0f
    private var sc = 1f; private var scx = 1f; private var scy = 1f; private var tx = 0f; private var ty = 0f
    private var float = 0f
    private var fin = 0f
    private val out = FloatArray(5)

    private fun xf(x0: Float, y0: Float, z0: Float): FloatArray {
        val x = x0 * scx
        val y = y0 * scy
        val z = z0 * sc
        val x1 = x * cyr + z * syr
        val z1 = -x * syr + z * cyr
        val y1 = y * cxr - z1 * sxr
        val z2 = y * sxr + z1 * cxr
        val x2 = x1 * czr - y1 * szr + tx
        val y2 = x1 * szr + y1 * czr + ty
        val f = F / (F + z2)
        out[0] = w / 2 + x2 * f * ppu; out[1] = h / 2 + y2 * f * ppu; out[2] = x2; out[3] = y2; out[4] = z2
        return out
    }

    private fun flatAt(u: Float, v: Float): Offset { val o = xf(u - 0.5f, (v - 0.5f) * ph, 0f); return Offset(o[0], o[1]) }

    private fun at(u: Float, v: Float): Offset {
        val fi = clamp(u) * NX
        val fj = clamp(v) * NY
        val i = min(NX - 1, floor(fi).toInt())
        val j = min(NY - 1, floor(fj).toInt())
        val a = fi - i
        val b = fj - j
        val k = j * (NX + 1) + i
        val k2 = k + NX + 1
        return Offset(mix(mix(sx[k], sx[k + 1], a), mix(sx[k2], sx[k2 + 1], a), b), mix(mix(sy[k], sy[k + 1], a), mix(sy[k2], sy[k2 + 1], a), b))
    }

    private fun computeMesh(tracing: Boolean, lv: Float) {
        val peel = sm(0.05f, 0.11f, vp) * (1 - sm(0.12f, 0.18f, vp))
        val popT = sm(0.09f, 0.17f, vp)
        val pop = if (popT <= 0f) 0f else backOut(popT)
        val ut = sm(0.2f, 0.3f, vp)
        val c = (1 - ut).pow(2) * cos(ut * PI.toFloat() * 1.6f)
        fin = sm(0.95f, 1f, vp)
        float = lv * sm(0.12f, 0.2f, vp) * (1 - fin)
        val ct = clamp((vp - 0.85f) / 0.05f)
        val boing = if (ct > 0f && ct < 1f) 0.022f * sin(ct * PI.toFloat() * 3) * (1 - ct) else 0f
        val land = clamp((fin - 0.45f) / 0.4f)
        val squash = lv * 0.035f * sin(PI.toFloat() * land)
        val k1 = 1 - pop
        val rx = -0.42f * k1
        val ry = 0.11f * k1
        val rz = -0.06f * k1 + 0.01f * sin(time * 1.13f) * float
        cxr = cos(rx); sxr = sin(rx); cyr = cos(ry); syr = sin(ry); czr = cos(rz); szr = sin(rz)
        sc = (0.64f + 0.36f * pop) * (1 - 0.018f * fin) * (1 + boing)
        scx = sc * (1 + squash * 0.6f)
        scy = sc * (1 - squash)
        tx = 0.1f * k1
        ty = 0.025f * k1 - 0.012f * float + 0.008f * sin(time * 1.7f) * float + squash * ph * 0.5f * sc
        ppu = min(w * 0.8f, h * 0.84f / ph)
        val pen = penAt(penPos)
        val done = penPos >= l - 1e-4f
        val flutter = lv * 0.022f * sin(PI.toFloat() * ut)
        for (j in 0..NY) {
            val v = j.toFloat() / NY
            for (i in 0..NX) {
                val u = i.toFloat() / NX
                val k = j * (NX + 1) + i
                val s = if (tracing) mix(bandS(rowA[j], u, pen, done), bandS(rowB[j], u, pen, done), rowT[j]) else 0f
                val r = 1 - s
                val env = 0.55f + 0.9f * (if (book) (1 - u) * (1 - u) else 0.5f + 0.5f * v * v)
                val cz = creaseZ[k] * r
                var x = u - 0.5f
                var y = (v - 0.5f) * ph
                var z = cz
                y += r * 0.012f * env * (0.6f * sin(u * 2.6f + v * 4.1f + 0.7f) + 0.4f * sin(u * 5.3f - v * 2.7f + 2.1f)) + cz * 0.5f
                x += r * 0.005f * sin(v * 5.1f + 1.3f) + cz * 0.3f
                if (book) {
                    val gg = max(0f, 1 - u / 0.42f)
                    z += c * (0.24f * gg * gg - 0.05f * sin(PI.toFloat() * u)); x += c * 0.05f * gg * gg
                } else {
                    val b = max(0f, v - 0.55f) / 0.45f
                    z += c * (-0.08f * sin(PI.toFloat() * u) - 0.17f * b * b); y -= c * 0.035f * b * b
                }
                z -= c * 0.11f * u.pow(4) * v.pow(4)
                z += flutter * sin(time * 9 + u * 3) * u * v
                val d = max(0f, u + v - 1.15f)
                z -= peel * 0.6f * d * d; x -= peel * 0.07f * d * d; y -= peel * 0.07f * d * d
                val o = xf(x, y, z)
                p3[3 * k] = o[2]; p3[3 * k + 1] = o[3]; p3[3 * k + 2] = o[4]
                sx[k] = o[0]; sy[k] = o[1]
                verts[2 * k] = o[0]; verts[2 * k + 1] = o[1]
            }
        }
        // the light (from the upper left, in front), and the photo's shadow and yellow cast until the band of light
        // has wiped them off: one colour per point, multiplied into the page
        val lx = -0.38f; val ly = -0.55f; val lz = -0.74f
        val ln = hypot(hypot(lx, ly), lz)
        val flatDot = -lz / ln
        val cut = if (vp < 0.92f) -0.2f else mix(-0.05f, 1.2f, clamp((vp - 0.92f) / 0.03f))
        for (j in 0..NY) for (i in 0..NX) {
            val k = j * (NX + 1) + i
            val kl = j * (NX + 1) + max(0, i - 1)
            val kr = j * (NX + 1) + min(NX, i + 1)
            val ku = max(0, j - 1) * (NX + 1) + i
            val kd = min(NY, j + 1) * (NX + 1) + i
            val ax = p3[3 * kr] - p3[3 * kl]; val ay = p3[3 * kr + 1] - p3[3 * kl + 1]; val az = p3[3 * kr + 2] - p3[3 * kl + 2]
            val bx = p3[3 * kd] - p3[3 * ku]; val by = p3[3 * kd + 1] - p3[3 * ku + 1]; val bz = p3[3 * kd + 2] - p3[3 * ku + 2]
            var nx = -(ay * bz - az * by); var ny = -(az * bx - ax * bz); var nz = -(ax * by - ay * bx)
            val nl = hypot(hypot(nx, ny), nz).takeIf { it > 0f } ?: 1f
            nx /= nl; ny /= nl; nz /= nl
            val shade = clamp(1 + 1.3f * ((nx * lx + ny * ly + nz * lz) / ln - flatDot), 0.5f, 1.15f)
            val u = i.toFloat() / NX
            val v = j.toFloat() / NY
            val keep = 1 - sm(cut - 0.14f, cut, u)     // 1: the band has passed, the page's own colours
            val g = 1 - sm(0f, 0.4f, u)
            val side = sm(0.1f, 1.3f, u * 0.55f + v * 0.75f)
            val sh = 1 - gutter * 0.42f * g * g - (1 - gutter) * 0.24f * side - 0.1f * sm(0.45f, 1f, v)
            val rr = min(1f, shade * mix(sh * 0.95f, 1f, keep))
            val gg = min(1f, shade * mix(sh * 0.965f * 0.95f, 1f, keep))
            val bb = min(1f, shade * mix(sh * 0.9f * 0.95f, 1f, keep))
            colors[k] = android.graphics.Color.argb(255, (rr * 255).toInt(), (gg * 255).toInt(), (bb * 255).toInt())
        }
    }

    // ---- drawing ----
    private val meshPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND }
    private val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val path = android.graphics.Path()
    // made once, not per frame: each native object a frame allocates brings on a blocking GC while the core runs
    private val rect = RectF()
    private val ironBody = android.graphics.Path().apply {
        moveTo(-28f, 10f); lineTo(28f, 10f); lineTo(18f, -6f); cubicTo(14f, -12f, 9f, -14f, 2f, -14f); lineTo(-28f, -14f); close()
    }
    private val ironHandle = android.graphics.Path().apply { moveTo(-18f, -14f); cubicTo(-18f, -26f, -12f, -28f, -4f, -28f); lineTo(10f, -28f); lineTo(10f, -14f) }
    // the sheen behind the iron: white to clear along x 0..1, placed by its local matrix
    private val sheen = LinearGradient(0f, 0f, 1f, 0f, android.graphics.Color.argb(115, 255, 255, 255), 0x00FFFFFF, Shader.TileMode.CLAMP)
    private val sheenAt = Matrix()
    private var blur: BlurMaskFilter? = null
    private var blurR = 0f

    private fun outline(): android.graphics.Path {
        path.reset()
        for (i in 0..NX) if (i == 0) path.moveTo(sx[i], sy[i]) else path.lineTo(sx[i], sy[i])
        for (j in 1..NY) { val k = j * (NX + 1) + NX; path.lineTo(sx[k], sy[k]) }
        for (i in NX - 1 downTo 0) { val k = NY * (NX + 1) + i; path.lineTo(sx[k], sy[k]) }
        for (j in NY - 1 downTo 1) { val k = j * (NX + 1); path.lineTo(sx[k], sy[k]) }
        path.close()
        return path
    }

    private fun line(c: android.graphics.Canvas, n: Int, p: (Float) -> Offset) {
        path.reset()
        for (s in 0..n) { val q = p(s.toFloat() / n); if (s == 0) path.moveTo(q.x, q.y) else path.lineTo(q.x, q.y) }
        c.drawPath(path, stroke)
    }

    private fun frameCorners(): List<Offset> {
        val fw = w * 0.94f; val fh = h * 0.9f; val cx = w / 2; val cy = h / 2
        val cs = cos(FRAME_ROT); val sn = sin(FRAME_ROT); val ins = 16f * w / 340f
        return listOf(-1 to -1, 1 to -1, 1 to 1, -1 to 1).map { (a, b) ->
            val x = a * (fw / 2 - ins); val y = b * (fh / 2 - ins)
            Offset(cx + x * cs - y * sn, cy + x * sn + y * cs)
        }
    }

    private fun sparkle(c: android.graphics.Canvas, x: Float, y: Float, r: Float, a: Float) {
        if (a <= 0.01f) return
        fill.color = 0xFFF2B705.toInt(); fill.alpha = (a * 255).toInt().coerceIn(0, 255); fill.shader = null
        path.reset()
        for (i in 0 until 8) {
            val ang = i * PI.toFloat() / 4
            val rad = if (i % 2 == 1) r * 0.28f else r
            if (i == 0) path.moveTo(x + rad * cos(ang), y + rad * sin(ang)) else path.lineTo(x + rad * cos(ang), y + rad * sin(ang))
        }
        path.close()
        c.drawPath(path, fill)
    }

    private fun puff(c: android.graphics.Canvas, x: Float, y: Float, r: Float, a: Float, s: Float) {
        if (a <= 0.01f) return
        fill.shader = null
        fill.color = android.graphics.Color.argb((0.6f * a * 255).toInt().coerceIn(0, 255), 255, 255, 255)
        c.drawCircle(x, y, r, fill)
        stroke.color = android.graphics.Color.argb((0.55f * a * 255).toInt().coerceIn(0, 255), 110, 116, 130)
        stroke.strokeWidth = 1.2f * max(0.8f, s)
        stroke.pathEffect = null
        c.drawCircle(x, y, r, stroke)
    }

    /** The app's iron (Doodles.drawIron), [face] flipping it as it turns, lifted [lift] off the page with its shadow. */
    private fun drawIron(c: android.graphics.Canvas, x: Float, y: Float, s: Float, a: Float, face: Float, lift: Float, accent: Int) {
        val f = if (abs(face) < 0.12f) (if (face < 0) -0.12f else 0.12f) else face
        val al = (a * 255).toInt().coerceIn(0, 255)
        fill.shader = null
        fill.color = android.graphics.Color.argb(((0.24f - min(0.14f, lift * 0.006f)) * a * 255).toInt().coerceIn(0, 255), 20, 25, 40)
        val rx = 31 * s * max(0.35f, abs(f))
        rect.set(x + lift * 0.3f - rx, y + 11 * s + lift * 0.4f - 5.5f * s, x + lift * 0.3f + rx, y + 11 * s + lift * 0.4f + 5.5f * s)
        c.drawOval(rect, fill)
        c.save()
        c.translate(x, y - lift)
        c.scale(f * s, s)
        val ink = Palette.Ink.toArgb()
        stroke.pathEffect = null
        stroke.strokeWidth = 2.4f
        stroke.color = ink; stroke.alpha = al
        val body = ironBody
        fill.color = android.graphics.Color.WHITE; fill.alpha = al
        c.drawPath(body, fill)
        c.drawPath(body, stroke)
        fill.color = accent; fill.alpha = al
        c.drawRect(-28f, 4f, 28f, 10f, fill)
        c.drawRect(-28f, 4f, 28f, 10f, stroke)
        c.drawPath(ironHandle, stroke)
        fill.color = Palette.Highlighter.toArgb(); fill.alpha = al
        c.drawCircle(-8f, -4f, 3.2f, fill)
        stroke.strokeWidth = 1.2f
        c.drawCircle(-8f, -4f, 3.2f, stroke)
        c.restore()
    }

    fun draw(
        c: android.graphics.Canvas, width: Float, height: Float, tex: Bitmap, photo: Bitmap?, corners: List<Offset>, photoWidth: Int,
        accent: Int, muted: Int, reduced: Boolean,
    ) {
        w = width; h = height
        val lv = if (live) 1f else 0f
        val tracing = vp >= 0.3f
        computeMesh(tracing, lv)
        val s = ppu / 260f
        val u1 = w / 340f                                   // the design's px at this size

        // under: the photo on the desk, then the page's shadow
        val photoA = 1 - sm(0.07f, 0.16f, vp)
        if (photoA > 0.01f) {
            val fw = w * 0.94f; val fh = h * 0.9f
            c.save()
            c.translate(w / 2, h / 2)
            c.rotate(FRAME_ROT * 180f / PI.toFloat())
            fill.shader = null
            fill.color = 0xFFFBFBF8.toInt(); fill.alpha = (photoA * 255).toInt()
            fill.setShadowLayer(14 * u1, 0f, 4 * u1, android.graphics.Color.argb((0.25f * photoA * 255).toInt(), 20, 25, 40))
            c.drawRoundRect(RectF(-fw / 2 - 6 * u1, -fh / 2 - 6 * u1, fw / 2 + 6 * u1, fh / 2 + 6 * u1), 4 * u1, 4 * u1, fill)
            fill.clearShadowLayer()
            val inner = RectF(-fw / 2, -fh / 2, fw / 2, fh / 2)
            c.clipRect(inner)
            fill.color = 0xFF3A332D.toInt(); fill.alpha = (photoA * 255).toInt()
            c.drawRect(inner, fill)
            if (photo != null && !photo.isRecycled) {
                // the photo filling the print (cover)
                val k = max(fw / photo.width, fh / photo.height)
                val pw = photo.width * k; val phh = photo.height * k
                bmpPaint.alpha = (photoA * 255).toInt()
                c.drawBitmap(photo, null, RectF(-pw / 2, -phh / 2, pw / 2, phh / 2), bmpPaint)
                // a little darker than the page that peels off it
                fill.color = android.graphics.Color.argb((0.18f * photoA * 255).toInt(), 20, 18, 16)
                c.drawRect(inner, fill)
                // where the page was: empty once it has popped off (the photo is the real one, the page would be twice)
                val gone = sm(0.09f, 0.15f, vp)
                if (corners.size == 4 && gone > 0f) {
                    val kk = if (photoWidth > 0) photo.width.toFloat() / photoWidth * k else k
                    path.reset()
                    corners.forEachIndexed { i, q ->
                        val x = q.x * kk - pw / 2; val y = q.y * kk - phh / 2
                        if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
                    }
                    path.close()
                    fill.color = android.graphics.Color.argb((gone * photoA * 0.9f * 255).toInt().coerceIn(0, 255), 42, 37, 33)
                    c.drawPath(path, fill)
                }
            }
            c.restore()
        }
        val lift = clamp(sm(0.09f, 0.17f, vp)) * (1 - fin) + float * 0.3f * sin(time * 1.7f)
        fill.shader = null
        fill.color = 0xFFF2F0EA.toInt()
        fill.setShadowLayer((8 + 28 * lift) * s, 0f, (3 + 16 * lift) * s, android.graphics.Color.argb(((0.15f + 0.12f * clamp(lift)) * 255).toInt(), 20, 25, 40))
        c.drawPath(outline(), fill)
        fill.clearShadowLayer()

        // the sheet: the page on the mesh; the paper whitens as it lands
        meshPaint.colorFilter = if (fin > 0f) ColorMatrixColorFilter(ColorMatrix().apply {
            val k = 1 + 0.06f * fin
            val o = 10f * fin
            set(floatArrayOf(k, 0f, 0f, 0f, o, 0f, k, 0f, 0f, o, 0f, 0f, k, 0f, o, 0f, 0f, 0f, 1f, 0f))
        }) else null
        c.drawBitmapMesh(tex, NX, NY, verts, 0, colors, 0, meshPaint)

        // a pencil mesh drapes over the floating sheet, its dashes marching
        val gA = sm(0.1f, 0.13f, vp) * (1 - sm(0.26f, 0.31f, vp)) * 0.45f
        if (gA > 0.01f) {
            val reveal = sm(0.1f, 0.19f, vp)
            stroke.color = 0xFF6B7180.toInt(); stroke.alpha = (gA * 255).toInt(); stroke.strokeWidth = 1f * u1
            stroke.pathEffect = DashPathEffect(floatArrayOf(3 * s, 4 * s), if (live) time * 14 else 0f)
            for (i in 0..NX step 5) line(c, 20) { q -> at(i.toFloat() / NX, mix(0f, reveal, q)) }
            for (j in 0..NY step 7) if (j.toFloat() / NY <= reveal + 1e-6f) line(c, 14) { q -> at(q, j.toFloat() / NY) }
            stroke.pathEffect = null
        }

        // the iron: a warm sheen behind it, steam, its cord, and the iron itself
        if (penA > 0.01f && tracing && !reduced) {
            val iv = iron()
            val ex = penExit
            val isz = 0.26f * ppu * sc / 60f
            if (mode == 0 && penPos < l) {
                val pen = penAt(penPos)
                if (pen.f <= RULE) {
                    val d = dirOf(pen.k)
                    val rear = ironU(pen.k, pen.tf) - d * HW
                    val tail = rear - d * 0.45f
                    val a = clamp(rear); val b = clamp(tail)
                    val v0 = bandV[pen.k] - 0.5f / l; val v1 = bandV[pen.k] + 0.5f / l
                    if (abs(a - b) > 0.01f) {
                        val pa = flatAt(a, bandV[pen.k]); val pb = flatAt(b, bandV[pen.k])
                        sheenAt.setScale(hypot(pb.x - pa.x, pb.y - pa.y), 1f)
                        sheenAt.postRotate(atan2(pb.y - pa.y, pb.x - pa.x) * 180f / PI.toFloat())
                        sheenAt.postTranslate(pa.x, pa.y)
                        sheen.setLocalMatrix(sheenAt)
                        fill.shader = sheen
                        fill.color = android.graphics.Color.WHITE
                        fill.alpha = (penA * 255).toInt().coerceIn(0, 255)
                        val q = listOf(flatAt(a, v0), flatAt(b, v0), flatAt(b, v1), flatAt(a, v1))
                        path.reset(); path.moveTo(q[0].x, q[0].y); for (p in q.drop(1)) path.lineTo(p.x, p.y); path.close()
                        c.drawPath(path, fill)
                        fill.shader = null
                    }
                }
            }
            for (pf in puffs) {
                val p = flatAt(pf.u, pf.v)
                puff(c, p.x + pf.drift * pf.age * 16 * s, p.y - (10 + pf.age * 36) * s, (3 + pf.age * 6) * s, (1 - pf.age / 1.1f) * 0.8f * penA, s)
            }
            val p = flatAt(iv.u, iv.v)
            val x = p.x + ex * 90 * s
            val y = p.y + 2 * isz - ex * 120 * s
            val lf = iv.lift * s + ex * 30 * s
            val a = penA * (1 - ex)
            val fc = if (abs(iv.face) < 0.12f) (if (iv.face < 0) -0.12f else 0.12f) else iv.face
            val rx = x - fc * 27 * isz; val ry = y - lf - 2 * isz
            stroke.color = muted; stroke.alpha = (a * 255).toInt().coerceIn(0, 255); stroke.strokeWidth = 2 * max(0.8f, s)
            path.reset(); path.moveTo(rx, ry)
            path.cubicTo(rx - fc * 26 * isz, ry + 20 * isz, w * 0.88f, h * 0.3f + 6 * sin(time * 1.3f) * u1, w + 14 * u1, h * 0.1f)
            c.drawPath(path, stroke)
            drawIron(c, x, y, isz, a, iv.face, lf, accent)
        }

        // composing: the iron lifts off and the page gives a springy final thump
        val ct = clamp((vp - 0.85f) / 0.05f)
        if (ct > 0f && ct < 1f) {
            fill.shader = null
            fill.color = android.graphics.Color.argb((0.28f * sin(PI.toFloat() * ct) * 255).toInt().coerceIn(0, 255), 255, 255, 255)
            c.drawPath(outline(), fill)
            if (live) listOf(0f to 0f, 0.5f to -0.02f, 1f to 0f, 1.03f to 0.5f, 1f to 1f, 0.5f to 1.02f, 0f to 1f, -0.03f to 0.5f).forEachIndexed { i, (u, v) ->
                val q = at(clamp(u), clamp(v))
                val dx = q.x - w / 2; val dy = q.y - h / 2
                val dl = hypot(dx, dy).takeIf { it > 0f } ?: 1f
                val o = (6 + 14 * ct) * s
                sparkle(c, q.x + dx / dl * o, q.y + dy / dl * o, (4 + (i % 2) * 2) * s, sin(PI.toFloat() * ct))
            }
        }

        // corner brackets: from the photo's edges onto the page, back as crop marks at the end
        val bA = 1 - sm(0.17f, 0.24f, vp)
        val endT = sm(0.955f, 1f, vp)
        if (bA > 0.01f || endT > 0.01f) {
            val e = backOut(clamp((vp - 0.012f) / 0.045f))
            val pc = listOf(Offset(sx[0], sy[0]), Offset(sx[NX], sy[NX]), Offset(sx[NV - 1], sy[NV - 1]), Offset(sx[NY * (NX + 1)], sy[NY * (NX + 1)]))
            val fcs = frameCorners()
            val ending = bA <= 0.01f
            val q = pc.mapIndexed { i, p -> if (ending) p else if (vp <= 0.012f) fcs[i] else Offset(mix(fcs[i].x, p.x, e), mix(fcs[i].y, p.y, e)) }
            val waiting = if (vp < 0.012f && live) 0.55f + 0.45f * sin(time * 6) else 1f
            stroke.color = accent; stroke.strokeWidth = 2.6f * max(0.8f, s); stroke.pathEffect = null
            stroke.alpha = (((if (ending) clamp(endT * 2) else bA) * waiting) * 255).toInt().coerceIn(0, 255)
            val arm = 16 * max(0.8f, s)
            val off = if (ending) 8 * s * (if (live) backOut(endT) else 1f) else 0f
            for (i in 0 until 4) {
                val p = q[i]; val a = q[(i + 1) % 4]; val b = q[(i + 3) % 4]
                var ax = a.x - p.x; var ay = a.y - p.y; var bx = b.x - p.x; var by = b.y - p.y
                val la = hypot(ax, ay).takeIf { it > 0f } ?: 1f
                val lb = hypot(bx, by).takeIf { it > 0f } ?: 1f
                ax /= la; ay /= la; bx /= lb; by /= lb
                val ox = p.x - (ax + bx) * off; val oy = p.y - (ay + by) * off
                path.reset(); path.moveTo(ox + ax * arm, oy + ay * arm); path.lineTo(ox, oy); path.lineTo(ox + bx * arm, oy + by * arm)
                c.drawPath(path, stroke)
            }
        }

        // resampling: the last of the steam lifts off, top to bottom, and leaves the page sharp
        val sv = clamp((vp - 0.9f) / 0.02f)
        if (sv > 0f && sv < 1f && !reduced) {
            val v = mix(-0.02f, 1.04f, sv)
            val a = sin(PI.toFloat() * sv)
            for (i in 0 until 9) {
                val u = (i + 0.5f) / 9 + 0.03f * sin(i * 7.3f)
                val p = at(u, clamp(v))
                puff(c, p.x, p.y - (2 + (i % 3) * 3) * s, (6 + 3 * sin(i * 3.1f)) * s, 0.75f * a, s)
            }
            for (sp in spark) {
                val d = v - sp[1]
                if (d > 0f && d < 0.22f) {
                    val p = at(sp[0], sp[1]); val k = d / 0.22f
                    puff(c, p.x + sin(sp[2] * 9) * 8 * s * k, p.y - k * 46 * s, (4 + 7 * k) * sp[2] * s, (1 - k) * 0.7f, s)
                }
            }
        }

        // removing shadows: a band of light
        val cutT = clamp((vp - 0.92f) / 0.03f)
        if (cutT > 0f && cutT < 1f && !reduced) {
            val u = mix(-0.05f, 1.2f, cutT) - 0.06f
            if (u > 0f && u < 1f) {
                stroke.color = android.graphics.Color.WHITE; stroke.alpha = (0.55f * sin(PI.toFloat() * cutT) * 255).toInt().coerceIn(0, 255)
                stroke.strokeWidth = 12 * s
                if (blur == null || blurR != 4 * s) { blurR = 4 * s; blur = BlurMaskFilter(blurR, BlurMaskFilter.Blur.NORMAL) }
                stroke.maskFilter = blur
                line(c, 6) { q -> at(u, q) }
                stroke.maskFilter = null
                for (sp in spark) {
                    val d = u - sp[0]
                    if (d > 0f && d < 0.16f) { val p = at(sp[0], sp[1]); sparkle(c, p.x, p.y, 5 * sp[2] * s, sin(PI.toFloat() * d / 0.16f)) }
                }
            }
        }

        // finishing: a puff of dust where the page lands
        val pt = clamp((fin - 0.55f) / 0.45f)
        if (pt > 0f && pt < 1f && live) {
            val pc = listOf(Offset(sx[0], sy[0]), Offset(sx[NX], sy[NX]), Offset(sx[NV - 1], sy[NV - 1]), Offset(sx[NY * (NX + 1)], sy[NY * (NX + 1)]))
            stroke.color = muted; stroke.strokeWidth = 1.3f * max(0.8f, s); stroke.pathEffect = null
            for (p in pc) {
                val base = atan2(p.y - h / 2, p.x - w / 2)
                for (j in 0 until 3) {
                    val ang = base + (j - 1) * 0.55f
                    val dist = (6 + pt * (16 + 6 * j)) * s
                    stroke.alpha = ((1 - pt) * 0.7f * 255).toInt().coerceIn(0, 255)
                    c.drawCircle(p.x + cos(ang) * dist, p.y + sin(ang) * dist, (1.5f + pt * (4 + j)) * s, stroke)
                }
            }
        }
    }
}

