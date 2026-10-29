package com.albus.flatscan

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import androidx.compose.ui.geometry.Offset
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The shape of the finished page. Since round 17 the core renders the page at the paper's measured
 * shape (A4 or US Letter when within 4 % of one: PipelineOptions::render_true_shape; before, always
 * A4); the shapes resize the finished page on export and on screen when it differs (page fractions,
 * so protected, erased and ignored areas stay where they are).
 */
enum class PageShape(val label: String, val caption: String) {
    Auto("Auto", "A4 or US Letter when the page is close to one, otherwise its own proportions"),
    AsScanned("A4", "A4 proportions (210 x 297 mm)"),   // the name is stored: pages saved before shapes stay A4
    Measured("As photographed", "The page's own proportions, measured from its corners (receipts, cards, letters)"),
    Letter("US Letter", "8.5 x 11 in"),
    IdCard("ID card", "85.6 x 54 mm (credit card size)"),
}

/**
 * Per-page look applied after the core's filter, in Kotlin: light and colour sliders (a colour
 * matrix: free on screen, one pass on export) and the page shape. All sliders are -1..1 with 0 =
 * unchanged; at the defaults nothing runs and the page is byte-identical to the core's render.
 * [measuredHw]: height / width of the page in the page's own (unrotated) frame, measured from the
 * photo corners when the page was processed (0: unknown).
 */
data class PageLook(
    val brightness: Float = 0f,
    val contrast: Float = 0f,
    val saturation: Float = 0f,
    val warmth: Float = 0f,
    val shape: PageShape = PageShape.AsScanned,
    val measuredHw: Float = 0f,
    val transform: PageTransform = PageTransform(),   // the page's perspective (Page > Perspective), after the shape
) {
    val colorIdentity: Boolean get() = brightness == 0f && contrast == 0f && saturation == 0f && warmth == 0f

    /** The look with the user's colour choices of [other] (shape and measurement kept). */
    fun withColorsOf(other: PageLook) = copy(brightness = other.brightness, contrast = other.contrast, saturation = other.saturation, warmth = other.warmth)

    /**
     * The 4 x 5 colour matrix (android.graphics.ColorMatrix layout, offsets in 0..255), or null at the
     * defaults. Order: saturation, contrast around mid-grey, brightness, warmth.
     */
    fun colorMatrix(): FloatArray? {
        if (colorIdentity) return null
        val m = ColorMatrix()
        if (saturation != 0f) m.setSaturation(1f + saturation)
        if (contrast != 0f) {
            val s = 2f.pow(contrast)            // 0.5x .. 2x
            val t = 127.5f * (1f - s)
            m.postConcat(ColorMatrix(floatArrayOf(s, 0f, 0f, 0f, t, 0f, s, 0f, 0f, t, 0f, 0f, s, 0f, t, 0f, 0f, 0f, 1f, 0f)))
        }
        if (brightness != 0f) {
            val t = 64f * brightness
            m.postConcat(ColorMatrix(floatArrayOf(1f, 0f, 0f, 0f, t, 0f, 1f, 0f, 0f, t, 0f, 0f, 1f, 0f, t, 0f, 0f, 0f, 1f, 0f)))
        }
        if (warmth != 0f) {
            val r = 1f + 0.10f * warmth
            val b = 1f - 0.16f * warmth
            val g = 1f + 0.02f * warmth
            m.postConcat(ColorMatrix(floatArrayOf(r, 0f, 0f, 0f, 0f, 0f, g, 0f, 0f, 0f, 0f, 0f, b, 0f, 0f, 0f, 0f, 0f, 1f, 0f)))
        }
        return m.array.copyOf()
    }

    /**
     * Height / width the finished page has in its own frame for a core page of [w] x [h], or null for
     * the core's own shape.
     */
    fun targetHw(w: Int, h: Int): Float? {
        if (w <= 0 || h <= 0) return null
        val portrait = w <= h
        return when (shape) {
            PageShape.Auto -> autoHw(w, h)
            PageShape.AsScanned -> if (portrait) 297f / 210f else 210f / 297f
            PageShape.Measured -> measuredHw.takeIf { it > 0f }
            PageShape.Letter -> if (portrait) 11f / 8.5f else 8.5f / 11f
            PageShape.IdCard -> if (portrait) 85.6f / 54f else 54f / 85.6f
        }?.takeIf { abs(it * w / h - 1f) > 0.005f }
    }

    // the core's page when the measured shape is within kSnap of it, US Letter when within kSnap of that, else
    // the measured shape: a cropped page, a coupon or a receipt is not stretched to A4, and the small error of a
    // measurement from corners does not make an A4 sheet a little off A4. A page from a round-17 core already has
    // this shape (it snaps the same way), so nothing is resampled; a page from an earlier core is A4 and is
    // reshaped here
    private fun autoHw(w: Int, h: Int): Float? {
        val m = measuredHw.takeIf { it > 0f } ?: return null
        val letter = if (w <= h) 11f / 8.5f else 8.5f / 11f
        return when {
            abs(m * w / h - 1f) < kSnap -> null
            abs(m / letter - 1f) < kSnap -> letter
            else -> m
        }
    }

    /**
     * The finished page size for a core page of [w] x [h]: the same number of pixels at the target
     * shape (a receipt gets taller and narrower), or [w] x [h].
     */
    fun outputSize(w: Int, h: Int): Pair<Int, Int> {
        val hw = targetHw(w, h) ?: return w to h
        val area = w.toDouble() * h
        val nw = sqrt(area / hw).roundToInt().coerceAtLeast(1)
        return nw to (nw * hw).roundToInt().coerceAtLeast(1)
    }

    fun toJson(): JSONObject = JSONObject().put("brightness", brightness.toDouble()).put("contrast", contrast.toDouble())
        .put("saturation", saturation.toDouble()).put("warmth", warmth.toDouble()).put("shape", shape.name)
        .put("measured_hw", measuredHw.toDouble())
        .apply { if (transform != PageTransform()) put("transform", transform.toJson()) }

    companion object {
        private const val kSnap = 0.04f

        fun fromJson(j: JSONObject?): PageLook {
            if (j == null) return PageLook()
            fun f(k: String) = j.optDouble(k, 0.0).toFloat().takeIf { !it.isNaN() } ?: 0f
            return PageLook(
                brightness = f("brightness").coerceIn(-1f, 1f), contrast = f("contrast").coerceIn(-1f, 1f),
                saturation = f("saturation").coerceIn(-1f, 1f), warmth = f("warmth").coerceIn(-1f, 1f),
                shape = PageShape.entries.firstOrNull { it.name == j.optString("shape") } ?: PageShape.AsScanned,
                measuredHw = f("measured_hw").coerceIn(0f, 8f),
                transform = PageTransform.fromJson(j.optJSONObject("transform")),
            )
        }
    }
}

/**
 * The finished page for export and thumbnails: the core's render [src] with [look] (shape, perspective, colour)
 * turned [rotation] clockwise quarter turns. At the default look this is exactly [rotatedCopy].
 */
fun finishPage(src: Bitmap, rotation: Int, look: PageLook): Bitmap {
    val cm = look.colorMatrix()
    val (w, h) = look.outputSize(src.width, src.height)
    val t = look.transform.matrix(h.toFloat() / w)
    if (cm == null && t == null && w == src.width && h == src.height) return src.rotatedCopy(rotation)
    val q = ((rotation % 4) + 4) % 4
    val outW = if (q % 2 == 0) w else h
    val outH = if (q % 2 == 0) h else w
    val out = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
    val m = Matrix()
    m.postScale(w.toFloat() / src.width, h.toFloat() / src.height)
    if (t != null) m.postConcat(pagePxMatrix(t, w.toFloat(), h.toFloat()))
    m.postTranslate(-w / 2f, -h / 2f)
    m.postRotate(90f * q)
    m.postTranslate(outW / 2f, outH / 2f)
    val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    if (cm != null) paint.colorFilter = ColorMatrixColorFilter(cm)
    val canvas = Canvas(out)
    // a perspective change can bare a corner of the page: the paper's colour there, as the core fills beyond the page
    if (t != null) canvas.drawRect(0f, 0f, outW.toFloat(), outH.toFloat(), Paint(paint).apply { color = paperColor(src) })
    canvas.drawBitmap(src, m, paint)
    return out
}

/** [t] (PageTransform.matrix: page fractions to page fractions) as a matrix on a page of [w] x [h] px. */
fun pagePxMatrix(t: DoubleArray, w: Float, h: Float): Matrix {
    val d = doubleArrayOf(w.toDouble(), 0.0, 0.0, 0.0, h.toDouble(), 0.0, 0.0, 0.0, 1.0)
    val dInv = doubleArrayOf(1.0 / w, 0.0, 0.0, 0.0, 1.0 / h, 0.0, 0.0, 0.0, 1.0)
    val p = PageTransform.mul(d, PageTransform.mul(t, dInv))
    return Matrix().apply { setValues(FloatArray(9) { p[it].toFloat() }) }
}

/** The page's paper colour: the median of 64 points along each side, 2 % in from its edges. */
fun paperColor(b: Bitmap): Int {
    if (b.width < 4 || b.height < 4) return android.graphics.Color.WHITE
    val r = ArrayList<Int>(); val g = ArrayList<Int>(); val bl = ArrayList<Int>()
    val ix = (b.width * 0.02f).toInt().coerceAtLeast(1)
    val iy = (b.height * 0.02f).toInt().coerceAtLeast(1)
    for (k in 0 until 64) {
        val fx = ix + (b.width - 1 - 2 * ix) * k / 63
        val fy = iy + (b.height - 1 - 2 * iy) * k / 63
        for (c in intArrayOf(b.getPixel(fx, iy), b.getPixel(fx, b.height - 1 - iy), b.getPixel(ix, fy), b.getPixel(b.width - 1 - ix, fy))) {
            r += android.graphics.Color.red(c); g += android.graphics.Color.green(c); bl += android.graphics.Color.blue(c)
        }
    }
    fun med(v: ArrayList<Int>) = v.sorted()[v.size / 2]
    return android.graphics.Color.rgb(med(r), med(g), med(bl))
}

/**
 * Height / width of the page outlined by [corners] (TL, TR, BR, BL, photo px) as the rectangle it
 * is in the world: Zhang and He's rectangle-from-perspective method ("Whiteboard scanning and image
 * enhancement", 2007) with the principal point at the photo centre ([photoW] x [photoH]) and the
 * focal length [focalPx] (estimated from the quad when 0). Falls back to the mean side lengths.
 */
fun measuredPageHw(corners: List<Offset>, photoW: Int, photoH: Int, focalPx: Float): Float {
    if (corners.size != 4) return 0f
    val (tl, tr, br, bl) = corners
    val top = hypot(tr.x - tl.x, tr.y - tl.y)
    val bottom = hypot(br.x - bl.x, br.y - bl.y)
    val left = hypot(bl.x - tl.x, bl.y - tl.y)
    val right = hypot(br.x - tr.x, br.y - tr.y)
    if (top + bottom < 1f) return 0f
    val simple = ((left + right) / (top + bottom)).toDouble()
    val u0 = photoW / 2.0
    val v0 = photoH / 2.0
    // homogeneous image points: m1 TL, m2 TR, m3 BL, m4 BR (the paper's numbering)
    val m1 = doubleArrayOf(tl.x.toDouble(), tl.y.toDouble(), 1.0)
    val m2 = doubleArrayOf(tr.x.toDouble(), tr.y.toDouble(), 1.0)
    val m3 = doubleArrayOf(bl.x.toDouble(), bl.y.toDouble(), 1.0)
    val m4 = doubleArrayOf(br.x.toDouble(), br.y.toDouble(), 1.0)
    fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    val d2 = dot(cross(m2, m4), m3)
    val d3 = dot(cross(m3, m4), m2)
    if (abs(d2) < 1e-9 || abs(d3) < 1e-9) return simple.toFloat()
    val k2 = dot(cross(m1, m4), m3) / d2
    val k3 = dot(cross(m1, m4), m2) / d3
    val n2 = DoubleArray(3) { k2 * m2[it] - m1[it] }
    val n3 = DoubleArray(3) { k3 * m3[it] - m1[it] }
    // the camera correction always applies: a page tilted one way only (no convergence across) still has its
    // other direction foreshortened (skipping it made an A4 tilted 15 deg 1.62 instead of 1.41). The focal length:
    // the given one, else from the quad when both directions converge enough to tell, else 0.72 x the long side
    // (surface.cpp's default; the result depends little on it)
    val longSide = maxOf(photoW, photoH).toDouble()
    fun converges(n: DoubleArray) = abs(n[2]) * longSide / maxOf(1e-9, hypot(n[0], n[1])) > 0.02
    var f2 = focalPx.toDouble().let { it * it }
    if (f2 <= 0 && converges(n2) && converges(n3)) {
        f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0) +
            (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)) / (n2[2] * n3[2])
    }
    // a quad that barely converges measures the focal length badly (0.29 to 1.78 x the long side on real photos;
    // phone cameras are about 0.7 - 1.0, 26 mm equivalent: 0.72): outside 0.5 - 2 x, the default
    if (focalPx <= 0f && !(f2 >= 0.25 * longSide * longSide && f2 <= 4.0 * longSide * longSide)) f2 = (0.72 * longSide).let { it * it }
    if (!(f2 > 0)) f2 = (0.72 * longSide).let { it * it }
    // n^T (A^-T A^-1) n with A = [[f, 0, u0], [0, f, v0], [0, 0, 1]]
    fun q(n: DoubleArray): Double {
        val x = (n[0] - u0 * n[2])
        val y = (n[1] - v0 * n[2])
        return (x * x + y * y) / f2 + n[2] * n[2]
    }
    val ratioWh = sqrt(q(n2) / q(n3))
    val hw = 1.0 / ratioWh
    // a wild estimate (near-degenerate quad) is worse than the side lengths
    if (hw.isNaN() || hw <= 0 || abs(kotlin.math.ln(hw / simple)) > 0.7) return simple.toFloat()
    return hw.toFloat().coerceIn(0.15f, 6f)
}
