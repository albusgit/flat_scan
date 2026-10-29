package com.albus.flatscan

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/**
 * The finished page's perspective, as Lightroom's Transform panel: a view of the page through a virtual camera that
 * can be tilted ([vertical]: about the page's horizontal axis, the top widened for > 0; [horizontal]: about its
 * vertical axis, the left widened for > 0; each -1..1 = -30..30 degrees), then turned [rotate] degrees (-10..10),
 * stretched ([aspect] -1..1: wider for > 0), zoomed ([scale] -1..1: 0.5x .. 1.5x) and shifted ([offsetX], [offsetY]
 * -1..1: a quarter of the page each way). [quad] (TL, TR, BR, BL as page fractions x, y, or null): the part of the page
 * that becomes the whole page first (Upright, Guided, the corner warp). [constrain]: zoomed in just enough that no
 * blank corner shows (Lightroom's Constrain crop). [upright]: the Upright mode that set the sliders (-1 none; as
 * core/src/upright.h UprightMode), [guides]: Guided's lines (x0, y0, x1, y1 page fractions of the core's page,
 * repeated). Applied in Kotlin after the core's render, as the page shape: on
 * screen while no area tool is open, and on export ([finishPage]); the area tools edit the page as the core rendered it.
 */
data class PageTransform(
    val vertical: Float = 0f,
    val horizontal: Float = 0f,
    val rotate: Float = 0f,
    val aspect: Float = 0f,
    val scale: Float = 0f,
    val offsetX: Float = 0f,
    val offsetY: Float = 0f,
    val quad: List<Float>? = null,
    val constrain: Boolean = false,
    val upright: Int = -1,
    val guides: List<Float> = emptyList(),
) {
    val isIdentity: Boolean
        get() = vertical == 0f && horizontal == 0f && rotate == 0f && aspect == 0f && scale == 0f && offsetX == 0f && offsetY == 0f && quad == null

    /** The sliders back to zero ([quad] and [constrain] kept). */
    fun slidersReset() = copy(vertical = 0f, horizontal = 0f, rotate = 0f, aspect = 0f, scale = 0f, offsetX = 0f, offsetY = 0f)

    /**
     * The forward map from page fractions of the core's page to page fractions of the finished page, as a 3 x 3
     * homography (row major), for a page of height / width [hw]; null when it is the identity.
     */
    fun matrix(hw: Float): DoubleArray? {
        if (isIdentity || hw <= 0f) return null
        val h = hw.toDouble()
        // page fractions <-> metric coordinates (page widths) around the page's centre
        val m = doubleArrayOf(1.0, 0.0, -0.5, 0.0, h, -0.5 * h, 0.0, 0.0, 1.0)
        val mInv = doubleArrayOf(1.0, 0.0, 0.5, 0.0, 1 / h, 0.5, 0.0, 0.0, 1.0)
        var t = identity()
        quad?.takeIf { it.size == 8 }?.let { q ->
            val src = DoubleArray(8) { i -> if (i % 2 == 0) q[i] - 0.5 else (q[i] - 0.5) * h }
            val dst = doubleArrayOf(-0.5, -0.5 * h, 0.5, -0.5 * h, 0.5, 0.5 * h, -0.5, 0.5 * h)
            homography(src, dst)?.let { t = it }
        }
        val f = CAMERA
        val tv = vertical.coerceIn(-1f, 1f) * MAX_TILT
        val th = horizontal.coerceIn(-1f, 1f) * MAX_TILT
        // a tilted plane seen from f page widths (the centre keeps its scale): the side tilted away gets smaller
        t = mul(doubleArrayOf(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, sin(tv), f), t)
        t = mul(doubleArrayOf(f, 0.0, 0.0, 0.0, f, 0.0, sin(th), 0.0, f), t)
        val a = Math.toRadians(rotate.coerceIn(-10f, 10f).toDouble())
        t = mul(doubleArrayOf(cos(a), -sin(a), 0.0, sin(a), cos(a), 0.0, 0.0, 0.0, 1.0), t)
        val sa = 2.0.pow(0.3 * aspect.coerceIn(-1f, 1f))
        val s = 1 + 0.5 * scale.coerceIn(-1f, 1f)
        t = mul(doubleArrayOf(s * sa, 0.0, 0.25 * offsetX.coerceIn(-1f, 1f), 0.0, s / sa, 0.25 * h * offsetY.coerceIn(-1f, 1f), 0.0, 0.0, 1.0), t)
        if (constrain) t = mul(zoomToFill(t, h), t)
        return mul(mInv, mul(t, m)).let { r -> normalised(r) }
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("vertical", vertical.toDouble()); put("horizontal", horizontal.toDouble()); put("rotate", rotate.toDouble())
        put("aspect", aspect.toDouble()); put("scale", scale.toDouble())
        put("offset_x", offsetX.toDouble()); put("offset_y", offsetY.toDouble())
        quad?.let { q -> put("quad", JSONArray().also { a -> q.forEach { a.put(it.toDouble()) } }) }
        if (constrain) put("constrain", true)
        if (upright >= 0) put("upright", upright)
        if (guides.isNotEmpty()) put("guides", JSONArray().also { a -> guides.forEach { a.put(it.toDouble()) } })
    }

    companion object {
        /** The virtual camera's distance (page widths): a 53-degree view across the page. */
        private const val CAMERA = 1.0
        private val MAX_TILT = Math.toRadians(30.0)

        fun fromJson(j: JSONObject?): PageTransform {
            if (j == null) return PageTransform()
            fun f(k: String) = j.optDouble(k, 0.0).toFloat().takeIf { !it.isNaN() }?.coerceIn(-1f, 1f) ?: 0f
            val q = j.optJSONArray("quad")?.let { a -> List(a.length()) { a.optDouble(it, 0.0).toFloat() } }?.takeIf { it.size == 8 }
            return PageTransform(
                vertical = f("vertical"), horizontal = f("horizontal"),
                rotate = j.optDouble("rotate", 0.0).toFloat().takeIf { !it.isNaN() }?.coerceIn(-10f, 10f) ?: 0f,
                aspect = f("aspect"), scale = f("scale"), offsetX = f("offset_x"), offsetY = f("offset_y"),
                quad = q, constrain = j.optBoolean("constrain"), upright = j.optInt("upright", -1).coerceIn(-1, 4),
                guides = j.optJSONArray("guides")?.let { a -> List(a.length()) { a.optDouble(it, 0.0).toFloat() } }?.takeIf { it.size % 4 == 0 } ?: emptyList(),
            )
        }

        fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        fun mul(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { k ->
            val r = k / 3
            val c = k % 3
            a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c]
        }

        /** [h] applied to (x, y). */
        fun apply(h: DoubleArray, x: Double, y: Double): Pair<Double, Double> {
            val w = h[6] * x + h[7] * y + h[8]
            return (h[0] * x + h[1] * y + h[2]) / w to (h[3] * x + h[4] * y + h[5]) / w
        }

        fun invert(a: DoubleArray): DoubleArray? {
            val c00 = a[4] * a[8] - a[5] * a[7]
            val c01 = a[5] * a[6] - a[3] * a[8]
            val c02 = a[3] * a[7] - a[4] * a[6]
            val det = a[0] * c00 + a[1] * c01 + a[2] * c02
            if (abs(det) < 1e-12) return null
            val inv = doubleArrayOf(
                c00, a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
                c01, a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
                c02, a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
            )
            return DoubleArray(9) { inv[it] / det }
        }

        private fun normalised(h: DoubleArray) = if (abs(h[8]) > 1e-12) DoubleArray(9) { h[it] / h[8] } else h

        /** The homography taking the 4 points [src] (x0, y0, ... x3, y3) to [dst], or null when they are degenerate. */
        fun homography(src: DoubleArray, dst: DoubleArray): DoubleArray? {
            // h8 = 1: 8 equations in h0..h7
            val a = Array(8) { DoubleArray(9) }
            for (i in 0 until 4) {
                val x = src[2 * i]; val y = src[2 * i + 1]; val u = dst[2 * i]; val v = dst[2 * i + 1]
                a[2 * i] = doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u)
                a[2 * i + 1] = doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v)
            }
            for (c in 0 until 8) {   // Gaussian elimination with partial pivoting
                var p = c
                for (r in c + 1 until 8) if (abs(a[r][c]) > abs(a[p][c])) p = r
                if (abs(a[p][c]) < 1e-12) return null
                val tmp = a[c]; a[c] = a[p]; a[p] = tmp
                for (r in 0 until 8) {
                    if (r == c) continue
                    val k = a[r][c] / a[c][c]
                    for (j in c until 9) a[r][j] -= k * a[c][j]
                }
            }
            return DoubleArray(9) { if (it == 8) 1.0 else a[it][8] / a[it][it] }
        }

        /**
         * The zoom about the page's centre (metric coordinates, page height [h] page widths) that [t] needs so that the
         * whole finished page shows the core's page: the corners of the finished page, taken back through the zoomed
         * map, all fall inside the core's page. As a matrix (the identity when nothing is blank).
         */
        private fun zoomToFill(t: DoubleArray, h: Double): DoubleArray {
            val inv0 = invert(t) ?: return identity()
            fun fits(z: Double): Boolean {
                // corners of the output, through (zoom z o t)^-1 = t^-1 o zoom(1/z)
                for ((x, y) in listOf(-0.5 to -0.5 * h, 0.5 to -0.5 * h, 0.5 to 0.5 * h, -0.5 to 0.5 * h)) {
                    val (sx, sy) = apply(inv0, x / z, y / z)
                    if (sx < -0.5 - 1e-6 || sx > 0.5 + 1e-6 || sy < -0.5 * h - 1e-6 || sy > 0.5 * h + 1e-6) return false
                }
                return true
            }
            if (fits(1.0)) return identity()
            var lo = 1.0
            var hi = 8.0
            if (!fits(hi)) return identity()
            repeat(40) { val mid = (lo + hi) / 2; if (fits(mid)) hi = mid else lo = mid }
            return doubleArrayOf(hi, 0.0, 0.0, 0.0, hi, 0.0, 0.0, 0.0, 1.0)
        }

        /**
         * The transform whose [quad] makes the 4 page points [corners] (TL, TR, BR, BL page fractions) the page's corners,
         * sliders at zero: the corner warp, and the solved Upright modes.
         */
        fun fromQuad(corners: List<Float>, constrain: Boolean = false) = PageTransform(quad = corners.toList(), constrain = constrain)

        /** The quad of a forward map [h] (page fractions): where the finished page's corners come from on the core's page. */
        fun quadOf(h: DoubleArray): List<Float>? {
            val inv = invert(h) ?: return null
            return listOf(0.0 to 0.0, 1.0 to 0.0, 1.0 to 1.0, 0.0 to 1.0).flatMap { (x, y) ->
                val (u, v) = apply(inv, x, y)
                listOf(u.toFloat(), v.toFloat())
            }
        }

        /** Clamps a page fraction so a quad corner stays near the page (a little beyond it: Upright can need that). */
        fun clampCorner(v: Float) = max(-0.5f, min(1.5f, v))
    }
}
