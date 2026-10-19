package com.albus.flatscan

import android.content.Context

/** User settings and remembered choices (SharedPreferences). */
class Settings(ctx: Context) {
    private val p = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)

    /** Filter of a new scan: the last one chosen on the result screen (or in Settings). */
    var defaultFilter: FilterKind
        get() {
            val name = p.getString("filter", null)
            FilterKind.entries.firstOrNull { it.name == name }?.let { return it }
            // older versions stored a grayscale switch
            return if (p.getBoolean("grayscale", false)) FilterKind.Grayscale else RenderParams().filter
        }
        set(v) = p.edit().putString("filter", v.name).apply()

    /** Filter strength of a new scan, 0..1 (the last one chosen). */
    var defaultStrength: Float
        get() = p.getFloat("strength", RenderParams().strength).coerceIn(0f, 1f)
        set(v) = p.edit().putFloat("strength", v.coerceIn(0f, 1f)).apply()

    var solidColors: Boolean
        get() = p.getBoolean("solid", false)
        set(v) = p.edit().putBoolean("solid", v).apply()

    /** Turn pages upright automatically (autoOrient of NativeLib.process). */
    var autoRotate: Boolean
        get() = p.getBoolean("auto_rotate", true)
        set(v) = p.edit().putBoolean("auto_rotate", v).apply()

    /** Output page width in px (portrait A4); analysis always runs at 1600 px. */
    var outputWidth: Int
        get() = p.getInt("output_width", 1600)
        set(v) = p.edit().putInt("output_width", v).apply()

    /** The one-time "Press and hold to see the original" hint was dismissed. */
    var holdHintShown: Boolean
        get() = p.getBoolean("hold_hint_shown", false)
        set(v) = p.edit().putBoolean("hold_hint_shown", v).apply()

    /** Render parameters of a new page. */
    fun defaultParams() = RenderParams(filter = defaultFilter, strength = defaultStrength, solid = solidColors)

    companion object {
        val OUTPUT_WIDTHS = listOf(1240 to "Compact (1240 px, 150 dpi)", 1600 to "Standard (1600 px)", 2480 to "High (2480 px, 300 dpi)")
    }
}
