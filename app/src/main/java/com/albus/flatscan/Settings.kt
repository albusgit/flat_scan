package com.albus.flatscan

import android.content.Context

/** The three user settings (APP_BUILD_BRIEF.md 1.5). */
class Settings(ctx: Context) {
    private val p = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var grayscale: Boolean
        get() = p.getBoolean("grayscale", false)
        set(v) = p.edit().putBoolean("grayscale", v).apply()

    var solidColors: Boolean
        get() = p.getBoolean("solid", false)
        set(v) = p.edit().putBoolean("solid", v).apply()

    /** Output page width in px (portrait A4); analysis always runs at 1600 px. */
    var outputWidth: Int
        get() = p.getInt("output_width", 1600)
        set(v) = p.edit().putInt("output_width", v).apply()

    companion object {
        val OUTPUT_WIDTHS = listOf(1240 to "Compact (1240 px, 150 dpi)", 1600 to "Standard (1600 px)", 2480 to "High (2480 px, 300 dpi)")
    }
}
