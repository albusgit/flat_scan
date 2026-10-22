package com.albus.flatscan

import android.content.Context
import android.text.TextUtils
import android.view.View

/** Compile-time defaults of feature flags (each can be changed in Settings). */
object Features {
    /** The page-finder net (assets/models/pagefinder.tflite); off: the colour-threshold paper mask. */
    const val PAGE_FINDER_NET = true
    /** Flattening nets (assets/models/uvdoc_static.tflite, syntheticdoc_hr_static.tflite); off: the page outline's surface model. */
    const val UVDOC = true
    /** The shadow net (assets/models/shadenet.tflite) divides shadows out of the page. */
    const val NEURAL_SHADOWS = true
    /** The rule net (assets/models/rulenet.onnx) tells the line refinement which lines are printed rules. */
    const val RULE_NET = true
}

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

    /** Feature flag: find the page with the net (falls back to the colour threshold if it cannot run). */
    var pageFinderNet: Boolean
        get() = p.getBoolean("page_finder_net", Features.PAGE_FINDER_NET)
        set(v) = p.edit().putBoolean("page_finder_net", v).apply()

    /** Feature flag: flatten with the nets (falls back to the outline's surface model when they disagree). */
    var uvDoc: Boolean
        get() = p.getBoolean("uvdoc", Features.UVDOC)
        set(v) = p.edit().putBoolean("uvdoc", v).apply()

    /** Which flattening nets run: 0 both (the straighter page wins), 1 UVDoc only, 2 SyntheticDoc high-res only. */
    var flatNets: Int
        get() = p.getInt("flat_nets", 0).coerceIn(0, 2)
        set(v) = p.edit().putInt("flat_nets", v.coerceIn(0, 2)).apply()

    /** Remove shadows with the shadow net. */
    var neuralShadows: Boolean
        get() = p.getBoolean("neural_shadows", Features.NEURAL_SHADOWS)
        set(v) = p.edit().putBoolean("neural_shadows", v).apply()

    /** Where the nets run ([Accel]: auto, cpu, gpu); Auto unless chosen (an older NPU / NNAPI choice is Auto now). */
    var acceleration: String
        get() = p.getString("acceleration", Accel.AUTO).takeIf { s -> Accel.CHOICES.any { it.first == s } } ?: Accel.AUTO
        set(v) = p.edit().putString("acceleration", v).apply()

    /** Auto's last benchmark: JSON soc, app (version), at, bench (NativeLib.benchBackends); null: none yet. */
    var autoChoice: String?
        get() = p.getString("accel_auto", null)
        set(v) = p.edit().apply { if (v == null) remove("accel_auto") else putString("accel_auto", v) }.apply()

    /** Let the rule net decide which lines the line refinement straightens. */
    var ruleNet: Boolean
        get() = p.getBoolean("rule_net", Features.RULE_NET)
        set(v) = p.edit().putBoolean("rule_net", v).apply()

    /** Show the crop screen for every page; otherwise only doubtful crops (quality gate) stop there. */
    var reviewCrops: Boolean
        get() = p.getBoolean("review_crops", false)
        set(v) = p.edit().putBoolean("review_crops", v).apply()

    /** Book spreads: the right page comes first (Hebrew, Arabic). Default: from the app language. */
    var bookRightToLeft: Boolean
        get() = if (p.contains("book_rtl")) p.getBoolean("book_rtl", false) else appLanguageRtl
        set(v) = p.edit().putBoolean("book_rtl", v).apply()

    private val appLanguageRtl = TextUtils.getLayoutDirectionFromLocale(ctx.resources.configuration.locales[0]) == View.LAYOUT_DIRECTION_RTL

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
