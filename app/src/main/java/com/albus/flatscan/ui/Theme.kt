package com.albus.flatscan.ui

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import com.albus.flatscan.R

/**
 * Recursive (SIL OFL 1.1, see assets/licenses/Recursive-OFL.txt): one variable font, Latin only,
 * weights 300-900 with its Casual and Mono axes. Three families read it at different axis values.
 */
@OptIn(ExperimentalTextApi::class)
private fun recursive(casual: Float, mono: Float) = FontFamily(
    listOf(400, 500, 600, 700, 800).map { w ->
        Font(
            R.font.recursive, FontWeight(w),
            variationSettings = FontVariation.Settings(
                FontVariation.weight(w), FontVariation.Setting("CASL", casual), FontVariation.Setting("MONO", mono),
            ),
        )
    },
)

/** Headings: a touch of the Casual axis. */
val RecursiveDisplay = recursive(casual = 0.45f, mono = 0f)

/** Body and controls: the plain (Linear) style. */
val RecursiveText = recursive(casual = 0f, mono = 0f)

/** Typed metadata: page counts, processing stages, the self-test report. */
val RecursiveMono = recursive(casual = 0f, mono = 1f)

/** Colours the design uses outside the Material roles: printer paper, ballpoint and a highlighter. */
object Palette {
    val Accent = Color(0xFF2747A8)
    val AccentPressed = Color(0xFF1D3888)
    val AccentContainer = Color(0xFFDCE4F7)
    val Ink = Color(0xFF1B2030)
    val Secondary = Color(0xFF5A6070)
    val Tertiary = Color(0xFF6B7180)
    val Outline = Color(0xFFCDD5E1)
    val OutlineSoft = Color(0xFFDDE3EC)
    val Dashed = Color(0xFFC3CCD9)
    val Paper = Color(0xFFFFFFFF)
    val GraphPaper = Color(0xFFEEF1F4)
    val Ruled = Color(0xFFC3D3EC)
    val Highlighter = Color(0xFFFFE45C)
    val RedPen = Color(0xFFD9473A)
}

/** The light screens' dark theme: neutral graphite. */
object Graphite {
    val Ground = Color(0xFF161719)
    val Sheet = Color(0xFF222327)
    val Raised = Color(0xFF2A2B30)
    val Ruled = Color(0xFF36383E)
    val Text = Color(0xFFECECEE)
    val Pencil = Color(0xFFA4A6AD)
    val Accent = Color(0xFF9FB5F2)
    val Highlighter = Color(0xFFFFE066)
}

/** The always-dark editor (camera, crop, result): a neutral ink well with a highlighter accent. */
object Editor {
    val Ground = Color(0xFF111318)
    val Panel = Color(0xFF1A1D24)
    val Raised = Color(0xFF20232B)
    val Divider = Color(0xFF262A32)
    val Outline = Color(0xFF2C313B)
    val Text = Color(0xFFF1F3F7)
    val Muted = Color(0xFFA2A9B6)
    val Inactive = Color(0xFFC3C8D1)
    val Accent = Color(0xFFFFE15A)
    val Histogram = Color(0xFF3A3F49)
    /** Top corners of the bottom panel. */
    val PanelShape = RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)
}

private val Light = lightColorScheme(
    primary = Palette.Accent, onPrimary = Color.White,
    primaryContainer = Palette.AccentContainer, onPrimaryContainer = Color(0xFF0F2466),
    secondary = Palette.Secondary, onSecondary = Color.White,
    secondaryContainer = Color(0xFFE3E8EE), onSecondaryContainer = Palette.Ink,
    background = Palette.GraphPaper, onBackground = Palette.Ink,
    surface = Palette.GraphPaper, onSurface = Palette.Ink,
    surfaceVariant = Color(0xFFE3E8EE), onSurfaceVariant = Palette.Secondary,
    surfaceContainerLowest = Palette.Paper, surfaceContainerLow = Palette.Paper,
    surfaceContainer = Palette.Paper, surfaceContainerHigh = Palette.Paper, surfaceContainerHighest = Color(0xFFE3E8EE),
    outline = Palette.Outline, outlineVariant = Palette.OutlineSoft,
    error = Palette.RedPen, onError = Color.White,
    inverseSurface = Palette.Ink, inverseOnSurface = Color(0xFFF1F3F7), inversePrimary = Graphite.Accent,
)

private val Dark = darkColorScheme(
    primary = Graphite.Accent, onPrimary = Editor.Ground,
    primaryContainer = Color(0xFF2E3F70), onPrimaryContainer = Color(0xFFDCE4FF),
    secondary = Graphite.Pencil, onSecondary = Graphite.Ground,
    secondaryContainer = Graphite.Raised, onSecondaryContainer = Graphite.Text,
    background = Graphite.Ground, onBackground = Graphite.Text,
    surface = Graphite.Ground, onSurface = Graphite.Text,
    surfaceVariant = Graphite.Raised, onSurfaceVariant = Graphite.Pencil,
    surfaceContainerLowest = Graphite.Sheet, surfaceContainerLow = Graphite.Sheet,
    surfaceContainer = Graphite.Sheet, surfaceContainerHigh = Graphite.Raised, surfaceContainerHighest = Color(0xFF303136),
    outline = Color(0xFF45474D), outlineVariant = Graphite.Ruled,
    error = Color(0xFFFF7A6B), onError = Graphite.Ground,
    inverseSurface = Graphite.Text, inverseOnSurface = Graphite.Ground, inversePrimary = Palette.Accent,
)

/** The editor scheme: ink-well tokens, the highlighter as primary, the panel as the surface for sheets and menus. */
private val EditorScheme = darkColorScheme(
    primary = Editor.Accent, onPrimary = Editor.Ground,
    primaryContainer = Color(0xFF4A421A), onPrimaryContainer = Color(0xFFFFF1B0),
    secondary = Editor.Muted, onSecondary = Editor.Ground,
    secondaryContainer = Editor.Divider, onSecondaryContainer = Editor.Text,
    background = Editor.Ground, onBackground = Editor.Text,
    surface = Editor.Ground, onSurface = Editor.Text,
    surfaceVariant = Color(0xFF22252D), onSurfaceVariant = Editor.Muted,
    surfaceContainerLowest = Editor.Panel, surfaceContainerLow = Editor.Panel,
    surfaceContainer = Editor.Raised, surfaceContainerHigh = Editor.Raised, surfaceContainerHighest = Editor.Divider,
    outline = Editor.Outline, outlineVariant = Editor.Divider,
    error = Color(0xFFFF7A6B), onError = Editor.Ground,
    inverseSurface = Editor.Text, inverseOnSurface = Editor.Ground, inversePrimary = Palette.Accent,
)

private fun style(size: Int, weight: FontWeight, line: Int, spacing: Float = 0f, family: FontFamily = RecursiveText) = TextStyle(
    fontFamily = family, fontWeight = weight, fontSize = size.sp, lineHeight = line.sp, letterSpacing = spacing.em,
)

private val Type = Typography(
    displaySmall = style(34, FontWeight.ExtraBold, 38, -0.03f, RecursiveDisplay),
    headlineLarge = style(34, FontWeight.ExtraBold, 38, -0.03f, RecursiveDisplay),
    headlineMedium = style(28, FontWeight.ExtraBold, 34, -0.02f, RecursiveDisplay),
    headlineSmall = style(22, FontWeight.Bold, 28, -0.01f, RecursiveDisplay),
    titleLarge = style(20, FontWeight.ExtraBold, 26, -0.02f, RecursiveDisplay),
    titleMedium = style(16, FontWeight.Bold, 22),
    titleSmall = style(14, FontWeight.Bold, 20),
    bodyLarge = style(16, FontWeight.Medium, 24),
    bodyMedium = style(14, FontWeight.Normal, 20),
    bodySmall = style(12, FontWeight.Normal, 16),
    labelLarge = style(14, FontWeight.Bold, 20),
    labelMedium = style(12, FontWeight.Bold, 16),
    labelSmall = style(11, FontWeight.Bold, 14),
)

/** Paper has crisp corners: 4-16 dp. */
private val Shape = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(10.dp),
    large = RoundedCornerShape(12.dp),
    extraLarge = RoundedCornerShape(16.dp),
)

@Composable
fun FlatScanTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) Dark else Light, typography = Type, shapes = Shape, content = content)
}

/**
 * The editor screens are dark whatever the system theme: dark tokens, the editor ground behind
 * [content] and light status / navigation bar icons while it is shown.
 */
@Composable
fun EditorTheme(content: @Composable () -> Unit) {
    val view = LocalView.current
    if (!view.isInEditMode) {
        DisposableEffect(view) {
            val window = (view.context as? Activity)?.window
            val bars = window?.let { WindowCompat.getInsetsController(it, view) }
            val status = bars?.isAppearanceLightStatusBars
            val nav = bars?.isAppearanceLightNavigationBars
            bars?.isAppearanceLightStatusBars = false
            bars?.isAppearanceLightNavigationBars = false
            onDispose {
                if (status != null) bars.isAppearanceLightStatusBars = status
                if (nav != null) bars.isAppearanceLightNavigationBars = nav
            }
        }
    }
    MaterialTheme(colorScheme = EditorScheme, typography = Type, shapes = Shape) {
        Surface(Modifier.fillMaxSize(), color = Editor.Ground, contentColor = Editor.Text, content = content)
    }
}
