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
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import com.albus.flatscan.R

/** Manrope (SIL OFL 1.1, see assets/licenses/Manrope-OFL.txt), static weights 400-800. */
val Manrope = FontFamily(
    Font(R.font.manrope_regular, FontWeight.Normal),
    Font(R.font.manrope_medium, FontWeight.Medium),
    Font(R.font.manrope_semibold, FontWeight.SemiBold),
    Font(R.font.manrope_bold, FontWeight.Bold),
    Font(R.font.manrope_extrabold, FontWeight.ExtraBold),
)

/** Colours the design uses outside the Material roles. */
object Palette {
    val Accent = Color(0xFF2B5BE0)
    val AccentPressed = Color(0xFF1E46B8)
    val AccentContainer = Color(0xFFE4EAFB)
    val Ink = Color(0xFF17181B)
    val Secondary = Color(0xFF55575E)
    val Tertiary = Color(0xFF6A6C73)
    val Outline = Color(0xFFDAD8D2)
    val OutlineSoft = Color(0xFFE6E4DF)
    val Dashed = Color(0xFFCFCDC6)
}

/** The always-dark editor (camera, crop, result). */
object Editor {
    val Ground = Color(0xFF0F1012)
    val Panel = Color(0xFF17181B)
    val Raised = Color(0xFF1B1C20)
    val Divider = Color(0xFF26272C)
    val Outline = Color(0xFF34363C)
    val Text = Color(0xFFF2F2F0)
    val Muted = Color(0xFFA3A5AB)
    val Inactive = Color(0xFFC4C6CC)
    val Accent = Color(0xFF8FB0FF)
    val Histogram = Color(0xFF3A3C43)
    /** Top corners of the bottom panel. */
    val PanelShape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
}

private val Light = lightColorScheme(
    primary = Palette.Accent, onPrimary = Color.White,
    primaryContainer = Palette.AccentContainer, onPrimaryContainer = Color(0xFF0E2A80),
    secondary = Palette.Secondary, onSecondary = Color.White,
    secondaryContainer = Color(0xFFECEAE5), onSecondaryContainer = Palette.Ink,
    background = Color(0xFFF6F5F2), onBackground = Palette.Ink,
    surface = Color(0xFFF6F5F2), onSurface = Palette.Ink,
    surfaceVariant = Color(0xFFEEECE7), onSurfaceVariant = Palette.Secondary,
    surfaceContainerLowest = Color.White, surfaceContainerLow = Color.White,
    surfaceContainer = Color.White, surfaceContainerHigh = Color.White, surfaceContainerHighest = Color(0xFFEEECE7),
    outline = Palette.Outline, outlineVariant = Palette.OutlineSoft,
    inverseSurface = Palette.Ink, inverseOnSurface = Color(0xFFF2F2F0), inversePrimary = Editor.Accent,
)

private val Dark = darkColorScheme(
    primary = Editor.Accent, onPrimary = Editor.Ground,
    primaryContainer = Color(0xFF233A8C), onPrimaryContainer = Color(0xFFDCE4FF),
    secondary = Editor.Muted, onSecondary = Editor.Ground,
    secondaryContainer = Color(0xFF26272C), onSecondaryContainer = Editor.Text,
    background = Editor.Ground, onBackground = Editor.Text,
    surface = Editor.Ground, onSurface = Editor.Text,
    surfaceVariant = Color(0xFF222328), onSurfaceVariant = Editor.Muted,
    surfaceContainerLowest = Editor.Panel, surfaceContainerLow = Editor.Panel,
    surfaceContainer = Editor.Panel, surfaceContainerHigh = Color(0xFF1E1F23), surfaceContainerHighest = Color(0xFF26272C),
    outline = Editor.Outline, outlineVariant = Color(0xFF26272C),
    inverseSurface = Editor.Text, inverseOnSurface = Editor.Ground, inversePrimary = Palette.Accent,
)

/** The editor scheme: the dark tokens with the panel as the default surface for sheets and menus. */
private val EditorScheme = Dark.copy(surfaceContainer = Color(0xFF1E1F23))

private fun style(size: Int, weight: FontWeight, line: Int, spacing: Float = 0f) = TextStyle(
    fontFamily = Manrope, fontWeight = weight, fontSize = size.sp, lineHeight = line.sp, letterSpacing = spacing.em,
)

private val Type = Typography(
    displaySmall = style(34, FontWeight.ExtraBold, 38, -0.03f),
    headlineLarge = style(34, FontWeight.ExtraBold, 38, -0.03f),
    headlineMedium = style(28, FontWeight.ExtraBold, 34, -0.02f),
    headlineSmall = style(22, FontWeight.Bold, 28, -0.01f),
    titleLarge = style(20, FontWeight.ExtraBold, 26, -0.02f),
    titleMedium = style(16, FontWeight.Bold, 22),
    titleSmall = style(14, FontWeight.Bold, 20),
    bodyLarge = style(16, FontWeight.Medium, 24),
    bodyMedium = style(14, FontWeight.Normal, 20),
    bodySmall = style(12, FontWeight.Normal, 16),
    labelLarge = style(14, FontWeight.Bold, 20),
    labelMedium = style(12, FontWeight.Bold, 16),
    labelSmall = style(11, FontWeight.Bold, 14),
)

private val Shape = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(14.dp),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(24.dp),
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
