package com.albus.flatscan.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Light = lightColorScheme(
    primary = Color(0xFF1F4ED8), onPrimary = Color.White,
    primaryContainer = Color(0xFFDCE4FF), onPrimaryContainer = Color(0xFF0B1F5C),
    secondary = Color(0xFF475569), onSecondary = Color.White,
    secondaryContainer = Color(0xFFE2E8F0), onSecondaryContainer = Color(0xFF1E293B),
    background = Color(0xFFFAFAF9), onBackground = Color(0xFF1C1C1E),
    surface = Color(0xFFFAFAF9), onSurface = Color(0xFF1C1C1E),
    surfaceVariant = Color(0xFFEEEEEC), onSurfaceVariant = Color(0xFF55555A),
    surfaceContainer = Color(0xFFF2F2F0), surfaceContainerHigh = Color(0xFFECECEA),
    outline = Color(0xFFBDBDBA), outlineVariant = Color(0xFFDADAD7),
)

private val Dark = darkColorScheme(
    primary = Color(0xFF9DB4FF), onPrimary = Color(0xFF0B1F5C),
    primaryContainer = Color(0xFF233A8C), onPrimaryContainer = Color(0xFFDCE4FF),
    secondary = Color(0xFFB6C2D3), onSecondary = Color(0xFF1E293B),
    secondaryContainer = Color(0xFF334155), onSecondaryContainer = Color(0xFFE2E8F0),
    background = Color(0xFF121212), onBackground = Color(0xFFE7E7E5),
    surface = Color(0xFF121212), onSurface = Color(0xFFE7E7E5),
    surfaceVariant = Color(0xFF2A2A2C), onSurfaceVariant = Color(0xFFB4B4B0),
    surfaceContainer = Color(0xFF1C1C1E), surfaceContainerHigh = Color(0xFF242426),
    outline = Color(0xFF5A5A5E), outlineVariant = Color(0xFF3A3A3C),
)

@Composable
fun FlatScanTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) Dark else Light, content = content)
}
