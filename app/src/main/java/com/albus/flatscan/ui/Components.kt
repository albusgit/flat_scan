package com.albus.flatscan.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/** Large buttons are 56 dp pills. */
val PillShape = RoundedCornerShape(28.dp)

/** The main action of a screen: a filled 56 dp pill ([container] defaults to the scheme's primary). */
@Composable
fun PrimaryPill(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    container: Color = MaterialTheme.colorScheme.primary,
    content: Color = MaterialTheme.colorScheme.onPrimary,
    body: @Composable RowScope.() -> Unit,
) {
    Button(
        onClick = onClick, enabled = enabled, shape = PillShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = container, contentColor = content,
            disabledContainerColor = container.copy(alpha = 0.38f), disabledContentColor = content.copy(alpha = 0.6f),
        ),
        contentPadding = PaddingValues(horizontal = 20.dp),
        modifier = modifier.height(56.dp), content = body,
    )
}

/** A secondary 56 dp pill with a thin outline on [container]. */
@Composable
fun SecondaryPill(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    container: Color = MaterialTheme.colorScheme.surfaceContainerLowest,
    border: Color = MaterialTheme.colorScheme.outline,
    body: @Composable RowScope.() -> Unit,
) {
    OutlinedButton(
        onClick = onClick, enabled = enabled, shape = PillShape,
        colors = ButtonDefaults.outlinedButtonColors(containerColor = container, contentColor = MaterialTheme.colorScheme.onSurface),
        border = BorderStroke(1.5.dp, border),
        contentPadding = PaddingValues(horizontal = 20.dp),
        modifier = modifier.height(56.dp), content = body,
    )
}

/** Pill label text (16 sp bold). */
@Composable
fun PillText(text: String) {
    Text(text, style = MaterialTheme.typography.titleMedium, maxLines = 1)
}
