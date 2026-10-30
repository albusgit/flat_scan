package com.albus.flatscan.ui

import android.graphics.Bitmap
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ProvideTextStyle
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.ImageShader
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.TileMode
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawOutline
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import kotlin.random.Random

/** Large buttons: 56 dp, crisp 12 dp corners. */
val PillShape = RoundedCornerShape(12.dp)

/** Small buttons and chips of the editor panels (36-40 dp). */
val ChipShape = RoundedCornerShape(8.dp)

/** True on the light (paper) theme. */
@Composable
fun isPaperLight(): Boolean = MaterialTheme.colorScheme.background.luminance() > 0.5f

/** The hard shadow under pressable buttons: ink on paper, black on graphite and in the editor. */
@Composable
fun inkShadow(): Color = if (isPaperLight()) Palette.Ink else Color.Black

/** Ruled-line colour for dividers. */
@Composable
fun ruledColor(): Color = if (isPaperLight()) Palette.Ruled else MaterialTheme.colorScheme.outlineVariant

/**
 * The desk under the paper screens: graph paper, a fine 16 dp grid with a stronger line every 5
 * squares (cool grey on paper, a faint lighter grey on graphite).
 */
@Composable
fun Modifier.graphPaper(): Modifier {
    val light = isPaperLight()
    val minor = if (light) Color(0xFFE2E7EE) else Color(0xFF1E1F22)
    val major = if (light) Color(0xFFD5DCE6) else Color(0xFF292A2F)
    return drawBehind {
        val step = 16.dp.toPx()
        val w = 1.dp.toPx()
        var i = 0
        var x = 0f
        while (x < size.width) {
            drawLine(if (i % 5 == 0) major else minor, Offset(x, 0f), Offset(x, size.height), strokeWidth = w)
            x += step; i++
        }
        i = 0
        var y = 0f
        while (y < size.height) {
            drawLine(if (i % 5 == 0) major else minor, Offset(0f, y), Offset(size.width, y), strokeWidth = w)
            y += step; i++
        }
    }
}

/** A divider in the ruled-line colour. */
@Composable
fun RuledDivider(modifier: Modifier = Modifier) {
    HorizontalDivider(modifier, thickness = 1.dp, color = ruledColor())
}

/** True when the system animations are off (Developer options or Accessibility > Remove animations). */
@Composable
fun rememberReducedMotion(): Boolean {
    val ctx = LocalContext.current
    return remember {
        android.provider.Settings.Global.getFloat(ctx.contentResolver, android.provider.Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
}

/**
 * A button face over a hard shadow [depth] below it; pressing pushes the face down into the shadow.
 * The modifier's size includes the shadow.
 */
@Composable
fun InkPressable(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    shape: Shape = PillShape,
    container: Color,
    contentColor: Color,
    border: BorderStroke? = null,
    shadow: Color = inkShadow(),
    depth: Dp = 3.dp,
    onClickLabel: String? = null,
    content: @Composable BoxScope.() -> Unit,
) {
    val source = remember { MutableInteractionSource() }
    val pressed by source.collectIsPressedAsState()
    val push by animateDpAsState(if (pressed && enabled) depth else 0.dp, tween(70), label = "press")
    val shadowColor = if (enabled) shadow else shadow.copy(alpha = shadow.alpha * 0.25f)
    Box(
        modifier.drawBehind {
            val d = depth.toPx()
            if (d > 0f) translate(top = d) { drawOutline(shape.createOutline(Size(size.width, size.height - d), layoutDirection, this), shadowColor) }
        },
        propagateMinConstraints = true,
    ) {
        Box(
            Modifier.padding(bottom = depth).offset { IntOffset(0, push.roundToPx()) }
                .clip(shape)
                .background(if (enabled) container else container.copy(alpha = container.alpha * 0.45f))
                .then(if (border != null) Modifier.border(border, shape) else Modifier)
                .clickable(source, ripple(), enabled = enabled, onClickLabel = onClickLabel, role = Role.Button, onClick = onClick),
            contentAlignment = Alignment.Center,
        ) {
            CompositionLocalProvider(LocalContentColor provides if (enabled) contentColor else contentColor.copy(alpha = 0.6f)) { content() }
        }
    }
}

/** The main action of a screen: a filled 56 dp button ([container] defaults to the scheme's primary). */
@Composable
fun PrimaryPill(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    container: Color = MaterialTheme.colorScheme.primary,
    content: Color = MaterialTheme.colorScheme.onPrimary,
    body: @Composable RowScope.() -> Unit,
) {
    InkPressable(onClick, modifier.height(56.dp), enabled, container = container, contentColor = content) {
        PillRow(body)
    }
}

/** A secondary 56 dp button with a thin outline on [container]; its shadow is the outline colour. */
@Composable
fun SecondaryPill(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    container: Color = MaterialTheme.colorScheme.surfaceContainerLowest,
    border: Color = MaterialTheme.colorScheme.outline,
    body: @Composable RowScope.() -> Unit,
) {
    InkPressable(
        onClick, modifier.height(56.dp), enabled, container = container, contentColor = MaterialTheme.colorScheme.onSurface,
        border = BorderStroke(1.5.dp, border), shadow = border,
    ) { PillRow(body) }
}

@Composable
private fun PillRow(body: @Composable RowScope.() -> Unit) {
    ProvideTextStyle(MaterialTheme.typography.labelLarge) {
        Row(Modifier.padding(horizontal = 20.dp), verticalAlignment = Alignment.CenterVertically, content = body)
    }
}

/** Pill label text (16 sp bold). */
@Composable
fun PillText(text: String) {
    Text(text, style = MaterialTheme.typography.titleMedium, maxLines = 1)
}

/**
 * A sheet of paper: [color] with crisp corners, a soft shadow, a faint grain, optionally ruled
 * lines and a strip of tape over the top edge.
 */
@Composable
fun PaperSurface(
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    shape: Shape = MaterialTheme.shapes.extraSmall,
    color: Color = MaterialTheme.colorScheme.surfaceContainerLowest,
    contentColor: Color = MaterialTheme.colorScheme.onSurface,
    lift: Dp = 2.dp,
    ruled: Boolean = false,
    tape: Boolean = false,
    onClickLabel: String? = null,
    content: @Composable BoxScope.() -> Unit,
) {
    val light = isPaperLight()
    val edge = if (light) Palette.OutlineSoft else MaterialTheme.colorScheme.outlineVariant
    val rule = ruledColor()
    val shadow = if (light) Palette.Ink.copy(alpha = 0.35f) else Color.Black
    val tapeColor = if (light) Color(0x99F4F1E4) else Color(0x66D9D6C8)
    Box(
        modifier
            .drawWithContent {
                drawContent()
                if (tape) {
                    val w = 64.dp.toPx()
                    val h = 18.dp.toPx()
                    rotate(-2.5f, pivot = Offset(size.width / 2, 0f)) {
                        drawRect(tapeColor, topLeft = Offset(size.width / 2 - w / 2, -h / 2), size = Size(w, h))
                    }
                }
            }
            .shadow(lift, shape, ambientColor = shadow, spotColor = shadow)
            .clip(shape)
            .background(color)
            .drawBehind {
                drawRect(Grain.brush, alpha = if (light) 1f else 0.6f)
                if (ruled) {
                    val step = 28.dp.toPx()
                    var y = step
                    while (y < size.height) {
                        drawLine(rule, Offset(0f, y), Offset(size.width, y), strokeWidth = 1.dp.toPx())
                        y += step
                    }
                }
            }
            .border(1.dp, edge, shape)
            .then(if (onClick != null) Modifier.clickable(role = Role.Button, onClickLabel = onClickLabel, onClick = onClick) else Modifier),
    ) {
        CompositionLocalProvider(LocalContentColor provides contentColor) { content() }
    }
}

/** A tileable paper grain: dark specks at about 3% opacity. */
private object Grain {
    private const val N = 96
    private val bitmap: ImageBitmap by lazy {
        val r = Random(42)
        val px = IntArray(N * N) { (r.nextInt(18) shl 24) or 0x1B2030 }
        Bitmap.createBitmap(px, N, N, Bitmap.Config.ARGB_8888).asImageBitmap()
    }
    val brush: ShaderBrush by lazy { ShaderBrush(ImageShader(bitmap, TileMode.Repeated, TileMode.Repeated)) }
}

/** A receipt: straight top, torn (zigzag) bottom edge with [tooth]-wide teeth. */
class ReceiptShape(private val tooth: Dp = 10.dp) : Shape {
    override fun createOutline(size: Size, layoutDirection: LayoutDirection, density: Density): Outline {
        val t = with(density) { tooth.toPx() }
        val n = (size.width / t).toInt().coerceAtLeast(1)
        val step = size.width / n
        val path = Path().apply {
            moveTo(0f, 0f)
            lineTo(size.width, 0f)
            lineTo(size.width, size.height - step / 2)
            for (i in n - 1 downTo 0) {
                lineTo(i * step + step / 2, size.height)
                lineTo(i * step, size.height - step / 2)
            }
            close()
        }
        return Outline.Generic(path)
    }
}
