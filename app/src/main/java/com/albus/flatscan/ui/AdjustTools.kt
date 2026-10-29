package com.albus.flatscan.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AspectRatio
import androidx.compose.material.icons.outlined.BarChart
import androidx.compose.material.icons.outlined.Brightness6
import androidx.compose.material.icons.outlined.CleaningServices
import androidx.compose.material.icons.outlined.Contrast
import androidx.compose.material.icons.outlined.InvertColors
import androidx.compose.material.icons.outlined.Thermostat
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.albus.flatscan.PageLook
import com.albus.flatscan.PageShape
import com.albus.flatscan.RenderParams
import com.albus.flatscan.ScanViewModel
import java.util.Locale
import kotlin.math.roundToInt

/** The page's PageLook colours as a Compose colour filter (null at the defaults: nothing is drawn differently). */
fun PageLook.composeFilter(): ColorFilter? = colorMatrix()?.let { ColorFilter.colorMatrix(ColorMatrix(it)) }

/** One tool of the Adjust panel: a chip in the tool row, its control below. */
private enum class AdjustTool(val label: String, val icon: ImageVector) {
    Brightness("Brightness", Icons.Outlined.Brightness6),
    Contrast("Contrast", Icons.Outlined.Contrast),
    Saturation("Saturation", Icons.Outlined.InvertColors),
    Warmth("Warmth", Icons.Outlined.Thermostat),
    Cleanup("Background", Icons.Outlined.CleaningServices),
    Levels("Levels", Icons.Outlined.BarChart),
}

private fun AdjustTool.changed(p: RenderParams, l: PageLook): Boolean {
    val d = RenderParams()
    return when (this) {
        AdjustTool.Brightness -> l.brightness != 0f
        AdjustTool.Contrast -> l.contrast != 0f
        AdjustTool.Saturation -> l.saturation != 0f
        AdjustTool.Warmth -> l.warmth != 0f
        AdjustTool.Cleanup -> p.bgCleanup != d.bgCleanup
        AdjustTool.Levels -> p.black != d.black || p.white != d.white || p.gamma != d.gamma
    }
}

/**
 * The Adjust tab: a row of tools (light and colour sliders, background cleanup, levels, page shape)
 * and the selected tool's control, so the panel stays one slider high. Light and colour run on
 * screen as a colour filter (no re-render); cleanup and levels re-render in the core as before.
 */
@Composable
fun AdjustPanel(vm: ScanViewModel) {
    val p = vm.params
    val l = vm.look
    var tool by rememberSaveable { mutableStateOf(AdjustTool.Brightness) }
    Column(Modifier.fillMaxWidth().padding(top = 8.dp, bottom = 4.dp)) {
        Row(Modifier.padding(start = 20.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Adjust", style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold), modifier = Modifier.weight(1f))
            val any = AdjustTool.entries.any { it.changed(p, l) }
            TextButton(onClick = { vm.resetAdjustments(); vm.updateLook(l.copy(brightness = 0f, contrast = 0f, saturation = 0f, warmth = 0f)) }, enabled = any) {
                Text("Reset")
            }
        }
        Row(
            Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            for (t in AdjustTool.entries) ToolChip(t, selected = t == tool, changed = t.changed(p, l)) { tool = t }
        }
        Box(Modifier.fillMaxWidth().heightIn(min = 76.dp).padding(start = 20.dp, end = 20.dp, top = 4.dp), contentAlignment = Alignment.CenterStart) {
            when (tool) {
                AdjustTool.Brightness -> Bipolar("Brightness", l.brightness, "Lighter or darker page") { vm.updateLook(l.copy(brightness = it)) }
                AdjustTool.Contrast -> Bipolar("Contrast", l.contrast, "Darker text on the same paper, or softer") { vm.updateLook(l.copy(contrast = it)) }
                AdjustTool.Saturation -> Bipolar("Saturation", l.saturation, "Stronger or paler colours; all the way left: grey") { vm.updateLook(l.copy(saturation = it)) }
                AdjustTool.Warmth -> Bipolar("Warmth", l.warmth, "Warmer (yellow) or cooler (blue) paper") { vm.updateLook(l.copy(warmth = it)) }
                AdjustTool.Cleanup -> Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Background cleanup", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                        Text("${(p.bgCleanup * 100).roundToInt()}%", style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold), color = Editor.Muted)
                    }
                    Slider(
                        value = p.bgCleanup, onValueChange = { vm.setBgCleanup(it) }, colors = adjustSliderColors(),
                        modifier = Modifier.semantics { contentDescription = "Background cleanup" },
                    )
                    Text("Removes grey specks and shading on white paper", style = MaterialTheme.typography.bodySmall, color = Editor.Muted)
                }
                AdjustTool.Levels -> Column {
                    Row(Modifier.padding(bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text("Levels", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                        Text(
                            "Black ${(p.black * 100).roundToInt()} · Mid ${"%.2f".format(Locale.US, p.gamma)} · White ${(p.white * 100).roundToInt()}",
                            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold), color = Editor.Muted,
                        )
                    }
                    LevelsEditor(vm.histogram, p.black, p.white, p.gamma, onChange = { b, w, g -> vm.setLevels(b, w, g) })
                }
            }
        }
    }
}

@Composable
private fun ToolChip(t: AdjustTool, selected: Boolean, changed: Boolean, onClick: () -> Unit) {
    val color = if (selected) Editor.Accent else Editor.Inactive
    Column(
        Modifier.widthIn(min = 72.dp).clip(RoundedCornerShape(8.dp))
            .selectable(selected, role = Role.Tab, onClick = onClick)
            .semantics { if (changed) stateDescription = "changed" }
            .padding(horizontal = 6.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box {
            Box(
                Modifier.size(40.dp).clip(CircleShape)
                    .background(if (selected) Editor.Accent.copy(alpha = 0.18f) else Color.Transparent)
                    .border(1.dp, if (selected) Editor.Accent else Editor.Outline, CircleShape),
                contentAlignment = Alignment.Center,
            ) { Icon(t.icon, null, tint = color, modifier = Modifier.size(20.dp)) }
            // a dot: this tool changed the page
            if (changed) Box(Modifier.align(Alignment.TopEnd).size(9.dp).background(Editor.Accent, CircleShape).border(1.5.dp, Editor.Panel, CircleShape))
        }
        Spacer(Modifier.height(4.dp))
        Text(t.label, style = MaterialTheme.typography.labelSmall, color = color, maxLines = 1, softWrap = false, textAlign = TextAlign.Center)
    }
}

/** A -1..1 slider with 0 in the middle; the value reads -100..+100, and a tap on it goes back to 0. */
@Composable
internal fun Bipolar(label: String, value: Float, caption: String, onChange: (Float) -> Unit) {
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
            val v = (value * 100).roundToInt()
            TextButton(
                onClick = { onChange(0f) }, enabled = v != 0,
                modifier = Modifier.minimumInteractiveComponentSize().semantics { contentDescription = "$label ${if (v > 0) "+" else ""}$v, reset" },
            ) {
                Text(if (v > 0) "+$v" else "$v", style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.SemiBold, fontSize = 14.sp), color = Editor.Muted)
            }
        }
        Slider(
            value = value, onValueChange = { v -> onChange(if (kotlin.math.abs(v) < 0.03f) 0f else v) }, valueRange = -1f..1f,
            colors = adjustSliderColors(), modifier = Modifier.semantics { contentDescription = label },
        )
        Text(caption, style = MaterialTheme.typography.bodySmall, color = Editor.Muted)
    }
}

/** The page's shape (the result screen's Page tab). */
@Composable
internal fun ShapeChooser(l: PageLook, onPick: (PageShape) -> Unit) {
    Column {
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for (s in PageShape.entries) {
                val on = s == l.shape
                val enabled = s != PageShape.Measured || l.measuredHw > 0f
                val shape = ChipShape
                Box(
                    Modifier.minimumInteractiveComponentSize().height(36.dp).clip(shape)
                        .background(if (on) Editor.Accent.copy(alpha = 0.18f) else Color.Transparent)
                        .border(1.dp, if (on) Editor.Accent else Editor.Outline, shape)
                        .selectable(on, enabled = enabled, role = Role.RadioButton) { onPick(s) }
                        .padding(horizontal = 14.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(s.label, style = MaterialTheme.typography.labelLarge.copy(fontSize = 13.sp), color = if (enabled) Editor.Text else Editor.Muted, maxLines = 1)
                }
            }
        }
        Text(
            if (l.shape == PageShape.Measured && l.measuredHw > 0f) l.shape.caption + " (height / width %.2f)".format(Locale.US, l.measuredHw)
            else if (l.measuredHw <= 0f && l.shape == PageShape.AsScanned) l.shape.caption + ". Process the page again to measure its own shape."
            else l.shape.caption,
            style = MaterialTheme.typography.bodySmall, color = Editor.Muted, modifier = Modifier.padding(top = 4.dp),
        )
    }
}

@Composable
private fun adjustSliderColors() = SliderDefaults.colors(
    thumbColor = Editor.Accent, activeTrackColor = Editor.Accent, inactiveTrackColor = Editor.Outline,
    disabledThumbColor = Editor.Muted, disabledActiveTrackColor = Editor.Muted.copy(alpha = 0.5f), disabledInactiveTrackColor = Editor.Divider,
)
