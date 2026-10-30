package com.albus.flatscan.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.LocalOffer
import androidx.compose.material.icons.outlined.Remove
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.albus.flatscan.TagEdit
import com.albus.flatscan.TagState
import com.albus.flatscan.Tags

/**
 * Home's tag filter: a chip for every tag (most used first, [tags]); the documents shown have every
 * chip that is on ([picked], AND). A Clear chip comes first while any is on.
 */
@Composable
fun TagFilterRow(tags: List<String>, picked: List<String>, onToggle: (String) -> Unit, onClear: () -> Unit, modifier: Modifier = Modifier) {
    LazyRow(modifier.fillMaxWidth().testTag("tag_filter"), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (picked.isNotEmpty()) item("clear") {
            AssistChip(
                onClick = onClear, label = { Text("Clear") },
                leadingIcon = { Icon(Icons.Outlined.Close, null, Modifier.size(AssistChipDefaults.IconSize)) },
                modifier = Modifier.testTag("tag_filter_clear"),
            )
        }
        items(tags, key = { Tags.key(it) }) { t ->
            val on = Tags.has(picked, t)
            FilterChip(
                selected = on, onClick = { onToggle(t) },
                label = { Text(t, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                // the tag's highlighter ink as a dot, as on its labels
                leadingIcon = if (on) ({ Icon(Icons.Outlined.Check, null, Modifier.size(FilterChipDefaults.IconSize)) })
                else ({ Box(Modifier.size(10.dp).clip(CircleShape).background(tagInk(t))) }),
                modifier = Modifier.testTag("tag_chip_${Tags.key(t)}"),
            )
        }
    }
}

/**
 * The open document's tags on the pages screen, in one 48 dp row under the title; tapping it opens the
 * tag sheet. With no tags yet it says "Add tags".
 */
@Composable
fun DocumentTagsRow(tags: List<String>, onEdit: () -> Unit, modifier: Modifier = Modifier) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Row(
        modifier.fillMaxWidth().heightIn(min = 48.dp).clip(RoundedCornerShape(8.dp))
            .clickable(role = Role.Button, onClickLabel = "Edit the tags", onClick = onEdit)
            .padding(horizontal = 12.dp).testTag("document_tags"),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Outlined.LocalOffer, null, Modifier.size(18.dp), tint = muted)
        Spacer(Modifier.width(10.dp))
        if (tags.isEmpty()) {
            Text("Add tags", style = MaterialTheme.typography.labelLarge, color = muted, modifier = Modifier.weight(1f))
        } else {
            Row(Modifier.weight(1f).horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                tags.forEach { TagLabel(it) }
            }
            Spacer(Modifier.width(8.dp))
            Icon(Icons.Outlined.Edit, null, Modifier.size(16.dp), tint = muted)
        }
    }
}

/** Highlighter inks for tags: each tag keeps one, picked by its name ([Tags.key]), so it reads the same everywhere. */
private val TagInks = listOf(
    Color(0xFFFFE45C), Color(0xFF9BE58C), Color(0xFFFFA8C5), Color(0xFFFFC266), Color(0xFF96D3FF), Color(0xFFC9B3FF),
)

fun tagInk(tag: String): Color = TagInks[Math.floorMod(Tags.key(tag).hashCode(), TagInks.size)]

/**
 * A tag as a small label (not a button): its name over a highlighter stroke in the tag's own ink ([tagInk]), a
 * little ragged at the ends like a marker. On graphite the ink is dimmed and the name stays light.
 */
@Composable
fun TagLabel(tag: String, modifier: Modifier = Modifier, small: Boolean = false) {
    val light = isPaperLight()
    val ink = tagInk(tag).copy(alpha = if (light) 0.85f else 0.32f)
    Text(
        tag, style = (if (small) MaterialTheme.typography.labelSmall else MaterialTheme.typography.labelMedium).copy(fontWeight = FontWeight.SemiBold),
        maxLines = 1, overflow = TextOverflow.Ellipsis,
        color = if (light) Palette.Ink else MaterialTheme.colorScheme.onSurface,
        modifier = modifier.drawBehind {
            val r = 3.dp.toPx()
            drawRoundRect(ink, Offset(-1.dp.toPx(), size.height * 0.08f), Size(size.width + 2.dp.toPx(), size.height * 0.86f), CornerRadius(r, r))
        }.padding(horizontal = if (small) 5.dp else 7.dp, vertical = if (small) 1.dp else 2.dp),
    )
}

/** A document's tags in a list row or a card: the first [max] as labels, then "+n" for the rest; nothing without tags. */
@Composable
fun TagMarks(tags: List<String>, modifier: Modifier = Modifier, max: Int = 2) {
    if (tags.isEmpty()) return
    Row(
        modifier.semantics(mergeDescendants = true) { contentDescription = "Tags: " + tags.joinToString(", ") },
        horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        tags.take(max).forEach { TagLabel(it, Modifier.weight(1f, fill = false), small = true) }
        if (tags.size > max) Text(
            "+${tags.size - max}", style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1,
        )
    }
}

/**
 * The tags of one document or several ([edit]; [subtitle] names them): type a new tag and add it, or
 * tap a tag to put it on all of them, or, when they all have it, to take it off. A tag only some of
 * them have shows a dash. Every change is saved at once ([onSet]: tag, on); [known] are the library's
 * tags, most used first. The chips keep their order while the sheet is open.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TagSheet(subtitle: String?, edit: TagEdit, known: List<String>, onSet: (String, Boolean) -> Unit, onDismiss: () -> Unit) {
    val order = remember { mutableStateListOf<String>().apply { addAll(edit.choices(known)) } }
    var text by remember { mutableStateOf("") }
    val several = edit.docs.size > 1
    fun add() {
        val typed = Tags.clean(text)
        if (typed.isEmpty()) return
        // an existing tag keeps its spelling
        val t = order.firstOrNull { Tags.key(it) == Tags.key(typed) } ?: known.firstOrNull { Tags.key(it) == Tags.key(typed) } ?: typed
        if (order.none { Tags.key(it) == Tags.key(t) }) order.add(0, t)
        onSet(t, true)
        text = ""
    }
    val state = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = state, containerColor = MaterialTheme.colorScheme.surfaceContainerLowest) {
        Column(Modifier.navigationBarsPadding().verticalScroll(rememberScrollState()).padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
            Text("Tags", style = MaterialTheme.typography.titleLarge)
            if (subtitle != null) Text(subtitle, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Spacer(Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = text,
                    // a comma ends the tag (it is the separator)
                    onValueChange = { v -> if (',' in v) { text = v.substringBefore(','); add() } else text = v.take(Tags.MAX_LENGTH) },
                    singleLine = true, label = { Text("New tag") },
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.None, imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = { add() }),
                    modifier = Modifier.weight(1f).testTag("tag_input"),
                )
                Spacer(Modifier.width(8.dp))
                TextButton(onClick = { add() }, enabled = Tags.clean(text).isNotEmpty(), modifier = Modifier.heightIn(min = 48.dp).testTag("tag_add")) { Text("Add") }
            }
            Spacer(Modifier.height(12.dp))
            if (order.isEmpty()) {
                Text(
                    "No tags yet. A tag groups documents, like “tax” or “receipts”; Home can then show just those.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    order.forEach { t ->
                        val s = edit.state(t)
                        FilterChip(
                            selected = s != TagState.None, onClick = { onSet(t, edit.tapTurnsOn(t)) },
                            label = { Text(t, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                            leadingIcon = when (s) {
                                TagState.All -> ({ Icon(Icons.Outlined.Check, null, Modifier.size(FilterChipDefaults.IconSize)) })
                                TagState.Some -> ({ Icon(Icons.Outlined.Remove, null, Modifier.size(FilterChipDefaults.IconSize)) })
                                TagState.None -> null
                            },
                            modifier = Modifier.testTag("tag_${Tags.key(t)}").semantics {
                                stateDescription = when (s) {
                                    TagState.All -> if (several) "On all of them" else "On"
                                    TagState.Some -> "On some of them"
                                    TagState.None -> "Off"
                                }
                            },
                        )
                    }
                }
                if (several) Text(
                    "Tap a tag to put it on all ${edit.docs.size} documents; tap it again to take it off all of them.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp),
                )
            }
            Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.End) {
                TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) { Text("Done") }
            }
        }
    }
}
