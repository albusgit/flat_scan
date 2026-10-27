package com.albus.flatscan.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.albus.flatscan.TagEdit
import com.albus.flatscan.TagState
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

/** Home's chip row and the tag sheet, driven as a user would (Robolectric). */
@RunWith(AndroidJUnit4::class)
@Config(qualifiers = "w411dp-h891dp")
class TagsUiTest {
    @get:Rule
    val rule = createComposeRule()

    @Test
    fun chipsToggleTheFilterAndClearEmptiesIt() {
        var picked by mutableStateOf(listOf<String>())
        rule.setContent {
            TagFilterRow(listOf("tax", "home"), picked, onToggle = { t -> picked = if (t in picked) picked - t else picked + t }, onClear = { picked = emptyList() })
        }
        rule.onNodeWithTag("tag_filter_clear").assertDoesNotExist()
        rule.onNodeWithTag("tag_chip_home").assertIsNotSelected().performClick()
        rule.onNodeWithTag("tag_chip_tax").performClick()
        assertEquals(listOf("home", "tax"), picked)
        rule.onNodeWithTag("tag_chip_home").assertIsSelected()
        rule.onNodeWithTag("tag_filter_clear").performClick()
        assertEquals(emptyList<String>(), picked)
    }

    @Test
    fun theSheetAddsANewTagToEveryDocumentAndTakesOneOffAll() {
        var edit by mutableStateOf(TagEdit(mapOf("a" to listOf("tax"), "b" to listOf("tax", "home"))))
        val calls = mutableListOf<Pair<String, Boolean>>()
        rule.setContent {
            TagSheet("2 documents", edit, known = listOf("tax", "home", "bills"), onSet = { t, on -> calls += t to on; edit = edit.set(t, on) }, onDismiss = {})
        }
        rule.onNodeWithText("2 documents").assertExists()
        // typed with a comma: the comma ends the tag
        rule.onNodeWithTag("tag_input").performTextInput(" Receipts ,")
        rule.waitForIdle()
        assertEquals(listOf("Receipts" to true), calls)
        assertEquals(TagState.All, edit.state("receipts"))
        rule.onNodeWithTag("tag_receipts").assertIsSelected()
        // home is on one of the two: a tap puts it on both
        rule.onNodeWithTag("tag_home").assertIsSelected().performClick()
        // tax is on both: a tap takes it off
        rule.onNodeWithTag("tag_tax").performClick()
        rule.waitForIdle()
        assertEquals(listOf("Receipts" to true, "home" to true, "tax" to false), calls)
        rule.onNodeWithTag("tag_tax").assertIsNotSelected()
        assertEquals(mapOf("a" to listOf("Receipts", "home"), "b" to listOf("home", "Receipts")), edit.docs)
    }
}
