package com.albus.flatscan

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** The page's processing parameters as stored in meta.json: the new fields round-trip, old files still load. */
@RunWith(AndroidJUnit4::class)
class PageMetaTest {
    private fun meta() = PageMeta("p", FloatArray(8) { it.toFloat() }, null, 0, false, 0f, RenderParams(), FloatArray(0), 0)

    @Test
    fun whiteboardModeAndFingerEditsRoundTrip() {
        val strokes = floatArrayOf(1f, 0.02f, 2f, 0.1f, 0.2f, 0.15f, 0.25f, 0f, 0.01f, 1f, 0.5f, 0.5f)
        val m = meta().copy(mode = CaptureMode.Whiteboard, fingersAuto = false, fingerStrokes = strokes,
            params = RenderParams(filter = FilterKind.ProjectorDark))
        val back = PageMeta.fromJson(JSONObject(m.toJson().toString()))
        assertEquals(CaptureMode.Whiteboard, back.mode)
        assertFalse(back.fingersAuto)
        assertTrue(strokes.contentEquals(back.fingerStrokes))
        assertEquals(FilterKind.ProjectorDark, back.params.filter)
        assertTrue(back.hasEdits)
    }

    @Test
    fun filesWrittenBeforeTheseFieldsLoadWithDefaults() {
        val j = meta().toJson()
        assertFalse(j.has("mode") || j.has("fingers_auto") || j.has("finger_strokes"))   // defaults are not written
        val back = PageMeta.fromJson(j)
        assertEquals(CaptureMode.Document, back.mode)
        assertTrue(back.fingersAuto)
        assertFalse(back.hasEdits)
    }

    @Test
    fun theModeIsPartOfTheGeometry() {
        val a = meta().geometryKey(1600, true, true)
        val b = meta().copy(mode = CaptureMode.Whiteboard).geometryKey(1600, true, true)
        assertTrue(a != b)
        // a document page's key is what it was before modes existed (stored sessions stay valid)
        assertFalse(a.contains("mode="))
    }

    @Test
    fun drawnLinesShadowsAndAiFillRoundTrip() {
        // two lines: group 1 with 2 points, group 2 with 3 points
        val drawn = floatArrayOf(1f, 2f, 0.1f, 0.5f, 0.9f, 0.52f, 2f, 3f, 0.3f, 0.1f, 0.31f, 0.5f, 0.3f, 0.9f)
        val m = meta().copy(userLines = drawn, removeShadows = false, fingerFill = 1)
        val back = PageMeta.fromJson(JSONObject(m.toJson().toString()))
        assertTrue(drawn.contentEquals(back.userLines))
        assertFalse(back.removeShadows)
        assertEquals(1, back.fingerFill)
        assertTrue(back.hasEdits)   // the AI fill is applied after processing, like the brush
        assertEquals(listOf(0, 6), drawnLineStarts(drawn.toList()))
        assertEquals(listOf(1, 2), drawnLineGroups(drawn.toList()))
        // both re-process the page; a page without them keeps the key it had (stored sessions stay valid)
        val plain = meta().geometryKey(1600, true, true)
        assertTrue(plain != meta().copy(userLines = drawn).geometryKey(1600, true, true))
        assertTrue(plain != meta().copy(removeShadows = false).geometryKey(1600, true, true))
        assertEquals(plain, meta().copy(fingerFill = 1).geometryKey(1600, true, true))
        assertFalse(meta().toJson().has("user_lines") || meta().toJson().has("remove_shadows") || meta().toJson().has("finger_fill"))
    }

    @Test
    fun aCutOffDrawnLineListIsReadUpToTheLastWholeLine() {
        assertEquals(listOf(0), drawnLineStarts(listOf(1f, 2f, 0.1f, 0.1f, 0.2f, 0.1f, 1f, 5f, 0.3f)))
    }

    @Test
    fun filtersOfferedPerMode() {
        assertFalse(FilterKind.Whiteboard in FilterKind.forMode(CaptureMode.Document))
        assertEquals(FilterKind.Whiteboard, FilterKind.forMode(CaptureMode.Whiteboard).first())
    }
}
