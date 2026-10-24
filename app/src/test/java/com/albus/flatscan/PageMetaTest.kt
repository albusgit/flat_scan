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
    fun filtersOfferedPerMode() {
        assertFalse(FilterKind.Whiteboard in FilterKind.forMode(CaptureMode.Document))
        assertEquals(FilterKind.Whiteboard, FilterKind.forMode(CaptureMode.Whiteboard).first())
    }
}
