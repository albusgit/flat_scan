package com.albus.flatscan.output

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * Writes the sample PDFs that tools/pdf_check/check.py opens in three engines (poppler, PDFium, pdf.js):
 * a Hebrew page (the bundled self-test photo as the app flattens it, with Tesseract heb+eng word boxes) and
 * an English page, as colour JPEG with the text layer (right-to-left lines in visual and in logical order),
 * and as black and white (CCITT G4 and Flate). Output: app/build/pdf_check/. The structural checks run here.
 */
@RunWith(AndroidJUnit4::class)
class PdfWriterTest {
    private val out = File("build/pdf_check").apply { mkdirs() }

    private fun res(name: String): File = File(javaClass.getResource("/output/$name")!!.toURI())

    private fun jpegPage(name: String, lines: List<OcrLine>): PdfPage {
        val f = res(name)
        val img = BitmapFactory.decodeFile(f.absolutePath)
        val (x, y, w, h) = PdfWriter.place(img.width, img.height, 595.0, 842.0, 0.0).toList()
        return PdfPage(595.0, 842.0, PdfImage.Jpeg(f.readBytes(), img.width, img.height, 3), x, y, w, h, img.width, img.height, lines)
    }

    private fun bilevelPage(name: String, lines: List<OcrLine>, g4: Boolean): PdfPage {
        val img = BitmapFactory.decodeFile(res(name).absolutePath)
        val argb = IntArray(img.width * img.height).also { img.getPixels(it, 0, img.width, 0, 0, img.width, img.height) }
        val packed = Bilevel.threshold(argb, img.width, img.height)
        val im = PdfImage.Bilevel(packed, img.width, img.height, if (g4) Bilevel.encodeG4(packed, img.width, img.height) else null)
        val (x, y, w, h) = PdfWriter.place(img.width, img.height, 595.0, 842.0, 0.0).toList()
        return PdfPage(595.0, 842.0, im, x, y, w, h, img.width, img.height, lines)
    }

    private fun lines(name: String) = OcrLayout.fromTsv(res(name).readText())

    private fun write(file: String, pages: List<PdfPage>, order: PdfWriter.RtlOrder = PdfWriter.RtlOrder.Visual, title: String = "Test"): ByteArray {
        val bytes = ByteArrayOutputStream().also { PdfWriter(it, order).write(pages, PdfInfo(title, lang = "he")) }.toByteArray()
        File(out, file).writeBytes(bytes)
        return bytes
    }

    @Test
    fun writesTheSamplePdfs() {
        val he = lines("hebrew_page.tsv")
        val en = lines("english_page.tsv")
        assertTrue("Hebrew lines found", he.count { it.rtl } > 20)
        assertTrue(en.none { it.rtl })
        write("hebrew_visual.pdf", listOf(jpegPage("hebrew_page.jpg", he)), PdfWriter.RtlOrder.Visual, "חשבון בדיקה")
        write("hebrew_logical.pdf", listOf(jpegPage("hebrew_page.jpg", he)), PdfWriter.RtlOrder.Logical)
        write("english.pdf", listOf(jpegPage("english_page.jpg", en)))
        write("both_color.pdf", listOf(jpegPage("hebrew_page.jpg", he), jpegPage("english_page.jpg", en)))
        val g4 = write("both_bw_g4.pdf", listOf(bilevelPage("hebrew_page.jpg", he, true), bilevelPage("english_page.jpg", en, true)))
        val flate = write("both_bw_flate.pdf", listOf(bilevelPage("hebrew_page.jpg", he, false), bilevelPage("english_page.jpg", en, false)))
        val text = String(g4, Charsets.ISO_8859_1)
        assertTrue(text.startsWith("%PDF-1.7"))
        assertTrue(text.contains("/CCITTFaxDecode"))
        assertTrue(text.trimEnd().endsWith("%%EOF"))
        assertTrue("G4 smaller than Flate: ${g4.size} vs ${flate.size}", g4.size < flate.size)
    }

    @Test
    fun xrefOffsetsPointAtTheirObjects() {
        val bytes = write("xref.pdf", listOf(jpegPage("english_page.jpg", lines("english_page.tsv"))))
        val s = String(bytes, Charsets.ISO_8859_1)
        val start = s.substringAfterLast("startxref\n").trim().substringBefore("\n").toInt()
        val table = s.substring(start).lineSequence().drop(2).takeWhile { it.matches(Regex("\\d{10} \\d{5} [nf] ?")) }.toList()
        table.drop(1).forEachIndexed { i, row ->
            val off = row.substring(0, 10).toInt()
            assertTrue("object ${i + 1} at $off", s.startsWith("${i + 1} 0 obj", off))
        }
    }

    @Test
    fun metadataStringsAndDates() {
        assertEquals("(A \\(b\\) c\\\\)", PdfWriter.pdfString("A (b) c\\"))
        assertEquals("<FEFF05D005D1>", PdfWriter.pdfString("אב"))
        assertTrue(PdfWriter.pdfDate(0).matches(Regex("D:\\d{14}[+-]\\d{2}'\\d{2}'")))
        assertEquals("cba", PdfWriter.reverseGraphemes("abc"))
        // a letter with its niqqud stays together
        assertEquals("ב" + "א\u05B8", PdfWriter.reverseGraphemes("א\u05B8ב"))
    }

    @Test
    fun placementCentresAndFills() {
        // an A4-shaped image fills an A4 page exactly
        assertEquals(listOf(0.0, 0.0, 595.0, 842.0), PdfWriter.place(1600, 2263, 595.0, 842.0, 0.0).toList())
        // a receipt is centred, full height
        val r = PdfWriter.place(500, 2000, 595.0, 842.0, 36.0)
        assertEquals(842.0 - 72, r[3], 1e-9)
        assertEquals((595.0 - r[2]) / 2, r[0], 1e-9)
    }

    @Test
    fun g4DecodesToTheSamePixels() {
        // a small image with runs of every kind: long white, short black, stripes, diagonal, edges
        val w = 97; val h = 41
        val argb = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val black = (x in 10..12) || (y % 7 == 0 && x > 30) || (x == y) || (x > 80 && (x + y) % 3 == 0) || x == 0 || (y == h - 1 && x < 50)
            argb[y * w + x] = if (black) 0xFF000000.toInt() else 0xFFFFFFFF.toInt()
        }
        // a plain 50% threshold (Bradley's would hollow out large black areas, not wanted here)
        val stride = (w + 7) / 8
        val packed = ByteArray(stride * h) { 0xFF.toByte() }
        for (y in 0 until h) for (x in 0 until w) if ((argb[y * w + x] and 0xFF) < 128) {
            packed[y * stride + x / 8] = (packed[y * stride + x / 8].toInt() and (0x80 ushr (x % 8)).inv()).toByte()
        }
        val g4 = Bilevel.encodeG4(packed, w, h)
        File(out, "g4_${w}x$h.raw").writeBytes(packed)
        File(out, "g4_${w}x$h.g4").writeBytes(g4)
        // decoded by tools/pdf_check/check.py (libtiff via Pillow) and compared with the .raw
        assertTrue(g4.size < packed.size)
    }
}
