package com.albus.flatscan.output

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.albus.flatscan.ColorMode
import com.albus.flatscan.ExportDoc
import com.albus.flatscan.Exporter
import com.albus.flatscan.OutputFormat
import com.albus.flatscan.OutputOptions
import com.albus.flatscan.PdfPageSize
import com.albus.flatscan.SizePreset
import com.albus.flatscan.data.Library
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * The export path as the app runs it (Android bitmaps through Robolectric's native graphics): presets,
 * colour modes, the exact size estimate, black and white pictures, and the plan's size target, black and
 * white at least 3x smaller than colour, measured on a sample set of finished pages: the two test pages,
 * plus every *.jpg in FS_SAMPLE_PAGES when set (pages the pipeline made from the test photos, testdata/).
 * The table goes to build/output_sizes.md.
 */
@RunWith(AndroidJUnit4::class)
class OutputTest {
    private val ctx: Context = ApplicationProvider.getApplicationContext()

    private fun res(name: String) = File(javaClass.getResource("/output/$name")!!.toURI())

    private fun samples(): List<File> {
        val extra = System.getenv("FS_SAMPLE_PAGES")?.let { File(it).listFiles { f -> f.extension == "jpg" }?.sorted() }.orEmpty()
        return listOf(res("hebrew_page.jpg"), res("english_page.jpg")) + extra
    }

    @Test
    fun blackAndWhiteIsAtLeastThreeTimesSmallerThanColour() {
        val rows = StringBuilder("| preset | pages | colour (PDF image bytes) | grayscale | black & white (G4) | colour / B&W |\n|---|---|---|---|---|---|\n")
        var balancedRatio = 0.0
        for (preset in SizePreset.entries) {
            var c = 0L; var g = 0L; var b = 0L
            for (f in samples()) {
                fun size(mode: ColorMode): Int = when (val im = Exporter.encodePage(f, OutputOptions(color = mode, size = preset)).first) {
                    is PdfImage.Jpeg -> im.bytes.size
                    is PdfImage.Bilevel -> im.g4!!.size
                }
                c += size(ColorMode.Color); g += size(ColorMode.Grayscale); b += size(ColorMode.BlackWhite)
            }
            val ratio = c.toDouble() / b
            if (preset == SizePreset.Balanced) balancedRatio = ratio
            rows.append("| ${preset.label} | ${samples().size} | $c | $g | $b | ${"%.1f".format(ratio)}x |\n")
        }
        File("build/output_sizes.md").writeText(rows.toString())
        println(rows)
        assertTrue("colour / black and white at Balanced: $balancedRatio", balancedRatio >= 3.0)
    }

    @Test
    fun presetsMakeSmallerFilesAndNeverEnlarge() {
        val f = res("english_page.jpg")
        val sizes = SizePreset.entries.map { p -> (Exporter.encodePage(f, OutputOptions(size = p)).first as PdfImage.Jpeg).bytes.size }
        assertTrue("small < balanced < high: $sizes", sizes[0] < sizes[1] && sizes[1] < sizes[2])
        // 1600 px across A4 is 193 dpi: Small (150 dpi) scales down, High (300 dpi) keeps the page as it is
        val small = Exporter.encodePage(f, OutputOptions(size = SizePreset.Small)).first
        assertEquals(1240, small.width)
        val high = Exporter.encodePage(f, OutputOptions(size = SizePreset.High)).first as PdfImage.Jpeg
        assertEquals(1600, high.width)
        assertTrue("High colour passes the finished page through", high.bytes.contentEquals(f.readBytes()))
    }

    @Test
    fun blackAndWhitePicturesAreOneBitPngs() {
        val (img, pic, ext) = Exporter.encodePage(res("hebrew_page.jpg"), OutputOptions(color = ColorMode.BlackWhite, format = OutputFormat.Png))
        assertEquals("png", ext)
        assertEquals(1, pic[24].toInt())          // IHDR bit depth
        assertEquals(0, pic[25].toInt())          // grayscale
        val b = BitmapFactory.decodeByteArray(pic, 0, pic.size)
        assertEquals(img.width, b.width)
        assertEquals(img.height, b.height)
        // the decoded picture holds black and white only, and some text is black
        val px = IntArray(b.width * b.height).also { b.getPixels(it, 0, b.width, 0, 0, b.width, b.height) }
        assertTrue(px.all { (it and 0xFFFFFF) == 0 || (it and 0xFFFFFF) == 0xFFFFFF })
        val dark = px.count { (it and 0xFFFFFF) == 0 }.toDouble() / px.size
        assertTrue("ink share $dark", dark in 0.01..0.3)
    }

    @Test
    fun theEstimateIsTheExactPdfSize() {
        val lib = Library.get(ctx)
        val ex = Exporter(ctx, lib)
        val o = OutputOptions(color = ColorMode.Grayscale, size = SizePreset.Small, pageSize = PdfPageSize.Letter)
        val pages = listOf("hebrew_page.jpg", "english_page.jpg").map { n ->
            val (img, pic, ext) = Exporter.encodePage(res(n), o)
            val lines = OcrLayout.fromTsv(res(n.replace(".jpg", ".tsv")).readText())
            Exporter.PreparedPage(img, pic, ext, Ocr.PageText(lines, 1600, 2263))
        }
        val p = Exporter.Prepared(ExportDoc("חשבון", 0, 0, listOf("a", "b")), o, pages)
        val written = ByteArrayOutputStream().also { ex.writePdf(p, it) }.size().toLong()
        // the file ID differs between two writes, never the length
        assertEquals(written, p.pdfBytes)
        assertEquals("2026-03-07_Bill - 2", ex.fileName(ExportDoc("Bill", java.util.Calendar.getInstance().apply { set(2026, 2, 7) }.timeInMillis, 0, listOf("x", "y")), OutputOptions(), 2))
    }
}
