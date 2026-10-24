package com.albus.flatscan.output

import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.zip.Deflater
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** A page image already encoded for the PDF. */
sealed class PdfImage(val width: Int, val height: Int) {
    /** JPEG bytes as they are (DCTDecode); [components] 1 (gray) or 3 (RGB). */
    class Jpeg(val bytes: ByteArray, width: Int, height: Int, val components: Int) : PdfImage(width, height)

    /** 1 bit per pixel, rows packed from the most significant bit, 1 = white ([Bilevel]); Flate or CCITT G4. */
    class Bilevel(val packed: ByteArray, width: Int, height: Int, val g4: ByteArray? = null) : PdfImage(width, height)
}

/** One recognised word: its text (logical order) and box in the page image's pixels (left, top, right, bottom). */
data class OcrWord(val text: String, val left: Int, val top: Int, val right: Int, val bottom: Int)

/** One line of words, in reading order; [rtl]: a right-to-left line (Hebrew, Arabic). */
data class OcrLine(val words: List<OcrWord>, val rtl: Boolean)

/** A page of the PDF: its size in points, the image and where it goes (points, from the bottom-left), and its text. */
data class PdfPage(
    val widthPt: Double,
    val heightPt: Double,
    val image: PdfImage,
    val x: Double, val y: Double, val w: Double, val h: Double,
    /** The image's pixel size the [lines] boxes refer to (the OCR image; any scale of the placed image). */
    val textWidth: Int = image.width,
    val textHeight: Int = image.height,
    val lines: List<OcrLine> = emptyList(),
)

/** Document information: Title, Author, Creator, Producer, CreationDate, ModDate; [lang] e.g. "he" or "en". */
data class PdfInfo(
    val title: String,
    val creator: String = "FlatScan",
    val producer: String = "FlatScan PDF writer",
    val created: Long = System.currentTimeMillis(),
    val modified: Long = created,
    val author: String? = null,
    val lang: String? = null,
)

/**
 * A small PDF 1.7 writer for scanned pages (no Android dependency: tested on the JVM, PdfWriterTest and
 * tools/pdf_check). Each page is one image, JPEG bytes passed through unchanged or a 1-bit image (Flate or
 * CCITT G4), with an invisible text layer over it (text render mode 3): every recognised word is placed on
 * its own box, in a glyph-less CID font whose ToUnicode map gives the text back to search and copy. The text
 * layer is drawn in the coordinates of the very image that is placed, so it lines up with it by construction.
 * Right-to-left lines are written in visual order, as text PDFs from word processors store Hebrew: the
 * line's words left to right, each word's characters reversed. poppler, PDFium and pdf.js then give the
 * words back in logical order (search finds them); the logical order is found by none of the three
 * (tools/pdf_check/check.py, docs/results/output/).
 */
class PdfWriter(private val out: OutputStream, private val rtlOrder: RtlOrder = RtlOrder.Visual) {
    /** How right-to-left lines are written in the text layer (see the class comment). */
    enum class RtlOrder { Visual, Logical }

    private var pos = 0L
    private val offsets = mutableListOf<Long>()
    private var nextId = 1

    private fun write(s: String) = write(s.toByteArray(Charsets.ISO_8859_1))
    private fun write(b: ByteArray) {
        out.write(b)
        pos += b.size
    }

    private fun newId(): Int = nextId++.also { offsets += -1L }

    private fun beginObj(id: Int) {
        offsets[id - 1] = pos
        write("$id 0 obj\n")
    }

    private fun obj(id: Int, body: String) {
        beginObj(id)
        write(body)
        write("\nendobj\n")
    }

    private fun stream(id: Int, dict: String, data: ByteArray) {
        beginObj(id)
        write("<< $dict /Length ${data.size} >>\nstream\n")
        write(data)
        write("\nendstream\nendobj\n")
    }

    fun write(pages: List<PdfPage>, info: PdfInfo) {
        require(pages.isNotEmpty()) { "no pages" }
        write("%PDF-1.7\n%âãÏÓ\n")
        val catalog = newId()
        val pagesId = newId()
        val infoId = newId()
        val hasText = pages.any { p -> p.lines.any { it.words.isNotEmpty() } }
        val fontId = if (hasText) writeFont() else 0
        val kids = mutableListOf<Int>()
        for (p in pages) kids += writePage(p, pagesId, fontId)
        obj(pagesId, "<< /Type /Pages /Kids [${kids.joinToString(" ") { "$it 0 R" }}] /Count ${kids.size} >>")
        val lang = info.lang?.let { " /Lang ${pdfString(it)}" } ?: ""
        obj(catalog, "<< /Type /Catalog /Pages $pagesId 0 R$lang >>")
        val sb = StringBuilder("<< /Title ${pdfString(info.title)} /Creator ${pdfString(info.creator)} /Producer ${pdfString(info.producer)}")
        info.author?.let { sb.append(" /Author ${pdfString(it)}") }
        sb.append(" /CreationDate ${pdfString(pdfDate(info.created))} /ModDate ${pdfString(pdfDate(info.modified))} >>")
        obj(infoId, sb.toString())
        val xref = pos
        write("xref\n0 ${offsets.size + 1}\n0000000000 65535 f \n")
        for (o in offsets) write("%010d 00000 n \n".format(Locale.US, o))
        val idHex = java.util.UUID.randomUUID().toString().replace("-", "")
        write("trailer\n<< /Size ${offsets.size + 1} /Root $catalog 0 R /Info $infoId 0 R /ID [<$idHex> <$idHex>] >>\nstartxref\n$xref\n%%EOF\n")
        out.flush()
    }

    private fun writePage(p: PdfPage, parent: Int, font: Int): Int {
        val img = newId()
        when (val im = p.image) {
            is PdfImage.Jpeg -> stream(
                img,
                "/Type /XObject /Subtype /Image /Width ${im.width} /Height ${im.height} /ColorSpace /${if (im.components == 1) "DeviceGray" else "DeviceRGB"} /BitsPerComponent 8 /Filter /DCTDecode",
                im.bytes,
            )
            is PdfImage.Bilevel -> if (im.g4 != null) stream(
                img,
                "/Type /XObject /Subtype /Image /Width ${im.width} /Height ${im.height} /ColorSpace /DeviceGray /BitsPerComponent 1 " +
                    "/Filter /CCITTFaxDecode /DecodeParms << /K -1 /Columns ${im.width} /Rows ${im.height} /BlackIs1 false >>",
                im.g4,
            ) else stream(
                img,
                "/Type /XObject /Subtype /Image /Width ${im.width} /Height ${im.height} /ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /FlateDecode",
                deflate(im.packed),
            )
        }
        val content = StringBuilder()
        content.append("q ${n(p.w)} 0 0 ${n(p.h)} ${n(p.x)} ${n(p.y)} cm /Im0 Do Q\n")
        if (font != 0) content.append(textLayer(p))
        val contentId = newId()
        stream(contentId, "/Filter /FlateDecode", deflate(content.toString().toByteArray(Charsets.ISO_8859_1)))
        val page = newId()
        val fontRes = if (font != 0) " /Font << /F1 $font 0 R >>" else ""
        obj(
            page,
            "<< /Type /Page /Parent $parent 0 R /MediaBox [0 0 ${n(p.widthPt)} ${n(p.heightPt)}] " +
                "/Resources << /XObject << /Im0 $img 0 R >>$fontRes >> /Contents $contentId 0 R >>",
        )
        return page
    }

    /** The invisible words of [p]: each on its box, scaled horizontally (Tz) to the box width. */
    private fun textLayer(p: PdfPage): String {
        if (p.lines.isEmpty()) return ""
        val sx = p.w / p.textWidth
        val sy = p.h / p.textHeight
        val sb = StringBuilder("BT 3 Tr\n")
        for (line in p.lines) {
            val all = line.words.filter { it.text.isNotBlank() && it.right > it.left && it.bottom > it.top }
            if (all.isEmpty()) continue
            val hPx = all.map { it.bottom - it.top }.sorted()[all.size / 2]
            val size = max(1.0, hPx * sy)
            // OCR "lines" can run across table columns: a gap wider than 1.5 line heights starts a new segment,
            // written without a space across the gap (extractors would join the columns otherwise)
            val byX = all.sortedBy { it.left }
            val segments = mutableListOf(mutableListOf(byX[0]))
            for (w in byX.drop(1)) {
                if (w.left - segments.last().last().right > 1.5 * hPx) segments += mutableListOf(w) else segments.last() += w
            }
            val visual = line.rtl && rtlOrder == RtlOrder.Visual
            // visual order: left to right, right-to-left words' characters reversed (as text PDFs from word
            // processors store Hebrew); logical: as recognised (right to left for a right-to-left line)
            val ordered = if (line.rtl && !visual) segments.asReversed().map { it.asReversed() } else segments
            for (seg in ordered) for ((i, w) in seg.withIndex()) {
                val last = i == seg.lastIndex
                // a left-to-right run (Latin, digits) keeps its order inside a right-to-left line (bidi)
                val body = if (visual && OcrLayout.isRtl(w.text)) reverseGraphemes(w.text) else w.text
                val text = if (last) body else "$body "
                val x0 = p.x + w.left * sx
                val wordPt = (w.right - w.left) * sx
                val advPt = if (last) wordPt else {
                    val next = seg[i + 1]
                    val gap = if (line.rtl && !visual) (w.left - next.right) * sx else (next.left - w.right) * sx
                    wordPt + max(0.0, gap)
                }
                val natural = text.length * GLYPH_ADVANCE * size
                val tz = (100.0 * advPt / natural).coerceIn(1.0, 10_000.0)
                // the glyph box runs from -0.2 to 0.8 em: with the baseline 0.2 of the height up, it covers the word
                val baseY = p.y + p.h - w.bottom * sy + 0.2 * (w.bottom - w.top) * sy
                sb.append("/F1 ${n(size)} Tf ${n(tz)} Tz 1 0 0 1 ${n(x0)} ${n(baseY)} Tm <").append(hex(text)).append("> Tj\n")
            }
        }
        sb.append("ET\n")
        return sb.toString()
    }

    /** The glyph-less CID font (Type0, Identity-H, CID = UTF-16 code unit, every CID drawn as glyph 1). */
    private fun writeFont(): Int {
        val ttf = PdfWriter::class.java.getResourceAsStream("/flatscan/glyphless.ttf")?.use { it.readBytes() } ?: error("glyphless.ttf missing")
        val type0 = newId()
        val cidFont = newId()
        val descriptor = newId()
        val file = newId()
        val cidToGid = newId()
        val toUnicode = newId()
        obj(type0, "<< /Type /Font /Subtype /Type0 /BaseFont /GlyphLessFont /Encoding /Identity-H /DescendantFonts [$cidFont 0 R] /ToUnicode $toUnicode 0 R >>")
        obj(
            cidFont,
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GlyphLessFont /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> " +
                "/FontDescriptor $descriptor 0 R /DW ${(GLYPH_ADVANCE * 1000).roundToInt()} /CIDToGIDMap $cidToGid 0 R >>",
        )
        obj(
            descriptor,
            "<< /Type /FontDescriptor /FontName /GlyphLessFont /Flags 5 /FontBBox [0 0 ${(GLYPH_ADVANCE * 1000).roundToInt()} 1000] " +
                "/ItalicAngle 0 /Ascent 1000 /Descent 0 /CapHeight 1000 /StemV 80 /FontFile2 $file 0 R >>",
        )
        stream(file, "/Length1 ${ttf.size} /Filter /FlateDecode", deflate(ttf))
        // every CID -> glyph 1 (the blank glyph)
        val map = ByteArray(65536 * 2) { if (it % 2 == 1) 1 else 0 }
        stream(cidToGid, "/Filter /FlateDecode", deflate(map))
        stream(toUnicode, "/Filter /FlateDecode", deflate(TO_UNICODE.toByteArray(Charsets.ISO_8859_1)))
        return type0
    }

    companion object {
        /** Advance of the blank glyph, in em (500 / 1000 units). */
        const val GLYPH_ADVANCE = 0.5

        /** CID = UTF-16 code unit: 255 ranges (a bfrange may only change its last byte), surrogates left out. */
        private val TO_UNICODE: String = buildString {
            append("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n")
            append("/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n")
            append("1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n")
            val highs = (0..255).filter { it !in 0xD8..0xDF }
            for (chunk in highs.chunked(100)) {
                append("${chunk.size} beginbfrange\n")
                for (hb in chunk) append("<%02X00> <%02XFF> <%02X00>\n".format(hb, hb, hb))
                append("endbfrange\n")
            }
            append("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n")
        }

        fun deflate(b: ByteArray): ByteArray {
            val d = Deflater(Deflater.BEST_COMPRESSION)
            d.setInput(b)
            d.finish()
            val o = ByteArrayOutputStream(max(64, b.size / 4))
            val buf = ByteArray(65536)
            while (!d.finished()) o.write(buf, 0, d.deflate(buf))
            d.end()
            return o.toByteArray()
        }

        /** [s] with its user-perceived characters (base letter + marks such as niqqud) in reverse order. */
        fun reverseGraphemes(s: String): String {
            val it = java.text.BreakIterator.getCharacterInstance(Locale.ROOT)
            it.setText(s)
            val parts = mutableListOf<String>()
            var start = it.first()
            var end = it.next()
            while (end != java.text.BreakIterator.DONE) {
                parts += s.substring(start, end)
                start = end
                end = it.next()
            }
            return parts.asReversed().joinToString("")
        }

        private fun hex(s: String): String {
            val sb = StringBuilder(s.length * 4)
            for (ch in s) {
                // non-BMP characters are written as their surrogate halves, which the map leaves unmapped
                sb.append("%04X".format(ch.code))
            }
            return sb.toString()
        }

        /** A PDF text string: ASCII in parentheses (escaped), anything else UTF-16BE with a byte-order mark. */
        fun pdfString(s: String): String {
            if (s.all { it.code in 32..126 }) return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"
            return "<FEFF" + s.map { "%04X".format(it.code) }.joinToString("") + ">"
        }

        /** D:YYYYMMDDHHmmSS+hh'mm' in the phone's time zone. */
        fun pdfDate(t: Long): String {
            val tz = TimeZone.getDefault()
            val off = tz.getOffset(t) / 60000
            val base = SimpleDateFormat("yyyyMMddHHmmss", Locale.US).apply { timeZone = tz }.format(Date(t))
            val sign = if (off >= 0) "+" else "-"
            return "D:$base$sign%02d'%02d'".format(Locale.US, abs(off) / 60, abs(off) % 60)
        }

        private fun n(v: Double): String {
            val r = (v * 1000).roundToInt() / 1000.0
            return if (r == r.toLong().toDouble()) r.toLong().toString() else "%.3f".format(Locale.US, r).trimEnd('0').trimEnd('.')
        }

        /**
         * Where an image of [iw] x [ih] px goes on a page of [pw] x [ph] pt with [margin] pt: as large as fits,
         * centred; an image within 1% of the box's shape fills it exactly. Returns x, y, w, h in points.
         */
        fun place(iw: Int, ih: Int, pw: Double, ph: Double, margin: Double): DoubleArray {
            val bw = max(1.0, pw - 2 * margin)
            val bh = max(1.0, ph - 2 * margin)
            if (abs(iw.toDouble() / ih / (bw / bh) - 1) < 0.01) return doubleArrayOf(margin, margin, bw, bh)
            val k = min(bw / iw, bh / ih)
            val w = iw * k
            val h = ih * k
            return doubleArrayOf((pw - w) / 2, (ph - h) / 2, w, h)
        }
    }
}
