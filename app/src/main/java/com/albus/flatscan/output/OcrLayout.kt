package com.albus.flatscan.output

/** Turning OCR output into lines of words for the PDF text layer and plain text for search. */
object OcrLayout {
    /** Whether [text] reads right to left: its first strong character is Hebrew, Arabic or another RTL script. */
    fun isRtl(text: String): Boolean {
        for (ch in text) {
            when (Character.getDirectionality(ch)) {
                Character.DIRECTIONALITY_RIGHT_TO_LEFT, Character.DIRECTIONALITY_RIGHT_TO_LEFT_ARABIC -> return true
                Character.DIRECTIONALITY_LEFT_TO_RIGHT -> return false
            }
        }
        return false
    }

    /**
     * Lines from Tesseract's TSV output (`tesseract IMAGE out -l heb+eng tsv`, or TessBaseAPI.getTSVText):
     * level-5 rows are words, grouped by block / paragraph / line, in Tesseract's reading order. A line is
     * right to left when most of its letters are.
     */
    fun fromTsv(tsv: String, minConf: Double = 0.0): List<OcrLine> {
        val groups = LinkedHashMap<String, MutableList<OcrWord>>()
        for (row in tsv.lineSequence().drop(1)) {
            val c = row.split('\t')
            if (c.size < 12 || c[0] != "5") continue
            val text = c[11].trim()
            if (text.isEmpty()) continue
            if ((c[10].toDoubleOrNull() ?: 0.0) < minConf) continue
            val l = c[6].toInt(); val t = c[7].toInt(); val w = c[8].toInt(); val h = c[9].toInt()
            groups.getOrPut("${c[2]}/${c[3]}/${c[4]}") { mutableListOf() } += OcrWord(text, l, t, l + w, t + h)
        }
        return groups.values.map { ws -> OcrLine(ws, rtlShare(ws) > 0.5) }
    }

    private fun rtlShare(ws: List<OcrWord>): Double {
        var rtl = 0; var all = 0
        for (w in ws) for (ch in w.text) when (Character.getDirectionality(ch)) {
            Character.DIRECTIONALITY_RIGHT_TO_LEFT, Character.DIRECTIONALITY_RIGHT_TO_LEFT_ARABIC -> { rtl++; all++ }
            Character.DIRECTIONALITY_LEFT_TO_RIGHT -> all++
        }
        return if (all == 0) 0.0 else rtl.toDouble() / all
    }

    /** The page's text, one line per line (search). */
    fun plainText(lines: List<OcrLine>): String = lines.joinToString("\n") { l -> l.words.joinToString(" ") { it.text } }

    /** The language most of the text is in, for the PDF's /Lang: "he" when Hebrew letters dominate, else "en". */
    fun language(lines: List<OcrLine>): String? {
        var he = 0; var lat = 0
        for (l in lines) for (w in l.words) for (ch in w.text) {
            if (ch in '\u0590'..'\u05FF') he++ else if (ch.isLetter()) lat++
        }
        return when { he == 0 && lat == 0 -> null; he >= lat -> "he"; else -> "en" }
    }
}
