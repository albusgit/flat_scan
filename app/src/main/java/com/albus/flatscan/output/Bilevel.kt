package com.albus.flatscan.output

import kotlin.math.max
import kotlin.math.min

/**
 * Black-and-white output: a page turned into 1 bit per pixel and compressed with CCITT Group 4 (ITU-T T.6,
 * what fax and scanners use; usually several times smaller than Flate). Pure Kotlin, tested on the JVM
 * (BilevelTest decodes the G4 data with libtiff through Pillow when it is installed).
 */
object Bilevel {
    /**
     * Black where a pixel is clearly darker than its neighbourhood (Bradley's adaptive threshold, a
     * [window]-px mean, [t] darker) and darker than [maxBlack]: text and lines stay crisp on a page that is
     * not perfectly white, and pale backgrounds go white. [argb]: the page, row by row.
     * Returns rows packed from the most significant bit, 1 = white (PDF DeviceGray with 1 bit).
     */
    fun threshold(argb: IntArray, w: Int, h: Int, window: Int = max(16, w / 24), t: Double = 0.15, maxBlack: Int = 200): ByteArray {
        val lum = IntArray(w * h)
        for (i in lum.indices) {
            val c = argb[i]
            lum[i] = (((c shr 16) and 255) * 299 + ((c shr 8) and 255) * 587 + (c and 255) * 114) / 1000
        }
        // integral image (longs: a 4000 x 6000 page sums past Int)
        val ii = LongArray((w + 1) * (h + 1))
        for (y in 0 until h) {
            var row = 0L
            for (x in 0 until w) {
                row += lum[y * w + x]
                ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + row
            }
        }
        val r = window / 2
        val stride = (w + 7) / 8
        val out = ByteArray(stride * h) { 0xFF.toByte() }
        for (y in 0 until h) {
            val y0 = max(0, y - r); val y1 = min(h, y + r + 1)
            for (x in 0 until w) {
                val x0 = max(0, x - r); val x1 = min(w, x + r + 1)
                val sum = ii[y1 * (w + 1) + x1] - ii[y0 * (w + 1) + x1] - ii[y1 * (w + 1) + x0] + ii[y0 * (w + 1) + x0]
                val mean = sum.toDouble() / ((x1 - x0) * (y1 - y0))
                val v = lum[y * w + x]
                if (v < maxBlack && v < mean * (1 - t)) {
                    val i = y * stride + x / 8
                    out[i] = (out[i].toInt() and (0x80 ushr (x % 8)).inv()).toByte()
                }
            }
        }
        return out
    }

    /**
     * A 1-bit grayscale PNG of a packed image (1 = white, as [threshold] makes it): black and white pages as
     * pictures, 8 pixels per byte.
     */
    fun png(packed: ByteArray, w: Int, h: Int): ByteArray {
        val stride = (w + 7) / 8
        val raw = ByteArray((stride + 1) * h)
        for (y in 0 until h) System.arraycopy(packed, y * stride, raw, y * (stride + 1) + 1, stride)   // filter byte 0 (none) per row
        val o = java.io.ByteArrayOutputStream()
        fun chunk(type: String, data: ByteArray) {
            val t = type.toByteArray(Charsets.US_ASCII)
            o.write(java.nio.ByteBuffer.allocate(4).putInt(data.size).array())
            o.write(t)
            o.write(data)
            val crc = java.util.zip.CRC32().apply { update(t); update(data) }
            o.write(java.nio.ByteBuffer.allocate(4).putInt(crc.value.toInt()).array())
        }
        o.write(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 13, 10, 26, 10))
        // width, height, bit depth 1, colour type 0 (grayscale), deflate, adaptive filtering, no interlace
        chunk("IHDR", java.nio.ByteBuffer.allocate(13).putInt(w).putInt(h).put(1).put(0).put(0).put(0).put(0).array())
        chunk("IDAT", PdfWriter.deflate(raw))
        chunk("IEND", ByteArray(0))
        return o.toByteArray()
    }

    /** Whether pixel [x] of a packed row is black. */
    private fun black(row: ByteArray, off: Int, x: Int): Boolean = (row[off + x / 8].toInt() and (0x80 ushr (x % 8))) == 0

    /**
     * CCITT G4 (T.6) of a packed 1-bit image (1 = white, as [threshold] makes it); decoded with
     * /K -1 /Columns w /BlackIs1 false. Ends with EOFB.
     */
    fun encodeG4(packed: ByteArray, w: Int, h: Int): ByteArray {
        val stride = (w + 7) / 8
        val bits = BitWriter(packed.size / 8 + 64)
        // changing elements of the reference and coding lines (x positions where the colour changes, then w, w)
        var ref = IntArray(2) { w }
        var refN = 0
        val cur = IntArray(w + 2)
        for (y in 0 until h) {
            val off = y * stride
            var n = 0
            var prev = false   // imaginary white pixel before the line
            for (x in 0 until w) {
                val b = black(packed, off, x)
                if (b != prev) { cur[n++] = x; prev = b }
            }
            cur[n] = w; cur[n + 1] = w
            var a0 = -1
            var colorBlack = false
            var ai = 0     // index into cur: first change after a0
            while (true) {
                // a1: next change after a0 on the coding line; b1: first change on the reference line after a0 of the opposite colour to a0's
                while (ai < n && cur[ai] <= a0) ai++
                val a1 = if (ai < n) cur[ai] else w
                val a2 = if (ai + 1 < n) cur[ai + 1] else w
                var bi = 0
                // reference changes alternate white->black (even index) and black->white (odd index)
                while (bi < refN && (ref[bi] <= a0 || ((bi % 2 == 0) == colorBlack))) bi++
                val b1 = if (bi < refN) ref[bi] else w
                val b2 = if (bi + 1 < refN) ref[bi + 1] else w
                if (b2 < a1) {
                    bits.put(0b0001, 4)              // pass
                    a0 = b2
                } else if (kotlin.math.abs(a1 - b1) <= 3) {
                    val d = a1 - b1
                    when (d) {                         // vertical
                        0 -> bits.put(0b1, 1)
                        1 -> bits.put(0b011, 3)
                        2 -> bits.put(0b000011, 6)
                        3 -> bits.put(0b0000011, 7)
                        -1 -> bits.put(0b010, 3)
                        -2 -> bits.put(0b000010, 6)
                        -3 -> bits.put(0b0000010, 7)
                    }
                    a0 = a1
                    colorBlack = !colorBlack
                } else {
                    bits.put(0b001, 3)               // horizontal
                    val start = max(0, a0)
                    run(bits, a1 - start, colorBlack)
                    run(bits, a2 - a1, !colorBlack)
                    a0 = a2
                }
                if (a0 >= w) break
            }
            // the coding line becomes the reference line
            if (ref.size < n + 2) ref = IntArray(n + 2)
            System.arraycopy(cur, 0, ref, 0, n + 2)
            refN = n
        }
        bits.put(0b000000000001, 12)   // EOFB
        bits.put(0b000000000001, 12)
        return bits.toByteArray()
    }

    private fun run(bits: BitWriter, len: Int, blackRun: Boolean) {
        val t = if (blackRun) BLACK else WHITE
        var l = len
        while (l >= 2560) { bits.put(t.getValue(2560)); l -= 2560 }
        if (l >= 64) {
            val m = l / 64 * 64
            bits.put(t.getValue(m))
            l -= m
        }
        bits.put(t.getValue(l))
    }

    private class BitWriter(cap: Int) {
        private var buf = ByteArray(max(16, cap))
        private var n = 0          // bytes complete
        private var acc = 0
        private var nbits = 0

        fun put(code: String) {
            for (c in code) put(if (c == '1') 1 else 0, 1)
        }

        fun put(value: Int, len: Int) {
            for (i in len - 1 downTo 0) {
                acc = (acc shl 1) or ((value ushr i) and 1)
                if (++nbits == 8) {
                    if (n == buf.size) buf = buf.copyOf(buf.size * 2)
                    buf[n++] = acc.toByte()
                    acc = 0
                    nbits = 0
                }
            }
        }

        fun toByteArray(): ByteArray {
            val extra = if (nbits > 0) 1 else 0
            val out = buf.copyOf(n + extra)
            if (extra == 1) out[n] = (acc shl (8 - nbits)).toByte()
            return out
        }
    }

    /** White run-length codes (ITU-T T.4 tables 2-3, from pdfminer.six's ccitt.py): run -> bit string. */
    private val WHITE: Map<Int, String> = mapOf(
        0 to "00110101", 1 to "000111", 2 to "0111", 3 to "1000", 4 to "1011", 5 to "1100", 6 to "1110",
        7 to "1111", 8 to "10011", 9 to "10100", 10 to "00111", 11 to "01000", 12 to "001000", 13 to "000011",
        14 to "110100", 15 to "110101", 16 to "101010", 17 to "101011", 18 to "0100111", 19 to "0001100",
        20 to "0001000", 21 to "0010111", 22 to "0000011", 23 to "0000100", 24 to "0101000", 25 to "0101011",
        26 to "0010011", 27 to "0100100", 28 to "0011000", 29 to "00000010", 30 to "00000011", 31 to "00011010",
        32 to "00011011", 33 to "00010010", 34 to "00010011", 35 to "00010100", 36 to "00010101",
        37 to "00010110", 38 to "00010111", 39 to "00101000", 40 to "00101001", 41 to "00101010",
        42 to "00101011", 43 to "00101100", 44 to "00101101", 45 to "00000100", 46 to "00000101",
        47 to "00001010", 48 to "00001011", 49 to "01010010", 50 to "01010011", 51 to "01010100",
        52 to "01010101", 53 to "00100100", 54 to "00100101", 55 to "01011000", 56 to "01011001",
        57 to "01011010", 58 to "01011011", 59 to "01001010", 60 to "01001011", 61 to "00110010",
        62 to "00110011", 63 to "00110100", 64 to "11011", 128 to "10010", 192 to "010111", 256 to "0110111",
        320 to "00110110", 384 to "00110111", 448 to "01100100", 512 to "01100101", 576 to "01101000",
        640 to "01100111", 704 to "011001100", 768 to "011001101", 832 to "011010010", 896 to "011010011",
        960 to "011010100", 1024 to "011010101", 1088 to "011010110", 1152 to "011010111", 1216 to "011011000",
        1280 to "011011001", 1344 to "011011010", 1408 to "011011011", 1472 to "010011000", 1536 to "010011001",
        1600 to "010011010", 1664 to "011000", 1728 to "010011011", 1792 to "00000001000", 1856 to "00000001100",
        1920 to "00000001101", 1984 to "000000010010", 2048 to "000000010011", 2112 to "000000010100",
        2176 to "000000010101", 2240 to "000000010110", 2304 to "000000010111", 2368 to "000000011100",
        2432 to "000000011101", 2496 to "000000011110", 2560 to "000000011111",
    )
    /** Black run-length codes (ITU-T T.4 tables 2-3, from pdfminer.six's ccitt.py): run -> bit string. */
    private val BLACK: Map<Int, String> = mapOf(
        0 to "0000110111", 1 to "010", 2 to "11", 3 to "10", 4 to "011", 5 to "0011", 6 to "0010", 7 to "00011",
        8 to "000101", 9 to "000100", 10 to "0000100", 11 to "0000101", 12 to "0000111", 13 to "00000100",
        14 to "00000111", 15 to "000011000", 16 to "0000010111", 17 to "0000011000", 18 to "0000001000",
        19 to "00001100111", 20 to "00001101000", 21 to "00001101100", 22 to "00000110111", 23 to "00000101000",
        24 to "00000010111", 25 to "00000011000", 26 to "000011001010", 27 to "000011001011",
        28 to "000011001100", 29 to "000011001101", 30 to "000001101000", 31 to "000001101001",
        32 to "000001101010", 33 to "000001101011", 34 to "000011010010", 35 to "000011010011",
        36 to "000011010100", 37 to "000011010101", 38 to "000011010110", 39 to "000011010111",
        40 to "000001101100", 41 to "000001101101", 42 to "000011011010", 43 to "000011011011",
        44 to "000001010100", 45 to "000001010101", 46 to "000001010110", 47 to "000001010111",
        48 to "000001100100", 49 to "000001100101", 50 to "000001010010", 51 to "000001010011",
        52 to "000000100100", 53 to "000000110111", 54 to "000000111000", 55 to "000000100111",
        56 to "000000101000", 57 to "000001011000", 58 to "000001011001", 59 to "000000101011",
        60 to "000000101100", 61 to "000001011010", 62 to "000001100110", 63 to "000001100111",
        64 to "0000001111", 128 to "000011001000", 192 to "000011001001", 256 to "000001011011",
        320 to "000000110011", 384 to "000000110100", 448 to "000000110101", 512 to "0000001101100",
        576 to "0000001101101", 640 to "0000001001010", 704 to "0000001001011", 768 to "0000001001100",
        832 to "0000001001101", 896 to "0000001110010", 960 to "0000001110011", 1024 to "0000001110100",
        1088 to "0000001110101", 1152 to "0000001110110", 1216 to "0000001110111", 1280 to "0000001010010",
        1344 to "0000001010011", 1408 to "0000001010100", 1472 to "0000001010101", 1536 to "0000001011010",
        1600 to "0000001011011", 1664 to "0000001100100", 1728 to "0000001100101", 1792 to "00000001000",
        1856 to "00000001100", 1920 to "00000001101", 1984 to "000000010010", 2048 to "000000010011",
        2112 to "000000010100", 2176 to "000000010101", 2240 to "000000010110", 2304 to "000000010111",
        2368 to "000000011100", 2432 to "000000011101", 2496 to "000000011110", 2560 to "000000011111",
    )
}
