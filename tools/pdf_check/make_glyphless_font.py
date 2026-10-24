"""Builds app/src/main/resources/flatscan/glyphless.ttf: a TrueType font with one glyph (advance 500/1000 em)
used by the PDF writer's invisible OCR text layer (the idea of Tesseract's pdf.ttf). Every CID maps to glyph 1
(CIDToGIDMap), so any Unicode text (Hebrew, Latin) can be written and extracted through the ToUnicode map.
Nothing is ever drawn (text render mode 3), but the glyph's outline is a full box from -200 to 800 units:
viewers size search highlights and selections from glyph boxes, and an empty glyph gives them zero height.
Usage: python3 -I tools/pdf_check/make_glyphless_font.py OUT.ttf"""
import sys
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

out = sys.argv[1]
fb = FontBuilder(1000, isTTF=True)
fb.setupGlyphOrder([".notdef", "blank"])
fb.setupCharacterMap({0x20: "blank"})
empty = TTGlyphPen(None).glyph()
pen = TTGlyphPen(None)
pen.moveTo((0, -200)); pen.lineTo((0, 800)); pen.lineTo((500, 800)); pen.lineTo((500, -200)); pen.closePath()
fb.setupGlyf({".notdef": empty, "blank": pen.glyph()})
fb.setupHorizontalMetrics({".notdef": (500, 0), "blank": (500, 0)})
fb.setupHorizontalHeader(ascent=800, descent=-200)
fb.setupNameTable({"familyName": "GlyphLessFont", "styleName": "Regular"})
fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
fb.setupPost()
fb.save(out)
