// Prints, as JSON, the text pdf.js (Firefox's PDF engine) extracts from each page of a PDF and its title.
// Usage: PDFJS_MODULES=.../node_modules node pdfjs_text.mjs FILE.pdf (npm install pdfjs-dist@4.10.38 there).
const { getDocument } = await import(process.env.PDFJS_MODULES + "/pdfjs-dist/legacy/build/pdf.mjs");
import { readFileSync } from "fs";
const data = new Uint8Array(readFileSync(process.argv[2]));
const doc = await getDocument({ data, verbosity: 0 }).promise;
const out = [];
for (let i = 1; i <= doc.numPages; i++) {
  const page = await doc.getPage(i);
  const tc = await page.getTextContent();
  out.push(tc.items.map((it) => it.str + (it.hasEOL ? "\n" : "")).join(""));
}
const meta = await doc.getMetadata();
process.stdout.write(JSON.stringify({ pages: out, title: meta.info.Title || null, producer: meta.info.Producer || null }));
