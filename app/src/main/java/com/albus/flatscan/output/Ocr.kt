package com.albus.flatscan.output

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.albus.flatscan.data.Library
import com.googlecode.tesseract.android.TessBaseAPI
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * On-device OCR: Tesseract 5 (Tesseract4Android) with the Hebrew and English LSTM models (tessdata_fast,
 * bundled in assets/tessdata, copied to filesDir/ocr/tessdata on first use). ML Kit has no Hebrew.
 * A page is recognised on its finished image (out.jpg: rotation and page shape applied), the very image the
 * PDF places, so the text layer's boxes line up with it. Results are kept beside the page (ocr.json, keyed by
 * out.jpg's size and time) and its plain text in the database (search).
 */
class Ocr(private val ctx: Context) {
    private val dir = File(ctx.filesDir, "ocr")

    /** The traineddata files in place (copied from the assets once per app version). */
    private fun ensureData(): File {
        val td = File(dir, "tessdata").apply { mkdirs() }
        for (lang in LANGS) {
            val f = File(td, "$lang.traineddata")
            val size = runCatching { ctx.assets.openFd("tessdata/$lang.traineddata").use { it.length } }.getOrDefault(-1L)
            if (f.exists() && (size < 0 || f.length() == size)) continue
            val tmp = File(td, "$lang.traineddata.tmp")
            ctx.assets.open("tessdata/$lang.traineddata").use { i -> tmp.outputStream().use { i.copyTo(it) } }
            if (!tmp.renameTo(f)) error("cannot install $lang.traineddata")
        }
        return dir
    }

    /** Lines of words found on [page] (any size), in its pixel coordinates. Slow (about a second per page): call off the main thread. */
    @Synchronized
    fun recognise(page: Bitmap): List<OcrLine> {
        val api = TessBaseAPI()
        try {
            if (!api.init(ensureData().absolutePath, LANGS.joinToString("+"), TessBaseAPI.OEM_LSTM_ONLY)) error("Tesseract did not start")
            api.pageSegMode = TessBaseAPI.PageSegMode.PSM_AUTO
            api.setImage(page)
            api.getUTF8Text()   // runs the recognition
            val it = api.resultIterator ?: return emptyList()
            val lines = mutableListOf<OcrLine>()
            var words = mutableListOf<OcrWord>()
            fun flush() {
                if (words.isNotEmpty()) lines += OcrLine(words, rtlShare(words) > 0.5)
                words = mutableListOf()
            }
            it.begin()
            do {
                if (it.isAtBeginningOf(TessBaseAPI.PageIteratorLevel.RIL_TEXTLINE)) flush()
                val text = it.getUTF8Text(TessBaseAPI.PageIteratorLevel.RIL_WORD)?.trim().orEmpty()
                val conf = it.confidence(TessBaseAPI.PageIteratorLevel.RIL_WORD)
                val r = it.getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_WORD)
                if (text.isNotEmpty() && conf >= MIN_CONF) words += OcrWord(text, r.left, r.top, r.right, r.bottom)
            } while (it.next(TessBaseAPI.PageIteratorLevel.RIL_WORD))
            flush()
            it.delete()
            return lines
        } finally {
            api.recycle()
        }
    }

    /**
     * The words of page [id] of [lib]: from its cache when out.jpg has not changed, else recognised now
     * (and cached, and its text stored for search). Null when the page has no rendered image.
     */
    fun pageWords(lib: Library, id: String): PageText? {
        val out = lib.store.outputFile(id)
        if (!out.exists()) return null
        val key = "${out.length()}-${out.lastModified()}"
        val cache = File(lib.store.dir(id), "ocr.json")
        runCatching {
            val j = JSONObject(cache.readText())
            if (j.optString("key") == key) return PageText(fromJson(j.getJSONArray("lines")), j.getInt("w"), j.getInt("h"))
        }
        val bmp = BitmapFactory.decodeFile(out.absolutePath) ?: return null
        val lines = try { recognise(bmp) } finally { bmp.recycle() }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }.also { BitmapFactory.decodeFile(out.absolutePath, it) }
        val j = JSONObject().put("key", key).put("w", bounds.outWidth).put("h", bounds.outHeight).put("lines", toJson(lines))
        val tmp = File(cache.parentFile, "ocr.json.tmp")
        tmp.writeText(j.toString())
        tmp.renameTo(cache)
        return PageText(lines, bounds.outWidth, bounds.outHeight)
    }

    /** OCR words of a page and the size of the image they were found on. */
    data class PageText(val lines: List<OcrLine>, val width: Int, val height: Int) {
        val text: String get() = OcrLayout.plainText(lines)
    }

    companion object {
        val LANGS = listOf("heb", "eng")
        const val MIN_CONF = 30f
        private const val TAG = "FlatScanOcr"

        private fun rtlShare(ws: List<OcrWord>): Double {
            val all = ws.sumOf { w -> w.text.count { it.isLetter() } }
            val rtl = ws.sumOf { w -> w.text.count { it in '֐'..'ࣿ' } }
            return if (all == 0) 0.0 else rtl.toDouble() / all
        }

        fun toJson(lines: List<OcrLine>): JSONArray = JSONArray().also { a ->
            for (l in lines) a.put(JSONObject().put("rtl", l.rtl).put("w", JSONArray().also { ws ->
                for (w in l.words) ws.put(JSONArray().put(w.text).put(w.left).put(w.top).put(w.right).put(w.bottom))
            }))
        }

        fun fromJson(a: JSONArray): List<OcrLine> = List(a.length()) { i ->
            val l = a.getJSONObject(i)
            val ws = l.getJSONArray("w")
            OcrLine(List(ws.length()) { k -> ws.getJSONArray(k).let { OcrWord(it.getString(0), it.getInt(1), it.getInt(2), it.getInt(3), it.getInt(4)) } }, l.getBoolean("rtl"))
        }

        /** Recognises the pages of [doc] in the background (search), once per change of a page. */
        fun enqueue(ctx: Context, doc: String) {
            val req = OneTimeWorkRequestBuilder<OcrWorker>().setInputData(workDataOf("doc" to doc)).addTag("ocr").build()
            WorkManager.getInstance(ctx).enqueueUniqueWork("ocr-$doc", ExistingWorkPolicy.REPLACE, req)
        }
    }
}

/** Background OCR of one document's pages (input "doc"); the text goes to the database for search. */
class OcrWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result {
        val doc = inputData.getString("doc") ?: return Result.failure()
        val lib = Library.get(applicationContext)
        val ocr = Ocr(applicationContext)
        for (id in lib.pageIds(doc)) {
            if (isStopped) return Result.retry()
            runCatching { ocr.pageWords(lib, id) }
                .onSuccess { t -> if (t != null) lib.setPageText(id, t.text) }
                .onFailure { Log.e("FlatScanOcr", "OCR of $id failed", it) }
        }
        return Result.success()
    }
}
