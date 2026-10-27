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
import android.graphics.Rect
import com.albus.flatscan.data.Library
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * On-device OCR: Tesseract 5 (Tesseract4Android) with the Hebrew and English LSTM models (tessdata_fast). Both
 * come in the text recognition pack the app downloads ([OcrPack]); without it there is no OCR ([available]).
 * ML Kit has no Hebrew.
 * A page is recognised on its finished image (out.jpg: rotation and page shape applied), the very image the
 * PDF places, so the text layer's boxes line up with it. Results are kept beside the page (ocr.json, keyed by
 * out.jpg's size and time) and its plain text in the database (search).
 */
class Ocr(private val ctx: Context) {
    /** True when the text recognition pack is installed (else [recognise] fails and [pageWords] gives null). */
    val available: Boolean get() = OcrPack.isReady(ctx)

    /** Lines of words found on [page] (any size), in its pixel coordinates. Slow (about a second per page): call off the main thread. */
    @Synchronized
    fun recognise(page: Bitmap): List<OcrLine> {
        val data = OcrPack.dataDir(ctx) ?: error("the text recognition pack is not installed")
        val t = Tess(OcrPack.loader(ctx) ?: error("the text recognition pack is not installed"))
        val api = t.create()
        try {
            if (!t.init(api, data.absolutePath, LANGS.joinToString("+"))) error("Tesseract did not start")
            t.setPageSegMode(api, t.psmAuto)
            t.setImage(api, page)
            t.utf8Text(api)   // runs the recognition
            val it = t.resultIterator(api) ?: return emptyList()
            val lines = mutableListOf<OcrLine>()
            var words = mutableListOf<OcrWord>()
            fun flush() {
                if (words.isNotEmpty()) lines += OcrLine(words, rtlShare(words) > 0.5)
                words = mutableListOf()
            }
            t.begin(it)
            do {
                if (t.isAtBeginningOf(it, t.rilTextline)) flush()
                val text = t.wordText(it, t.rilWord)?.trim().orEmpty()
                val conf = t.confidence(it, t.rilWord)
                val r = t.boundingRect(it, t.rilWord)
                if (text.isNotEmpty() && conf >= MIN_CONF) words += OcrWord(text, r.left, r.top, r.right, r.bottom)
            } while (t.next(it, t.rilWord))
            flush()
            t.delete(it)
            return lines
        } finally {
            t.recycle(api)
        }
    }

    /**
     * Tesseract4Android's API (TessBaseAPI, ResultIterator) through the pack's class loader ([OcrPack.loader]):
     * the app's own loader must never load these classes (their static initialiser loads the native libraries
     * from the APK, where they are not).
     */
    private class Tess(private val loader: ClassLoader) {
        private val api = loader.loadClass("com.googlecode.tesseract.android.TessBaseAPI")
        private val iter = loader.loadClass("com.googlecode.tesseract.android.ResultIterator")
        private fun const(cls: String, name: String) = loader.loadClass(cls).getField(name).getInt(null)
        private val oemLstmOnly = api.getField("OEM_LSTM_ONLY").getInt(null)
        val psmAuto = const("com.googlecode.tesseract.android.TessBaseAPI\$PageSegMode", "PSM_AUTO")
        val rilTextline = const("com.googlecode.tesseract.android.TessBaseAPI\$PageIteratorLevel", "RIL_TEXTLINE")
        val rilWord = const("com.googlecode.tesseract.android.TessBaseAPI\$PageIteratorLevel", "RIL_WORD")
        private val mInit = api.getMethod("init", String::class.java, String::class.java, Int::class.javaPrimitiveType)
        private val mPsm = api.getMethod("setPageSegMode", Int::class.javaPrimitiveType)
        private val mImage = api.getMethod("setImage", Bitmap::class.java)
        private val mText = api.getMethod("getUTF8Text")
        private val mIter = api.getMethod("getResultIterator")
        private val mRecycle = api.getMethod("recycle")
        private val mBegin = iter.getMethod("begin")
        private val mAtBegin = iter.getMethod("isAtBeginningOf", Int::class.javaPrimitiveType)
        private val mWord = iter.getMethod("getUTF8Text", Int::class.javaPrimitiveType)
        private val mConf = iter.getMethod("confidence", Int::class.javaPrimitiveType)
        private val mRect = iter.getMethod("getBoundingRect", Int::class.javaPrimitiveType)
        private val mNext = iter.getMethod("next", Int::class.javaPrimitiveType)
        private val mDelete = iter.getMethod("delete")

        fun create(): Any = api.getConstructor().newInstance()
        fun init(a: Any, data: String, langs: String) = mInit.invoke(a, data, langs, oemLstmOnly) as Boolean
        fun setPageSegMode(a: Any, m: Int) { mPsm.invoke(a, m) }
        fun setImage(a: Any, b: Bitmap) { mImage.invoke(a, b) }
        fun utf8Text(a: Any) = mText.invoke(a) as String?
        fun resultIterator(a: Any): Any? = mIter.invoke(a)
        fun recycle(a: Any) { mRecycle.invoke(a) }
        fun begin(i: Any) { mBegin.invoke(i) }
        fun isAtBeginningOf(i: Any, level: Int) = mAtBegin.invoke(i, level) as Boolean
        fun wordText(i: Any, level: Int) = mWord.invoke(i, level) as String?
        fun confidence(i: Any, level: Int) = mConf.invoke(i, level) as Float
        fun boundingRect(i: Any, level: Int) = mRect.invoke(i, level) as Rect
        fun next(i: Any, level: Int) = mNext.invoke(i, level) as Boolean
        fun delete(i: Any) { mDelete.invoke(i) }
    }

    /**
     * The words of page [id] of [lib]: from its cache when out.jpg has not changed, else recognised now
     * (and cached, and its text stored for search). Null when the page has no rendered image, or when it has
     * no cached text and the text recognition pack is not installed.
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
        if (!available) return null
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

        /** Recognises the pages of [doc] in the background (search), once per change of a page; nothing without the pack. */
        fun enqueue(ctx: Context, doc: String) {
            if (!OcrPack.isReady(ctx)) return
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
