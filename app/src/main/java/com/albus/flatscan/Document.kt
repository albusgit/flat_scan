package com.albus.flatscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.util.UUID
import kotlin.math.max

/**
 * Everything needed to rebuild one page from its source photo without the native session:
 * corners (in pixels of the source photo after [cropRotation] clockwise quarter turns), the
 * render parameters and the protected picture rectangles (page fractions x0, y0, x1, y1 ...).
 */
data class PageMeta(
    val id: String,
    val corners: FloatArray,
    val detected: FloatArray?,       // corners found by detection (null: detection failed)
    val cropRotation: Int,           // quarter turns the user applied on the crop screen
    val userRotated: Boolean,        // any manual rotation -> no auto orientation
    val focalPx: Float,
    val params: RenderParams,
    val protect: FloatArray,
    val rotation: Int,               // quarter turns of the result (applied to the output)
    val pageIndex: Int = 0,          // page of a book spread in the source photo (0 = the larger one)
    val forceHomography: Boolean = false,   // the quality gate flagged the automatic crop
    val spreadWith: String? = null,  // the other page of the same book spread
    val ignore: FloatArray = FloatArray(0),   // areas whose lines the unwarp must not use (page fractions)
    val perspectiveOnly: Boolean = false,     // the user turned the unwarp off for this page
    val erase: FloatArray = FloatArray(0),    // Erase tool rectangles (page fractions), applied after processing
    val look: PageLook = PageLook(),          // light, colour and page shape (Kotlin, after the core's render)
) {
    /**
     * Everything the processed geometry depends on: a stored session (DocumentStore.saveSession)
     * is reused only while this key is unchanged.
     */
    fun geometryKey(renderWidth: Int, autoOrient: Boolean, net: Boolean): String = listOf(
        corners.joinToString(","), detected?.joinToString(",") ?: "-", cropRotation, userRotated, focalPx, pageIndex,
        forceHomography, ignore.joinToString(","), perspectiveOnly, renderWidth, autoOrient, net,
    ).joinToString("|")

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("corners", corners.toJson())
        if (detected != null) put("detected", detected.toJson())
        put("crop_rotation", cropRotation)
        put("user_rotated", userRotated)
        put("focal_px", focalPx.toDouble())
        put("params", JSONObject().apply {
            put("filter", params.filter.name)
            put("strength", params.strength.toDouble())
            put("solid", params.solid)
            put("bg_cleanup", params.bgCleanup.toDouble())
            put("black", params.black.toDouble())
            put("white", params.white.toDouble())
            put("gamma", params.gamma.toDouble())
            put("screen", params.screen.name)
        })
        put("protect", protect.toJson())
        put("rotation", rotation)
        put("page_index", pageIndex)
        put("force_homography", forceHomography)
        if (spreadWith != null) put("spread_with", spreadWith)
        put("ignore", ignore.toJson())
        put("perspective_only", perspectiveOnly)
        put("erase", erase.toJson())
        if (look != PageLook()) put("look", look.toJson())
    }

    companion object {
        fun fromJson(j: JSONObject): PageMeta {
            val p = j.getJSONObject("params")
            val d = RenderParams()
            return PageMeta(
                id = j.getString("id"),
                corners = j.getJSONArray("corners").toFloats(),
                detected = j.optJSONArray("detected")?.toFloats(),
                cropRotation = j.optInt("crop_rotation"),
                userRotated = j.optBoolean("user_rotated"),
                focalPx = j.optDouble("focal_px", 0.0).toFloat(),
                params = RenderParams(
                    filter = FilterKind.entries.firstOrNull { it.name == p.optString("filter") } ?: d.filter,
                    strength = p.optDouble("strength", d.strength.toDouble()).toFloat(),
                    solid = p.optBoolean("solid", d.solid),
                    bgCleanup = p.optDouble("bg_cleanup", d.bgCleanup.toDouble()).toFloat(),
                    black = p.optDouble("black", d.black.toDouble()).toFloat(),
                    white = p.optDouble("white", d.white.toDouble()).toFloat(),
                    gamma = p.optDouble("gamma", d.gamma.toDouble()).toFloat(),
                    screen = ScreenMode.entries.firstOrNull { it.name == p.optString("screen") } ?: d.screen,
                ),
                protect = j.optJSONArray("protect")?.toFloats() ?: FloatArray(0),
                rotation = j.optInt("rotation"),
                pageIndex = j.optInt("page_index"),
                forceHomography = j.optBoolean("force_homography"),
                spreadWith = j.optString("spread_with").ifEmpty { null },
                ignore = j.optJSONArray("ignore")?.toFloats() ?: FloatArray(0),
                perspectiveOnly = j.optBoolean("perspective_only"),
                erase = j.optJSONArray("erase")?.toFloats() ?: FloatArray(0),
                look = PageLook.fromJson(j.optJSONObject("look")),
            )
        }

        private fun FloatArray.toJson() = JSONArray().also { a -> forEach { a.put(it.toDouble()) } }
        private fun JSONArray.toFloats() = FloatArray(length()) { getDouble(it).toFloat() }
    }
}

/**
 * The pages' files, one folder per page under filesDir/pages/<id>/ (app-private storage, not the cache,
 * which Android may clear): original.<ext> (the photo exactly as captured or imported; every
 * processing run decodes it the same way, [loadSource]), meta.json, out.jpg (the rendered, rotated
 * page, JPEG q95: export), thumb.jpg (a small copy of out.jpg: lists) and session/ (the processed page,
 * so opening it again does not re-run the geometry). Which pages make which document, and in what
 * order, is in the database ([com.albus.flatscan.data.LibraryDb]). Pages made before history existed
 * have source.jpg (the decoded photo, re-encoded at q95) instead of an original.
 * All calls do file IO: call them off the main thread.
 */
class DocumentStore(val root: File) {
    constructor(ctx: Context) : this(File(ctx.filesDir, "pages"))

    fun newId(): String = UUID.randomUUID().toString().substring(0, 12)

    fun dir(id: String) = File(root, id)
    /** The original photo of [id] (any image format), or null when there is none (dropped, or a legacy page). */
    fun originalFile(id: String): File? = dir(id).listFiles()?.firstOrNull { it.isFile && it.name.startsWith("original.") && !it.name.endsWith(".tmp") }
    private fun legacySource(id: String) = File(dir(id), "source.jpg")
    fun outputFile(id: String) = File(dir(id), "out.jpg")
    fun thumbFile(id: String) = File(dir(id), "thumb.jpg")
    private fun metaFile(id: String) = File(dir(id), "meta.json")

    fun exists(id: String) = metaFile(id).exists()

    /** The page can be processed again (it has a photo to process). */
    fun hasSource(id: String) = originalFile(id) != null || legacySource(id).exists()

    /** Page folders on disk. */
    fun pageIds(): List<String> = root.listFiles()?.filter { it.isDirectory }?.map { it.name }.orEmpty()

    /** Removes page folders that are not in [keep] (abandoned imports, purged documents). */
    fun prune(keep: Collection<String>) {
        val k = keep.toSet()
        root.listFiles()?.forEach { if (it.isDirectory && it.name !in k) it.deleteRecursively() }
    }

    /**
     * Stores the photo of a new page byte for byte from [input] ([ext]: its file extension, e.g. "jpg"),
     * and returns the file. The page is processed from this file, never from an in-memory copy, so
     * processing it again later starts from exactly the same pixels.
     */
    fun saveOriginal(id: String, ext: String, input: InputStream): File {
        dir(id).mkdirs()
        val e = ext.lowercase().filter { it.isLetterOrDigit() }.take(5).ifEmpty { "jpg" }
        val f = File(dir(id), "original.$e")
        val tmp = File(dir(id), "original.$e.tmp")
        tmp.outputStream().buffered().use { input.copyTo(it) }
        if (!tmp.renameTo(f)) error("cannot write original of $id")
        return f
    }

    /** The other page of a book spread shares its photo: a copy of [from]'s original (or legacy source). */
    fun copySource(from: String, to: String) {
        dir(to).mkdirs()
        val src = originalFile(from) ?: legacySource(from)
        val target = if (src.name.startsWith("original.")) File(dir(to), src.name) else legacySource(to)
        val tmp = File(dir(to), target.name + ".tmp")
        src.copyTo(tmp, overwrite = true)
        if (!tmp.renameTo(target)) error("cannot write source of $to")
    }

    /** A copy of page [from] as page [to]: every file, session included (duplicate page). */
    fun copyPage(from: String, to: String) {
        val d = dir(to)
        d.deleteRecursively()
        dir(from).copyRecursively(d, overwrite = true)
        if (exists(to)) saveMeta(loadMeta(to).copy(id = to, spreadWith = null))
    }

    /** The upright photo of [id] as processing sees it (EXIF orientation applied, at most [MAX_PHOTO_SIDE]). */
    fun loadSource(id: String): LoadedPhoto {
        originalFile(id)?.let { return loadPhotoFile(it) }
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val b = BitmapFactory.decodeFile(legacySource(id).absolutePath, opts) ?: error("cannot read page $id")
        return LoadedPhoto(if (b.config == Bitmap.Config.ARGB_8888) b else b.copy(Bitmap.Config.ARGB_8888, false), 0f)
    }

    /** Deletes the original photo (and legacy source) of [id]; returns the bytes freed. */
    fun dropOriginal(id: String): Long {
        var n = 0L
        listOfNotNull(originalFile(id), legacySource(id).takeIf { it.exists() }).forEach { n += it.length(); it.delete() }
        return n
    }

    fun saveMeta(m: PageMeta) {
        dir(m.id).mkdirs()
        val tmp = File(dir(m.id), "meta.json.tmp")
        tmp.writeText(m.toJson().toString())
        tmp.renameTo(metaFile(m.id))
    }

    fun loadMeta(id: String): PageMeta = PageMeta.fromJson(JSONObject(metaFile(id).readText()))

    /** A stored session of [id] made with geometry [key]: its directory, info JSON and details, or null. */
    fun loadSession(id: String, key: String): Triple<File, JSONObject, String>? = runCatching {
        val d = File(dir(id), "session")
        val j = JSONObject(File(d, "ready.json").readText())
        if (j.optString("key") != key) null else Triple(d, j.getJSONObject("info"), j.optString("details"))
    }.getOrNull()

    /** The stored session of [id] whatever its key (a page without its original can only be shown as it was made). */
    fun loadAnySession(id: String): Triple<File, JSONObject, String>? = runCatching {
        val d = File(dir(id), "session")
        val j = JSONObject(File(d, "ready.json").readText())
        Triple(d, j.getJSONObject("info"), j.optString("details"))
    }.getOrNull()

    /**
     * Stores a processed page: [write] fills a fresh directory (false: give up), which then
     * replaces session/ of [id] with its key, info JSON and details.
     */
    fun saveSession(id: String, key: String, info: JSONObject, details: String, write: (File) -> Boolean): Boolean {
        if (!dir(id).exists()) return false
        val tmp = File(dir(id), "session.tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()
        if (!write(tmp)) {
            tmp.deleteRecursively()
            return false
        }
        File(tmp, "ready.json").writeText(JSONObject().put("key", key).put("info", info).put("details", details).toString())
        val d = File(dir(id), "session")
        d.deleteRecursively()
        return tmp.renameTo(d)
    }

    /** The page's stored session directory whatever its geometry key (the geometry out.jpg was made with), or null. */
    fun sessionDir(id: String): File? = File(dir(id), "session").takeIf { File(it, "ready.json").exists() }

    /** Writes the rendered page and its thumbnail. */
    fun saveOutput(id: String, page: Bitmap) {
        dir(id).mkdirs()
        writeJpeg(page, outputFile(id), 95)
        val s = THUMB_SIDE.toFloat() / max(page.width, page.height)
        val t = if (s < 1f) Bitmap.createScaledBitmap(page, (page.width * s).toInt().coerceAtLeast(1), (page.height * s).toInt().coerceAtLeast(1), true) else page
        writeJpeg(t, thumbFile(id), 85)
        if (t !== page) t.recycle()
    }

    /** The rendered page scaled to at most [maxSide] px (from the thumbnail when it is big enough), or null when there is none yet. */
    fun loadThumbnail(id: String, maxSide: Int): Bitmap? {
        val f = thumbFile(id).takeIf { it.exists() && maxSide <= THUMB_SIDE } ?: outputFile(id)
        if (!f.exists()) return null
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(f.absolutePath, bounds)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        return BitmapFactory.decodeFile(f.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample })
    }

    fun delete(id: String) {
        dir(id).deleteRecursively()
    }

    /** Bytes on disk: (everything, originals only) for [ids] (all pages when null). */
    fun usage(ids: Collection<String>? = null): Pair<Long, Long> {
        var all = 0L
        var orig = 0L
        for (id in ids ?: pageIds()) {
            dir(id).walkTopDown().filter { it.isFile }.forEach {
                all += it.length()
                if (it.parentFile == dir(id) && (it.name.startsWith("original.") || it.name == "source.jpg")) orig += it.length()
            }
        }
        return all to orig
    }

    private fun writeJpeg(b: Bitmap, f: File, quality: Int) {
        val tmp = File(f.parentFile, f.name + ".tmp")
        tmp.outputStream().buffered().use { b.compress(Bitmap.CompressFormat.JPEG, quality, it) }
        if (!tmp.renameTo(f)) error("cannot write $f")
    }

    companion object {
        const val THUMB_SIDE = 480

        /**
         * The single document of versions before history (cacheDir/document: doc.json, name.txt, one folder
         * per page), or null. [LegacyDocument.dir] is where its page folders are.
         */
        fun legacy(ctx: Context): LegacyDocument? = legacyAt(File(ctx.cacheDir, "document"))

        fun legacyAt(dir: File): LegacyDocument? {
            val order = runCatching {
                val a = JSONObject(File(dir, "doc.json").readText()).getJSONArray("pages")
                List(a.length()) { a.getString(it) }
            }.getOrNull() ?: return null
            val name = runCatching { File(dir, "name.txt").readText().trim() }.getOrNull()?.takeIf { it.isNotEmpty() }
            return LegacyDocument(dir, order.filter { File(File(dir, it), "meta.json").exists() }, name)
        }
    }
}

/** The pre-history document found in the cache: its folder, page order and name. */
class LegacyDocument(val dir: File, val pages: List<String>, val name: String?)
