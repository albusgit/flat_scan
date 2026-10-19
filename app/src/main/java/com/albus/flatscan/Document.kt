package com.albus.flatscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
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
) {
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
        })
        put("protect", protect.toJson())
        put("rotation", rotation)
        put("page_index", pageIndex)
        put("force_homography", forceHomography)
        if (spreadWith != null) put("spread_with", spreadWith)
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
                ),
                protect = j.optJSONArray("protect")?.toFloats() ?: FloatArray(0),
                rotation = j.optInt("rotation"),
                pageIndex = j.optInt("page_index"),
                forceHomography = j.optBoolean("force_homography"),
                spreadWith = j.optString("spread_with").ifEmpty { null },
            )
        }

        private fun FloatArray.toJson() = JSONArray().also { a -> forEach { a.put(it.toDouble()) } }
        private fun JSONArray.toFloats() = FloatArray(length()) { getDouble(it).toFloat() }
    }
}

/**
 * The current document on disk: cacheDir/document/doc.json (page order) and one folder per
 * page with source.jpg (the upright photo, JPEG q95), meta.json and out.jpg (the rendered,
 * rotated page, JPEG q95: thumbnails and export). All calls do file IO: call them off the
 * main thread.
 */
class DocumentStore(ctx: Context) {
    val root = File(ctx.cacheDir, "document")

    fun newId(): String = UUID.randomUUID().toString().substring(0, 12)

    private fun dir(id: String) = File(root, id)
    fun sourceFile(id: String) = File(dir(id), "source.jpg")
    fun outputFile(id: String) = File(dir(id), "out.jpg")
    private fun metaFile(id: String) = File(dir(id), "meta.json")

    fun loadOrder(): List<String> = runCatching {
        val a = JSONObject(File(root, "doc.json").readText()).getJSONArray("pages")
        List(a.length()) { a.getString(it) }.filter { metaFile(it).exists() && sourceFile(it).exists() }
    }.getOrDefault(emptyList())

    fun saveOrder(ids: List<String>) {
        root.mkdirs()
        val tmp = File(root, "doc.json.tmp")
        tmp.writeText(JSONObject().put("pages", JSONArray(ids)).toString())
        tmp.renameTo(File(root, "doc.json"))
    }

    /** Removes page folders that are not in [keep] (abandoned imports). */
    fun prune(keep: Collection<String>) {
        root.listFiles()?.forEach { if (it.isDirectory && it.name !in keep) it.deleteRecursively() }
    }

    fun saveSource(id: String, photo: Bitmap) {
        dir(id).mkdirs()
        writeJpeg(photo, sourceFile(id))
    }

    /** The other page of a book spread shares its photo: a copy of [from]'s source. */
    fun copySource(from: String, to: String) {
        dir(to).mkdirs()
        val tmp = File(dir(to), "source.jpg.tmp")
        sourceFile(from).copyTo(tmp, overwrite = true)
        if (!tmp.renameTo(sourceFile(to))) error("cannot write source of $to")
    }

    fun loadSource(id: String): Bitmap {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val b = BitmapFactory.decodeFile(sourceFile(id).absolutePath, opts) ?: error("cannot read page $id")
        return if (b.config == Bitmap.Config.ARGB_8888) b else b.copy(Bitmap.Config.ARGB_8888, false)
    }

    fun saveMeta(m: PageMeta) {
        dir(m.id).mkdirs()
        val tmp = File(dir(m.id), "meta.json.tmp")
        tmp.writeText(m.toJson().toString())
        tmp.renameTo(metaFile(m.id))
    }

    fun loadMeta(id: String): PageMeta = PageMeta.fromJson(JSONObject(metaFile(id).readText()))

    fun saveOutput(id: String, page: Bitmap) {
        dir(id).mkdirs()
        writeJpeg(page, outputFile(id))
    }

    /** The rendered page scaled to at most [maxSide] px, or null when there is none yet. */
    fun loadThumbnail(id: String, maxSide: Int): Bitmap? {
        val f = outputFile(id)
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

    fun deleteAll() {
        root.deleteRecursively()
    }

    private fun writeJpeg(b: Bitmap, f: File) {
        val tmp = File(f.parentFile, f.name + ".tmp")
        tmp.outputStream().buffered().use { b.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        if (!tmp.renameTo(f)) error("cannot write $f")
    }
}
