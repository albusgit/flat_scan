package com.albus.flatscan.output

import android.content.Context
import android.content.pm.ApplicationInfo
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dalvik.system.BaseDexClassLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

/**
 * Text recognition is a download, not part of the APK: Tesseract4Android's native libraries (libjpeg, libpngx,
 * libleptonica, libtesseract) and the heb + eng tessdata_fast models come as one zip per ABI
 * (flatscan-ocr-VERSION-ABI.zip, built by app/build.gradle.kts task ocrPacks, published as a GitHub release asset
 * of the tag ocr-pack-VERSION). The APK carries each pack's SHA-256 and size (assets/ocr_pack.json, task
 * ocrPackInfo): a download whose hash differs is thrown away. The pack is unpacked into noBackupFilesDir/ocr_pack/
 * (lib/, tessdata/) and its native code loaded only by [loader]: Tesseract4Android's Java classes (in the APK)
 * loaded again by a class loader whose library path is the pack's lib/, so their System.loadLibrary calls find it.
 */
object OcrPack {
    sealed interface State {
        data object Missing : State
        data class Downloading(val done: Long, val total: Long) : State
        data object Ready : State
        data class Failed(val message: String) : State
    }

    /** One pack: its file name in the release, SHA-256 (hex) and size in bytes. */
    data class Pack(val version: Int, val abi: String, val file: String, val sha256: String, val size: Long, val base: String) {
        val url: String get() = base.trimEnd('/') + "/" + file
    }

    private const val TAG = "FlatScanOcr"
    private const val PREFS = "ocr_pack"
    private const val KEY_BASE = "base_url"

    var state by mutableStateOf<State>(State.Missing)
        private set

    @Volatile private var classLoader: ClassLoader? = null

    /** The pack for this app's ABI (assets/ocr_pack.json), null when the build has none. */
    fun pack(ctx: Context): Pack? = runCatching {
        val j = JSONObject(ctx.assets.open("ocr_pack.json").bufferedReader().use { it.readText() })
        val abi = appAbi(ctx) ?: return null
        val p = j.getJSONObject("packs").optJSONObject(abi) ?: return null
        val base = debugBase(ctx) ?: j.getString("base")
        Pack(j.getInt("version"), abi, p.getString("file"), p.getString("sha256"), p.getLong("size"), base)
    }.onFailure { Log.w(TAG, "no OCR pack info", it) }.getOrNull()

    /** The ABI of the app's native code (its lib folder in the APK), as the packs name it. */
    private fun appAbi(ctx: Context): String? = when (File(ctx.applicationInfo.nativeLibraryDir).name) {
        "arm64" -> "arm64-v8a"
        "x86_64" -> "x86_64"
        "arm" -> "armeabi-v7a"
        "x86" -> "x86"
        else -> null
    }

    /**
     * Debuggable builds only: another place to fetch the pack from (tests: a local server through `adb reverse`),
     * set by the debug intent `--es ocrPackBase URL` (MainActivity); empty clears it.
     */
    fun setDebugBase(ctx: Context, url: String) {
        if (ctx.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_BASE, url.ifBlank { null }).apply()
    }

    private fun debugBase(ctx: Context): String? =
        if (ctx.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) null
        else ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_BASE, null)

    private fun root(ctx: Context) = File(ctx.noBackupFilesDir, "ocr_pack")

    /** Where pack [p] is unpacked: lib/ and tessdata/, and `ok` holding its hash once complete. */
    private fun dir(ctx: Context, p: Pack) = File(root(ctx), "${p.version}-${p.abi}")

    /** The folder Tesseract reads its models from (it holds tessdata/); null until the pack is installed. */
    fun dataDir(ctx: Context): File? = pack(ctx)?.let { p -> dir(ctx, p).takeIf { ready(ctx, p) } }

    fun isReady(ctx: Context): Boolean = pack(ctx)?.let { ready(ctx, it) } == true

    private fun ready(ctx: Context, p: Pack): Boolean = runCatching { File(dir(ctx, p), "ok").readText() == p.sha256 }.getOrDefault(false)

    /** The size on the phone of the installed pack (bytes), 0 when none. */
    fun installedBytes(ctx: Context): Long = root(ctx).walkBottomUp().filter { it.isFile }.sumOf { it.length() }

    /** [state] from what is on the phone (no download in progress). */
    fun refresh(ctx: Context) {
        if (state is State.Downloading) return
        state = if (isReady(ctx)) State.Ready else if (state is State.Failed) state else State.Missing
    }

    /**
     * Downloads, checks and unpacks the pack (any thread; [state] follows). True when it is installed. Older
     * versions are removed.
     */
    suspend fun download(ctx: Context): Boolean = withContext(Dispatchers.IO) {
        val p = pack(ctx) ?: run {
            state = State.Failed("Text recognition is not available for this phone's processor")
            return@withContext false
        }
        if (ready(ctx, p)) {
            state = State.Ready
            return@withContext true
        }
        if (state is State.Downloading) return@withContext false
        state = State.Downloading(0, p.size)
        val root = root(ctx).apply { mkdirs() }
        val zip = File(root, p.file + ".part")
        try {
            val client = OkHttpClient.Builder().connectTimeout(20, TimeUnit.SECONDS).readTimeout(60, TimeUnit.SECONDS).build()
            val md = MessageDigest.getInstance("SHA-256")
            client.newCall(Request.Builder().url(p.url).build()).execute().use { r ->
                if (!r.isSuccessful) error("the server answered ${r.code}")
                val body = r.body ?: error("empty answer")
                var done = 0L
                var shown = 0L
                body.byteStream().use { i ->
                    zip.outputStream().use { o ->
                        val buf = ByteArray(64 * 1024)
                        while (true) {
                            val n = i.read(buf)
                            if (n < 0) break
                            if (done + n > p.size) error("the file is larger than expected")
                            o.write(buf, 0, n)
                            md.update(buf, 0, n)
                            done += n
                            if (done - shown >= 256 * 1024) {
                                shown = done
                                state = State.Downloading(done, p.size)
                            }
                        }
                    }
                }
                if (done != p.size) error("the download was cut off ($done of ${p.size} bytes)")
            }
            val sha = md.digest().joinToString("") { "%02x".format(it) }
            if (sha != p.sha256) error("the file is not the expected one (SHA-256 $sha)")
            install(ctx, p, zip)
            classLoader = null
            state = State.Ready
            true
        } catch (e: Exception) {
            Log.e(TAG, "OCR pack download failed", e)
            state = State.Failed(e.message ?: e.javaClass.simpleName)
            false
        } finally {
            zip.delete()
        }
    }

    /** Unpacks a checked zip: lib/NAME.so and tessdata/NAME.traineddata only, into a fresh folder. */
    private fun install(ctx: Context, p: Pack, zip: File) {
        val d = dir(ctx, p)
        val tmp = File(root(ctx), d.name + ".tmp")
        tmp.deleteRecursively()
        File(tmp, "lib").mkdirs()
        File(tmp, "tessdata").mkdirs()
        var libs = 0
        var models = 0
        ZipInputStream(zip.inputStream().buffered()).use { z ->
            while (true) {
                val e = z.nextEntry ?: break
                if (e.isDirectory) continue
                val m = Regex("""(lib|tessdata)/([A-Za-z0-9_.-]+\.(so|traineddata))""").matchEntire(e.name) ?: error("unexpected file ${e.name} in the pack")
                val (folder, name, ext) = m.destructured
                if ((folder == "lib") != (ext == "so")) error("unexpected file ${e.name} in the pack")
                val f = File(File(tmp, folder), name)
                f.outputStream().use { z.copyTo(it) }
                f.setReadOnly()
                if (ext == "so") libs++ else models++
            }
        }
        if (libs == 0 || models == 0) error("the pack has no libraries or no models")
        File(tmp, "ok").writeText(p.sha256)
        d.deleteRecursively()
        if (!tmp.renameTo(d)) error("cannot install the pack")
        // other versions and ABIs
        root(ctx).listFiles()?.filter { it.name != d.name }?.forEach { it.deleteRecursively() }
    }

    /** Removes the pack (Settings); recognised text already stored stays. */
    fun remove(ctx: Context) {
        classLoader = null
        root(ctx).deleteRecursively()
        state = State.Missing
    }

    /**
     * The class loader for Tesseract4Android: its classes from the APK, defined anew by a loader whose native
     * library path is the pack's lib/ (the classes' static initialiser loads jpeg, pngx, leptonica and tesseract by
     * name). Every other class comes from the app's loader. Null when the pack is not installed.
     */
    fun loader(ctx: Context): ClassLoader? {
        classLoader?.let { return it }
        val d = dataDir(ctx) ?: return null
        return synchronized(this) {
            classLoader ?: PackLoader(ctx.applicationInfo.sourceDir, File(d, "lib").absolutePath, OcrPack::class.java.classLoader!!).also { classLoader = it }
        }
    }

    /** Child first for Tesseract4Android's packages, parent first for the rest. */
    private class PackLoader(apk: String, libs: String, parent: ClassLoader) : BaseDexClassLoader(apk, null, libs, parent) {
        override fun loadClass(name: String, resolve: Boolean): Class<*> {
            if (name.startsWith("com.googlecode.tesseract.") || name.startsWith("com.googlecode.leptonica.")) {
                synchronized(this) { return findLoadedClass(name) ?: findClass(name) }
            }
            return super.loadClass(name, resolve)
        }
    }
}
