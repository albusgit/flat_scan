package com.albus.flatscan.depth

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.Surface
import android.view.WindowManager
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.Executors
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The depth test (experimental, Settings > Advanced > Depth capture): can ARCore measure a page's shape well enough
 * to help flattening, and on which phones? It shows ARCore's depth over the camera, measures how far the depth in the
 * middle of the view strays from a flat plane (on a flat sheet that is the depth's noise; on a curled page, noise plus
 * the curl), and saves frames for desktop analysis: the camera image, the smoothed and raw depth with its confidence,
 * the camera intrinsics and ARCore's camera pose (one world frame per session, so frames from a few angles can be
 * combined).
 */
class DepthCapture(private val ctx: Context) {
    /** What the screen shows; posted on the main thread. */
    data class Live(
        val tracking: String = "",
        val distanceMm: Float = 0f,
        val smoothRmsMm: Float = 0f,
        val rawRmsMm: Float = 0f,
        val rawPoints: Int = 0,
        val depthSize: String = "",
        val saved: Int = 0,
        val note: String = "",
    )

    @Volatile var session: Session? = null
        private set
    var info: String = ""
        private set
    var onLive: (Live) -> Unit = {}
    @Volatile var showDepth = true
    @Volatile private var saveRequested = false
    private val main = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private var live = Live()
    private val dir: File = File(ctx.filesDir, "depth/" + System.currentTimeMillis())

    /** Creates the session on the main thread: depth on, the largest camera image the phone offers to the CPU. */
    fun open(): String? {
        val s = try {
            Session(ctx)
        } catch (e: Exception) {
            return "ARCore could not start: ${e.javaClass.simpleName}" + (e.message?.let { ": $it" } ?: "")
        }
        val filter = CameraConfigFilter(s)
        val all = s.getSupportedCameraConfigs(filter)
        val tof = s.getSupportedCameraConfigs(CameraConfigFilter(s).setDepthSensorUsage(EnumSet.of(CameraConfig.DepthSensorUsage.REQUIRE_AND_USE)))
        val pick = (tof.ifEmpty { all }).maxByOrNull { it.imageSize.width * it.imageSize.height }
        if (pick != null) s.cameraConfig = pick
        val depthOk = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        val cfg = Config(s).apply {
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            depthMode = if (depthOk) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
        }
        s.configure(cfg)
        info = buildString {
            append("${Build.MANUFACTURER} ${Build.MODEL}")
            if (Build.VERSION.SDK_INT >= 31) append(" (${Build.SOC_MODEL})")
            append("\nDepth: ").append(if (depthOk) "yes" else "not supported on this phone")
            append(" · depth sensor (ToF): ").append(if (tof.isNotEmpty()) "yes" else "no")
            pick?.let { append("\nCamera image ${it.imageSize.width} x ${it.imageSize.height}, preview ${it.textureSize.width} x ${it.textureSize.height}") }
        }
        session = s
        return null
    }

    fun resume() = session?.resume()
    fun pause() = session?.pause()

    fun close() {
        session?.close()
        session = null
        io.shutdown()
    }

    fun requestSave() {
        saveRequested = true
    }

    /** The saved frames in one zip in the shared cache (for the share sheet), or null when nothing was saved. */
    fun zipSaved(): File? {
        val files = dir.listFiles()?.sortedBy { it.name }.orEmpty()
        if (files.isEmpty()) return null
        val out = File(ctx.cacheDir, "shared/depth_${dir.name}.zip").apply { parentFile?.mkdirs() }
        ZipOutputStream(out.outputStream().buffered()).use { z ->
            for (f in files) {
                z.putNextEntry(ZipEntry(f.name))
                f.inputStream().use { it.copyTo(z) }
                z.closeEntry()
            }
        }
        return out
    }

    private fun post(l: Live) {
        live = l
        main.post { onLive(l) }
    }

    // ---------------------------------------------------------------- drawing (GL thread)

    val renderer = object : GLSurfaceView.Renderer {
        private var camTex = 0
        private var depthTex = 0
        private var camProg = 0
        private var depthProg = 0
        private var texSet = false
        private var width = 0
        private var height = 0
        private var frames = 0
        private val quad = floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        private val texCoords = floatBuffer(FloatArray(8))
        private var nearMm = 200f
        private var farMm = 1200f

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            val t = IntArray(2)
            GLES20.glGenTextures(2, t, 0)
            camTex = t[0]
            depthTex = t[1]
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, camTex)
            texParams(GLES11Ext.GL_TEXTURE_EXTERNAL_OES)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
            texParams(GLES20.GL_TEXTURE_2D)
            camProg = program(VS, CAM_FS)
            depthProg = program(VS, DEPTH_FS)
            texSet = false
        }

        override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
            GLES20.glViewport(0, 0, w, h)
            width = w
            height = h
            session?.setDisplayGeometry(displayRotation(), w, h)
        }

        override fun onDrawFrame(gl: GL10?) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            val s = session ?: return
            if (!texSet) {
                s.setCameraTextureName(camTex)
                s.setDisplayGeometry(displayRotation(), width, height)
                texSet = true
            }
            val frame = try {
                s.update()
            } catch (e: Exception) {
                return
            }
            if (frame.hasDisplayGeometryChanged() || frames == 0) {
                quad.position(0)
                texCoords.position(0)
                frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quad, Coordinates2d.TEXTURE_NORMALIZED, texCoords)
            }
            frames++
            if (frame.timestamp == 0L) return
            draw(camProg, GLES11Ext.GL_TEXTURE_EXTERNAL_OES, camTex, 1f)
            val tracking = frame.camera.trackingState
            var depth: Image? = null
            try {
                depth = frame.acquireDepthImage16Bits()
            } catch (_: NotYetAvailableException) {
            } catch (_: IllegalStateException) {   // depth off on this phone
            }
            depth?.use { d ->
                if (showDepth) {
                    upload(d)
                    draw(depthProg, GLES20.GL_TEXTURE_2D, depthTex, 0.55f)
                }
                if (frames % 15 == 0) measure(frame, d, tracking)
            }
            if (depth == null && frames % 15 == 0) post(live.copy(tracking = trackingText(frame), note = "No depth yet: move the phone slowly"))
            if (saveRequested) {
                saveRequested = false
                save(frame)
            }
        }

        private fun draw(prog: Int, target: Int, tex: Int, alpha: Float) {
            GLES20.glUseProgram(prog)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(target, tex)
            GLES20.glUniform1i(GLES20.glGetUniformLocation(prog, "tex"), 0)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(prog, "alpha"), alpha)
            GLES20.glUniform2f(GLES20.glGetUniformLocation(prog, "range"), nearMm, farMm)
            val pos = GLES20.glGetAttribLocation(prog, "pos")
            val uv = GLES20.glGetAttribLocation(prog, "uv")
            quad.position(0)
            texCoords.position(0)
            GLES20.glVertexAttribPointer(pos, 2, GLES20.GL_FLOAT, false, 0, quad)
            GLES20.glVertexAttribPointer(uv, 2, GLES20.GL_FLOAT, false, 0, texCoords)
            GLES20.glEnableVertexAttribArray(pos)
            GLES20.glEnableVertexAttribArray(uv)
            if (alpha < 1f) {
                GLES20.glEnable(GLES20.GL_BLEND)
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            }
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glDisable(GLES20.GL_BLEND)
            GLES20.glDisableVertexAttribArray(pos)
            GLES20.glDisableVertexAttribArray(uv)
        }

        /** the 16-bit depth (mm) as a two-channel byte texture: low byte in L, high byte in A */
        private fun upload(d: Image) {
            val p = d.planes[0]
            val w = d.width
            val h = d.height
            val buf = tight(p.buffer, w * 2, p.rowStride, h)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
            GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1)
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE_ALPHA, w, h, 0, GLES20.GL_LUMINANCE_ALPHA, GLES20.GL_UNSIGNED_BYTE, buf)
        }

        private fun measure(frame: Frame, d: Image, tracking: TrackingState) {
            val intr = frame.camera.imageIntrinsics
            val f = intr.focalLength
            val c = intr.principalPoint
            val dims = intr.imageDimensions
            val smooth = plane(d, null, f, c, dims)
            var raw: Plane? = null
            try {
                frame.acquireRawDepthImage16Bits().use { r ->
                    frame.acquireRawDepthConfidenceImage().use { conf -> raw = plane(r, conf, f, c, dims) }
                }
            } catch (_: Exception) {
            }
            if (smooth != null && smooth.distance > 0) {
                nearMm = (smooth.distance * 0.6f).coerceAtLeast(100f)
                farMm = smooth.distance * 1.6f
            }
            post(
                live.copy(
                    tracking = trackingText(frame),
                    distanceMm = smooth?.distance ?: 0f,
                    smoothRmsMm = smooth?.rms ?: 0f,
                    rawRmsMm = raw?.rms ?: 0f,
                    rawPoints = raw?.n ?: 0,
                    depthSize = "${d.width} x ${d.height}",
                    note = if (tracking != TrackingState.TRACKING) "Move the phone slowly until tracking starts" else "",
                ),
            )
        }

        private fun save(frame: Frame) {
            if (!dir.isDirectory) dir.mkdirs()
            val k = live.saved
            val base = File(dir, "f%02d".format(k)).path
            val cam = frame.camera
            val intr = cam.imageIntrinsics
            val meta = StringBuilder("{")
            fun add(key: String, v: String) {
                if (meta.length > 1) meta.append(", ")
                meta.append('"').append(key).append("\": ").append(v)
            }
            add("device", "\"${Build.MANUFACTURER} ${Build.MODEL}\"")
            add("timestamp_ns", frame.timestamp.toString())
            add("tracking", "\"${cam.trackingState}\"")
            add("focal_px", arr(intr.focalLength))
            add("principal_px", arr(intr.principalPoint))
            add("image_size", intr.imageDimensions.joinToString(", ", "[", "]"))
            // physical camera in world (OpenGL: +X right, +Y up, looking along -Z, relative to the image readout)
            val pose = cam.pose
            add("pose_t", arr(pose.translation))
            add("pose_q_xyzw", arr(pose.rotationQuaternion))
            add("display_rotation", displayRotation().toString())
            var jpeg: ByteArray? = null
            try {
                frame.acquireCameraImage().use { img -> jpeg = yuvToJpeg(img) }
            } catch (_: Exception) {
            }
            val blobs = mutableListOf<Pair<String, ByteArray>>()
            fun grab(name: String, acquire: () -> Image, bpp: Int) {
                try {
                    acquire().use { im ->
                        val p = im.planes[0]
                        val b = tight(p.buffer, im.width * bpp, p.rowStride, im.height)
                        blobs += "$name.raw" to ByteArray(b.remaining()).also { b.get(it) }
                        add("${name}_size", "[${im.width}, ${im.height}]")
                    }
                } catch (_: Exception) {
                }
            }
            grab("depth_mm_u16", { frame.acquireDepthImage16Bits() }, 2)
            grab("raw_depth_mm_u16", { frame.acquireRawDepthImage16Bits() }, 2)
            grab("raw_confidence_u8", { frame.acquireRawDepthConfidenceImage() }, 1)
            meta.append("}\n")
            val n = k + 1
            post(live.copy(saved = n, note = "Saved frame $n"))
            val j = jpeg
            io.execute {
                File("$base.json").writeText(meta.toString())
                j?.let { File("$base.jpg").writeBytes(it) }
                for ((name, b) in blobs) File("${base}_$name").writeBytes(b)
            }
        }
    }

    private class Plane(val distance: Float, val rms: Float, val n: Int)

    /**
     * Least-squares plane z = a x + b y + c through the depth points in the middle 40 % of the view (mm, camera
     * frame; intrinsics scaled from the camera image to the depth image, which covers the same view), the raw depth
     * kept where its confidence is over half. -> the median distance and the RMS of the points' distance from it.
     */
    private fun plane(d: Image, conf: Image?, f: FloatArray, c: FloatArray, dims: IntArray): Plane? {
        val w = d.width
        val h = d.height
        val sx = w.toFloat() / dims[0]
        val sy = h.toFloat() / dims[1]
        val fx = f[0] * sx
        val fy = f[1] * sy
        val cx = c[0] * sx
        val cy = c[1] * sy
        val dp = d.planes[0]
        val db = dp.buffer.order(ByteOrder.LITTLE_ENDIAN)
        val cb = conf?.planes?.get(0)
        val xs = ArrayList<Float>()
        val ys = ArrayList<Float>()
        val zs = ArrayList<Float>()
        for (v in (h * 3 / 10) until (h * 7 / 10)) for (u in (w * 3 / 10) until (w * 7 / 10)) {
            val z = (db.getShort(v * dp.rowStride + u * 2).toInt() and 0xFFFF).toFloat()
            if (z <= 0f) continue
            if (cb != null && (cb.buffer.get(v * cb.rowStride + u * cb.pixelStride).toInt() and 0xFF) < 128) continue
            xs += (u - cx) * z / fx
            ys += (v - cy) * z / fy
            zs += z
        }
        val n = zs.size
        if (n < 20) return null
        // normal equations of z = a x + b y + c, solved by Cramer's rule
        var sxx = 0.0; var sxy = 0.0; var syy = 0.0; var sxs = 0.0; var sys = 0.0
        var sxz = 0.0; var syz = 0.0; var sz = 0.0
        for (i in 0 until n) {
            val x = xs[i].toDouble(); val y = ys[i].toDouble(); val z = zs[i].toDouble()
            sxx += x * x; sxy += x * y; syy += y * y; sxs += x; sys += y
            sxz += x * z; syz += y * z; sz += z
        }
        val m = arrayOf(doubleArrayOf(sxx, sxy, sxs), doubleArrayOf(sxy, syy, sys), doubleArrayOf(sxs, sys, n.toDouble()))
        val r = doubleArrayOf(sxz, syz, sz)
        val det = det3(m)
        if (abs(det) < 1e-9) return null
        val sol = DoubleArray(3) { k -> det3(Array(3) { i -> DoubleArray(3) { j -> if (j == k) r[i] else m[i][j] } }) / det }
        var ss = 0.0
        for (i in 0 until n) {
            val e = zs[i] - (sol[0] * xs[i] + sol[1] * ys[i] + sol[2])
            ss += e * e
        }
        // along z; the distance from the plane is shorter by the tilt
        val tilt = sqrt(1 + sol[0] * sol[0] + sol[1] * sol[1])
        return Plane(zs.sorted()[n / 2], (sqrt(ss / n) / tilt).toFloat(), n)
    }

    private fun det3(m: Array<DoubleArray>) =
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])

    private fun trackingText(frame: Frame) = when (frame.camera.trackingState) {
        TrackingState.TRACKING -> "Tracking"
        TrackingState.PAUSED -> "Tracking paused" + (frame.camera.trackingFailureReason.takeIf { it.name != "NONE" }?.let { ": ${it.name.lowercase().replace('_', ' ')}" } ?: "")
        else -> "Not tracking"
    }

    @Suppress("DEPRECATION")
    private fun displayRotation(): Int = (ctx.getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay?.rotation ?: Surface.ROTATION_0

    companion object {
        private const val VS = "attribute vec4 pos; attribute vec2 uv; varying vec2 v; void main() { gl_Position = pos; v = uv; }"
        private const val CAM_FS = "#extension GL_OES_EGL_image_external : require\nprecision mediump float; varying vec2 v; " +
            "uniform samplerExternalOES tex; uniform float alpha; uniform vec2 range; void main() { gl_FragColor = vec4(texture2D(tex, v).rgb, alpha); }"
        // depth mm = L * 255 + A * 255 * 256; blue near -> red far over range; no depth: clear
        private const val DEPTH_FS = "precision highp float; varying vec2 v; uniform sampler2D tex; uniform float alpha; uniform vec2 range; " +
            "void main() { vec4 t = texture2D(tex, v); float mm = t.r * 255.0 + t.a * 255.0 * 256.0; " +
            "float k = clamp((mm - range.x) / (range.y - range.x), 0.0, 1.0); " +
            "vec3 c = clamp(vec3(1.5 - abs(4.0 * k - 3.0), 1.5 - abs(4.0 * k - 2.0), 1.5 - abs(4.0 * k - 1.0)), 0.0, 1.0); " +
            "gl_FragColor = vec4(c, mm > 0.0 ? alpha : 0.0); }"

        private fun floatBuffer(a: FloatArray): FloatBuffer =
            ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }

        private fun texParams(target: Int) {
            GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
            GLES20.glTexParameteri(target, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        }

        private fun program(vs: String, fs: String): Int {
            fun shader(type: Int, src: String) = GLES20.glCreateShader(type).also {
                GLES20.glShaderSource(it, src)
                GLES20.glCompileShader(it)
            }
            return GLES20.glCreateProgram().also {
                GLES20.glAttachShader(it, shader(GLES20.GL_VERTEX_SHADER, vs))
                GLES20.glAttachShader(it, shader(GLES20.GL_FRAGMENT_SHADER, fs))
                GLES20.glLinkProgram(it)
            }
        }

        /** a plane's rows without their padding */
        private fun tight(src: ByteBuffer, rowBytes: Int, rowStride: Int, rows: Int): ByteBuffer {
            val b = src.duplicate().order(ByteOrder.LITTLE_ENDIAN)
            if (rowStride == rowBytes) return b.apply { position(0); limit(rowBytes * rows) }.slice()
            val out = ByteBuffer.allocateDirect(rowBytes * rows)
            val row = ByteArray(rowBytes)
            for (r in 0 until rows) {
                b.position(r * rowStride)
                b.get(row)
                out.put(row)
            }
            out.position(0)
            return out
        }

        private fun arr(a: FloatArray) = a.joinToString(", ", "[", "]")

        /** YUV_420_888 -> NV21 -> JPEG (quality 95) */
        private fun yuvToJpeg(img: Image): ByteArray {
            val w = img.width
            val h = img.height
            val nv21 = ByteArray(w * h * 3 / 2)
            val y = img.planes[0]
            val yb = y.buffer
            for (r in 0 until h) {
                yb.position(r * y.rowStride)
                yb.get(nv21, r * w, w)
            }
            val u = img.planes[1]
            val v = img.planes[2]
            var o = w * h
            for (r in 0 until h / 2) for (cc in 0 until w / 2) {
                nv21[o++] = v.buffer.get(r * v.rowStride + cc * v.pixelStride)
                nv21[o++] = u.buffer.get(r * u.rowStride + cc * u.pixelStride)
            }
            val bos = ByteArrayOutputStream()
            YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 95, bos)
            return bos.toByteArray()
        }
    }
}
