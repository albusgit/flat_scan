package com.albus.flatscan.sync

import java.io.Closeable
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * A TCP proxy in front of a test server that can lose the network the way a phone does:
 * [Mode.CutRequest] closes the connection after a number of request bytes (offline mid-upload),
 * [Mode.DropResponse] lets the whole request through but never delivers the server's answer (the upload
 * arrived, the phone did not hear it). Each fault happens once, then the proxy passes everything again.
 */
class FlakyProxy(private val host: String, private val port: Int) : Closeable {
    sealed interface Mode {
        data object Pass : Mode
        data class CutRequest(val afterBytes: Long) : Mode
        data object DropResponse : Mode
    }

    @Volatile var mode: Mode = Mode.Pass
    val faults = AtomicLong()
    private val server = ServerSocket(0, 50, InetAddress.getLoopbackAddress())
    val localPort: Int get() = server.localPort

    init {
        thread(isDaemon = true, name = "proxy-accept") {
            while (!server.isClosed) {
                val client = runCatching { server.accept() }.getOrNull() ?: break
                thread(isDaemon = true) { handle(client) }
            }
        }
    }

    private fun handle(client: Socket) {
        val upstream = Socket(host, port)
        val m = mode
        fun closeAll() { runCatching { client.close() }; runCatching { upstream.close() } }
        // request: client -> server
        thread(isDaemon = true) {
            pump(client.getInputStream(), upstream.getOutputStream()) { sent ->
                if (m is Mode.CutRequest && sent >= m.afterBytes) {
                    mode = Mode.Pass
                    faults.incrementAndGet()
                    closeAll()
                    false
                } else true
            }
        }
        // answer: server -> client
        thread(isDaemon = true) {
            if (m is Mode.DropResponse) {
                val first = runCatching { upstream.getInputStream().read() }.getOrDefault(-1)
                if (first >= 0) {
                    mode = Mode.Pass
                    faults.incrementAndGet()
                    Thread.sleep(200)   // the server has finished the request: its answer is lost
                }
                closeAll()
            } else {
                pump(upstream.getInputStream(), client.getOutputStream()) { true }
                closeAll()
            }
        }
    }

    private fun pump(i: InputStream, o: OutputStream, keepGoing: (Long) -> Boolean) {
        val b = ByteArray(8192)
        var total = 0L
        try {
            while (true) {
                val n = i.read(b)
                if (n < 0) break
                o.write(b, 0, n)
                o.flush()
                total += n
                if (!keepGoing(total)) break
            }
        } catch (_: Exception) {
        }
    }

    override fun close() = server.close()
}
