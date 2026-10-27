package com.albus.flatscan.sync

import okhttp3.OkHttpClient
import okhttp3.internal.tls.OkHostnameVerifier
import java.io.IOException
import java.security.KeyStore
import java.security.MessageDigest
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.concurrent.TimeUnit
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509TrustManager

/** Why an upload or a connection test failed, and whether trying again later can help. */
class SyncException(val kind: Kind, message: String, cause: Throwable? = null) : IOException(message, cause) {
    enum class Kind {
        /** The server rejected the credentials (401 / 403): fix the settings, no retry. */
        Auth,
        /** Wrong address, missing folder, unexpected answer: fix the settings, no retry. */
        Config,
        /** The server's certificate is not trusted and not pinned ([UntrustedCertificateException]). */
        Certificate,
        /** Network or server trouble (timeouts, 5xx): try again later. */
        Transient,
    }

    val retryable: Boolean get() = kind == Kind.Transient
}

/**
 * The server presented a certificate the system does not trust and the user has not pinned. The UI shows
 * [sha256] (and [subject]) and, if the user trusts it, pins it for [host] (trust on first use).
 */
class UntrustedCertificateException(val host: String, val sha256: String, val subject: String) :
    CertificateException("untrusted certificate for $host, SHA-256 $sha256")

/** HTTP clients for the sync targets: HTTPS with the system's trust, plus certificates the user pinned. */
object Http {
    /** SHA-256 of a certificate as colon-separated upper-case hex (what the UI shows and the pins store). */
    fun fingerprint(c: X509Certificate): String =
        MessageDigest.getInstance("SHA-256").digest(c.encoded).joinToString(":") { "%02X".format(it) }

    /**
     * A client for [host] that trusts the system's CAs, and also a certificate whose fingerprint is in
     * [pins] (host -> SHA-256). An unknown, untrusted certificate fails with [UntrustedCertificateException]
     * as the cause. Never logs: no logging interceptor, and the credentials go only in headers.
     */
    fun client(host: String, pins: Map<String, String>, timeoutSec: Long = 30): OkHttpClient {
        val system = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm()).apply { init(null as KeyStore?) }
            .trustManagers.filterIsInstance<X509TrustManager>().first()
        val pin = pins[host.lowercase()]
        // trust on first use (the plan's rule for self-signed servers): the system's own validation always runs
        // first; only a certificate it rejects is compared with the fingerprint the user pinned
        @android.annotation.SuppressLint("CustomX509TrustManager")
        val tm = object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<out X509Certificate>, authType: String) = system.checkClientTrusted(chain, authType)
            override fun checkServerTrusted(chain: Array<out X509Certificate>, authType: String) {
                try {
                    system.checkServerTrusted(chain, authType)
                } catch (e: CertificateException) {
                    val leaf = chain.firstOrNull() ?: throw e
                    val fp = fingerprint(leaf)
                    if (pin != null && fp.equals(pin, ignoreCase = true)) return
                    throw UntrustedCertificateException(host, fp, leaf.subjectX500Principal.name).apply { initCause(e) }
                }
            }
            override fun getAcceptedIssuers(): Array<X509Certificate> = system.acceptedIssuers
        }
        val ssl = SSLContext.getInstance("TLS").apply { init(null, arrayOf(tm), null) }
        // a pinned self-signed certificate often names another host (or none, for an IP address): the pin
        // identifies the server, so its name is not checked; every other certificate goes through the usual check
        val verifier = HostnameVerifier { h, session ->
            val leaf = runCatching { session.peerCertificates.firstOrNull() as? X509Certificate }.getOrNull()
            if (pin != null && leaf != null && fingerprint(leaf).equals(pin, ignoreCase = true)) true
            else OkHostnameVerifier.verify(h, session)
        }
        return OkHttpClient.Builder()
            .sslSocketFactory(ssl.socketFactory, tm)
            .hostnameVerifier(verifier)
            .connectTimeout(timeoutSec, TimeUnit.SECONDS)
            .readTimeout(timeoutSec * 4, TimeUnit.SECONDS)
            .writeTimeout(timeoutSec * 4, TimeUnit.SECONDS)
            .retryOnConnectionFailure(false)   // retries are the sync engine's, with its own idempotency
            .build()
    }

    /** The untrusted-certificate cause of [t], if that is why it failed. */
    fun untrusted(t: Throwable?): UntrustedCertificateException? {
        var e = t
        while (e != null) {
            if (e is UntrustedCertificateException) return e
            e = e.cause
        }
        return null
    }

    /** [t] as a [SyncException]: certificate, network (transient) or as it is. */
    fun classify(t: Throwable): SyncException = when {
        t is SyncException -> t
        untrusted(t) != null -> SyncException(SyncException.Kind.Certificate, "The server's certificate is not trusted (SHA-256 ${untrusted(t)!!.sha256})", t)
        t is java.net.UnknownHostException -> SyncException(SyncException.Kind.Transient, "Server not found: ${t.message}", t)
        t is javax.net.ssl.SSLException -> SyncException(SyncException.Kind.Config, "Secure connection failed: ${t.message}", t)
        else -> SyncException(SyncException.Kind.Transient, t.message ?: t.javaClass.simpleName, t)
    }

    /** A failed HTTP answer as a [SyncException]: 401/403 credentials, 404/405/409 settings, others transient. */
    fun status(code: Int, what: String): SyncException = when (code) {
        401, 403 -> SyncException(SyncException.Kind.Auth, "$what: the server refused the credentials (HTTP $code)")
        400, 404, 405, 409, 413, 415, 422 -> SyncException(SyncException.Kind.Config, "$what: HTTP $code")
        else -> SyncException(SyncException.Kind.Transient, "$what: HTTP $code")
    }

    /**
     * The address as typed, with https:// added when no scheme is given (HTTPS by default) and a trailing
     * slash; null when it is not an http(s) URL.
     */
    fun normalizeUrl(s: String): String? {
        val t = s.trim()
        if (t.isEmpty()) return null
        val withScheme = if ("://" in t) t else "https://$t"
        val u = okhttp3.HttpUrl.Companion.run { withScheme.toHttpUrlOrNull() } ?: return null
        return u.toString().let { if (it.endsWith("/")) it else "$it/" }
    }
}
