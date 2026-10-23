package com.albus.flatscan.sync

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.provider.DocumentsContract
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.albus.flatscan.FileNameTemplate
import org.json.JSONObject

/** Where finished documents are uploaded. */
enum class SyncKind(val label: String) { Off("Off"), WebDav("WebDAV (Nextcloud, ownCloud, Synology, NAS)"), Paperless("paperless-ngx"), Folder("A folder on this phone") }

/**
 * The sync settings. Addresses, user name, folder and options live in plain preferences ("sync");
 * the password, the API token and the pinned certificates in EncryptedSharedPreferences ("sync_secrets",
 * AES-256 with a key in the Android Keystore). Nothing here is ever logged.
 */
class SyncConfig(private val ctx: Context) {
    private val p: SharedPreferences = ctx.getSharedPreferences("sync", Context.MODE_PRIVATE)
    private val secrets: SharedPreferences by lazy {
        runCatching {
            EncryptedSharedPreferences.create(
                ctx, "sync_secrets",
                MasterKey.Builder(ctx).setKeyScheme(MasterKey.KeyScheme.AES256_GCM).build(),
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
            )
        }.getOrElse {
            // a Keystore that cannot be used (some emulators, a restored backup with a lost key): secrets are
            // then not stored at all rather than stored in clear
            NoSecrets
        }
    }

    var kind: SyncKind
        get() = SyncKind.entries.firstOrNull { it.name == p.getString("kind", null) } ?: SyncKind.Off
        set(v) = p.edit().putString("kind", v.name).apply()

    /** WebDAV: the folder's full URL; paperless: the server's base URL. */
    var url: String
        get() = p.getString("url", "") ?: ""
        set(v) = p.edit().putString("url", v.trim()).apply()

    var user: String
        get() = p.getString("user", "") ?: ""
        set(v) = p.edit().putString("user", v.trim()).apply()

    /** WebDAV password or paperless API token. */
    var secret: String
        get() = secrets.getString("secret", "") ?: ""
        set(v) = secrets.edit().putString("secret", v).apply()

    /** The SAF folder (OpenDocumentTree) for [SyncKind.Folder]. */
    var folderUri: String?
        get() = p.getString("folder_uri", null)
        set(v) = p.edit().apply { if (v == null) remove("folder_uri") else putString("folder_uri", v) }.apply()

    /** paperless: tags added to every upload (comma separated), and the correspondent (name). */
    var paperlessTags: String
        get() = p.getString("pl_tags", "") ?: ""
        set(v) = p.edit().putString("pl_tags", v).apply()

    var paperlessCorrespondent: String
        get() = p.getString("pl_corr", "") ?: ""
        set(v) = p.edit().putString("pl_corr", v.trim()).apply()

    /** Upload a document by itself when it is closed after a scan or an edit; off: only by hand. */
    var autoUpload: Boolean
        get() = p.getBoolean("auto", false)
        set(v) = p.edit().putBoolean("auto", v).apply()

    var wifiOnly: Boolean
        get() = p.getBoolean("wifi_only", true)
        set(v) = p.edit().putBoolean("wifi_only", v).apply()

    var chargingOnly: Boolean
        get() = p.getBoolean("charging_only", false)
        set(v) = p.edit().putBoolean("charging_only", v).apply()

    /** Certificates the user chose to trust (trust on first use): host -> SHA-256. */
    var pins: Map<String, String>
        get() = runCatching { JSONObject(secrets.getString("pins", "{}") ?: "{}").let { j -> j.keys().asSequence().associateWith { j.getString(it) } } }.getOrDefault(emptyMap())
        set(v) = secrets.edit().putString("pins", JSONObject(v as Map<*, *>).toString()).apply()

    fun pin(host: String, sha256: String) {
        pins = pins + (host.lowercase() to sha256)
    }

    val configured: Boolean
        get() = when (kind) {
            SyncKind.Off -> false
            SyncKind.WebDav, SyncKind.Paperless -> Http.normalizeUrl(url) != null && secret.isNotEmpty()
            SyncKind.Folder -> folderUri != null
        }

    /** The uploader for the current settings, or null when sync is off or incomplete. */
    fun uploader(): Uploader? {
        if (!configured) return null
        return when (kind) {
            SyncKind.Off -> null
            SyncKind.WebDav -> {
                val u = Http.normalizeUrl(url)!!
                WebDavUploader(WebDavClient(Http.client(Uri.parse(u).host.orEmpty(), pins), u, user, secret))
            }
            SyncKind.Paperless -> {
                val u = Http.normalizeUrl(url)!!
                PaperlessUploader(
                    PaperlessClient(Http.client(Uri.parse(u).host.orEmpty(), pins), u, secret),
                    paperlessTags.split(",").map { it.trim() }.filter { it.isNotEmpty() },
                    paperlessCorrespondent.ifBlank { null },
                )
            }
            SyncKind.Folder -> SafFolderUploader(ctx, Uri.parse(folderUri))
        }
    }

    /** File-name template and date format of uploads and exports (feature 3 shares them). */
    var nameTemplate: String
        get() = p.getString("name_template", FileNameTemplate.DEFAULT) ?: FileNameTemplate.DEFAULT
        set(v) = p.edit().putString("name_template", v).apply()

    var dateFormat: String
        get() = p.getString("date_format", FileNameTemplate.DEFAULT_DATE) ?: FileNameTemplate.DEFAULT_DATE
        set(v) = p.edit().putString("date_format", v).apply()

    private object NoSecrets : SharedPreferences {
        override fun getAll(): MutableMap<String, *> = mutableMapOf<String, Any>()
        override fun getString(key: String?, defValue: String?) = defValue
        override fun getStringSet(key: String?, defValues: MutableSet<String>?) = defValues
        override fun getInt(key: String?, defValue: Int) = defValue
        override fun getLong(key: String?, defValue: Long) = defValue
        override fun getFloat(key: String?, defValue: Float) = defValue
        override fun getBoolean(key: String?, defValue: Boolean) = defValue
        override fun contains(key: String?) = false
        override fun edit(): SharedPreferences.Editor = object : SharedPreferences.Editor {
            override fun putString(key: String?, value: String?) = this
            override fun putStringSet(key: String?, values: MutableSet<String>?) = this
            override fun putInt(key: String?, value: Int) = this
            override fun putLong(key: String?, value: Long) = this
            override fun putFloat(key: String?, value: Float) = this
            override fun putBoolean(key: String?, value: Boolean) = this
            override fun remove(key: String?) = this
            override fun clear() = this
            override fun commit() = false
            override fun apply() {}
        }
        override fun registerOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}
        override fun unregisterOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}
    }
}

/**
 * A folder picked with the Storage Access Framework (Syncthing, Dropbox-style folder apps, SMB apps): no
 * network code at all. The document is created first (its name reserved: a taken name gets " (2)"...) and
 * its URI recorded; a retry writes into that same document again.
 */
class SafFolderUploader(private val ctx: Context, private val tree: Uri) : Uploader {
    private val cr get() = ctx.contentResolver
    private val treeDoc: Uri get() = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))

    private fun names(): Set<String> {
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val out = mutableSetOf<String>()
        cr.query(children, arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)?.use { c ->
            while (c.moveToNext()) out += c.getString(0)
        }
        return out
    }

    override suspend fun upload(job: UploadJob, ledger: SyncLedger): String = try {
        val reserved = ledger.get(job.docId)?.takeIf { it.startsWith(PREFIX) }?.removePrefix(PREFIX)?.let(Uri::parse)
        val target = reserved ?: run {
            val taken = names()
            val name = FileNameTemplate.unique(job.fileName, "pdf") { it in taken }
            val doc = DocumentsContract.createDocument(cr, treeDoc, "application/pdf", name)
                ?: throw SyncException(SyncException.Kind.Config, "Could not create $name in the folder")
            ledger.set(job.docId, PREFIX + doc)
            doc
        }
        (cr.openOutputStream(target, "wt") ?: throw SyncException(SyncException.Kind.Config, "Could not write to the folder")).use { o ->
            job.file.inputStream().use { it.copyTo(o) }
        }
        ledger.set(job.docId, null)
        target.lastPathSegment?.substringAfterLast('/') ?: "the folder"
    } catch (e: SecurityException) {
        throw SyncException(SyncException.Kind.Config, "No access to the folder any more: pick it again", e)
    } catch (e: java.io.FileNotFoundException) {
        ledger.set(job.docId, null)
        throw SyncException(SyncException.Kind.Transient, "The file in the folder went missing: ${e.message}", e)
    }

    override suspend fun test(): String = try {
        val doc = DocumentsContract.createDocument(cr, treeDoc, "text/plain", ".flatscan-test.txt")
            ?: throw SyncException(SyncException.Kind.Config, "Could not create a file in the folder")
        cr.openOutputStream(doc, "wt")?.use { it.write("FlatScan connection test".toByteArray()) }
        DocumentsContract.deleteDocument(cr, doc)
        "Created and deleted a test file in the folder"
    } catch (e: SecurityException) {
        throw SyncException(SyncException.Kind.Config, "No access to the folder: pick it again", e)
    }

    companion object {
        const val PREFIX = "saf:"
    }
}
