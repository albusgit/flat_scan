package com.albus.flatscan

import com.albus.flatscan.data.DocumentSummary
import java.util.Locale

/**
 * Document tags (feature round 5): free text, kept in the documents.tags column as one comma-separated
 * string ([com.albus.flatscan.data.DocumentEntity.tags], no schema change). A tag is trimmed, its inner
 * whitespace collapsed to one space and commas (the separator) turned into spaces, at most [MAX_LENGTH]
 * characters. On one document tags are unique ignoring case; the first spelling stays. Pure (TagsTest).
 */
object Tags {
    const val MAX_LENGTH = 32

    private val space = Regex("\\s+")

    /** [raw] as a tag ("" when nothing is left). */
    fun clean(raw: String): String {
        val s = raw.replace(',', ' ').replace(space, " ").trim()
        if (s.length <= MAX_LENGTH) return s
        // not between the two halves of a surrogate pair (an emoji)
        val end = if (Character.isHighSurrogate(s[MAX_LENGTH - 1])) MAX_LENGTH - 1 else MAX_LENGTH
        return s.substring(0, end).trim()
    }

    /** What two tags are compared by: the same key is the same tag. */
    fun key(tag: String): String = clean(tag).lowercase(Locale.ROOT)

    /** [tags] cleaned, without empty ones and repeats (ignoring case), in their order. */
    fun normalise(tags: Iterable<String>): List<String> {
        val seen = HashSet<String>()
        return tags.map(::clean).filter { it.isNotEmpty() && seen.add(it.lowercase(Locale.ROOT)) }
    }

    /** The tags of a documents.tags value. */
    fun parse(column: String): List<String> = if (column.isBlank()) emptyList() else normalise(column.split(','))

    /** [tags] as a documents.tags value. */
    fun format(tags: Iterable<String>): String = normalise(tags).joinToString(",")

    fun has(tags: List<String>, tag: String): Boolean = key(tag).let { k -> tags.any { key(it) == k } }

    /** [tags] with [tag] at the end (unchanged when it is there already, in any case, or blank). */
    fun plus(tags: List<String>, tag: String): List<String> = normalise(tags + tag)

    /** [tags] without [tag] (any case). */
    fun minus(tags: List<String>, tag: String): List<String> = key(tag).let { k -> normalise(tags).filter { key(it) != k } }

    /**
     * Every tag of [docs], most used first, ties alphabetical. A tag spelt differently on different
     * documents is one tag, shown as the first document has it (the list is newest first).
     */
    fun ranked(docs: List<DocumentSummary>): List<String> {
        val spelling = LinkedHashMap<String, String>()
        val count = HashMap<String, Int>()
        for (d in docs) for (t in parse(d.tags)) {
            val k = t.lowercase(Locale.ROOT)
            spelling.getOrPut(k) { t }
            count[k] = (count[k] ?: 0) + 1
        }
        return spelling.keys.sortedWith(compareByDescending<String> { count[it] }.thenBy { it }).map { spelling.getValue(it) }
    }

    /**
     * The documents that have every tag of [filter] (AND: each chip picked narrows the list further);
     * all of [docs] when [filter] is empty.
     */
    fun filter(docs: List<DocumentSummary>, filter: Collection<String>): List<DocumentSummary> {
        val want = filter.map(::key).filter { it.isNotEmpty() }.toSet()
        if (want.isEmpty()) return docs
        return docs.filter { d -> parse(d.tags).map { it.lowercase(Locale.ROOT) }.containsAll(want) }
    }
}

/** A tag over the documents being tagged: on all of them, on some, on none. */
enum class TagState { All, Some, None }

/**
 * The tag sheet's view of the documents it tags ([docs]: document id -> its tags), for one document or
 * a selection of several. Pure (TagEditTest); the library applies a change with the same [Tags.plus] /
 * [Tags.minus] ([com.albus.flatscan.data.Library.setTag]).
 */
data class TagEdit(val docs: Map<String, List<String>>) {
    fun state(tag: String): TagState {
        val n = docs.values.count { Tags.has(it, tag) }
        return when {
            docs.isEmpty() || n == 0 -> TagState.None
            n == docs.size -> TagState.All
            else -> TagState.Some
        }
    }

    /**
     * What tapping [tag] does: true puts it on every document (it was on none or only some of them),
     * false takes it off all of them (they all had it).
     */
    fun tapTurnsOn(tag: String): Boolean = state(tag) != TagState.All

    /** [tag] put on ([on]) or taken off every document. */
    fun set(tag: String, on: Boolean): TagEdit = TagEdit(docs.mapValues { (_, t) -> if (on) Tags.plus(t, tag) else Tags.minus(t, tag) })

    /**
     * The tags the sheet offers, in a fixed order: the documents' own tags first (in [known]'s order,
     * then any [known] does not have), then the rest of [known] (the library's tags, most used first).
     */
    fun choices(known: List<String>): List<String> {
        val own = Tags.normalise(docs.values.flatten())
        val ownKeys = own.map(Tags::key).toSet()
        val knownOwn = known.filter { Tags.key(it) in ownKeys }
        return Tags.normalise(knownOwn + own + known)
    }
}
