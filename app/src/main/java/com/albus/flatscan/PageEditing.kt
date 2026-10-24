package com.albus.flatscan

/**
 * The page structure an editing session can change: the page order of every document it touched
 * ([docs], document id -> page ids) and the result rotation of pages it turned ([rotations], page id ->
 * quarter turns, only pages whose rotation it changed). Immutable: every edit makes a new state.
 */
data class PagesState(val current: String, val docs: Map<String, List<String>>, val rotations: Map<String, Int> = emptyMap()) {
    val pages: List<String> get() = docs[current].orEmpty()
}

/**
 * Page operations of the page editor, as pure functions on [PagesState] (unit tested, PageEditingTest).
 * Indices are positions in the current document; selections keep their document order.
 */
object PageOps {
    /** [sel] taken out and put back as one block so that its first page lands at [to] (index in the list without [sel]). */
    fun move(s: PagesState, sel: Collection<String>, to: Int): PagesState {
        val moved = s.pages.filter { it in sel }
        if (moved.isEmpty()) return s
        val rest = s.pages.filter { it !in sel }
        val at = to.coerceIn(0, rest.size)
        return s.withPages(rest.take(at) + moved + rest.drop(at))
    }

    /** One page from position [from] to position [to] (drag and drop: the result has it at [to]). */
    fun moveOne(s: PagesState, from: Int, to: Int): PagesState {
        val p = s.pages
        if (from !in p.indices || to !in p.indices || from == to) return s
        val l = p.toMutableList()
        val x = l.removeAt(from)
        l.add(to, x)
        return s.withPages(l)
    }

    fun reverse(s: PagesState): PagesState = s.withPages(s.pages.reversed())

    fun delete(s: PagesState, sel: Collection<String>): PagesState = s.withPages(s.pages.filter { it !in sel })

    /** [ids] (new pages) inserted at [at]. */
    fun insert(s: PagesState, at: Int, ids: List<String>): PagesState {
        val p = s.pages
        val i = at.coerceIn(0, p.size)
        return s.withPages(p.take(i) + ids + p.drop(i))
    }

    /** Each selected page followed by its copy; [copyId] names the copies (the caller copies the files). */
    fun duplicate(s: PagesState, sel: Collection<String>, copyId: (String) -> String): Pair<PagesState, Map<String, String>> {
        val copies = mutableMapOf<String, String>()
        val out = s.pages.flatMap { if (it in sel) listOf(it, copyId(it).also { c -> copies[it] = c }) else listOf(it) }
        val rot = s.rotations + copies.mapNotNull { (o, c) -> s.rotations[o]?.let { c to it } }
        return s.copy(docs = s.docs + (s.current to out), rotations = rot) to copies
    }

    /** The selected pages turned [quarters] clockwise; [base] gives a page's rotation before the session. */
    fun rotate(s: PagesState, sel: Collection<String>, quarters: Int, base: (String) -> Int): PagesState {
        val r = s.rotations.toMutableMap()
        for (id in s.pages) if (id in sel) r[id] = Math.floorMod((r[id] ?: base(id)) + quarters, 4)
        return s.copy(rotations = r)
    }

    /** Pages from [at] on go to the new document [newDoc] (in order); the current document keeps the rest. */
    fun split(s: PagesState, at: Int, newDoc: String): PagesState {
        val p = s.pages
        if (at <= 0 || at >= p.size) return s
        return s.copy(docs = s.docs + (s.current to p.take(at)) + (newDoc to p.drop(at)))
    }

    /** [sel] leave the current document and are appended to [target] (whose pages are [targetPages] when not in the state yet). */
    fun moveToDocument(s: PagesState, sel: Collection<String>, target: String, targetPages: List<String>): PagesState {
        if (target == s.current) return s
        val moved = s.pages.filter { it in sel }
        if (moved.isEmpty()) return s
        val t = (s.docs[target] ?: targetPages).filter { it !in sel } + moved
        return s.copy(docs = s.docs + (s.current to s.pages.filter { it !in sel }) + (target to t))
    }

    private fun PagesState.withPages(p: List<String>) = copy(docs = docs + (current to p))
}

/**
 * Undo / redo of a whole page-editing session: every structural edit ([apply]) pushes the state before
 * it; [undo] / [redo] walk the history. [limit] states are kept.
 */
class PagesHistory(initial: PagesState, private val limit: Int = 100) {
    var state: PagesState = initial
        private set
    private val past = ArrayDeque<PagesState>()
    private val future = ArrayDeque<PagesState>()

    val canUndo: Boolean get() = past.isNotEmpty()
    val canRedo: Boolean get() = future.isNotEmpty()

    /** Records [next] as one step (nothing when it equals the current state). */
    fun apply(next: PagesState): PagesState {
        if (next == state) return state
        past.addLast(state)
        while (past.size > limit) past.removeFirst()
        future.clear()
        state = next
        return state
    }

    fun undo(): PagesState? {
        val prev = past.removeLastOrNull() ?: return null
        future.addLast(state)
        state = prev
        return prev
    }

    fun redo(): PagesState? {
        val next = future.removeLastOrNull() ?: return null
        past.addLast(state)
        state = next
        return next
    }

    /** Every page any state of this session refers to (files that must not be deleted while it lasts). */
    fun referenced(): Set<String> = (past + future + state).flatMap { s -> s.docs.values.flatten() }.toSet()
}
