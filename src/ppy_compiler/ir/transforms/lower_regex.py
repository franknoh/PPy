"""`lower-regex`: every regex operation as a call to a matcher compiled from its pattern.

A pattern becomes one private function of core operations, so the LLVM
and C backends run it as they run anything else. The matcher backtracks
the way CPython's does, with an explicit stack instead of recursion: an
entry records where to resume, the position, and every loop's count, so
popping one puts the matcher back exactly. Alternatives are tried in
order, a greedy repeat gives back one iteration at a time, a lazy one
takes one more, and a group keeps the position of its last iteration. A
repeat of a single byte class does not push per byte: it scans the run
and gives back one byte per backtrack. A match that would need more
stack than the matcher keeps falls back to Python, which is `re`.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..dialects import core, regex
from ..dialects.regex import Alt, Anchor, Chars, Compiled, Group, Node, Repeat, Seq
from ..model import Attribute, Block, Builder, IRFunction, IRModule, Operation, Successor, Value
from ..passes import Pass, PassContext
from ..types import BOOL, I64, U8, BufferType, IRType

__all__ = ["STACK_ENTRIES", "LowerRegex"]

#: Backtrack entries a matcher keeps room for; needing more is the fallback.
STACK_ENTRIES = 4096

_WORD = regex.analyse(rb"\w").tree
assert isinstance(_WORD, Chars)


class LowerRegex(Pass):
    """Rewrite every `regex.*` operation as a call to its pattern's matcher."""

    name = "lower-regex"
    invalidates = ("dominance", "uses")

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        changed = False
        matchers: dict[tuple[str, bytes, int], IRFunction] = {}
        for function in list(module.functions.values()):
            if function.is_declaration:
                continue
            for block in list(function.body.blocks):
                for op in list(block.operations):
                    if op.dialect != "regex":
                        continue
                    pattern, flags = regex.pattern_of(op)
                    key = (op.local_name, pattern, flags)
                    matcher = matchers.get(key)
                    if matcher is None:
                        compiled = regex.analyse(pattern, flags)
                        index = sum(1 for f in module.functions if f.startswith("ppy.regex."))
                        matcher = _Matcher(module, op.local_name, compiled, index).build()
                        matchers[key] = matcher
                    b = Builder().before(op)
                    call = core.call(b, matcher.name, tuple(op.operands), tuple(matcher.results))
                    for old, new in zip(op.results, call.results, strict=True):
                        old.replace_all_uses_with(new)
                    op.erase()
                    changed = True
        if changed:
            ctx.invalidate()
        return changed


def _walk(node: Node) -> Iterator[Node]:
    yield node
    if isinstance(node, Seq):
        for item in node.items:
            yield from _walk(item)
    elif isinstance(node, Alt):
        for branch in node.branches:
            yield from _walk(branch)
    elif isinstance(node, (Group, Repeat)):
        yield from _walk(node.body)


def _simple(node: Repeat) -> bool:
    """A repeat of one byte class: scanned, not backtracked per byte."""
    return isinstance(node.body, Chars)


class _Matcher:
    """One pattern's matcher function."""

    def __init__(self, module: IRModule, mode: str, compiled: Compiled, index: int) -> None:
        self.module = module
        self.mode = mode
        self.compiled = compiled
        self.name = f"ppy.regex.{index}"
        self.loops = [n for n in _walk(compiled.tree) if isinstance(n, Repeat) and not _simple(n)]
        self.loop_index = {id(loop): k for k, loop in enumerate(self.loops)}
        #: An entry: the handler, the position, one value of the handler's own,
        #: then each loop's count and iteration start.
        self.entry_size = 3 + 2 * len(self.loops)
        self.handlers: list[Block] = []
        self.restorers: dict[int, int] = {}
        self.blocks = 0
        self.b = Builder()
        spans = compiled.spans
        params: list[tuple[str, IRType]] = [("buf", BufferType(U8)), ("pos", I64), ("endpos", I64)]
        results: list[IRType] = [BOOL, *([I64] * spans)]
        attributes: dict[str, Attribute] = {
            "ppy.synthesized": f"regex.{mode}",
            "ppy.regex": compiled.pattern.decode("latin-1"),
        }
        self.function = module.add_function(
            self.name, params, results, visibility="private", attributes=attributes
        )
        entry = self.function.add_entry_block()
        b = self.at(entry)
        self.buf, pos, endpos = entry.arguments[0], entry.arguments[1], entry.arguments[2]
        # The string is `buf[:endpos]`; a start past the end is the end.
        length = core.cast(b, core.buffer_len(b, self.buf), I64)
        zero = self.const(0)
        endpos = core.select(b, self.cmp("lt", endpos, zero), zero, endpos)
        self.endpos = core.select(b, self.cmp("gt", endpos, length), length, endpos)
        pos = core.select(b, self.cmp("lt", pos, zero), zero, pos)
        self.pos = core.select(b, self.cmp("gt", pos, length), length, pos)
        self.p = core.alloca(b, I64, name="p")
        self.sp = core.alloca(b, I64, name="sp")
        self.start = core.alloca(b, I64, name="start") if mode == "search" else None
        self.stack = core.alloca(b, I64, count=STACK_ENTRIES * self.entry_size, name="stack")
        self.caps = core.alloca(b, I64, count=spans, name="caps")
        self.counts = [core.alloca(b, I64, name=f"count{k}") for k in range(len(self.loops))]
        self.starts = [core.alloca(b, I64, name=f"iter{k}") for k in range(len(self.loops))]
        self.attempt = self.block("attempt", (("s", I64),))
        self.fail = self.block("fail")
        self.exhausted = self.block("exhausted")
        self.notfound = self.block("notfound")
        self.success = self.block("success")

    # -- pieces ------------------------------------------------------------------

    def block(self, hint: str, arguments: tuple[tuple[str | None, IRType], ...] = ()) -> Block:
        self.blocks += 1
        return self.function.body.add_block(f"{hint}{self.blocks}", arguments)

    def at(self, block: Block) -> Builder:
        self.b = Builder(block)
        return self.b

    def const(self, value: int) -> Value:
        return core.const(self.b, value, I64)

    def load(self, slot: Value) -> Value:
        return core.load(self.b, slot)

    def store(self, value: Value, slot: Value) -> None:
        core.store(self.b, value, slot)

    def add(self, a: Value, n: int) -> Value:
        return core.add(self.b, a, self.const(n), overflow="proven")

    def sub(self, a: Value, b: Value) -> Value:
        return core.sub(self.b, a, b, overflow="proven")

    def cmp(self, predicate: str, a: Value, b: Value) -> Value:
        return core.cmp(self.b, predicate, a, b)

    def byte(self, index: Value) -> Value:
        return core.cast(self.b, core.buffer_load(self.b, self.buf, index), I64)

    def cap(self, slot: int) -> Value:
        return core.ptr_offset(self.b, self.caps, self.const(slot))

    def jump(self, block: Block, *arguments: Value) -> None:
        core.br(self.b, Successor(block, arguments))

    def branch(self, condition: Value, then: Block, otherwise: Block) -> None:
        core.cond_br(self.b, condition, Successor(then), Successor(otherwise))

    def handler(self, hint: str, held: bool = False) -> tuple[int, Block]:
        """A block backtracking resumes at; with `held`, given the entry's own value."""
        block = self.block(hint, (("held", I64),) if held else ())
        self.handlers.append(block)
        return len(self.handlers) - 1, block

    def push(self, label: int, held: Value | int) -> None:
        """An entry: resume at `label` with the position and every count of now.

        Every handler is statically reachable from every failure, so nothing
        defined in a block before this one may be read here: `held` is made
        here or comes from this block.
        """
        b = self.b
        if isinstance(held, int):
            held = self.const(held)
        sp = self.load(self.sp)
        room = self.cmp("lt", sp, self.const(STACK_ENTRIES))
        core.guard(b, room, "range", label="regex.stack.ok")
        stride = core.mul(b, sp, self.const(self.entry_size), overflow="proven")
        base = core.ptr_offset(b, self.stack, stride)
        fields = [self.const(label), self.load(self.p), held]
        for count, start in zip(self.counts, self.starts, strict=True):
            fields.extend((self.load(count), self.load(start)))
        for offset, value in enumerate(fields):
            self.store(value, core.ptr_offset(b, base, self.const(offset)))
        self.store(self.add(sp, 1), self.sp)

    def test(self, x: Value, chars: Chars) -> Value:
        """Whether byte `x` (as i64) is in `chars`."""
        members = [c for c in range(256) if chars.has(c)]
        if len(members) == 256:
            return core.const(self.b, True, BOOL)
        if not members:
            return core.const(self.b, False, BOOL)
        if len(members) == 1:
            return self.cmp("eq", x, self.const(members[0]))
        if len(members) == 255:
            missing = next(c for c in range(256) if not chars.has(c))
            return self.cmp("ne", x, self.const(missing))
        runs: list[tuple[int, int]] = []
        for c in members:
            if runs and runs[-1][1] == c - 1:
                runs[-1] = (runs[-1][0], c)
            else:
                runs.append((c, c))
        if len(runs) <= 3:
            tests = [self.run(x, low, high) for low, high in runs]
            found = tests[0]
            for other in tests[1:]:
                found = core.bitwise(self.b, "or", found, other)
            return found
        b = self.b
        words = [w - (1 << 64) if w >= (1 << 63) else w for w in chars.words]
        word = self.const(words[3])
        for bound, w in ((192, words[2]), (128, words[1]), (64, words[0])):
            word = core.select(b, self.cmp("lt", x, self.const(bound)), self.const(w), word)
        shift = core.bitwise(b, "and", x, self.const(63))
        bit = core.bitwise(b, "and", core.shift(b, "shr", word, shift), self.const(1))
        return self.cmp("ne", bit, self.const(0))

    def run(self, x: Value, low: int, high: int) -> Value:
        if low == high:
            return self.cmp("eq", x, self.const(low))
        if low == 0:
            return self.cmp("le", x, self.const(high))
        if high == 255:
            return self.cmp("ge", x, self.const(low))
        above = self.cmp("ge", x, self.const(low))
        below = self.cmp("le", x, self.const(high))
        return core.bitwise(self.b, "and", above, below)

    # -- the function --------------------------------------------------------------

    def build(self) -> IRFunction:
        compiled = self.compiled
        spans = compiled.spans
        b = self.b
        endpos = self.endpos
        zero = self.const(0)
        attempt, notfound, success = self.attempt, self.notfound, self.success
        pattern = self.block("pattern")
        if self.mode == "search":
            # A search from past the end finds nothing; a match there may
            # still match nothing, as `re` has it.
            core.cond_br(
                b,
                self.cmp("gt", self.pos, endpos),
                Successor(notfound),
                Successor(attempt, (self.pos,)),
            )
        else:
            self.jump(attempt, self.pos)

        b = self.at(attempt)
        s = attempt.arguments[0]
        self.store(s, self.p)
        if self.start is not None:
            self.store(s, self.start)
        self.store(zero, self.sp)
        minus_one = self.const(-1)
        for slot in range(spans):
            self.store(minus_one, self.cap(slot))
        self.store(s, self.cap(0))
        for count, start in zip(self.counts, self.starts, strict=True):
            self.store(zero, count)
            self.store(minus_one, start)
        self.jump(pattern)

        self.at(pattern)
        self.gen(compiled.tree)
        if self.mode == "fullmatch":
            self.branch(self.cmp("eq", self.load(self.p), endpos), success, self.fail)
        else:
            self.jump(success)

        b = self.at(success)
        self.store(self.load(self.p), self.cap(1))
        core.ret(
            b, core.const(b, True, BOOL), *[self.load(self.cap(slot)) for slot in range(spans)]
        )

        b = self.at(notfound)
        core.ret(b, core.const(b, False, BOOL), *[self.const(-1) for _ in range(spans)])

        b = self.at(self.exhausted)
        if self.start is not None:
            following = self.add(self.load(self.start), 1)
            core.cond_br(
                b,
                self.cmp("gt", following, endpos),
                Successor(notfound),
                Successor(attempt, (following,)),
            )
        else:
            self.jump(notfound)

        self.dispatcher()
        self.prune()
        return self.function

    def prune(self) -> None:
        """Drop the blocks nothing reaches: a pattern that cannot fail has no backtracking."""
        region = self.function.body
        reachable: set[int] = set()
        work = [region.blocks[0]]
        while work:
            block = work.pop()
            if id(block) in reachable:
                continue
            reachable.add(id(block))
            work.extend(block.successors)
        for block in list(region.blocks):
            if id(block) in reachable:
                continue
            for op in list(block.operations):
                op.erase()
            region.blocks.remove(block)
        # A slot only ever written -- the stack of a pattern that never
        # backtracks -- goes, with its writes.
        for op in list(region.blocks[0].operations):
            if op.name != "core.alloca":
                continue
            uses = list(op.result.uses)
            if all(isinstance(u, Operation) and u.name == "core.store" and i == 1 for u, i in uses):
                for user, _index in uses:
                    assert isinstance(user, Operation)
                    user.erase()
                op.erase()

    def dispatcher(self) -> None:
        """`fail`: pop an entry, put the matcher back, resume at its handler."""
        b = self.at(self.fail)
        sp = self.load(self.sp)
        pop = self.block("pop")
        self.branch(self.cmp("eq", sp, self.const(0)), self.exhausted, pop)
        b = self.at(pop)
        sp = self.sub(sp, self.const(1))
        self.store(sp, self.sp)
        stride = core.mul(b, sp, self.const(self.entry_size), overflow="proven")
        base = core.ptr_offset(b, self.stack, stride)

        def field(offset: int) -> Value:
            return self.load(core.ptr_offset(self.b, base, self.const(offset)))

        label = field(0)
        self.store(field(1), self.p)
        held = field(2)
        for k, (count, start) in enumerate(zip(self.counts, self.starts, strict=True)):
            self.store(field(3 + 2 * k), count)
            self.store(field(4 + 2 * k), start)
        for index, handler in enumerate(self.handlers):
            following = self.block("dispatch")
            core.cond_br(
                self.b,
                self.cmp("eq", label, self.const(index)),
                Successor(handler, (held,) if handler.arguments else ()),
                Successor(following),
            )
            self.at(following)
        core.unreachable(self.b)

    # -- the pattern -------------------------------------------------------------

    def gen(self, node: Node) -> None:
        if isinstance(node, Seq):
            for item in node.items:
                self.gen(item)
        elif isinstance(node, Chars):
            self.gen_chars(node)
        elif isinstance(node, Anchor):
            self.gen_anchor(node)
        elif isinstance(node, Group):
            self.gen_group(node)
        elif isinstance(node, Alt):
            self.gen_alt(node)
        elif isinstance(node, Repeat):
            if _simple(node):
                self.gen_simple(node)
            else:
                self.gen_repeat(node)
        else:  # pragma: no cover
            raise TypeError(f"no matcher for {node!r}")

    def gen_chars(self, chars: Chars) -> None:
        p = self.load(self.p)
        read = self.block("read")
        advance = self.block("advance")
        self.branch(self.cmp("lt", p, self.endpos), read, self.fail)
        self.at(read)
        self.branch(self.test(self.byte(p), chars), advance, self.fail)
        self.at(advance)
        self.store(self.add(p, 1), self.p)

    def gen_anchor(self, anchor: Anchor) -> None:
        p = self.load(self.p)
        ok = self.block("anchored")
        kind = anchor.kind
        newline = self.const(10)
        if kind in {"start", "string_start"}:
            self.branch(self.cmp("eq", p, self.const(0)), ok, self.fail)
        elif kind == "string_end":
            self.branch(self.cmp("eq", p, self.endpos), ok, self.fail)
        elif kind == "end":
            # `$`: the end, or before a newline that ends the string.
            before = self.block("beforelast")
            last = self.block("last")
            self.branch(self.cmp("eq", p, self.endpos), ok, before)
            self.at(before)
            self.branch(self.cmp("eq", self.add(p, 1), self.endpos), last, self.fail)
            self.at(last)
            self.branch(self.cmp("eq", self.byte(p), newline), ok, self.fail)
        elif kind == "line_start":
            after = self.block("afternewline")
            self.branch(self.cmp("eq", p, self.const(0)), ok, after)
            self.at(after)
            self.branch(
                self.cmp("eq", self.byte(self.sub(p, self.const(1))), newline), ok, self.fail
            )
        elif kind == "line_end":
            at = self.block("atnewline")
            self.branch(self.cmp("eq", p, self.endpos), ok, at)
            self.at(at)
            self.branch(self.cmp("eq", self.byte(p), newline), ok, self.fail)
        else:
            before = self.word_at(self.sub(p, self.const(1)), self.cmp("gt", p, self.const(0)))
            after = self.word_at(p, self.cmp("lt", p, self.endpos))
            differ = self.cmp("ne", before, after)
            if kind == "boundary":
                self.branch(differ, ok, self.fail)
            else:
                self.branch(differ, self.fail, ok)
        self.at(ok)

    def word_at(self, index: Value, in_range: Value) -> Value:
        """Whether the byte at `index` is a word byte; False outside the string."""
        check = self.block("wordcheck")
        merge = self.block("word", (("w", BOOL),))
        core.cond_br(
            self.b,
            in_range,
            Successor(check),
            Successor(merge, (core.const(self.b, False, BOOL),)),
        )
        self.at(check)
        self.jump(merge, self.test(self.byte(index), _WORD))
        self.at(merge)
        return merge.arguments[0]

    def restorer(self, slot: int) -> int:
        """The handler that puts a capture slot back and keeps backtracking."""
        label = self.restorers.get(slot)
        if label is None:
            saved = self.b
            label, block = self.handler("restore", held=True)
            self.restorers[slot] = label
            self.at(block)
            self.store(block.arguments[0], self.cap(slot))
            self.jump(self.fail)
            self.b = saved
        return label

    def gen_group(self, group: Group) -> None:
        if group.index is None:
            self.gen(group.body)
            return
        for slot, inner in ((2 * group.index, True), (2 * group.index + 1, False)):
            if inner:
                self.mark(slot)
                self.gen(group.body)
            else:
                self.mark(slot)

    def mark(self, slot: int) -> None:
        """Record the position in a capture slot, restorable on backtracking."""
        old = self.load(self.cap(slot))
        self.push(self.restorer(slot), old)
        self.store(self.load(self.p), self.cap(slot))

    def gen_alt(self, alt: Alt) -> None:
        join = self.block("join")
        for index, branch in enumerate(alt.branches):
            if index + 1 < len(alt.branches):
                label, handler = self.handler("alt")
                self.push(label, 0)
                self.gen(branch)
                self.jump(join)
                self.at(handler)
            else:
                self.gen(branch)
                self.jump(join)
        self.at(join)

    def gen_repeat(self, repeat: Repeat) -> None:
        k = self.loop_index[id(repeat)]
        count, start = self.counts[k], self.starts[k]
        head = self.block("head")
        body = self.block("body")
        exit_ = self.block("exit")
        self.store(self.const(0), count)
        self.store(self.const(-1), start)
        self.jump(head)

        self.at(head)
        c = self.load(count)
        more = self.block("more")
        self.branch(self.cmp("lt", c, self.const(repeat.minimum)), body, more)
        self.at(more)
        if repeat.greedy:
            if repeat.maximum is not None:
                again = self.block("again")
                self.branch(self.cmp("ge", c, self.const(repeat.maximum)), exit_, again)
                self.at(again)
            label, handler = self.handler("giveup")
            self.push(label, 0)
            self.jump(body)
            self.at(handler)
            self.jump(exit_)
        else:
            label, handler = self.handler("onemore")
            self.push(label, 0)
            self.jump(exit_)
            self.at(handler)
            # Where the last iteration took nothing, another would too.
            progressed = self.block("progressed")
            self.branch(self.cmp("eq", self.load(self.p), self.load(start)), self.fail, progressed)
            self.at(progressed)
            if repeat.maximum is not None:
                self.branch(
                    self.cmp("ge", self.load(count), self.const(repeat.maximum)), self.fail, body
                )
            else:
                self.jump(body)

        self.at(body)
        self.store(self.load(self.p), start)
        self.gen(repeat.body)
        # An iteration that took nothing stands, but is the last one tried
        # once the minimum is met: the loop would never end, and `re` stops
        # there too.
        c = self.load(count)
        self.store(self.add(c, 1), count)
        advance = self.block("advance")
        empty = self.block("empty")
        self.branch(self.cmp("eq", self.load(self.p), self.load(start)), empty, advance)
        self.at(empty)
        self.branch(self.cmp("ge", c, self.const(repeat.minimum)), exit_, advance)
        self.at(advance)
        self.jump(head)

        self.at(exit_)

    def gen_simple(self, repeat: Repeat) -> None:
        chars = repeat.body
        assert isinstance(chars, Chars)
        if repeat.greedy:
            self.gen_simple_greedy(chars, repeat.minimum, repeat.maximum)
        else:
            self.gen_simple_lazy(chars, repeat.minimum, repeat.maximum)

    def gen_simple_greedy(self, chars: Chars, minimum: int, maximum: int | None) -> None:
        p0 = self.load(self.p)
        # `re` wants room for the minimum first, even a minimum of nothing:
        # a match from past the end fails here where an empty group would not.
        fits = self.block("fits")
        self.branch(self.cmp("le", self.add(p0, minimum), self.endpos), fits, self.fail)
        self.at(fits)
        scan = self.block("scan", (("q", I64),))
        bound = self.block("bound")
        probe = self.block("probe")
        done = self.block("scanned", (("q", I64),))
        exit_ = self.block("exit")
        self.jump(scan, p0)
        self.at(scan)
        q = scan.arguments[0]
        core.cond_br(
            self.b, self.cmp("lt", q, self.endpos), Successor(bound), Successor(done, (q,))
        )
        self.at(bound)
        if maximum is not None:
            taken = self.sub(q, p0)
            core.cond_br(
                self.b,
                self.cmp("lt", taken, self.const(maximum)),
                Successor(probe),
                Successor(done, (q,)),
            )
        else:
            self.jump(probe)
        self.at(probe)
        core.cond_br(
            self.b,
            self.test(self.byte(q), chars),
            Successor(scan, (self.add(q, 1),)),
            Successor(done, (q,)),
        )
        self.at(done)
        q = done.arguments[0]
        low = self.add(p0, minimum)
        enough = self.block("enough")
        self.branch(self.cmp("lt", q, low), self.fail, enough)
        self.at(enough)
        self.store(q, self.p)
        if maximum == minimum:
            self.jump(exit_)
            self.at(exit_)
            return
        # Backtracking gives one byte back, down to the minimum.
        label, handler = self.handler("giveback", held=True)
        push = self.block("keep")
        self.branch(self.cmp("gt", q, low), push, exit_)
        self.at(push)
        self.push(label, low)
        self.jump(exit_)
        self.at(handler)
        low = handler.arguments[0]
        shorter = self.sub(self.load(self.p), self.const(1))
        self.store(shorter, self.p)
        again = self.block("keep")
        self.branch(self.cmp("gt", shorter, low), again, exit_)
        self.at(again)
        self.push(label, low)
        self.jump(exit_)
        self.at(exit_)

    def gen_simple_lazy(self, chars: Chars, minimum: int, maximum: int | None) -> None:
        p0 = self.load(self.p)
        low = self.add(p0, minimum)
        exit_ = self.block("exit")
        # The minimum, first.
        fits = self.block("fits")
        self.branch(self.cmp("le", low, self.endpos), fits, self.fail)
        self.at(fits)
        if minimum:
            loop = self.block("least", (("i", I64),))
            probe = self.block("probe")
            done = self.block("leastdone")
            self.jump(loop, p0)
            self.at(loop)
            i = loop.arguments[0]
            self.branch(self.cmp("lt", i, low), probe, done)
            self.at(probe)
            core.cond_br(
                self.b,
                self.test(self.byte(i), chars),
                Successor(loop, (self.add(i, 1),)),
                Successor(self.fail),
            )
            self.at(done)
        self.store(low, self.p)
        if maximum is not None and maximum <= minimum:
            self.jump(exit_)
            self.at(exit_)
            return
        label, handler = self.handler("onemore", held=True)
        self.push(label, p0)
        self.jump(exit_)
        # Backtracking takes one byte more, up to the maximum.
        self.at(handler)
        p0 = handler.arguments[0]
        q = self.load(self.p)
        room = self.block("room")
        if maximum is not None:
            self.branch(self.cmp("lt", self.sub(q, p0), self.const(maximum)), room, self.fail)
        else:
            self.jump(room)
        self.at(room)
        probe = self.block("probe")
        self.branch(self.cmp("lt", q, self.endpos), probe, self.fail)
        self.at(probe)
        took = self.block("took")
        self.branch(self.test(self.byte(q), chars), took, self.fail)
        self.at(took)
        q = self.add(q, 1)
        self.store(q, self.p)
        if maximum is not None:
            again = self.block("again")
            self.branch(self.cmp("lt", self.sub(q, p0), self.const(maximum)), again, exit_)
            self.at(again)
        self.push(label, p0)
        self.jump(exit_)
        self.at(exit_)
