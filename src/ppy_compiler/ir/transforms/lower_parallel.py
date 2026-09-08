"""`lower-parallel`: the parallel dialect as core and concurrency operations.

The backend the configuration selected decides the shape: `serial` calls
the body over the whole range; `threads` splits the range into as many
chunks as there are workers, runs all but the last on spawned threads and
the last on the calling one, then joins and folds; `simd` is the serial
shape handed to the vectorizer; `openmp` leaves the operations for the C
backend, which spells them as OpenMP regions. A range too small to be
worth the threads runs serially however the build was configured, and a
floating-point reduction that may not reassociate is never split.
"""

from __future__ import annotations

from ..dialects import concurrency, core, parallel
from ..model import Block, Builder, IRFunction, IRModule, Operation, Successor, Value
from ..passes import Pass, PassContext
from ..types import BOOL, I64, FloatType, IntType, PtrType

__all__ = ["BACKENDS", "LowerParallel"]

BACKENDS = ("serial", "threads", "simd", "openmp")


class LowerParallel(Pass):
    """Rewrite every `parallel.*` operation for the selected backend."""

    name = "lower-parallel"
    invalidates = ("dominance", "uses")

    def __init__(self, backend: str = "threads", threads: int = 1, minimum: int = 4096) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"parallel backend is one of {BACKENDS}, not {backend!r}")
        self.backend = backend
        self.threads = max(1, int(threads))
        self.minimum = max(1, int(minimum))

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        if self.backend == "openmp":
            return False
        changed = False
        for function in list(module.functions.values()):
            if function.is_declaration:
                continue
            for block in list(function.body.blocks):
                for op in list(block.operations):
                    if op.dialect != "parallel":
                        continue
                    _Rewriter(module, function, self, ctx).rewrite(op)
                    changed = True
        if changed:
            ctx.invalidate()
        return changed


class _Rewriter:
    def __init__(
        self, module: IRModule, function: IRFunction, pass_: LowerParallel, ctx: PassContext
    ) -> None:
        self.module = module
        self.function = function
        self.pass_ = pass_
        self.ctx = ctx
        self.counter = 0

    def fresh(self, hint: str) -> str:
        self.counter += 1
        return f"{hint}{self.counter}"

    def rewrite(self, op: Operation) -> None:
        callee = self.module.functions[op.attributes["callee"].name]  # type: ignore[union-attr]
        threads = self.pass_.threads if self.pass_.backend == "threads" else 1
        if op.local_name == "reduce":
            result = op.results[0]
            floating = isinstance(result.type, FloatType)
            if floating and not op.attributes.get("reassociate", True):
                threads = 1
            self.ctx.remark(
                f"parallel reduction over @{callee.name} lowered to "
                f"{'one chunk' if threads == 1 else f'{threads} chunks'}"
            )
        else:
            self.ctx.remark(
                f"parallel loop over @{callee.name} lowered to "
                f"{'one chunk' if threads == 1 else f'{threads} chunks'}"
            )
        if threads == 1:
            self.serial(op, callee)
        else:
            self.chunked(op, callee, threads)

    # -- serial ----------------------------------------------------------------

    def serial(self, op: Operation, callee: IRFunction) -> None:
        b = Builder().before(op)
        if op.local_name == "for":
            core.call(b, callee.name, tuple(op.operands), ())
            op.erase()
            return
        if op.local_name == "reduce":
            value = core.call(b, callee.name, tuple(op.operands), callee.results).results[0]
            op.results[0].replace_all_uses_with(value)
            op.erase()
            return
        wrapper = self.map_wrapper(callee, op.operands[-1].type)
        core.call(b, wrapper.name, tuple(op.operands), ())
        op.erase()

    def map_wrapper(self, callee: IRFunction, out_type) -> IRFunction:  # type: ignore[no-untyped-def]
        """`@callee__map(captures..., begin, end, out)`: the store loop."""
        name = f"{callee.name}__map"
        existing = self.module.functions.get(name)
        if existing is not None:
            return existing
        captures = list(callee.params[:-1])
        params = [*captures, ("begin", I64), ("end", I64), ("out", out_type)]
        wrapper = self.module.add_function(
            name, params, [], visibility="private", attributes={"ppy.synthesized": "parallel.map"}
        )
        entry = wrapper.add_entry_block()
        arguments = list(entry.arguments)
        caps, begin, end, out = arguments[:-3], arguments[-3], arguments[-2], arguments[-1]
        head = wrapper.body.add_block("map.head", [("i", I64)])
        body = wrapper.body.add_block("map.body")
        done = wrapper.body.add_block("map.done")
        b = Builder(entry)
        core.br(b, Successor(head, (begin,)))
        b = Builder(head)
        i = head.arguments[0]
        core.cond_br(b, core.cmp(b, "lt", i, end), Successor(body), Successor(done))
        b = Builder(body)
        value = core.call(b, callee.name, (*caps, i), callee.results).results[0]
        core.store(b, value, core.ptr_offset(b, out, i))
        core.br(b, Successor(head, (core.add(b, i, core.const(b, 1, I64), overflow="wrap"),)))
        b = Builder(done)
        core.ret(b)
        return wrapper

    # -- chunks on threads -------------------------------------------------------

    def chunked(self, op: Operation, callee: IRFunction, threads: int) -> None:
        """Split at the operation: a serial path for small ranges, a spawned
        path otherwise, both joining at the continuation."""
        self.module.require("concurrency", 1)
        libraries = self.module.attributes.get("ppy.libraries", ())
        assert isinstance(libraries, tuple)
        if "pthread" not in libraries:
            self.module.attributes["ppy.libraries"] = (*libraries, "pthread")
        block = op.parent
        assert block is not None
        continuation = self._split_after(op, block)
        kind = op.local_name
        trailing = {"for": 2, "reduce": 3, "map": 3}[kind]
        captures = tuple(op.operands[: len(op.operands) - trailing])
        begin, end = op.operands[-trailing], op.operands[-trailing + 1]
        extra = op.operands[-1] if kind != "for" else None
        result_type = op.results[0].type if kind == "reduce" else None
        # `continuation` takes the result as a block argument for a reduce.
        joined_value = (
            continuation.add_argument(result_type, "reduced") if kind == "reduce" else None
        )

        b = Builder().before(op)
        n = core.sub(b, end, begin, overflow="wrap")
        small = core.cmp(b, "lt", n, core.const(b, self.pass_.minimum, I64))
        serial_block = self.function.body.add_block(self.fresh("par.serial"))
        spawn_block = self.function.body.add_block(self.fresh("par.spawn"))
        core.cond_br(b, small, Successor(serial_block), Successor(spawn_block))

        # The serial path: one chunk on this thread.
        s = Builder(serial_block)
        if kind == "for":
            core.call(s, callee.name, (*captures, begin, end), ())
            core.br(s, Successor(continuation))
        elif kind == "reduce":
            whole = core.call(
                s, callee.name, (*captures, begin, end, extra), callee.results
            ).results[0]
            core.br(s, Successor(continuation, (whole,)))
        else:
            wrapper = self.map_wrapper(callee, extra.type)  # type: ignore[union-attr]
            core.call(s, wrapper.name, (*captures, begin, end, extra), ())
            core.br(s, Successor(continuation))

        # The spawned path: threads - 1 chunks on new threads, the last here.
        p = Builder(spawn_block)
        count = core.const(p, threads, I64)
        size = core.div(
            p,
            core.add(
                p, n, core.sub(p, count, core.const(p, 1, I64), overflow="wrap"), overflow="wrap"
            ),
            count,
            overflow="wrap",
            rounding="trunc",
        )
        handles: list[Value] = []
        partials: list[Value] = []
        task = None
        if kind == "reduce":
            task = self.reduce_task(callee, result_type)
        elif kind == "map":
            task = self.map_wrapper(callee, extra.type)  # type: ignore[union-attr]
        for j in range(threads - 1):
            chunk_begin = core.add(
                p, begin, core.mul(p, core.const(p, j, I64), size, overflow="wrap"), overflow="wrap"
            )
            chunk_end = self._min(p, core.add(p, chunk_begin, size, overflow="wrap"), end)
            if kind == "for":
                handles.append(
                    concurrency.spawn(p, callee.name, (*captures, chunk_begin, chunk_end))
                )
            elif kind == "reduce":
                slot = core.alloca(p, result_type, name="partial")
                start = self._identity(p, str(op.attributes["op"]), result_type, extra)
                partials.append(slot)
                handles.append(
                    concurrency.spawn(
                        p, task.name, (*captures, chunk_begin, chunk_end, start, slot)
                    )
                )  # type: ignore[union-attr]
            else:
                handles.append(
                    concurrency.spawn(p, task.name, (*captures, chunk_begin, chunk_end, extra))
                )  # type: ignore[union-attr]
        last_begin = core.add(
            p,
            begin,
            core.mul(p, core.const(p, threads - 1, I64), size, overflow="wrap"),
            overflow="wrap",
        )
        last_begin = self._min(p, last_begin, end)
        # The chunk run here reports its status like the joined ones do: a
        # guard that fired in it must not take the fallback while the others
        # still run on this frame.
        accumulator: Value | None = None
        if kind == "for":
            inline = core.call(
                p, callee.name, (*captures, last_begin, end), (), capture_status=True
            )
            own_status = inline.results[0]
        elif kind == "reduce":
            start = self._identity(p, str(op.attributes["op"]), result_type, extra)
            inline = core.call(
                p,
                callee.name,
                (*captures, last_begin, end, start),
                callee.results,
                capture_status=True,
            )
            accumulator, own_status = inline.results[0], inline.results[1]
        else:
            inline = core.call(
                p, task.name, (*captures, last_begin, end, extra), (), capture_status=True
            )  # type: ignore[union-attr]
            own_status = inline.results[0]
        # Every chunk is joined before any is judged: a guard that fired in
        # one must not send this function back to Python while the others
        # still run on its stack.
        failed: Value = core.cmp(p, "ne", own_status, core.const(p, 0, I64))
        for handle in handles:
            status = concurrency.join(p, handle)
            bad = core.cmp(p, "ne", status, core.const(p, 0, I64))
            failed = core.bitwise(p, "or", failed, bad)
        ok = core.bitwise(p, "xor", failed, core.const(p, True, BOOL))
        core.guard(p, ok, "contract", "a parallel chunk failed")
        if kind == "reduce":
            kind_name = str(op.attributes["op"])
            assert accumulator is not None
            for slot in partials:
                accumulator = self._combine(p, kind_name, accumulator, core.load(p, slot))
            if kind_name in {"add", "mul"}:
                accumulator = self._combine(p, kind_name, extra, accumulator)  # type: ignore[arg-type]
            core.br(p, Successor(continuation, (accumulator,)))
            op.results[0].replace_all_uses_with(joined_value)  # type: ignore[arg-type]
        else:
            core.br(p, Successor(continuation))
        op.erase()

    def reduce_task(self, callee: IRFunction, result_type) -> IRFunction:  # type: ignore[no-untyped-def]
        """`@callee__task(captures..., begin, end, start, out)`: a chunk's fold, stored."""
        name = f"{callee.name}__task"
        existing = self.module.functions.get(name)
        if existing is not None:
            return existing
        captures = list(callee.params[:-3])
        params = [
            *captures,
            ("begin", I64),
            ("end", I64),
            ("start", result_type),
            # The caller hands a slot of its own stack, so the pointer keeps
            # that address space.
            ("out", PtrType(result_type, "stack")),
        ]
        task = self.module.add_function(
            name,
            params,
            [],
            visibility="private",
            attributes={"ppy.synthesized": "parallel.reduce"},
        )
        entry = task.add_entry_block()
        arguments = list(entry.arguments)
        caps, begin, end, start, out = (
            arguments[:-4],
            arguments[-4],
            arguments[-3],
            arguments[-2],
            arguments[-1],
        )
        b = Builder(entry)
        value = core.call(b, callee.name, (*caps, begin, end, start), callee.results).results[0]
        core.store(b, value, out)
        core.ret(b)
        return task

    # -- helpers ----------------------------------------------------------------------

    def _split_after(self, op: Operation, block: Block) -> Block:
        """Move everything after `op` into a new block; return it."""
        continuation = self.function.body.add_block(self.fresh("par.join"))
        index = block.operations.index(op)
        trailing = list(block.operations[index + 1 :])
        for later in trailing:
            block.remove(later)
            continuation.append(later)
        return continuation

    @staticmethod
    def _min(b: Builder, a: Value, c: Value) -> Value:
        return core.select(b, core.cmp(b, "lt", a, c), a, c)

    @staticmethod
    def _identity(b: Builder, op: str, t, init: Value) -> Value:  # type: ignore[no-untyped-def]
        start = parallel.identity(op, t, init)
        if isinstance(start, Value):
            return start
        return core.const(b, start, t)

    @staticmethod
    def _combine(b: Builder, op: str, a: Value, c: Value) -> Value:
        t = a.type
        integer = isinstance(t, IntType) or t == I64
        if op == "add":
            return core.add(b, a, c, overflow="wrap") if integer else core.add(b, a, c)
        if op == "mul":
            return core.mul(b, a, c, overflow="wrap") if integer else core.mul(b, a, c)
        better = core.cmp(b, "lt" if op == "min" else "gt", c, a)
        return core.select(b, better, c, a)
