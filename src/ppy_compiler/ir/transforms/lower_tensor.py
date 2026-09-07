"""`lower-tensor`: tensors as memory, tensor operations as loops.

A tensor value becomes a view -- a buffer of its elements, a stride per
axis, an offset -- and each operation becomes core loops over that memory:
`load`, `fill`, and `broadcast`, `reshape` of a contiguous tensor,
`transpose`, and `slice` are views onto memory that already exists;
`store`, the arithmetic, `fused`, `concat`, `reduce`, `matmul`, and
`convert` write freshly allocated memory -- or, when a `tensor.store` is
the result's only reader and no operand is broadcast, straight into the
store's buffer, with no temporary and no copy. Small static tensors live
on the stack, large and symbolic ones on the heap and are freed where the
function returns.

A symbolic dimension is bound where a tensor naming it is loaded: `N` in a
load of `tensor<f64, N, 3>` is the buffer's length over 3, guarded to
divide exactly, and every later extent, stride, and allocation over `N`
is arithmetic on that value. A shape that names an unbound symbol, or
two unknown dimensions in one load, is refused with the reason; so is a
tensor that crosses a call, which travels as a buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import shape as shapes
from ..dialects import core, sparse
from ..dialects import tensor as tensors
from ..model import Block, Builder, IRFunction, IRModule, Operation, Successor, Value
from ..passes import Pass, PassContext
from ..types import I32, I64, U8, BufferType, FloatType, IndexType, IntType, IRType, PtrType

__all__ = ["STACK_LIMIT", "LowerTensor", "LoweringError"]

#: Tensors up to this many bytes live on the stack; larger ones on the heap.
STACK_LIMIT = 1 << 16


class LoweringError(ValueError):
    """Tensor IR the lowering cannot make memory of, with the reason."""


@dataclass(slots=True)
class _View:
    """A tensor value as memory: the buffer, and how indices reach elements."""

    buffer: Value
    info: tensors.TensorInfo
    strides: tuple[shapes.Dim, ...]
    offset: shapes.Dim

    @property
    def shape(self) -> shapes.Shape:
        return self.info.shape

    def contiguous(self) -> bool:
        return self.offset == 0 and self.strides == shapes.strides_of(self.shape)


#: The dialects whose values are tensors and whose operations this pass lowers.
DIALECTS = frozenset({"tensor", "linalg", "fft", "sparse"})


@dataclass(slots=True)
class _Sparse:
    """A sparse matrix as its parts: values, the minor indices (or COO rows),
    and the compressed pointers (or COO columns)."""

    info: sparse.SparseInfo
    values: Value
    first: Value
    second: Value


class LowerTensor(Pass):
    """Rewrite every tensor, linalg, fft, and sparse operation as core loops.

    `lapack` says whether the factorizations may call LAPACK; without it
    they are refused with the reason.
    """

    name = "lower-tensor"
    requires = ("dominators",)
    invalidates = ("dominators",)

    def __init__(self, lapack: bool = True) -> None:
        self.lapack = lapack

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        changed = False
        for function in list(module.functions.values()):
            if function.is_declaration:
                continue
            if any(tensors.describe(t) is not None for _n, t in function.params) or any(
                tensors.describe(t) is not None for t in function.results
            ):
                raise LoweringError(
                    f"@{function.name}: a tensor crosses a call as a buffer; "
                    "`tensor.load` it inside"
                )
            if not any(op.dialect in DIALECTS for op in function.operations()):
                continue
            _FunctionLowering(module, function, ctx, self.lapack).run()
            changed = True
        if changed:
            ctx.invalidate()
        return changed


class _FunctionLowering:
    def __init__(
        self, module: IRModule, function: IRFunction, ctx: PassContext, lapack: bool = True
    ) -> None:
        self.module = module
        self.function = function
        self.ctx = ctx
        self.lapack = lapack
        self.views: dict[int, _View] = {}
        self.sparse_values: dict[int, _Sparse] = {}
        self.erase: list[Operation] = []
        self.heap: list[tuple[Value, Block]] = []
        self.counter = 0
        self._while_parts: tuple[Block, Block, Block] | None = None
        #: Each symbolic dimension's value, once a load has bound it.
        self.symbols: dict[str, Value] = {}
        #: Stores an operation wrote straight into: done, and erased unlowered.
        self.absorbed: set[int] = set()

    def fresh(self, hint: str) -> str:
        self.counter += 1
        return f"{hint}{self.counter}"

    def run(self) -> None:
        for block in list(self.function.body.blocks):
            for argument in block.arguments:
                if tensors.describe(argument.type) is not None:
                    raise LoweringError(
                        f"@{self.function.name}: a tensor cannot be a block argument"
                    )
        # Blocks split as loops are emitted, so walk a snapshot and follow.
        pending = [
            op
            for block in self.function.body.blocks
            for op in block.operations
            if op.dialect in DIALECTS
        ]
        for op in pending:
            self.lower(op)
        self.free_at_returns()
        for op in reversed(self.erase):
            op.erase()

    # -- views and memory ---------------------------------------------------------

    def view(self, value: Value) -> _View:
        found = self.views.get(id(value))
        if found is None:
            raise LoweringError(
                f"@{self.function.name}: tensor %{value.name or '?'} was never made"
            )
        return found

    def info(self, value: Value) -> tensors.TensorInfo:
        info = tensors.describe(value.type)
        if info is None:
            raise LoweringError(f"{value.type} is not a tensor")
        return info

    def extent(self, b: Builder, dim: shapes.Dim) -> Value:
        """A dimension as an i64: a number, a bound symbol, or arithmetic on them."""
        if isinstance(dim, int):
            return core.const(b, dim, I64)
        if isinstance(dim, shapes.Symbol):
            bound = self.symbols.get(dim.name)
            if bound is None:
                raise LoweringError(
                    f"@{self.function.name}: dimension {dim.name} is not bound; "
                    "load a tensor whose shape names it first"
                )
            return bound
        values = [self.extent(b, a) for a in dim.args]
        if dim.op == "add":
            total = values[0]
            for v in values[1:]:
                total = core.add(b, total, v, overflow="wrap")
            return total
        if dim.op == "mul":
            product = values[0]
            for v in values[1:]:
                product = core.mul(b, product, v, overflow="wrap")
            return product
        if dim.op == "ceil_div":
            top, bottom = values
            up = core.sub(
                b, core.add(b, top, bottom, overflow="wrap"), core.const(b, 1, I64), overflow="wrap"
            )
            return core.div(b, up, bottom, overflow="wrap", rounding="floor")
        left, right = values
        return core.select(b, core.cmp(b, "ge", left, right), left, right)

    def bind_symbols(self, b: Builder, buffer: Value, info: tensors.TensorInfo) -> None:
        """Bind the one symbol `info.shape` leaves unknown from `buffer`'s length.

        The symbol must stand as a dimension on its own, once: `N` in
        `tensor<f64, N, 3>` is the length over 3, and the length must
        divide exactly. Two unknown dimensions cannot be told apart by one
        length, and are refused.
        """
        unknown = sorted(_symbols_of(info.shape) - set(self.symbols))
        if not unknown:
            return
        spelled = shapes.spell_shape(info.shape)
        if len(unknown) > 1:
            raise LoweringError(
                f"@{self.function.name}: {spelled} has {len(unknown)} unknown dimensions "
                f"({', '.join(unknown)}); one buffer length binds one"
            )
        name = unknown[0]
        bare = [
            index
            for index, dim in enumerate(info.shape)
            if isinstance(dim, shapes.Symbol) and dim.name == name
        ]
        elsewhere = any(
            name in _symbols_of((dim,)) for index, dim in enumerate(info.shape) if index not in bare
        )
        if len(bare) != 1 or elsewhere:
            raise LoweringError(
                f"@{self.function.name}: {name} in {spelled} cannot be read from a buffer "
                "length; a dimension is bound where it stands on its own, once"
            )
        rest = shapes.numel(tuple(dim for index, dim in enumerate(info.shape) if index != bare[0]))
        length = self.length(b, buffer)
        if rest == 1:
            self.symbols[name] = length
            return
        divisor = self.extent(b, rest)
        nonzero = core.cmp(b, "ne", divisor, core.const(b, 0, I64))
        core.guard(b, nonzero, "bounds", f"a zero extent leaves {name} unknown")
        remainder = core.mod(b, length, divisor, overflow="wrap", rounding="floor")
        exact = core.cmp(b, "eq", remainder, core.const(b, 0, I64))
        core.guard(
            b, exact, "bounds", f"the buffer's length is not a multiple of {shapes.spell(rest)}"
        )
        self.symbols[name] = core.div(b, length, divisor, overflow="wrap", rounding="floor")

    def allocate(self, b: Builder, info: tensors.TensorInfo, hint: str) -> _View:
        """Fresh contiguous memory for a tensor of `info`'s shape."""
        if not info.static:
            count = self.extent(b, shapes.numel(info.shape))
            buffer = self.allocate_dynamic(b, info.dtype, count, hint)
            return _View(buffer, info, shapes.strides_of(info.shape), 0)
        count = max(shapes.static_numel(info.shape), 1)
        width = _width(info.dtype)
        if count * width <= STACK_LIMIT:
            pointer = core.alloca(b, info.dtype, count=count, name=self.fresh(hint))
        else:
            malloc = core.call_extern(
                b, "malloc", (core.const(b, count * width, I64),), (PtrType(U8),)
            ).results[0]
            pointer = core.cast(b, malloc, PtrType(info.dtype))
            block = b.block
            assert block is not None
            self.heap.append((malloc, block))
        buffer = core.call_intrinsic(
            b,
            "ppy.buffer_from_parts",
            (pointer, core.const(b, count, I64)),
            (BufferType(info.dtype),),
        ).results[0]
        return _View(buffer, info, shapes.strides_of(info.shape), 0)

    def free_at_returns(self) -> None:
        if not self.heap:
            return
        # The lowering rewrote the control flow; an earlier dominator tree is stale.
        self.ctx.invalidate(self.function)
        dominators = self.ctx.analysis("dominators", self.function)
        for block in self.function.body.blocks:
            terminator = block.terminator
            if terminator is None or terminator.name != "core.ret":
                continue
            b = Builder().before(terminator)
            for pointer, home in self.heap:
                if home is block or dominators.dominates(home, block):  # type: ignore[attr-defined]
                    core.call_extern(b, "free", (pointer,), ())
                else:
                    self.ctx.remark(
                        f"@{self.function.name}: heap tensor memory is not freed on every path"
                    )

    # -- loops -----------------------------------------------------------------------

    def loops(self, op: Operation, extents: shapes.Shape) -> tuple[Builder, list[Value], Block]:
        """Nested counted loops over `extents`, placed where `op` stands.

        `op` and everything after it move to a fresh continuation block the
        loops exit into, so a second call for the same `op` adds another set
        of loops after the first. Returns the builder positioned in the
        innermost body, the index values, and the continuation.
        """
        block = op.parent
        assert block is not None
        continuation = self.function.body.add_block(self.fresh("t.next"))
        index = block.operations.index(op)
        for later in list(block.operations[index:]):
            block.remove(later)
            continuation.append(later)
        b = Builder(block)
        if not extents:
            leave = core.br(b, Successor(continuation))
            return Builder().before(leave), [], continuation
        indices: list[Value] = []
        heads: list[Block] = []
        dones: list[Block] = []
        for depth, extent in enumerate(extents):
            head = self.function.body.add_block(self.fresh(f"t.head{depth}_"), [(f"i{depth}", I64)])
            body = self.function.body.add_block(self.fresh(f"t.body{depth}_"))
            done = self.function.body.add_block(self.fresh(f"t.done{depth}_"))
            core.br(b, Successor(head, (core.const(b, 0, I64),)))
            h = Builder(head)
            index = head.arguments[0]
            core.cond_br(
                h,
                core.cmp(h, "lt", index, self.extent(h, extent)),
                Successor(body),
                Successor(done),
            )
            indices.append(index)
            heads.append(head)
            dones.append(done)
            b = Builder(body)
        # The innermost body ends in a branch to its latch; the body's
        # operations go before that branch. Each `done` steps the enclosing
        # index and returns to its head; the outermost `done` leaves.
        latch = self.function.body.add_block(self.fresh("t.latch"))
        leave = core.br(b, Successor(latch))
        for depth in reversed(range(len(extents))):
            stepper = Builder(latch if depth == len(extents) - 1 else dones[depth + 1])
            step = core.add(stepper, indices[depth], core.const(stepper, 1, I64), overflow="wrap")
            core.br(stepper, Successor(heads[depth], (step,)))
        core.br(Builder(dones[0]), Successor(continuation))
        return Builder().before(leave), indices, continuation

    def address(
        self,
        b: Builder,
        view: _View,
        indices: list[Value],
        strides: tuple[shapes.Dim, ...] | None = None,
    ) -> Value:
        """The element of `view` at `indices`, as a pointer."""
        strides = view.strides if strides is None else strides
        offset = self.extent(b, view.offset)
        for index, stride in zip(indices, strides, strict=True):
            if stride == 0:
                continue
            term = core.mul(b, index, self.extent(b, stride), overflow="wrap")
            offset = core.add(b, offset, term, overflow="wrap")
        return core.ptr_offset(b, core.buffer_data(b, view.buffer), offset)

    def load(
        self,
        b: Builder,
        view: _View,
        indices: list[Value],
        strides: tuple[shapes.Dim, ...] | None = None,
    ) -> Value:
        return core.load(b, self.address(b, view, indices, strides))

    def store(self, b: Builder, view: _View, indices: list[Value], value: Value) -> None:
        core.store(b, value, self.address(b, view, indices))

    # -- operations ----------------------------------------------------------------------

    def lower(self, op: Operation) -> None:
        name = op.local_name if op.dialect == "tensor" else f"{op.dialect}_{op.local_name}"
        handler = getattr(self, f"op_{name}", None)
        if handler is None:
            raise LoweringError(f"{op.name} has no lowering")
        if id(op) in self.absorbed:
            self.erase.append(op)
            return
        if op.dialect != "tensor":
            # The algorithms index by number: a factorization's n, a transform's
            # length. They want the shapes they work on known.
            for value in (*op.operands, *op.results):
                described = tensors.describe(value.type)
                if described is not None and not described.static:
                    raise LoweringError(
                        f"@{self.function.name}: {op.name} works on static shapes; "
                        f"{value.type} is symbolic"
                    )
        handler(op)
        self.erase.append(op)

    # -- loops with state -----------------------------------------------------------

    def slot(self, b: Builder, t: IRType, initial: Value, hint: str) -> Value:
        """A stack slot holding `initial`: state a loop carries across iterations."""
        pointer = core.alloca(b, t, name=self.fresh(hint))
        core.store(b, initial, pointer)
        return pointer

    def while_(self, op: Operation) -> tuple[Builder, Builder, Block]:
        """A `while` where `op` stands: (condition builder, body builder, continuation).

        The condition builder's block must end in `core.cond_br %c, ^body,
        ^continuation` made by `self.close_while`; the body flows back to the head.
        """
        block = op.parent
        assert block is not None
        continuation = self.function.body.add_block(self.fresh("t.next"))
        index = block.operations.index(op)
        for later in list(block.operations[index:]):
            block.remove(later)
            continuation.append(later)
        head = self.function.body.add_block(self.fresh("w.head"))
        body = self.function.body.add_block(self.fresh("w.body"))
        core.br(Builder(block), Successor(head))
        back = core.br(Builder(body), Successor(head))
        self._while_parts = (head, body, continuation)
        return Builder(head), Builder().before(back), continuation

    def close_while(self, condition_builder: Builder, condition: Value) -> None:
        assert self._while_parts is not None
        _head, body, continuation = self._while_parts
        core.cond_br(condition_builder, condition, Successor(body), Successor(continuation))

    def require_library(self, name: str) -> None:
        libraries = self.module.attributes.get("ppy.libraries", ())
        assert isinstance(libraries, tuple)
        if name not in libraries:
            self.module.attributes["ppy.libraries"] = (*libraries, name)

    def op_empty(self, op: Operation) -> None:
        b = Builder().before(op)
        self.views[id(op.result)] = self.allocate(b, self.info(op.result), "empty")

    def op_load(self, op: Operation) -> None:
        info = self.info(op.result)
        b = Builder().before(op)
        buffer = op.operands[0]
        self.bind_symbols(b, buffer, info)
        count = self.extent(b, shapes.numel(info.shape))
        enough = core.cmp(b, "ge", self.length(b, buffer), count)
        core.guard(b, enough, "bounds", "the buffer holds fewer elements than the tensor")
        self.views[id(op.result)] = _View(buffer, info, shapes.strides_of(info.shape), 0)

    def op_store(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.operands[0])
        buffer = op.operands[1]
        b = Builder().before(op)
        count = self.extent(b, shapes.numel(info.shape))
        enough = core.cmp(b, "ge", self.length(b, buffer), count)
        core.guard(b, enough, "bounds", "the buffer holds fewer elements than the tensor")
        target = _View(buffer, info, shapes.strides_of(info.shape), 0)
        inner, indices, _next = self.loops(op, source.shape)
        self.store(inner, target, indices, self.load(inner, source, indices))

    def output(self, op: Operation, info: tensors.TensorInfo, hint: str) -> _View:
        """Where `op` writes its result: the buffer of a `tensor.store` that is
        the result's only reader, or fresh memory.

        Writing straight into the destination saves the temporary and the
        copy. It is safe when every element is read and written at one
        index in one iteration, so an input the destination aliases is
        consumed before it is overwritten: the operands must have the
        result's own shape, since a broadcast operand is read again.
        """
        uses = op.results[0].uses
        if len(uses) == 1:
            user, position = uses[0]
            if (
                isinstance(user, Operation)
                and user.name == "tensor.store"
                and position == 0
                and user.parent is op.parent
                and _defined_before(user.operands[1], op)
                and all(
                    tensors.describe(v.type) is None or tensors.describe(v.type).shape == info.shape  # type: ignore[union-attr]
                    for v in op.operands
                )
            ):
                buffer = user.operands[1]
                b = Builder().before(op)
                count = self.extent(b, shapes.numel(info.shape))
                enough = core.cmp(b, "ge", self.length(b, buffer), count)
                core.guard(b, enough, "bounds", "the buffer holds fewer elements than the tensor")
                self.absorbed.add(id(user))
                return _View(buffer, info, shapes.strides_of(info.shape), 0)
        return self.allocate(Builder().before(op), info, hint)

    def op_fused(self, op: Operation) -> None:
        """One loop over the broadcast shape; the body inlined per element."""
        info = self.info(op.result)
        body = op.regions[0].blocks[0]
        views: dict[int, _View] = {}
        iteration: shapes.Shape = ()
        for operand in op.operands:
            if tensors.describe(operand.type) is not None:
                view = self.view(operand)
                views[id(operand)] = view
                iteration = shapes.broadcast(iteration, view.shape)
        kind = op.attributes.get("reduce")
        result = self.output(op, info, "fused") if kind is None else None
        if result is None:
            result = self.allocate(Builder().before(op), info, "fused")
        if kind is not None:
            inner, indices, _next = self.loops(op, result.shape)
            self.store(inner, result, indices, _identity(inner, str(kind), info.dtype))
        inner, indices, _next = self.loops(op, iteration)
        mapping: dict[int, Value] = {}
        for argument, operand in zip(body.arguments, op.operands, strict=True):
            view = views.get(id(operand))
            if view is None:
                mapping[id(argument)] = operand
            else:
                strides = _broadcast_strides(view, iteration)
                mapping[id(argument)] = self.load(inner, view, indices, strides)
        produced: Value | None = None
        for inner_op in body.operations:
            if inner_op.name == "tensor.yield":
                produced = mapping.get(id(inner_op.operands[0]), inner_op.operands[0])
                break
            clone = inner.create(
                inner_op.name,
                tuple(mapping.get(id(v), v) for v in inner_op.operands),
                tuple(r.type for r in inner_op.results),
                dict(inner_op.attributes),
            )
            for original, copy in zip(inner_op.results, clone.results, strict=True):
                mapping[id(original)] = copy
        assert produced is not None
        if kind is None:
            self.store(inner, result, indices, produced)
        else:
            axes = tuple(int(a) for a in op.attributes["axes"])  # type: ignore[union-attr]
            keepdims = bool(op.attributes.get("keepdims", False))
            kept = [index for axis, index in enumerate(indices) if axis not in axes]
            if keepdims:
                kept = [
                    core.const(inner, 0, I64) if axis in axes else index
                    for axis, index in enumerate(indices)
                ]
            current = self.load(inner, result, kept)
            self.store(
                inner, result, kept, _combine(inner, str(kind), info.dtype, current, produced)
            )
        self.views[id(op.result)] = result

    def op_fill(self, op: Operation) -> None:
        """One element in memory, read through zero strides."""
        info = self.info(op.result)
        b = Builder().before(op)
        pointer = core.alloca(b, info.dtype, name=self.fresh("fill"))
        core.store(b, op.operands[0], pointer)
        buffer = core.call_intrinsic(
            b,
            "ppy.buffer_from_parts",
            (pointer, core.const(b, 1, I64)),
            (BufferType(info.dtype),),
        ).results[0]
        self.views[id(op.result)] = _View(buffer, info, (0,) * info.rank, 0)

    def op_unary(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        kind = str(op.attributes["op"])
        result = self.output(op, info, kind)
        inner, indices, _next = self.loops(op, result.shape)
        value = self.load(inner, source, indices)
        self.store(inner, result, indices, tensors.scalar_unary(inner, kind, value, self.module))
        self.views[id(op.result)] = result

    def _elementwise(self, op: Operation, name: str) -> None:
        left, right = self.view(op.operands[0]), self.view(op.operands[1])
        result_info = self.info(op.result)
        result = self.output(op, result_info, name)
        inner, indices, _next = self.loops(op, result.shape)
        a = self.load(inner, left, indices, _broadcast_strides(left, result.shape))
        c = self.load(inner, right, indices, _broadcast_strides(right, result.shape))
        value = tensors.scalar_binary(inner, name, a, c, self.module)
        self.store(inner, result, indices, value)
        self.views[id(op.result)] = result

    def op_add(self, op: Operation) -> None:
        self._elementwise(op, "add")

    def op_sub(self, op: Operation) -> None:
        self._elementwise(op, "sub")

    def op_mul(self, op: Operation) -> None:
        self._elementwise(op, "mul")

    def op_div(self, op: Operation) -> None:
        self._elementwise(op, "div")

    def op_pow(self, op: Operation) -> None:
        self._elementwise(op, "pow")

    def op_min(self, op: Operation) -> None:
        self._elementwise(op, "min")

    def op_max(self, op: Operation) -> None:
        self._elementwise(op, "max")

    def op_broadcast(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        self.views[id(op.result)] = _View(
            source.buffer, info, _broadcast_strides(source, info.shape), source.offset
        )

    def op_reshape(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        if not source.contiguous():
            source = self.materialize(op, source, "reshape")
        self.views[id(op.result)] = _View(
            source.buffer, info, shapes.strides_of(info.shape), source.offset
        )

    def materialize(self, op: Operation, view: _View, hint: str) -> _View:
        """A contiguous copy of `view`, made before `op`."""
        b = Builder().before(op)
        copy = self.allocate(b, view.info, hint)
        inner, indices, _next = self.loops(op, view.shape)
        self.store(inner, copy, indices, self.load(inner, view, indices))
        return copy

    def op_transpose(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        perm = tuple(int(p) for p in op.attributes["perm"])  # type: ignore[union-attr]
        self.views[id(op.result)] = _View(
            source.buffer, info, tuple(source.strides[p] for p in perm), source.offset
        )

    def op_slice(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        starts = tuple(int(v) for v in op.attributes["starts"])  # type: ignore[union-attr]
        steps = tuple(int(v) for v in op.attributes["steps"])  # type: ignore[union-attr]
        offset: shapes.Dim = source.offset
        for start, stride in zip(starts, source.strides, strict=True):
            offset = shapes.add(offset, shapes.mul(start, stride))
        strides = tuple(
            shapes.mul(stride, step) for stride, step in zip(source.strides, steps, strict=True)
        )
        self.views[id(op.result)] = _View(source.buffer, info, strides, offset)

    def op_concat(self, op: Operation) -> None:
        info = self.info(op.result)
        axis = int(op.attributes["axis"])  # type: ignore[call-overload]
        b = Builder().before(op)
        result = self.allocate(b, info, "concat")
        position: shapes.Dim = 0
        for operand in op.operands:
            part = self.view(operand)
            inner, indices, _next = self.loops(op, part.shape)
            shifted = list(indices)
            shifted[axis] = core.add(
                inner, indices[axis], self.extent(inner, position), overflow="wrap"
            )
            self.store(inner, result, shifted, self.load(inner, part, indices))
            position = shapes.add(position, part.shape[axis])
        self.views[id(op.result)] = result

    def op_reduce(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        axes = tuple(int(a) for a in op.attributes["axes"])  # type: ignore[union-attr]
        kind = str(op.attributes["op"])
        keepdims = bool(op.attributes.get("keepdims", False))
        b = Builder().before(op)
        result = self.allocate(b, info, "reduce")
        # The identity first, then every element folded in row-major order.
        inner, indices, _next = self.loops(op, result.shape)
        self.store(inner, result, indices, _identity(inner, kind, info.dtype))
        inner, indices, _next = self.loops(op, source.shape)
        kept = [index for axis, index in enumerate(indices) if axis not in axes]
        if keepdims:
            kept = [
                core.const(inner, 0, I64) if axis in axes else index
                for axis, index in enumerate(indices)
            ]
        current = self.load(inner, result, kept)
        value = self.load(inner, source, indices)
        self.store(inner, result, kept, _combine(inner, kind, info.dtype, current, value))
        self.views[id(op.result)] = result

    def op_matmul(self, op: Operation) -> None:
        left, right = self.view(op.operands[0]), self.view(op.operands[1])
        info = self.info(op.result)
        b = Builder().before(op)
        result = self.allocate(b, info, "matmul")
        rows, inner_dim = left.shape
        _k, cols = right.shape
        outer, ij, _next = self.loops(op, (rows, cols))
        self.store(outer, result, ij, _identity(outer, "add", info.dtype))
        outer, ijk, _next = self.loops(op, (rows, cols, inner_dim))
        i, j, k = ijk
        a = self.load(outer, left, [i, k])
        c = self.load(outer, right, [k, j])
        current = self.load(outer, result, [i, j])
        product = (
            core.mul(outer, a, c)
            if isinstance(info.dtype, FloatType)
            else core.mul(outer, a, c, overflow="wrap")
        )
        self.store(outer, result, [i, j], _combine(outer, "add", info.dtype, current, product))
        self.views[id(op.result)] = result

    def op_convert(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        b = Builder().before(op)
        result = self.allocate(b, info, "convert")
        inner, indices, _next = self.loops(op, source.shape)
        value = self.load(inner, source, indices)
        if value.type != info.dtype:
            value = core.cast(inner, value, info.dtype)
        self.store(inner, result, indices, value)
        self.views[id(op.result)] = result

    # -- linalg ----------------------------------------------------------------------------

    def op_linalg_dot(self, op: Operation) -> None:
        left, right = self.view(op.operands[0]), self.view(op.operands[1])
        info = self.info(op.result)
        b = Builder().before(op)
        result = self.allocate(b, info, "dot")
        inner, _none, _next = self.loops(op, ())
        self.store(inner, result, [], _identity(inner, "add", info.dtype))
        inner, indices, _next = self.loops(op, left.shape)
        product = core.mul(inner, self.load(inner, left, indices), self.load(inner, right, indices))
        self.store(inner, result, [], core.add(inner, self.load(inner, result, []), product))
        self.views[id(op.result)] = result

    def op_linalg_matmul(self, op: Operation) -> None:
        self.op_matmul(op)

    def op_linalg_cholesky(self, op: Operation) -> None:
        """Cholesky-Banachiewicz, row by row; a pivot that is not positive fails the guard."""
        source = self.view(op.operands[0])
        info = self.info(op.result)
        n = source.shape[0]
        b = Builder().before(op)
        result = self.allocate(b, info, "cholesky")
        inner, ij, _next = self.loops(op, (n, n))
        self.store(inner, result, ij, core.const(inner, 0.0, info.dtype))
        # L[i][j] = (A[i][j] - sum_k L[i][k] L[j][k]) / L[j][j], for j <= i.
        inner, ij, _next = self.loops(op, (n, n))
        i, j = ij
        lower = core.cmp(inner, "le", j, i)
        then_block = self.function.body.add_block(self.fresh("chol.then"))
        after = self.function.body.add_block(self.fresh("chol.after"))
        core.cond_br(inner, lower, Successor(then_block), Successor(after))
        t = Builder(then_block)
        total = self.slot(t, info.dtype, self.load(t, source, [i, j]), "acc")
        # k runs over [0, j): a counted loop with a runtime bound.
        k_head = self.function.body.add_block(self.fresh("chol.k"), [("k", I64)])
        k_body = self.function.body.add_block(self.fresh("chol.kbody"))
        k_done = self.function.body.add_block(self.fresh("chol.kdone"))
        core.br(t, Successor(k_head, (core.const(t, 0, I64),)))
        h = Builder(k_head)
        k = k_head.arguments[0]
        core.cond_br(h, core.cmp(h, "lt", k, j), Successor(k_body), Successor(k_done))
        kb = Builder(k_body)
        product = core.mul(kb, self.load(kb, result, [i, k]), self.load(kb, result, [j, k]))
        core.store(kb, core.sub(kb, core.load(kb, total), product), total)
        core.br(kb, Successor(k_head, (core.add(kb, k, core.const(kb, 1, I64), overflow="wrap"),)))
        d = Builder(k_done)
        remaining = core.load(d, total)
        diagonal = core.cmp(d, "eq", i, j)
        diag_block = self.function.body.add_block(self.fresh("chol.diag"))
        off_block = self.function.body.add_block(self.fresh("chol.off"))
        core.cond_br(d, diagonal, Successor(diag_block), Successor(off_block))
        dg = Builder(diag_block)
        positive = core.cmp(dg, "gt", remaining, core.const(dg, 0.0, info.dtype))
        core.guard(dg, positive, "contract", "the matrix is not positive definite")
        self.module.require("math", 1)
        root = dg.create("math.sqrt", (remaining,), (info.dtype,)).result
        self.store(dg, result, [i, j], root)
        core.br(dg, Successor(after))
        og = Builder(off_block)
        pivot = self.load(og, result, [j, j])
        self.store(og, result, [i, j], core.div(og, remaining, pivot))
        core.br(og, Successor(after))
        # `after` continues the (i, j) loop: the innermost body's own branch
        # follows, so hand the loop the rest of its body there.
        self._continue_in(inner, after)
        self.views[id(op.result)] = result

    def _continue_in(self, inner: Builder, block: Block) -> None:
        """Move the innermost body's trailing branch into `block`."""
        anchor = inner.anchor
        assert anchor is not None and anchor.parent is not None
        owner = anchor.parent
        owner.remove(anchor)
        block.append(anchor)

    def op_linalg_triangular_solve(self, op: Operation) -> None:
        matrix, rhs = self.view(op.operands[0]), self.view(op.operands[1])
        info = self.info(op.result)
        lower = bool(op.attributes.get("lower", True))
        unit = bool(op.attributes.get("unit", False))
        n = matrix.shape[0]
        columns = rhs.shape[1] if len(rhs.shape) == 2 else 1
        b = Builder().before(op)
        result = self.allocate(b, info, "trsolve")
        rhs_columns = (
            rhs
            if len(rhs.shape) == 2
            else _View(
                rhs.buffer,
                tensors.TensorInfo(rhs.info.dtype, (n, 1), rhs.info.layout),
                (rhs.strides[0], 0),
                rhs.offset,
            )
        )
        out_columns = _View(
            result.buffer,
            tensors.TensorInfo(info.dtype, (n, columns), info.layout),
            (result.strides[0], result.strides[1] if len(result.shape) == 2 else 0),
            0,
        )
        # Forward (lower) or backward (upper) substitution, one right-hand side at a time.
        inner, ci, _next = self.loops(op, (columns, n))
        c, step = ci
        i = step if lower else core.sub(inner, core.const(inner, n - 1, I64), step, overflow="wrap")
        total = self.slot(inner, info.dtype, self.load(inner, rhs_columns, [i, c]), "acc")
        k_head = self.function.body.add_block(self.fresh("trs.k"), [("k", I64)])
        k_body = self.function.body.add_block(self.fresh("trs.kbody"))
        k_done = self.function.body.add_block(self.fresh("trs.kdone"))
        start = (
            core.const(inner, 0, I64)
            if lower
            else core.add(inner, i, core.const(inner, 1, I64), overflow="wrap")
        )
        core.br(inner, Successor(k_head, (start,)))
        h = Builder(k_head)
        k = k_head.arguments[0]
        bound = i if lower else core.const(h, n, I64)
        core.cond_br(h, core.cmp(h, "lt", k, bound), Successor(k_body), Successor(k_done))
        kb = Builder(k_body)
        product = core.mul(kb, self.load(kb, matrix, [i, k]), self.load(kb, out_columns, [k, c]))
        core.store(kb, core.sub(kb, core.load(kb, total), product), total)
        core.br(kb, Successor(k_head, (core.add(kb, k, core.const(kb, 1, I64), overflow="wrap"),)))
        d = Builder(k_done)
        value = core.load(d, total)
        if not unit:
            pivot = self.load(d, matrix, [i, i])
            nonzero = core.cmp(d, "ne", pivot, core.const(d, 0.0, info.dtype))
            core.guard(d, nonzero, "zero_division", "the triangular matrix is singular")
            value = core.div(d, value, pivot)
        self.store(d, out_columns, [i, c], value)
        self._continue_in(inner, k_done)
        self.views[id(op.result)] = result

    def op_linalg_solve(self, op: Operation) -> None:
        """LU with partial pivoting on a copy, then two triangular solves."""
        matrix, rhs = self.view(op.operands[0]), self.view(op.operands[1])
        info = self.info(op.result)
        n = matrix.shape[0]
        columns = rhs.shape[1] if len(rhs.shape) == 2 else 1
        work = self.materialize(op, matrix, "lu")
        x = self.allocate(Builder().before(op), info, "solve")
        x_columns = _View(
            x.buffer,
            tensors.TensorInfo(info.dtype, (n, columns), info.layout),
            (x.strides[0], x.strides[1] if len(x.shape) == 2 else 0),
            0,
        )
        rhs_columns = (
            rhs
            if len(rhs.shape) == 2
            else _View(
                rhs.buffer,
                tensors.TensorInfo(rhs.info.dtype, (n, 1), rhs.info.layout),
                (rhs.strides[0], 0),
                rhs.offset,
            )
        )
        inner, ic, _next = self.loops(op, (n, columns))
        self.store(inner, x_columns, ic, self.load(inner, rhs_columns, ic))
        # Elimination column by column: pick the largest pivot, swap rows in
        # both the work matrix and the right-hand sides, then eliminate below.
        inner, (k,), _next = self.loops(op, (n,))
        best = self.slot(inner, I64, k, "pivot")
        magnitude = self.slot(
            inner, info.dtype, _absolute(inner, self.load(inner, work, [k, k])), "best"
        )
        r_head = self.function.body.add_block(self.fresh("lu.r"), [("r", I64)])
        r_body = self.function.body.add_block(self.fresh("lu.rbody"))
        r_done = self.function.body.add_block(self.fresh("lu.rdone"))
        core.br(
            inner,
            Successor(r_head, (core.add(inner, k, core.const(inner, 1, I64), overflow="wrap"),)),
        )
        h = Builder(r_head)
        r = r_head.arguments[0]
        core.cond_br(
            h, core.cmp(h, "lt", r, core.const(h, n, I64)), Successor(r_body), Successor(r_done)
        )
        rb = Builder(r_body)
        candidate = _absolute(rb, self.load(rb, work, [r, k]))
        better = core.cmp(rb, "gt", candidate, core.load(rb, magnitude))
        core.store(rb, core.select(rb, better, candidate, core.load(rb, magnitude)), magnitude)
        core.store(rb, core.select(rb, better, r, core.load(rb, best)), best)
        core.br(rb, Successor(r_head, (core.add(rb, r, core.const(rb, 1, I64), overflow="wrap"),)))
        d = Builder(r_done)
        pivot_row = core.load(d, best)
        singular = core.cmp(d, "ne", core.load(d, magnitude), core.const(d, 0.0, info.dtype))
        core.guard(d, singular, "zero_division", "the matrix is singular")
        # Swap rows k and pivot_row of the work matrix and of x.
        for target, width in ((work, n), (x_columns, columns)):
            s_head = self.function.body.add_block(self.fresh("lu.s"), [("j", I64)])
            s_body = self.function.body.add_block(self.fresh("lu.sbody"))
            s_done = self.function.body.add_block(self.fresh("lu.sdone"))
            core.br(d, Successor(s_head, (core.const(d, 0, I64),)))
            sh = Builder(s_head)
            j = s_head.arguments[0]
            core.cond_br(
                sh,
                core.cmp(sh, "lt", j, core.const(sh, width, I64)),
                Successor(s_body),
                Successor(s_done),
            )
            sb = Builder(s_body)
            first = self.load(sb, target, [k, j])
            second = self.load(sb, target, [pivot_row, j])
            self.store(sb, target, [k, j], second)
            self.store(sb, target, [pivot_row, j], first)
            core.br(
                sb, Successor(s_head, (core.add(sb, j, core.const(sb, 1, I64), overflow="wrap"),))
            )
            d = Builder(s_done)
        # Eliminate below the pivot.
        e_head = self.function.body.add_block(self.fresh("lu.e"), [("r", I64)])
        e_body = self.function.body.add_block(self.fresh("lu.ebody"))
        e_done = self.function.body.add_block(self.fresh("lu.edone"))
        core.br(d, Successor(e_head, (core.add(d, k, core.const(d, 1, I64), overflow="wrap"),)))
        eh = Builder(e_head)
        r = e_head.arguments[0]
        core.cond_br(
            eh, core.cmp(eh, "lt", r, core.const(eh, n, I64)), Successor(e_body), Successor(e_done)
        )
        eb = Builder(e_body)
        factor = core.div(eb, self.load(eb, work, [r, k]), self.load(eb, work, [k, k]))
        for target, width in ((work, n), (x_columns, columns)):
            f_head = self.function.body.add_block(self.fresh("lu.f"), [("j", I64)])
            f_body = self.function.body.add_block(self.fresh("lu.fbody"))
            f_done = self.function.body.add_block(self.fresh("lu.fdone"))
            core.br(eb, Successor(f_head, (core.const(eb, 0, I64),)))
            fh = Builder(f_head)
            j = f_head.arguments[0]
            core.cond_br(
                fh,
                core.cmp(fh, "lt", j, core.const(fh, width, I64)),
                Successor(f_body),
                Successor(f_done),
            )
            fb = Builder(f_body)
            updated = core.sub(
                fb,
                self.load(fb, target, [r, j]),
                core.mul(fb, factor, self.load(fb, target, [k, j])),
            )
            self.store(fb, target, [r, j], updated)
            core.br(
                fb, Successor(f_head, (core.add(fb, j, core.const(fb, 1, I64), overflow="wrap"),))
            )
            eb = Builder(f_done)
        core.br(eb, Successor(e_head, (core.add(eb, r, core.const(eb, 1, I64), overflow="wrap"),)))
        self._continue_in(inner, e_done)
        # Back substitution through the upper triangle now in `work`.
        inner, cs, _next = self.loops(op, (columns, n))
        c, step = cs
        i = core.sub(inner, core.const(inner, n - 1, I64), step, overflow="wrap")
        total = self.slot(inner, info.dtype, self.load(inner, x_columns, [i, c]), "acc")
        b_head = self.function.body.add_block(self.fresh("lu.b"), [("j", I64)])
        b_body = self.function.body.add_block(self.fresh("lu.bbody"))
        b_done = self.function.body.add_block(self.fresh("lu.bdone"))
        core.br(
            inner,
            Successor(b_head, (core.add(inner, i, core.const(inner, 1, I64), overflow="wrap"),)),
        )
        bh = Builder(b_head)
        j = b_head.arguments[0]
        core.cond_br(
            bh, core.cmp(bh, "lt", j, core.const(bh, n, I64)), Successor(b_body), Successor(b_done)
        )
        bb = Builder(b_body)
        product = core.mul(bb, self.load(bb, work, [i, j]), self.load(bb, x_columns, [j, c]))
        core.store(bb, core.sub(bb, core.load(bb, total), product), total)
        core.br(bb, Successor(b_head, (core.add(bb, j, core.const(bb, 1, I64), overflow="wrap"),)))
        bd = Builder(b_done)
        self.store(
            bd, x_columns, [i, c], core.div(bd, core.load(bd, total), self.load(bd, work, [i, i]))
        )
        self._continue_in(inner, b_done)
        self.views[id(op.result)] = x

    # -- LAPACK: the factorizations --------------------------------------------------------------

    def lapack_call(self, b: Builder, name: str, arguments: list[Value]) -> None:
        if not self.lapack:
            raise LoweringError(
                f"@{self.function.name}: linalg.{name} needs LAPACK, which this build does not link"
            )
        self.require_library("lapack")
        core.call_extern(b, f"d{name}_", tuple(arguments), ())

    def int_slot(self, b: Builder, value: int | Value) -> Value:
        constant = core.const(b, value, I32) if isinstance(value, int) else core.cast(b, value, I32)
        return self.slot(b, I32, constant, "lapack")

    def column_major(self, op: Operation, view: _View, hint: str) -> _View:
        """A copy of `view` with column-major strides: what LAPACK reads and writes."""
        b = Builder().before(op)
        rows = view.shape[0]
        copy = self.allocate(b, view.info, hint)
        copy = _View(copy.buffer, copy.info, (1, rows), 0)
        inner, indices, _next = self.loops(op, view.shape)
        self.store(inner, copy, indices, self.load(inner, view, indices))
        return copy

    def op_linalg_qr(self, op: Operation) -> None:
        """`dgeqrf` then `dorgqr`: Q from the reflectors, R from the upper triangle."""
        source = self.view(op.operands[0])
        m, n = source.shape
        k = min(m, n)
        q_info, r_info = self.info(op.results[0]), self.info(op.results[1])
        work = self.column_major(op, source, "qr")
        b = Builder().before(op)
        tau = self.allocate(b, tensors.TensorInfo(q_info.dtype, (k,), q_info.layout), "tau")
        lwork = max(64 * max(m, n), 1)
        scratch = self.allocate(
            b, tensors.TensorInfo(q_info.dtype, (lwork,), q_info.layout), "work"
        )
        info_slot = self.int_slot(b, 0)
        data = core.buffer_data(b, work.buffer)
        self.lapack_call(
            b,
            "geqrf",
            [
                self.int_slot(b, m),
                self.int_slot(b, n),
                data,
                self.int_slot(b, m),
                core.buffer_data(b, tau.buffer),
                core.buffer_data(b, scratch.buffer),
                self.int_slot(b, lwork),
                info_slot,
            ],
        )
        core.guard(
            b,
            core.cmp(b, "eq", core.load(b, info_slot), core.const(b, 0, I32)),
            "contract",
            "LAPACK dgeqrf failed",
        )
        # R: the upper triangle of the factored matrix, k by n.
        r = self.allocate(b, r_info, "r")
        inner, ij, _next = self.loops(op, (k, n))
        i, j = ij
        upper = core.cmp(inner, "le", i, j)
        self.store(
            inner,
            r,
            ij,
            core.select(
                inner, upper, self.load(inner, work, ij), core.const(inner, 0.0, r_info.dtype)
            ),
        )
        # Q: the first k reflectors applied to the identity, m by k, column-major in `work`.
        b = Builder().before(op)
        self.lapack_call(
            b,
            "orgqr",
            [
                self.int_slot(b, m),
                self.int_slot(b, k),
                self.int_slot(b, k),
                core.buffer_data(b, work.buffer),
                self.int_slot(b, m),
                core.buffer_data(b, tau.buffer),
                core.buffer_data(b, scratch.buffer),
                self.int_slot(b, lwork),
                info_slot,
            ],
        )
        core.guard(
            b,
            core.cmp(b, "eq", core.load(b, info_slot), core.const(b, 0, I32)),
            "contract",
            "LAPACK dorgqr failed",
        )
        q = self.allocate(b, q_info, "q")
        inner, ij, _next = self.loops(op, (m, k))
        self.store(inner, q, ij, self.load(inner, work, ij))
        self.views[id(op.results[0])] = q
        self.views[id(op.results[1])] = r

    def op_linalg_svd(self, op: Operation) -> None:
        """`dgesvd` with reduced U and Vt."""
        source = self.view(op.operands[0])
        m, n = source.shape
        k = min(m, n)
        u_info, s_info, vt_info = (self.info(r) for r in op.results)
        work = self.column_major(op, source, "svd")
        b = Builder().before(op)
        u_cm = self.allocate(b, u_info, "u")
        u_cm = _View(u_cm.buffer, u_cm.info, (1, m), 0)
        vt_cm = self.allocate(b, vt_info, "vt")
        vt_cm = _View(vt_cm.buffer, vt_cm.info, (1, k), 0)
        s = self.allocate(b, s_info, "s")
        lwork = max(5 * max(m, n) * 8, 128)
        scratch = self.allocate(
            b, tensors.TensorInfo(u_info.dtype, (lwork,), u_info.layout), "work"
        )
        info_slot = self.int_slot(b, 0)
        jobs = self.allocate(b, tensors.TensorInfo(U8, (2,), u_info.layout), "jobs")
        core.store(b, core.const(b, ord("S"), U8), core.buffer_data(b, jobs.buffer))
        core.store(
            b,
            core.const(b, ord("S"), U8),
            core.ptr_offset(b, core.buffer_data(b, jobs.buffer), core.const(b, 1, I64)),
        )
        job = core.buffer_data(b, jobs.buffer)
        self.lapack_call(
            b,
            "gesvd",
            [
                job,
                core.ptr_offset(b, job, core.const(b, 1, I64)),
                self.int_slot(b, m),
                self.int_slot(b, n),
                core.buffer_data(b, work.buffer),
                self.int_slot(b, m),
                core.buffer_data(b, s.buffer),
                core.buffer_data(b, u_cm.buffer),
                self.int_slot(b, m),
                core.buffer_data(b, vt_cm.buffer),
                self.int_slot(b, k),
                core.buffer_data(b, scratch.buffer),
                self.int_slot(b, lwork),
                info_slot,
            ],
        )
        core.guard(
            b,
            core.cmp(b, "eq", core.load(b, info_slot), core.const(b, 0, I32)),
            "contract",
            "LAPACK dgesvd failed",
        )
        u = self.allocate(b, u_info, "u_rm")
        inner, ij, _next = self.loops(op, (m, k))
        self.store(inner, u, ij, self.load(inner, u_cm, ij))
        vt = self.allocate(Builder().before(op), vt_info, "vt_rm")
        inner, ij, _next = self.loops(op, (k, n))
        self.store(inner, vt, ij, self.load(inner, vt_cm, ij))
        self.views[id(op.results[0])] = u
        self.views[id(op.results[1])] = s
        self.views[id(op.results[2])] = vt

    def op_linalg_eig(self, op: Operation) -> None:
        """`dgeev`: eigenvalues and right eigenvectors, complex as (re, im) pairs."""
        source = self.view(op.operands[0])
        n = source.shape[0]
        values_info, vectors_info = self.info(op.results[0]), self.info(op.results[1])
        dtype = values_info.dtype
        work = self.column_major(op, source, "eig")
        b = Builder().before(op)
        wr = self.allocate(b, tensors.TensorInfo(dtype, (n,), values_info.layout), "wr")
        wi = self.allocate(b, tensors.TensorInfo(dtype, (n,), values_info.layout), "wi")
        vr_cm = self.allocate(b, tensors.TensorInfo(dtype, (n, n), values_info.layout), "vr")
        vr_cm = _View(vr_cm.buffer, vr_cm.info, (1, n), 0)
        lwork = max(8 * n, 64)
        scratch = self.allocate(b, tensors.TensorInfo(dtype, (lwork,), values_info.layout), "work")
        info_slot = self.int_slot(b, 0)
        jobs = self.allocate(b, tensors.TensorInfo(U8, (2,), values_info.layout), "jobs")
        core.store(b, core.const(b, ord("N"), U8), core.buffer_data(b, jobs.buffer))
        core.store(
            b,
            core.const(b, ord("V"), U8),
            core.ptr_offset(b, core.buffer_data(b, jobs.buffer), core.const(b, 1, I64)),
        )
        job = core.buffer_data(b, jobs.buffer)
        dummy = self.allocate(b, tensors.TensorInfo(dtype, (1,), values_info.layout), "vl")
        self.lapack_call(
            b,
            "geev",
            [
                job,
                core.ptr_offset(b, job, core.const(b, 1, I64)),
                self.int_slot(b, n),
                core.buffer_data(b, work.buffer),
                self.int_slot(b, n),
                core.buffer_data(b, wr.buffer),
                core.buffer_data(b, wi.buffer),
                core.buffer_data(b, dummy.buffer),
                self.int_slot(b, 1),
                core.buffer_data(b, vr_cm.buffer),
                self.int_slot(b, n),
                core.buffer_data(b, scratch.buffer),
                self.int_slot(b, lwork),
                info_slot,
            ],
        )
        core.guard(
            b,
            core.cmp(b, "eq", core.load(b, info_slot), core.const(b, 0, I32)),
            "contract",
            "LAPACK dgeev failed",
        )
        values = self.allocate(b, values_info, "values")
        inner, (i,), _next = self.loops(op, (n,))
        self.store(inner, values, [i, core.const(inner, 0, I64)], self.load(inner, wr, [i]))
        self.store(inner, values, [i, core.const(inner, 1, I64)], self.load(inner, wi, [i]))
        # LAPACK packs a complex pair of eigenvectors as two real columns
        # (re, im); a real eigenvalue's vector is one real column.
        vectors = self.allocate(Builder().before(op), vectors_info, "vectors")
        inner, ij, _next = self.loops(op, (n, n))
        i, j = ij
        imaginary = self.load(inner, wi, [j])
        zero = core.const(inner, 0.0, dtype)
        is_real = core.cmp(inner, "eq", imaginary, zero)
        positive = core.cmp(inner, "gt", imaginary, zero)
        here = self.load(inner, vr_cm, [i, j])
        next_column = core.select(
            inner,
            core.cmp(inner, "lt", j, core.const(inner, n - 1, I64)),
            core.add(inner, j, core.const(inner, 1, I64), overflow="wrap"),
            j,
        )
        previous_column = core.select(
            inner,
            core.cmp(inner, "gt", j, core.const(inner, 0, I64)),
            core.sub(inner, j, core.const(inner, 1, I64), overflow="wrap"),
            j,
        )
        following = self.load(inner, vr_cm, [i, next_column])
        preceding = self.load(inner, vr_cm, [i, previous_column])
        real_part = core.select(inner, is_real, here, core.select(inner, positive, here, preceding))
        imag_part = core.select(
            inner, is_real, zero, core.select(inner, positive, following, core.neg(inner, here))
        )
        self.store(inner, vectors, [i, j, core.const(inner, 0, I64)], real_part)
        self.store(inner, vectors, [i, j, core.const(inner, 1, I64)], imag_part)
        self.views[id(op.results[0])] = values
        self.views[id(op.results[1])] = vectors

    # -- fft: the definition, in loops -----------------------------------------------------------

    def _dft(
        self,
        op: Operation,
        source: _View,
        result: _View,
        axis: int,
        inverse: bool,
        real_input: bool = False,
    ) -> None:
        """A transform along `axis` of `source` (complex unless `real_input`) into `result`.

        For every output frequency, the sum over every input sample of
        `x[j] * exp(-+ 2 pi i j k / n)`; the inverse divides by `n`.
        """
        dtype = result.info.dtype
        n = source.shape[axis]
        outputs = result.shape[axis]
        batch = [d for i, d in enumerate(result.shape[:-1]) if i != axis]
        self.module.require("math", 1)
        # Loops over the batch indices, the output frequency, then the input sample.
        inner, indices, _next = self.loops(op, (*batch, outputs, n))
        k, j = indices[-2], indices[-1]
        batch_indices = indices[:-2]

        def at(view: _View, along: Value, pair: int | None) -> list[Value]:
            """Indices into `view` at `along` on the transform axis; `pair`
            picks the real or imaginary half of a complex view."""
            full: list[Value] = []
            position = 0
            for dim in range(len(view.shape) - (0 if pair is None else 1)):
                if dim == axis:
                    full.append(along)
                else:
                    full.append(batch_indices[position])
                    position += 1
            if pair is not None:
                full.append(core.const(inner, pair, I64))
            return full

        angle_step = (2.0 if inverse else -2.0) * 3.141592653589793 / n
        jk = core.mul(inner, j, k, overflow="wrap")
        angle = core.mul(inner, core.cast(inner, jk, dtype), core.const(inner, angle_step, dtype))
        cosine = inner.create("math.cos", (angle,), (dtype,)).result
        sine = inner.create("math.sin", (angle,), (dtype,)).result
        if real_input:
            x_re = self.load(inner, source, at(source, j, None))
            x_im = core.const(inner, 0.0, dtype)
        else:
            x_re = self.load(inner, source, at(source, j, 0))
            x_im = self.load(inner, source, at(source, j, 1))
        term_re = core.sub(inner, core.mul(inner, x_re, cosine), core.mul(inner, x_im, sine))
        term_im = core.add(inner, core.mul(inner, x_re, sine), core.mul(inner, x_im, cosine))
        first = core.cmp(inner, "eq", j, core.const(inner, 0, I64))
        out_re = at(result, k, 0)
        out_im = at(result, k, 1)
        current_re = core.select(
            inner, first, core.const(inner, 0.0, dtype), self.load(inner, result, out_re)
        )
        current_im = core.select(
            inner, first, core.const(inner, 0.0, dtype), self.load(inner, result, out_im)
        )
        self.store(inner, result, out_re, core.add(inner, current_re, term_re))
        self.store(inner, result, out_im, core.add(inner, current_im, term_im))
        if inverse:
            inner, indices, _next = self.loops(op, result.shape)
            scaled = core.div(
                inner, self.load(inner, result, indices), core.const(inner, float(n), dtype)
            )
            self.store(inner, result, indices, scaled)

    def op_fft_fft(self, op: Operation, inverse: bool = False) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        result = self.allocate(Builder().before(op), info, "fft")
        self._dft(op, source, result, len(info.shape) - 2, inverse)
        self.views[id(op.result)] = result

    def op_fft_ifft(self, op: Operation) -> None:
        self.op_fft_fft(op, inverse=True)

    def op_fft_fftn(self, op: Operation, inverse: bool = False) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        current = source
        for axis in range(len(info.shape) - 1):
            result = self.allocate(Builder().before(op), info, "fftn")
            self._dft(op, current, result, axis, inverse)
            current = result
        self.views[id(op.result)] = current

    def op_fft_ifftn(self, op: Operation) -> None:
        self.op_fft_fftn(op, inverse=True)

    def op_fft_rfft(self, op: Operation) -> None:
        source = self.view(op.operands[0])
        info = self.info(op.result)
        result = self.allocate(Builder().before(op), info, "rfft")
        self._dft(op, source, result, len(info.shape) - 2, False, real_input=True)
        self.views[id(op.result)] = result

    def op_fft_irfft(self, op: Operation) -> None:
        """The Hermitian half spectrum completed, transformed back, and the real part kept."""
        source = self.view(op.operands[0])
        info = self.info(op.result)
        n = int(op.attributes["n"])  # type: ignore[call-overload]
        dtype = info.dtype
        full_info = tensors.TensorInfo(dtype, (*info.shape[:-1], n, 2), info.layout)
        full = self.allocate(Builder().before(op), full_info, "spectrum")
        half = source.shape[-2]
        inner, indices, _next = self.loops(op, (*info.shape[:-1], n))
        k = indices[-1]
        mirrored = core.sub(inner, core.const(inner, n, I64), k, overflow="wrap")
        in_half = core.cmp(inner, "lt", k, core.const(inner, half, I64))
        pick = core.select(inner, in_half, k, mirrored)
        re = self.load(inner, source, [*indices[:-1], pick, core.const(inner, 0, I64)])
        im = self.load(inner, source, [*indices[:-1], pick, core.const(inner, 1, I64)])
        self.store(inner, full, [*indices, core.const(inner, 0, I64)], re)
        self.store(
            inner,
            full,
            [*indices, core.const(inner, 1, I64)],
            core.select(inner, in_half, im, core.neg(inner, im)),
        )
        transformed = self.allocate(Builder().before(op), full_info, "irfft")
        self._dft(op, full, transformed, len(full_info.shape) - 2, True)
        result = self.allocate(Builder().before(op), info, "real")
        inner, indices, _next = self.loops(op, info.shape)
        self.store(
            inner,
            result,
            indices,
            self.load(inner, transformed, [*indices, core.const(inner, 0, I64)]),
        )
        self.views[id(op.result)] = result

    # -- runtime-bounded loops -------------------------------------------------------------------

    def range_loop(self, op: Operation, start: Value, stop: Value) -> tuple[Builder, Value, Block]:
        """One counted loop over `[start, stop)` with runtime bounds, placed where `op` stands."""
        block = op.parent
        assert block is not None
        continuation = self.function.body.add_block(self.fresh("t.next"))
        index = block.operations.index(op)
        for later in list(block.operations[index:]):
            block.remove(later)
            continuation.append(later)
        head = self.function.body.add_block(self.fresh("r.head"), [("p", I64)])
        body = self.function.body.add_block(self.fresh("r.body"))
        latch = self.function.body.add_block(self.fresh("r.latch"))
        core.br(Builder(block), Successor(head, (start,)))
        h = Builder(head)
        p = head.arguments[0]
        core.cond_br(h, core.cmp(h, "lt", p, stop), Successor(body), Successor(continuation))
        leave = core.br(Builder(body), Successor(latch))
        stepper = Builder(latch)
        core.br(
            stepper,
            Successor(head, (core.add(stepper, p, core.const(stepper, 1, I64), overflow="wrap"),)),
        )
        return Builder().before(leave), p, continuation

    def allocate_dynamic(self, b: Builder, dtype: IRType, count: Value, hint: str) -> Value:
        """A heap buffer of `count` elements, freed where the function returns."""
        width = core.const(b, _width(dtype), I64)
        raw = core.call_extern(
            b, "malloc", (core.mul(b, count, width, overflow="wrap"),), (PtrType(U8),)
        ).results[0]
        block = b.block
        assert block is not None
        self.heap.append((raw, block))
        pointer = core.cast(b, raw, PtrType(dtype))
        buffer = core.call_intrinsic(
            b, "ppy.buffer_from_parts", (pointer, count), (BufferType(dtype),)
        ).results[0]
        del hint
        return buffer

    def zero(self, op: Operation, view: _View) -> None:
        inner, indices, _next = self.loops(op, view.shape)
        self.store(
            inner,
            view,
            indices,
            core.const(
                inner, 0.0 if isinstance(view.info.dtype, FloatType) else 0, view.info.dtype
            ),
        )

    def length(self, b: Builder, buffer: Value) -> Value:
        """A buffer's element count, as an i64."""
        return core.cast(b, core.buffer_len(b, buffer), I64)

    def buffer_at(self, b: Builder, buffer: Value, index: Value) -> Value:
        return core.buffer_load(b, buffer, index)

    def buffer_set(self, b: Builder, buffer: Value, index: Value, value: Value) -> None:
        core.buffer_store(b, value, buffer, index)

    def as_index(self, b: Builder, value: Value) -> Value:
        return value if value.type == I64 else core.cast(b, value, I64)

    def as_stored_index(self, b: Builder, value: Value, index_type: IRType) -> Value:
        return value if value.type == index_type else core.cast(b, value, index_type)

    # -- sparse ----------------------------------------------------------------------------------

    def sparse(self, value: Value) -> _Sparse:
        found = self.sparse_values.get(id(value))
        if found is None:
            raise LoweringError(
                f"@{self.function.name}: sparse matrix %{value.name or '?'} was never made"
            )
        return found

    def op_sparse_from_parts(self, op: Operation) -> None:
        info = sparse.describe(op.result.type)
        assert info is not None
        values, first, second = op.operands
        self.sparse_values[id(op.result)] = _Sparse(info, values, first, second)

    def op_sparse_transpose(self, op: Operation) -> None:
        source = self.sparse(op.operands[0])
        info = sparse.describe(op.result.type)
        assert info is not None
        if source.info.format == "coo":
            self.sparse_values[id(op.result)] = _Sparse(
                info, source.values, source.second, source.first
            )
        else:
            self.sparse_values[id(op.result)] = _Sparse(
                info, source.values, source.first, source.second
            )

    def _each_nonzero(self, op: Operation, s: _Sparse):  # type: ignore[no-untyped-def]
        """Loops over every stored element: (builder, row index, column index, value)."""
        if s.info.format == "coo":
            b = Builder().before(op)
            count = self.length(b, s.values)
            inner, p, _next = self.range_loop(op, core.const(b, 0, I64), count)
            row = self.as_index(inner, self.buffer_at(inner, s.first, p))
            col = self.as_index(inner, self.buffer_at(inner, s.second, p))
            return inner, row, col, self.buffer_at(inner, s.values, p)
        outer_count = s.info.rows if s.info.format == "csr" else s.info.cols
        inner, (major,), _next = self.loops(op, (outer_count,))
        start = self.as_index(inner, self.buffer_at(inner, s.second, major))
        stop = self.as_index(
            inner,
            self.buffer_at(
                inner, s.second, core.add(inner, major, core.const(inner, 1, I64), overflow="wrap")
            ),
        )
        # The inner loop nests inside the outer body: its continuation is
        # where the outer body's trailing branch must go.
        hole = inner.anchor
        assert hole is not None
        body, p, _next = self.range_loop(hole, start, stop)
        minor = self.as_index(body, self.buffer_at(body, s.first, p))
        row, col = (major, minor) if s.info.format == "csr" else (minor, major)
        return body, row, col, self.buffer_at(body, s.values, p)

    def op_sparse_to_dense(self, op: Operation) -> None:
        s = self.sparse(op.operands[0])
        info = self.info(op.result)
        dense = self.allocate(Builder().before(op), info, "dense")
        self.zero(op, dense)
        body, row, col, value = self._each_nonzero(op, s)
        current = self.load(body, dense, [row, col])
        self.store(body, dense, [row, col], _combine(body, "add", info.dtype, current, value))
        self.views[id(op.result)] = dense

    def op_sparse_matmul(self, op: Operation) -> None:
        s = self.sparse(op.operands[0])
        right = self.view(op.operands[1])
        info = self.info(op.result)
        out = self.allocate(Builder().before(op), info, "spmm")
        self.zero(op, out)
        body, row, col, value = self._each_nonzero(op, s)
        hole = body.anchor
        assert hole is not None
        k = right.shape[1]
        inner, (j,), _next = self.loops(hole, (k,))
        product = (
            core.mul(inner, value, self.load(inner, right, [col, j]))
            if isinstance(info.dtype, FloatType)
            else core.mul(inner, value, self.load(inner, right, [col, j]), overflow="wrap")
        )
        current = self.load(inner, out, [row, j])
        self.store(inner, out, [row, j], _combine(inner, "add", info.dtype, current, product))
        self.views[id(op.result)] = out

    def op_sparse_reduce(self, op: Operation) -> None:
        s = self.sparse(op.operands[0])
        info = self.info(op.result)
        axis = int(op.attributes["axis"])  # type: ignore[call-overload]
        out = self.allocate(Builder().before(op), info, "spsum")
        self.zero(op, out)
        body, row, col, value = self._each_nonzero(op, s)
        target = [col] if axis == 0 else [row]
        current = self.load(body, out, target)
        self.store(body, out, target, _combine(body, "add", info.dtype, current, value))
        self.views[id(op.result)] = out

    def op_sparse_convert(self, op: Operation) -> None:
        s = self.sparse(op.operands[0])
        info = sparse.describe(op.result.type)
        assert info is not None
        if info.format == s.info.format:
            self.sparse_values[id(op.result)] = s
            return
        if info.format == "coo":
            self.sparse_values[id(op.result)] = self._expand(op, s, info)
            return
        self.sparse_values[id(op.result)] = self._compress(op, s, info)

    def _expand(self, op: Operation, s: _Sparse, info: sparse.SparseInfo) -> _Sparse:
        """CSR or CSC to COO: the compressed axis written out per element."""
        b = Builder().before(op)
        count = self.length(b, s.values)
        rows = self.allocate_dynamic(b, info.index, count, "rows")
        cols = self.allocate_dynamic(b, info.index, count, "cols")
        outer_count = s.info.rows if s.info.format == "csr" else s.info.cols
        inner, (major,), _next = self.loops(op, (outer_count,))
        start = self.as_index(inner, self.buffer_at(inner, s.second, major))
        stop = self.as_index(
            inner,
            self.buffer_at(
                inner, s.second, core.add(inner, major, core.const(inner, 1, I64), overflow="wrap")
            ),
        )
        hole = inner.anchor
        assert hole is not None
        body, p, _cont = self.range_loop(hole, start, stop)
        minor = self.buffer_at(body, s.first, p)
        major_stored = self.as_stored_index(body, major, info.index)
        if s.info.format == "csr":
            self.buffer_set(body, rows, p, major_stored)
            self.buffer_set(body, cols, p, minor)
        else:
            self.buffer_set(body, rows, p, minor)
            self.buffer_set(body, cols, p, major_stored)
        return _Sparse(info, s.values, rows, cols)

    def _compress(self, op: Operation, s: _Sparse, info: sparse.SparseInfo) -> _Sparse:
        """Any format to CSR or CSC by counting sort on the compressed axis."""
        b = Builder().before(op)
        count = self.length(b, s.values)
        majors = info.rows if info.format == "csr" else info.cols
        pointers = self.allocate(
            b, tensors.TensorInfo(info.index, (majors + 1,), tensors.layouts.Layout()), "pointers"
        )
        self.zero(op, pointers)
        values = self.allocate_dynamic(Builder().before(op), info.dtype, count, "values")
        indices = self.allocate_dynamic(Builder().before(op), info.index, count, "indices")
        # Count per major index, into pointers[major + 1].
        body, row, col, _value = self._each_nonzero(op, s)
        major = row if info.format == "csr" else col
        slot = [core.add(body, major, core.const(body, 1, I64), overflow="wrap")]
        one = core.const(body, 1, info.index)
        self.store(
            body,
            pointers,
            slot,
            core.add(body, self.load(body, pointers, slot), one, overflow="wrap"),
        )
        # Exclusive prefix sums make the row pointers.
        inner, (m,), _next = self.loops(op, (majors,))
        next_slot = [core.add(inner, m, core.const(inner, 1, I64), overflow="wrap")]
        total = core.add(
            inner,
            self.load(inner, pointers, [m]),
            self.load(inner, pointers, next_slot),
            overflow="wrap",
        )
        self.store(inner, pointers, next_slot, total)
        # A cursor per major index, then every element placed.
        cursors = self.allocate(
            Builder().before(op),
            tensors.TensorInfo(info.index, (majors,), tensors.layouts.Layout()),
            "cursors",
        )
        inner, (m,), _next = self.loops(op, (majors,))
        self.store(inner, cursors, [m], self.load(inner, pointers, [m]))
        body, row, col, value = self._each_nonzero(op, s)
        major = row if info.format == "csr" else col
        minor = col if info.format == "csr" else row
        position = self.as_index(body, self.load(body, cursors, [major]))
        self.buffer_set(body, values, position, value)
        self.buffer_set(body, indices, position, self.as_stored_index(body, minor, info.index))
        self.store(
            body,
            cursors,
            [major],
            core.add(
                body, self.load(body, cursors, [major]), one_of(body, info.index), overflow="wrap"
            ),
        )
        return _Sparse(info, values, indices, pointers.buffer)

    def op_sparse_add(self, op: Operation) -> None:
        """Two CSR (or CSC) matrices with sorted indices merged row by row."""
        left, right = self.sparse(op.operands[0]), self.sparse(op.operands[1])
        info = sparse.describe(op.result.type)
        assert info is not None
        majors = info.rows if info.format == "csr" else info.cols
        b = Builder().before(op)
        capacity = core.add(
            b, self.length(b, left.values), self.length(b, right.values), overflow="wrap"
        )
        values = self.allocate_dynamic(b, info.dtype, capacity, "values")
        indices = self.allocate_dynamic(b, info.index, capacity, "indices")
        pointers = self.allocate(
            b, tensors.TensorInfo(info.index, (majors + 1,), tensors.layouts.Layout()), "pointers"
        )
        written = self.slot(b, I64, core.const(b, 0, I64), "written")
        self.store(b, pointers, [core.const(b, 0, I64)], core.const(b, 0, info.index))
        outer, (m,), _next = self.loops(op, (majors,))
        m_next = core.add(outer, m, core.const(outer, 1, I64), overflow="wrap")
        pa = self.slot(
            outer, I64, self.as_index(outer, self.buffer_at(outer, left.second, m)), "pa"
        )
        pb = self.slot(
            outer, I64, self.as_index(outer, self.buffer_at(outer, right.second, m)), "pb"
        )
        end_a = self.as_index(outer, self.buffer_at(outer, left.second, m_next))
        end_b = self.as_index(outer, self.buffer_at(outer, right.second, m_next))
        hole = outer.anchor
        assert hole is not None
        condition, body, _cont = self.while_(hole)
        more_a = core.cmp(condition, "lt", core.load(condition, pa), end_a)
        more_b = core.cmp(condition, "lt", core.load(condition, pb), end_b)
        self.close_while(condition, core.bitwise(condition, "or", more_a, more_b))
        a_index = core.load(body, pa)
        b_index = core.load(body, pb)
        has_a = core.cmp(body, "lt", a_index, end_a)
        has_b = core.cmp(body, "lt", b_index, end_b)
        big = core.const(body, (1 << 62), I64)
        col_a = core.select(
            body,
            has_a,
            self.as_index(
                body,
                self.buffer_at(
                    body, left.first, core.select(body, has_a, a_index, core.const(body, 0, I64))
                ),
            ),
            big,
        )
        col_b = core.select(
            body,
            has_b,
            self.as_index(
                body,
                self.buffer_at(
                    body, right.first, core.select(body, has_b, b_index, core.const(body, 0, I64))
                ),
            ),
            big,
        )
        take_a = core.cmp(body, "le", col_a, col_b)
        take_b = core.cmp(body, "le", col_b, col_a)
        zero = core.const(body, 0.0 if isinstance(info.dtype, FloatType) else 0, info.dtype)
        value_a = core.select(
            body,
            take_a,
            self.buffer_at(
                body, left.values, core.select(body, has_a, a_index, core.const(body, 0, I64))
            ),
            zero,
        )
        value_b = core.select(
            body,
            take_b,
            self.buffer_at(
                body, right.values, core.select(body, has_b, b_index, core.const(body, 0, I64))
            ),
            zero,
        )
        total = _combine(body, "add", info.dtype, value_a, value_b)
        column = core.select(body, take_a, col_a, col_b)
        position = core.load(body, written)
        self.buffer_set(body, values, position, total)
        self.buffer_set(body, indices, position, self.as_stored_index(body, column, info.index))
        core.store(
            body, core.add(body, position, core.const(body, 1, I64), overflow="wrap"), written
        )
        step = core.const(body, 1, I64)
        core.store(
            body,
            core.select(body, take_a, core.add(body, a_index, step, overflow="wrap"), a_index),
            pa,
        )
        core.store(
            body,
            core.select(body, take_b, core.add(body, b_index, step, overflow="wrap"), b_index),
            pb,
        )
        # After the merge of this row: its end pointer.
        after = Builder().before(outer.anchor)
        self.store(
            after,
            pointers,
            [m_next],
            self.as_stored_index(after, core.load(after, written), info.index),
        )
        self.sparse_values[id(op.result)] = _Sparse(info, values, indices, pointers.buffer)


def one_of(b: Builder, t: IRType) -> Value:
    return core.const(b, 1, t)


def _broadcast_strides(view: _View, target: shapes.Shape) -> tuple[shapes.Dim, ...]:
    """The strides that read `view` as if it had `target`'s shape."""
    rank = len(target)
    padded_shape = (1,) * (rank - len(view.shape)) + view.shape
    padded_strides = (0,) * (rank - len(view.shape)) + view.strides
    return tuple(
        0 if source == 1 and result != 1 else stride
        for source, result, stride in zip(padded_shape, target, padded_strides, strict=True)
    )


def _identity(b: Builder, kind: str, dtype: IRType) -> Value:
    floating = isinstance(dtype, FloatType)
    if kind == "add":
        return core.const(b, 0.0 if floating else 0, dtype)
    if kind == "mul":
        return core.const(b, 1.0 if floating else 1, dtype)
    if floating:
        return core.const(b, float("inf") if kind == "min" else float("-inf"), dtype)
    width = dtype.width if isinstance(dtype, IntType) else 64
    signed = dtype.signed if isinstance(dtype, IntType) else True
    if kind == "min":
        return core.const(b, (1 << (width - 1)) - 1 if signed else (1 << width) - 1, dtype)
    return core.const(b, -(1 << (width - 1)) if signed else 0, dtype)


def _combine(b: Builder, kind: str, dtype: IRType, current: Value, value: Value) -> Value:
    floating = isinstance(dtype, FloatType)
    if kind == "add":
        return (
            core.add(b, current, value)
            if floating
            else core.add(b, current, value, overflow="wrap")
        )
    if kind == "mul":
        return (
            core.mul(b, current, value)
            if floating
            else core.mul(b, current, value, overflow="wrap")
        )
    del dtype
    return tensors.scalar_extremum(b, kind, current, value)


def _defined_before(value: Value, op: Operation) -> bool:
    """Is `value` available where `op` stands: a block argument, or an operation
    earlier in the same block? (The destination buffer is read there.)"""
    owner = value.owner
    if not isinstance(owner, Operation):
        return True
    block = op.parent
    if owner.parent is not block or block is None:
        return False
    return block.operations.index(owner) < block.operations.index(op)


def _symbols_of(shape: shapes.Shape) -> set[str]:
    """The names a shape depends on."""
    found: set[str] = set()

    def walk(dim: shapes.Dim) -> None:
        if isinstance(dim, shapes.Symbol):
            found.add(dim.name)
        elif isinstance(dim, shapes.Expr):
            for argument in dim.args:
                walk(argument)

    for dim in shape:
        walk(dim)
    return found


def _absolute(b: Builder, value: Value) -> Value:
    negative = core.cmp(b, "lt", value, core.const(b, 0.0, value.type))
    return core.select(b, negative, core.neg(b, value), value)


def _width(dtype: IRType) -> int:
    if isinstance(dtype, (IntType, FloatType)):
        return max(dtype.width // 8, 1)
    if isinstance(dtype, IndexType):
        return 8
    return 1
