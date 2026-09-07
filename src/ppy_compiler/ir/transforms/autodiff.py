"""Reverse-mode automatic differentiation over the canonical and tensor IR.

`differentiate` takes a function of one block whose result is a float, or
a rank-0 float tensor, and makes a second function: the same forward
computation, then the adjoint of every operation in reverse, and the
gradient with respect to the parameters asked for. Each operation has one
rule, applied in one order -- the reverse of the forward order, with
contributions to one adjoint added in the order they arise -- and the
runtime's reference implementation follows the same rules in the same
order, so the three execution paths agree bit for bit.

What is covered is what the rules cover: scalar arithmetic, the math
functions, `erf` and `erfc`, `select`, casts between floats; and over
tensors the elementwise arithmetic and unary functions, `fill`,
`broadcast`, `reshape`, `transpose`, `reduce` by sum, `matmul`, `convert`,
and `load` from a buffer parameter, whose gradient is written to a buffer
the caller passes. A function with control flow, a store, a call, or an
operation without a rule is refused with the reason (spec 44, 45).
"""

from __future__ import annotations

import math

from ..dialects import core
from ..dialects import tensor as tensors
from ..model import Builder, IRFunction, IRModule, Operation, Value
from ..types import I64, BufferType, FloatType, IRType
from .promote_slots import promote_slots

__all__ = ["AutodiffError", "differentiate", "gradient_name"]

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)
_TWO_OVER_ROOT_PI = 2.0 / math.sqrt(math.pi)


class AutodiffError(ValueError):
    """A function the transform cannot differentiate, with the reason."""


def gradient_name(name: str, wrt: tuple[int, ...], value: bool) -> str:
    """The symbol the derivative of `name` gets."""
    kind = "value_and_grad" if value else "grad"
    suffix = "" if wrt == (0,) else "_" + "_".join(str(i) for i in wrt)
    return f"{name}__{kind}{suffix}"


def differentiate(
    module: IRModule,
    function: IRFunction,
    *,
    wrt: tuple[int, ...] = (0,),
    value: bool = False,
    name: str | None = None,
) -> IRFunction:
    """The gradient of `function` with respect to the parameters at `wrt`.

    A scalar parameter's gradient is a result, in `wrt` order after the
    value when `value` is asked for; a buffer parameter's gradient is
    written into a buffer parameter added after the function's own, one
    per buffer in `wrt`, named `grad_<parameter>`.
    """
    if len(function.body.blocks) != 1:
        raise AutodiffError(
            f"@{function.name} has control flow; the derivative of a branch or loop is not "
            "taken yet"
        )
    if len(function.results) != 1 or not _differentiable(function.results[0]):
        raise AutodiffError(f"@{function.name} must return one float to be differentiated")
    if not wrt or any(i < 0 or i >= len(function.params) for i in wrt):
        raise AutodiffError(f"@{function.name}: `wrt` names a parameter position")
    for index in wrt:
        _name, t = function.params[index]
        if isinstance(t, FloatType):
            continue
        if isinstance(t, BufferType) and isinstance(t.element, FloatType):
            continue
        raise AutodiffError(
            f"@{function.name}: the gradient with respect to `{_name}: {t}` is not defined; "
            "a float or a buffer of floats is"
        )
    symbol = name or gradient_name(function.name, wrt, value)
    if symbol in module.functions:
        return module.functions[symbol]
    _Reverse(module, function, wrt, value, symbol).run()
    return module.functions[symbol]


def _scalar_type(t: IRType) -> IRType:
    info = tensors.describe(t)
    return t if info is None else info.dtype


def _differentiable(t: IRType) -> bool:
    if isinstance(t, FloatType):
        return True
    info = tensors.describe(t)
    return info is not None and isinstance(info.dtype, FloatType) and info.rank == 0


class _Reverse:
    def __init__(
        self, module: IRModule, source: IRFunction, wrt: tuple[int, ...], value: bool, name: str
    ) -> None:
        self.module = module
        self.source = source
        self.wrt = wrt
        self.value = value
        self.name = name
        self.adjoints: dict[int, Value] = {}
        self.buffer_adjoints: dict[int, Value] = {}
        self.mapping: dict[int, Value] = {}
        params = list(source.params)
        for index in wrt:
            pname, t = params[index]
            if isinstance(t, BufferType):
                params.append((f"grad_{pname}", t))
        results: list[IRType] = []
        if value:
            results.append(_scalar_type(source.results[0]))
        for index in wrt:
            _pname, t = source.params[index]
            if isinstance(t, FloatType):
                results.append(t)
        self.function = module.add_function(name, params, results)
        if source.attributes.get("fastmath"):
            self.function.attributes["fastmath"] = True
        self.entry = self.function.add_entry_block()
        self.b = Builder(self.entry)

    # -- the forward copy -----------------------------------------------------------------

    def run(self) -> None:
        entry = self.entry
        for old, new in zip(self.source.entry.arguments, entry.arguments, strict=False):
            self.mapping[id(old)] = new
        b = self.b
        returned: Value | None = None
        for op in self.source.body.blocks[0].operations:
            if op.name == "core.ret":
                returned = self.mapping[id(op.operands[0])]
                break
            if op.regions:
                raise AutodiffError(f"@{self.source.name}: {op.name} holds a region no rule reads")
            clone = b.create(
                op.name,
                tuple(self.mapping.get(id(v), v) for v in op.operands),
                tuple(r.type for r in op.results),
                dict(op.attributes),
                result_names=tuple(r.name for r in op.results),
            )
            for original, copy in zip(op.results, clone.results, strict=True):
                self.mapping[id(original)] = copy
        assert returned is not None
        promote_slots(self.function)
        forward = list(entry.operations)
        for op in forward:
            if op.name in {"core.alloca", "core.load", "core.store", "core.call"}:
                raise AutodiffError(
                    f"@{self.source.name}: {op.name} is memory or a call the derivative does "
                    "not follow"
                )
        # -- the reverse sweep ------------------------------------------------------
        self.adjoints[id(returned)] = self._one_like(returned)
        for op in reversed(forward):
            self._backward(op)
        outputs: list[Value] = []
        if self.value:
            # A rank-0 tensor's one element comes out as the scalar it is.
            outputs.append(
                returned if isinstance(returned.type, FloatType) else self._tensor_scalar(returned)
            )
        grad_buffers = iter(entry.arguments[len(self.source.params) :])
        for index in self.wrt:
            parameter = entry.arguments[index]
            if isinstance(parameter.type, BufferType):
                target = next(grad_buffers)
                adjoint = self.buffer_adjoints.get(id(parameter))
                if adjoint is None:
                    raise AutodiffError(
                        f"@{self.source.name}: `{self.source.params[index][0]}` is never read "
                        "as a tensor, so its gradient has no shape"
                    )
                tensors.store(b, adjoint, target)
            else:
                adjoint = self.adjoints.get(id(parameter))
                if adjoint is None:
                    adjoint = core.const(b, 0.0, parameter.type)
                outputs.append(adjoint)
        core.ret(b, *outputs)

    # -- accumulation -------------------------------------------------------------------

    def _one_like(self, value: Value) -> Value:
        b = self.b
        if isinstance(value.type, FloatType):
            return core.const(b, 1.0, value.type)
        info = tensors.describe(value.type)
        assert info is not None
        return tensors.fill(b, core.const(b, 1.0, info.dtype), value.type)  # type: ignore[arg-type]

    def _zero_like(self, value: Value) -> Value:
        b = self.b
        if isinstance(value.type, FloatType):
            return core.const(b, 0.0, value.type)
        info = tensors.describe(value.type)
        assert info is not None
        return tensors.fill(b, core.const(b, 0.0, info.dtype), value.type)  # type: ignore[arg-type]

    def _add(self, existing: Value, contribution: Value) -> Value:
        if isinstance(existing.type, FloatType):
            return core.add(self.b, existing, contribution)
        return tensors.elementwise(self.b, "add", existing, contribution)

    def _accumulate(self, operand: Value, contribution: Value) -> None:
        """Add `contribution` to `operand`'s adjoint: the first one is the adjoint."""
        if isinstance(operand.type, FloatType) or tensors.describe(operand.type) is not None:
            existing = self.adjoints.get(id(operand))
            self.adjoints[id(operand)] = (
                contribution if existing is None else self._add(existing, contribution)
            )

    # -- the rules ------------------------------------------------------------------------

    def _backward(self, op: Operation) -> None:
        if not op.results:
            return
        result = op.results[0]
        g = self.adjoints.get(id(result))
        if g is None:
            return
        if isinstance(result.type, FloatType):
            handler = _SCALAR_RULES.get(op.name)
        else:
            handler = _TENSOR_RULES.get(op.name)
        if handler is None:
            raise AutodiffError(f"@{self.source.name}: {op.name} has no derivative rule")
        handler(self, op, g)

    # scalar helpers, each one operation so the order of evaluation is fixed

    def _c(self, value: float, t: IRType) -> Value:
        return core.const(self.b, value, t)

    def _math(self, name: str, *operands: Value) -> Value:
        self.module.require("math", 1)
        return self.b.create(f"math.{name}", operands, (operands[0].type,)).result

    def _special(self, name: str, operand: Value) -> Value:
        self.module.require("special", 1)
        return self.b.create(f"special.{name}", (operand,), (operand.type,)).result

    # tensor helpers

    def _tensor_const(self, value: float, like: Value) -> Value:
        info = tensors.describe(like.type)
        assert info is not None
        return tensors.fill(self.b, core.const(self.b, value, info.dtype), like.type)  # type: ignore[arg-type]

    def _unbroadcast(self, g: Value, target: Value) -> Value:
        """`g`, summed over the axes broadcasting `target` grew, to `target`'s shape."""
        wanted = tensors.describe(target.type)
        have = tensors.describe(g.type)
        assert wanted is not None and have is not None
        if have.shape == wanted.shape:
            return g
        b = self.b
        self.module.require("tensor", 1)
        leading = tuple(range(have.rank - wanted.rank))
        if leading:
            g = tensors.reduce(b, g, leading, "add")
            have = tensors.describe(g.type)
            assert have is not None
        grown = tuple(
            axis
            for axis, (mine, theirs) in enumerate(zip(have.shape, wanted.shape, strict=True))
            if theirs == 1 and mine != 1
        )
        if grown:
            g = tensors.reduce(b, g, grown, "add", keepdims=True)
        return g

    def _tensor_scalar(self, t: Value) -> Value:
        """The one element of a rank-0 tensor, as a scalar."""
        b = self.b
        info = tensors.describe(t.type)
        assert info is not None
        slot = core.alloca(b, info.dtype, name="adj")
        buffer = core.call_intrinsic(
            b, "ppy.buffer_from_parts", (slot, core.const(b, 1, I64)), (BufferType(info.dtype),)
        ).results[0]
        tensors.store(b, t, buffer)
        return core.load(b, slot)


# -- scalar rules ---------------------------------------------------------------------------


def _s_add(r: _Reverse, op: Operation, g: Value) -> None:
    a, c = op.operands
    r._accumulate(a, g)
    r._accumulate(c, g)


def _s_sub(r: _Reverse, op: Operation, g: Value) -> None:
    a, c = op.operands
    r._accumulate(a, g)
    r._accumulate(c, core.neg(r.b, g))


def _s_mul(r: _Reverse, op: Operation, g: Value) -> None:
    a, c = op.operands
    r._accumulate(a, core.mul(r.b, g, c))
    r._accumulate(c, core.mul(r.b, g, a))


def _s_div(r: _Reverse, op: Operation, g: Value) -> None:
    a, c = op.operands
    result = op.results[0]
    r._accumulate(a, core.div(r.b, g, c))
    r._accumulate(c, core.neg(r.b, core.mul(r.b, g, core.div(r.b, result, c))))


def _s_neg(r: _Reverse, op: Operation, g: Value) -> None:
    r._accumulate(op.operands[0], core.neg(r.b, g))


def _s_select(r: _Reverse, op: Operation, g: Value) -> None:
    condition, a, c = op.operands
    zero = r._zero_like(g)
    r._accumulate(a, core.select(r.b, condition, g, zero))
    r._accumulate(c, core.select(r.b, condition, zero, g))


def _s_cast(r: _Reverse, op: Operation, g: Value) -> None:
    source = op.operands[0]
    if isinstance(source.type, FloatType):
        r._accumulate(source, core.cast(r.b, g, source.type))


def _s_math(name: str):  # type: ignore[no-untyped-def]
    def rule(r: _Reverse, op: Operation, g: Value) -> None:
        x = op.operands[0]
        result = op.results[0]
        t = x.type
        b = r.b
        if name == "sin":
            r._accumulate(x, core.mul(b, g, r._math("cos", x)))
        elif name == "cos":
            r._accumulate(x, core.neg(b, core.mul(b, g, r._math("sin", x))))
        elif name == "tan":
            one_plus = core.add(b, r._c(1.0, t), core.mul(b, result, result))
            r._accumulate(x, core.mul(b, g, one_plus))
        elif name == "exp":
            r._accumulate(x, core.mul(b, g, result))
        elif name == "exp2":
            r._accumulate(x, core.mul(b, g, core.mul(b, result, r._c(_LN2, t))))
        elif name == "log":
            r._accumulate(x, core.div(b, g, x))
        elif name == "log2":
            r._accumulate(x, core.div(b, g, core.mul(b, x, r._c(_LN2, t))))
        elif name == "log10":
            r._accumulate(x, core.div(b, g, core.mul(b, x, r._c(_LN10, t))))
        elif name == "sqrt":
            r._accumulate(x, core.div(b, g, core.mul(b, r._c(2.0, t), result)))
        elif name == "abs":
            zero = r._c(0.0, t)
            positive = core.cmp(b, "gt", x, zero)
            negative = core.cmp(b, "lt", x, zero)
            below = core.select(b, negative, core.neg(b, g), zero)
            r._accumulate(x, core.select(b, positive, g, below))
        elif name == "pow":
            base, exponent = op.operands
            lowered = core.sub(b, exponent, r._c(1.0, t))
            r._accumulate(
                base, core.mul(b, g, core.mul(b, exponent, r._math("pow", base, lowered)))
            )
            r._accumulate(exponent, core.mul(b, g, core.mul(b, result, r._math("log", base))))
        elif name in {"floor", "ceil", "trunc"}:
            return
        else:
            raise AutodiffError(f"@{r.source.name}: math.{name} has no derivative rule")

    return rule


def _s_erf(sign: float):  # type: ignore[no-untyped-def]
    def rule(r: _Reverse, op: Operation, g: Value) -> None:
        x = op.operands[0]
        b = r.b
        t = x.type
        bell = r._math("exp", core.neg(b, core.mul(b, x, x)))
        scaled = core.mul(b, g, core.mul(b, r._c(sign * _TWO_OVER_ROOT_PI, t), bell))
        r._accumulate(x, scaled)

    return rule


_MATH_NAMES = (
    "sin",
    "cos",
    "tan",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "sqrt",
    "abs",
    "floor",
    "ceil",
    "trunc",
    "pow",
)

_SCALAR_RULES = {
    "core.add": _s_add,
    "core.sub": _s_sub,
    "core.mul": _s_mul,
    "core.div": _s_div,
    "core.neg": _s_neg,
    "core.select": _s_select,
    "core.cast": _s_cast,
    "core.const": lambda r, op, g: None,
    "special.erf": _s_erf(1.0),
    "special.erfc": _s_erf(-1.0),
    **{f"math.{name}": _s_math(name) for name in _MATH_NAMES},
}


# -- tensor rules -----------------------------------------------------------------------------


def _t_elementwise(name: str):  # type: ignore[no-untyped-def]
    def rule(r: _Reverse, op: Operation, g: Value) -> None:
        a, c = op.operands
        result = op.results[0]
        b = r.b
        r.module.require("tensor", 1)
        ew = tensors.elementwise
        if name == "add":
            r._accumulate(a, r._unbroadcast(g, a))
            r._accumulate(c, r._unbroadcast(g, c))
        elif name == "sub":
            r._accumulate(a, r._unbroadcast(g, a))
            r._accumulate(c, r._unbroadcast(tensors.unary(b, "neg", g), c))
        elif name == "mul":
            r._accumulate(a, r._unbroadcast(ew(b, "mul", g, c), a))
            r._accumulate(c, r._unbroadcast(ew(b, "mul", g, a), c))
        elif name == "div":
            r._accumulate(a, r._unbroadcast(ew(b, "div", g, c), a))
            quotient = ew(b, "div", result, c)
            r._accumulate(c, r._unbroadcast(tensors.unary(b, "neg", ew(b, "mul", g, quotient)), c))
        elif name == "pow":
            one = r._tensor_const(1.0, c)
            lowered = ew(b, "sub", c, one)
            r._accumulate(
                a, r._unbroadcast(ew(b, "mul", g, ew(b, "mul", c, ew(b, "pow", a, lowered))), a)
            )
            r._accumulate(
                c,
                r._unbroadcast(
                    ew(b, "mul", g, ew(b, "mul", result, tensors.unary(b, "log", a))), c
                ),
            )
        else:
            raise AutodiffError(f"@{r.source.name}: tensor.{name} has no derivative rule")

    return rule


def _t_unary(r: _Reverse, op: Operation, g: Value) -> None:
    x = op.operands[0]
    result = op.results[0]
    kind = str(op.attributes["op"])
    b = r.b
    r.module.require("tensor", 1)
    ew = tensors.elementwise
    un = tensors.unary
    if kind == "neg":
        r._accumulate(x, un(b, "neg", g))
    elif kind == "sin":
        r._accumulate(x, ew(b, "mul", g, un(b, "cos", x)))
    elif kind == "cos":
        r._accumulate(x, un(b, "neg", ew(b, "mul", g, un(b, "sin", x))))
    elif kind == "tan":
        r._accumulate(
            x, ew(b, "mul", g, ew(b, "add", r._tensor_const(1.0, x), ew(b, "mul", result, result)))
        )
    elif kind == "exp":
        r._accumulate(x, ew(b, "mul", g, result))
    elif kind == "exp2":
        r._accumulate(x, ew(b, "mul", g, ew(b, "mul", result, r._tensor_const(_LN2, x))))
    elif kind == "log":
        r._accumulate(x, ew(b, "div", g, x))
    elif kind == "log2":
        r._accumulate(x, ew(b, "div", g, ew(b, "mul", x, r._tensor_const(_LN2, x))))
    elif kind == "log10":
        r._accumulate(x, ew(b, "div", g, ew(b, "mul", x, r._tensor_const(_LN10, x))))
    elif kind == "sqrt":
        r._accumulate(x, ew(b, "div", g, ew(b, "mul", r._tensor_const(2.0, x), result)))
    elif kind == "erf":
        bell = un(b, "exp", un(b, "neg", ew(b, "mul", x, x)))
        r._accumulate(x, ew(b, "mul", g, ew(b, "mul", r._tensor_const(_TWO_OVER_ROOT_PI, x), bell)))
    elif kind == "erfc":
        bell = un(b, "exp", un(b, "neg", ew(b, "mul", x, x)))
        r._accumulate(
            x, ew(b, "mul", g, ew(b, "mul", r._tensor_const(-_TWO_OVER_ROOT_PI, x), bell))
        )
    elif kind in {"floor", "ceil", "trunc"}:
        return
    else:
        raise AutodiffError(f"@{r.source.name}: tensor.unary {kind} has no derivative rule")


def _t_fill(r: _Reverse, op: Operation, g: Value) -> None:
    scalar = op.operands[0]
    if not isinstance(scalar.type, FloatType):
        return
    info = tensors.describe(g.type)
    assert info is not None
    r.module.require("tensor", 1)
    total = tensors.reduce(r.b, g, tuple(range(info.rank)), "add") if info.rank else g
    r._accumulate(scalar, r._tensor_scalar(total))


def _t_broadcast(r: _Reverse, op: Operation, g: Value) -> None:
    r._accumulate(op.operands[0], r._unbroadcast(g, op.operands[0]))


def _t_reshape(r: _Reverse, op: Operation, g: Value) -> None:
    source = op.operands[0]
    info = tensors.describe(source.type)
    assert info is not None
    r._accumulate(source, tensors.reshape(r.b, g, info.shape))


def _t_transpose(r: _Reverse, op: Operation, g: Value) -> None:
    perm = tuple(int(p) for p in op.attributes["perm"])  # type: ignore[union-attr]
    inverse = tuple(perm.index(axis) for axis in range(len(perm)))
    r._accumulate(op.operands[0], tensors.transpose(r.b, g, inverse))


def _t_reduce(r: _Reverse, op: Operation, g: Value) -> None:
    kind = op.attributes["op"]
    if kind != "add":
        raise AutodiffError(f"@{r.source.name}: tensor.reduce {kind} has no derivative rule")
    source = op.operands[0]
    info = tensors.describe(source.type)
    assert info is not None
    axes = tuple(int(a) for a in op.attributes["axes"])  # type: ignore[union-attr]
    kept = g
    if not op.attributes.get("keepdims", False):
        kept_shape = tuple(1 if axis in axes else dim for axis, dim in enumerate(info.shape))
        kept = tensors.reshape(r.b, g, kept_shape)
    r._accumulate(source, tensors.broadcast(r.b, kept, info.shape))


def _t_matmul(r: _Reverse, op: Operation, g: Value) -> None:
    a, c = op.operands
    left, right = tensors.describe(a.type), tensors.describe(c.type)
    assert left is not None and right is not None
    if left.rank != 2 or right.rank != 2:
        raise AutodiffError(f"@{r.source.name}: tensor.matmul is differentiated for matrices")
    b = r.b
    r._accumulate(a, tensors.matmul(b, g, tensors.transpose(b, c, (1, 0))))
    r._accumulate(c, tensors.matmul(b, tensors.transpose(b, a, (1, 0)), g))


def _t_convert(r: _Reverse, op: Operation, g: Value) -> None:
    source = op.operands[0]
    info = tensors.describe(source.type)
    assert info is not None
    if isinstance(info.dtype, FloatType):
        r._accumulate(source, tensors.convert(r.b, g, info.dtype))


def _t_load(r: _Reverse, op: Operation, g: Value) -> None:
    buffer = op.operands[0]
    parameters = list(r.function.entry.arguments[: len(r.source.params)])
    for index in r.wrt:
        if parameters[index] is buffer:
            existing = r.buffer_adjoints.get(id(buffer))
            r.buffer_adjoints[id(buffer)] = g if existing is None else r._add(existing, g)
            return
    # A tensor read from any other buffer is a constant of the function.


_TENSOR_RULES = {
    "tensor.unary": _t_unary,
    "tensor.fill": _t_fill,
    "tensor.broadcast": _t_broadcast,
    "tensor.reshape": _t_reshape,
    "tensor.transpose": _t_transpose,
    "tensor.reduce": _t_reduce,
    "tensor.matmul": _t_matmul,
    "tensor.convert": _t_convert,
    "tensor.load": _t_load,
    "tensor.empty": lambda r, op, g: None,
    **{f"tensor.{name}": _t_elementwise(name) for name in ("add", "sub", "mul", "div", "pow")},
}
