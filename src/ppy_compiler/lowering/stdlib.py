"""`random`, `heapq`, and `bisect` in native code.

The calls `analysis.native_stdlib` answers for, lowered over the runtime in
`random.c` and `stdlib.c`: CPython's Mersenne Twister and `random.py`'s
algorithms on top of it, draw for draw, and `_heapq`'s and `_bisect`'s
comparisons, one for one. Under `ppy run` the generator's state is the one
inside `random._inst`, so Python and native draws interleave as they would
in CPython; what CPython raises for is a guard, with its message.
"""

from __future__ import annotations

import ast

from ..analysis.native_stdlib import MODELS
from ..backend.llvm.lowering import Unsupported
from ..analysis.lexical import LexicalBindings
from ..ir import F64, I64, Value
from ..ir.dialects import core
from ..ir.dialects import math as math_dialect
from .collections import HANDLE, Kind, _pointer

__all__ = ["StdlibLowering"]

_FLOAT_ARGUMENTS = {
    "uniform": 2,
    "triangular": 3,
    "gauss": 2,
    "normalvariate": 2,
    "lognormvariate": 2,
    "expovariate": 1,
    "paretovariate": 1,
    "weibullvariate": 2,
}


class StdlibLowering:
    """The standard library's calls; mixed into `_FunctionLowering`."""

    def _stdlib_target(self, node: ast.Call) -> str | None:
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings):
            return None
        found = lexical.targets_at(node.func)
        if len(found) != 1:
            return None
        qualname = next(iter(found))
        return qualname if qualname in MODELS else None

    def _stdlib_call(self, node: ast.Call, discard_result: bool) -> Value | None:
        """A call to one of the modules here, lowered; None where it is not one."""
        qualname = self._stdlib_target(node)
        if qualname is None:
            return None
        module, _, name = qualname.partition(".")
        if node.keywords and qualname != "random.choices":
            raise Unsupported(f"`{qualname}` with keywords has no native lowering")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise Unsupported(f"`{qualname}` with `*` arguments has no native lowering")
        if module == "random":
            return self._random_call(name, node)
        if module == "heapq":
            return self._heapq_call(name, node)
        return self._bisect_call(name, node)

    # -- helpers -----------------------------------------------------------------

    def _int_argument(self, node: ast.expr) -> Value:
        return self._coerce(self._expr(node), "int")  # type: ignore[attr-defined]

    def _float_argument(self, node: ast.expr) -> Value:
        return self._coerce(self._expr(node), "float")  # type: ignore[attr-defined]

    def _float(self, value: float) -> Value:
        return core.const(self.b, value, F64)  # type: ignore[attr-defined]

    def _plain_list(self, node: ast.expr) -> tuple[Kind, Value, bool]:
        kind = self._builtin_of(node)  # type: ignore[attr-defined]
        if kind is None or kind.name != "List":
            raise Unsupported(f"`{ast.unparse(node)}` is not a native list")
        return self._receiver(node)  # type: ignore[attr-defined,no-any-return]

    def _math(self, name: str, value: Value) -> Value:
        self.frontend.module.require("math", 1)  # type: ignore[attr-defined]
        return math_dialect.call(self.b, name, value)  # type: ignore[attr-defined,no-any-return]

    def _state(self) -> Value:
        return self._rt("ppy_random_state", (), HANDLE)  # type: ignore[attr-defined,no-any-return]

    # -- random ------------------------------------------------------------------

    def _random_call(self, name: str, node: ast.Call) -> Value:
        args = node.args
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        require = self._require  # type: ignore[attr-defined]
        if name == "random":
            return rt("ppy_random_double", (self._state(),), F64)
        if name == "seed":
            given = word(1 if args else 0)
            value = self._int_argument(args[0]) if args else word(0)
            rt("ppy_random_reseed", (self._state(), given, value), None)
            return word(0)
        if name == "getrandbits":
            k = self._int_argument(args[0])
            require(
                core.cmp(b, "ge", k, word(0)),
                "getrandbits of a negative count",
                "ValueError: number of bits must be non-negative",
            )
            self._guard(  # type: ignore[attr-defined]
                core.cmp(b, "le", k, word(63)),
                "range",
                "getrandbits past 63 bits",
                raises="OverflowError: the result does not fit in a 64-bit integer",
            )
            return rt("ppy_random_bits", (self._state(), k))
        if name in {"randrange", "randint"}:
            return self._randrange(name, args)
        if name == "choice":
            return self._choice(args[0])
        if name == "shuffle":
            _, handle, owned = self._plain_list(args[0])
            rt("ppy_random_shuffle", (self._state(), handle), None)
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return word(0)
        if name == "sample":
            return self._sample(args[0], args[1])
        if name == "choices":
            return self._choices(node)
        return self._distribution(name, args)

    def _randrange(self, name: str, args: list[ast.expr]) -> Value:
        """`randrange` and `randint`, with CPython's checks in its order."""
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        require = self._require  # type: ignore[attr-defined]
        values = [self._int_argument(a) for a in args]
        if name == "randrange" and len(values) == 1:
            (stop,) = values
            require(
                core.cmp(b, "gt", stop, word(0)),
                "empty range for randrange()",
                "ValueError: empty range for randrange()",
            )
            return self._rt("ppy_random_below", (self._state(), stop))  # type: ignore[attr-defined,no-any-return]
        if name == "randint":
            start, last = values
            stop = self._checked_binary(last, word(1), "add")  # type: ignore[attr-defined]
            step = word(1)
            spelled = (start, stop)
        else:
            start, stop = values[0], values[1]
            step = values[2] if len(values) == 3 else word(1)
            spelled = (start, stop)
        width = self._checked_binary(stop, start, "sub")  # type: ignore[attr-defined]
        one = core.cmp(b, "eq", step, word(1))
        require(
            core.bitwise(b, "or", core.cmp(b, "ne", step, word(1)), core.cmp(b, "gt", width, word(0))),
            "empty range in randrange()",
            "ValueError: empty range in randrange({0}, {1})",
            spelled,
        )
        if len(values) == 3:
            require(
                core.bitwise(b, "or", one, core.cmp(b, "ne", step, word(0))),
                "zero step for randrange()",
                "ValueError: zero step for randrange()",
            )
            count = self._rt("ppy_random_range_count", (width, step))  # type: ignore[attr-defined]
            require(
                core.bitwise(b, "or", one, core.cmp(b, "gt", count, word(0))),
                "empty range in randrange()",
                "ValueError: empty range in randrange({0}, {1}, {2})",
                (start, stop, step),
            )
            count = core.select(b, one, width, count)
        else:
            count = width
        return self._rt("ppy_random_range_pick", (self._state(), start, count, step))  # type: ignore[attr-defined,no-any-return]

    def _choice(self, population: ast.expr) -> Value:
        kind, handle, owned = self._plain_list(population)
        shape = kind.value
        assert shape is not None
        if owned and shape.reference:
            raise Unsupported("an element chosen from a temporary list outlives it")
        length = self._rt("ppy_coll_len", (handle,))  # type: ignore[attr-defined]
        self._require(  # type: ignore[attr-defined]
            core.cmp(self.b, "gt", length, self._word(0)),  # type: ignore[attr-defined]
            "choice from an empty sequence",
            "IndexError: Cannot choose from an empty sequence",
        )
        at = self._rt("ppy_random_below", (self._state(), length))  # type: ignore[attr-defined]
        found = self._read(self._rt("ppy_seq_at", (handle, at), HANDLE), shape)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return found  # type: ignore[no-any-return]

    def _sample(self, population: ast.expr, count: ast.expr) -> Value:
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        bounds = _range_bounds(population)
        if bounds is None:
            _, handle, owned = self._plain_list(population)
            size = self._rt("ppy_coll_len", (handle,))  # type: ignore[attr-defined]
        else:
            start, stop, step = (self._int_argument(a) if a is not None else None for a in bounds)
            start = start if start is not None else word(0)
            step = step if step is not None else word(1)
            assert stop is not None
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "ne", step, word(0)),
                "range() arg 3 must not be zero",
                "ValueError: range() arg 3 must not be zero",
            )
            size = self._rt("ppy_random_range_len", (start, stop, step))  # type: ignore[attr-defined]
        k = self._int_argument(count)
        self._require(  # type: ignore[attr-defined]
            core.bitwise(b, "and", core.cmp(b, "ge", k, word(0)), core.cmp(b, "le", k, size)),
            "sample larger than population or negative",
            "ValueError: Sample larger than population or is negative",
        )
        positions = self._rt("ppy_random_sample_positions", (self._state(), size, k), HANDLE)  # type: ignore[attr-defined]
        if bounds is not None:
            self._rt("ppy_random_affine", (positions, start, step), None)  # type: ignore[attr-defined]
            return positions
        made = self._rt("ppy_random_gather", (handle, positions), HANDLE)  # type: ignore[attr-defined]
        self._release(positions)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return made  # type: ignore[no-any-return]

    def _choices(self, node: ast.Call) -> Value:
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        require = self._require  # type: ignore[attr-defined]
        keywords = {k.arg: k.value for k in node.keywords}
        if None in keywords:
            raise Unsupported("`**` in a call has no native lowering")
        weights = node.args[1] if len(node.args) > 1 else keywords.get("weights")
        cumulative = keywords.get("cum_weights")
        if weights is not None and cumulative is not None:
            raise Unsupported("`choices` takes weights or cumulative weights")
        # Python evaluates the arguments in order: the population, the
        # weights, then `k`.
        _, handle, owned = self._plain_list(node.args[0])
        given = weights if weights is not None else cumulative
        weighed = None
        if given is not None:
            weight_kind, weighed, weighed_owned = self._plain_list(given)
            integers = weight_kind.value is not None and weight_kind.value.kind == "int"
        k = self._int_argument(keywords["k"]) if "k" in keywords else word(1)
        size = rt("ppy_coll_len", (handle,))
        state = self._state()
        if weighed is None:
            require(
                core.bitwise(b, "or", core.cmp(b, "gt", size, word(0)), core.cmp(b, "le", k, word(0))),
                "choices from an empty population",
                "IndexError: list index out of range",
            )
            positions = rt("ppy_random_choice_positions", (state, size, k), HANDLE)
        else:
            require(
                core.cmp(b, "eq", rt("ppy_coll_len", (weighed,)), size),
                "weights do not match the population",
                "ValueError: The number of weights does not match the population",
            )
            require(
                core.cmp(b, "gt", size, word(0)),
                "choices from empty weights",
                "IndexError: list index out of range",
            )
            flags = (word(int(cumulative is not None)), word(int(integers)))
            total = rt("ppy_random_weights_total", (weighed, *flags), F64)
            fault = rt("ppy_random_total_fault", (total,))
            require(
                core.cmp(b, "ne", fault, word(1)),
                "weights total not above zero",
                "ValueError: Total of weights must be greater than zero",
            )
            require(
                core.cmp(b, "ne", fault, word(2)),
                "weights total not finite",
                "ValueError: Total of weights must be finite",
            )
            positions = rt("ppy_random_weighted_positions", (state, weighed, *flags, k, total), HANDLE)
            self._done_with(weighed, weighed_owned)  # type: ignore[attr-defined]
        made = rt("ppy_random_gather", (handle, positions), HANDLE)
        self._release(positions)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return made  # type: ignore[no-any-return]

    def _distribution(self, name: str, args: list[ast.expr]) -> Value:
        """The float distributions, over `random()` as `random.py` has them."""
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        if len(args) > _FLOAT_ARGUMENTS.get(name, 0):
            raise Unsupported(f"`random.{name}` with {len(args)} arguments")
        if name == "uniform":
            low, high = (self._expr(a) for a in args)  # type: ignore[attr-defined]
            if low.type == I64 and high.type == I64:
                # `b - a` is an integer's difference, then a float.
                span = self._coerce(self._checked_binary(high, low, "sub"), "float")  # type: ignore[attr-defined]
            else:
                span = core.sub(b, self._coerce(high, "float"), self._coerce(low, "float"))  # type: ignore[attr-defined]
            drawn = rt("ppy_random_double", (self._state(),), F64)
            return core.add(b, self._coerce(low, "float"), core.mul(b, span, drawn))  # type: ignore[attr-defined]
        floats = [self._float_argument(a) for a in args]
        if name == "triangular":
            defaults = [self._float(0.0), self._float(1.0), self._float(0.0)]
            given = floats + defaults[len(floats) :]
            has_mode = self._word(int(len(floats) == 3))  # type: ignore[attr-defined]
            return rt("ppy_random_triangular", (self._state(), *given, has_mode), F64)  # type: ignore[no-any-return]
        if name in {"gauss", "normalvariate", "lognormvariate"}:
            mu, sigma = floats if floats else (self._float(0.0), self._float(1.0))
            if name == "gauss":
                if not self.frontend.standalone:  # type: ignore[attr-defined]
                    raise Unsupported(
                        "`random.gauss` holds a value back in Python's generator under `ppy run`"
                    )
                return rt("ppy_random_gauss", (self._state(), mu, sigma), F64)  # type: ignore[no-any-return]
            drawn = rt("ppy_random_normal", (self._state(), mu, sigma), F64)
            if name == "lognormvariate":
                return self._math("exp", drawn)
            return drawn  # type: ignore[no-any-return]
        divisor = floats[-1] if floats else self._float(1.0)
        self._require(  # type: ignore[attr-defined]
            core.cmp(b, "ne", divisor, self._float(0.0)),
            "float division by zero",
            "ZeroDivisionError: float division by zero",
        )
        if name == "expovariate":
            return rt("ppy_random_expo", (self._state(), divisor), F64)  # type: ignore[no-any-return]
        if name == "paretovariate":
            drawn = rt("ppy_random_pareto", (self._state(), divisor), F64)
        else:
            drawn = rt("ppy_random_weibull", (self._state(), floats[0], divisor), F64)
        # A power past the doubles, or zero to a negative power, is an
        # exception in Python and not a number here.
        magnitude = self._math("abs", drawn)
        self._guard(  # type: ignore[attr-defined]
            core.cmp(b, "lt", magnitude, self._float(float("inf"))),
            "range",
            f"`random.{name}` past the doubles",
        )
        return drawn  # type: ignore[no-any-return]

    # -- heapq ---------------------------------------------------------------------

    def _heapq_call(self, name: str, node: ast.Call) -> Value:
        args = node.args
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        if name in {"nlargest", "nsmallest"}:
            count = self._int_argument(args[0])
            _, handle, owned = self._plain_list(args[1])
            made = rt("ppy_heapq_select", (handle, count, word(int(name == "nlargest"))), HANDLE)
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        kind, handle, owned = self._plain_list(args[0])
        shape = kind.value
        assert shape is not None
        if owned:
            raise Unsupported("a heap is a list with a name")
        if name == "heapify":
            rt("ppy_heapq_heapify", (handle,), None)
            return word(0)
        if name == "heappush":
            self._add_node(kind, handle, args[1])  # type: ignore[attr-defined]
            rt("ppy_heapq_push", (handle,), None)
            return word(0)
        if shape.reference or shape.handles:
            raise Unsupported("an element taken from a heap of references")
        if name == "heappop":
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "gt", rt("ppy_coll_len", (handle,)), word(0)),
                "heappop from an empty heap",
                "IndexError: index out of range",
            )
            return self._read(rt("ppy_heapq_pop", (handle,), HANDLE), shape)  # type: ignore[attr-defined,no-any-return]
        if name == "heapreplace":
            # The item is evaluated before the check, as Python passes it.
            value, _ = self._value(args[1], shape)  # type: ignore[attr-defined]
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "gt", rt("ppy_coll_len", (handle,)), word(0)),
                "heapreplace on an empty heap",
                "IndexError: index out of range",
            )
            self._write(rt("ppy_seq_push_back", (handle,), HANDLE), shape, value)  # type: ignore[attr-defined]
            return self._read(rt("ppy_heapq_replace", (handle,), HANDLE), shape)  # type: ignore[attr-defined,no-any-return]
        self._add_node(kind, handle, args[1])  # type: ignore[attr-defined]
        return self._read(rt("ppy_heapq_pushpop", (handle,), HANDLE), shape)  # type: ignore[attr-defined,no-any-return]

    # -- bisect --------------------------------------------------------------------

    def _bisect_call(self, name: str, node: ast.Call) -> Value:
        args = node.args
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        kind, handle, owned = self._plain_list(args[0])
        shape = kind.value
        assert shape is not None
        inserting = name.startswith("insort")
        if inserting and owned:
            raise Unsupported("`insort` into a temporary list")
        value, value_owned = self._value(args[1], shape)  # type: ignore[attr-defined]
        buffer = self._alloca(shape.ir_type(), "value")  # type: ignore[attr-defined]
        address = core.cast(b, buffer, _pointer(buffer, HANDLE.pointee))
        self._write(address, shape, value)  # type: ignore[attr-defined]
        lo = self._int_argument(args[2]) if len(args) > 2 else word(0)
        self._require(  # type: ignore[attr-defined]
            core.cmp(b, "ge", lo, word(0)),
            "lo must be non-negative",
            "ValueError: lo must be non-negative",
        )
        length = rt("ppy_coll_len", (handle,))
        if len(args) > 3:
            hi = self._int_argument(args[3])
            # Past the end, CPython reads an element that is not there.
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "le", hi, length),
                "bisect past the end of the list",
                "IndexError: list index out of range",
            )
        else:
            hi = length
        right = word(int(not name.endswith("_left")))
        position = rt("ppy_bisect", (handle, address, lo, hi, right))
        if not inserting:
            if value_owned:
                self._release(value)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return position  # type: ignore[no-any-return]
        slot = rt("ppy_seq_insert", (handle, position), HANDLE)
        self._put_value(slot, shape, value, value_owned, True)  # type: ignore[attr-defined]
        return word(0)  # type: ignore[no-any-return]


def _range_bounds(node: ast.expr) -> tuple[ast.expr | None, ast.expr, ast.expr | None] | None:
    """`range(stop)`, `range(start, stop)`, `range(start, stop, step)` spelled
    out: its start, stop, and step; None for anything else."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
        and 1 <= len(node.args) <= 3
    ):
        return None
    if len(node.args) == 1:
        return None, node.args[0], None
    step = node.args[2] if len(node.args) == 3 else None
    return node.args[0], node.args[1], step

