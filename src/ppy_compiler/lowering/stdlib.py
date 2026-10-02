"""`random`, `heapq`, `bisect`, and `math` in native code.

The calls `analysis.native_stdlib` answers for, lowered over the runtime in
`random.c` and `stdlib.c`: CPython's Mersenne Twister and `random.py`'s
algorithms on top of it, draw for draw; `_heapq`'s and `_bisect`'s
comparisons, one for one; `math`'s integer functions, `fsum`, and `hypot`
as `mathmodule.c` computes them, and libm's functions with the checks it
makes of what they give. Under `ppy run` the generator's state is the one
inside `random._inst`, so Python and native draws interleave as they would
in CPython; what CPython raises for is a guard, with its message.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from functools import cache

from ..analysis import types as T
from ..analysis.lexical import LexicalBindings
from ..analysis.native_stdlib import MATH_NATIVE, MODELS, STRING_CONSTANTS
from ..backend.llvm.lowering import _MATH_INTRINSICS, Unsupported
from ..ir import BOOL, F64, I64, Value
from ..ir.dialects import core
from ..ir.dialects import math as math_dialect
from ..ir.raising import OVERFLOW, said
from .collections import HANDLE, Kind, _pointer, kind_of
from .library import LibraryLowering
from .memo import MemoLowering

__all__ = ["StdlibLowering"]

#: `collections.deque`'s methods, as the runtime's `Deque` names them.
_DEQUE_METHODS = {
    "append": "push_back",
    "appendleft": "push_front",
    "pop": "pop_back",
    "popleft": "pop_front",
    "extend": "extend",
    "extendleft": "extendleft",
    "rotate": "rotate",
    "clear": "clear",
    "copy": "copy",
}

#: `math`'s constants.
_CONSTANTS = {
    "math.pi": math.pi,
    "math.e": math.e,
    "math.tau": math.tau,
    "math.inf": math.inf,
    "math.nan": math.nan,
}
_CONSTANT_NAMES = frozenset(q.rpartition(".")[2] for q in _CONSTANTS)

#: `ppy_math_unary`'s functions, by their number there.
_UNARY = (
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "log1p", "expm1", "erf", "erfc", "cbrt", "exp2", "fabs", "tan",
)  # fmt: skip

#: `ppy_math_binary`'s, likewise.
_BINARY = {"atan2": 0, "copysign": 1, "fmod": 2}

#: The ones whose infinity from a finite number is an `OverflowError`, not a
#: `ValueError` (CPython's `can_overflow`).
_OVERFLOWS = frozenset({"sinh", "cosh", "exp2", "expm1"})

_FLOAT_ARGUMENTS = {
    "uniform": 2,
    "triangular": 3,
    "gauss": 2,
    "normalvariate": 2,
    "lognormvariate": 2,
    "expovariate": 1,
    "paretovariate": 1,
    "weibullvariate": 2,
    "gammavariate": 2,
    "betavariate": 2,
}


class StdlibLowering(LibraryLowering, MemoLowering):
    """The standard library's calls; mixed into `_FunctionLowering`."""

    def _make_collection(self, name: str, value: ast.expr, declared: T.Type | None = None) -> bool:
        if isinstance(value, ast.Call) and self._stdlib_target_any(value) == "collections.deque":
            kind = kind_of(declared, self._records()) if declared is not None else None  # type: ignore[attr-defined]
            kind = kind or self._kind_of(value)
            if kind is None:
                # `r = deque([1, 2])`: the name's type says what the call's may not.
                analysis = self.frontend.analysis.functions.get(self.info.qualname)  # type: ignore[attr-defined]
                final = analysis.locals.get(name) if analysis is not None else None
                kind = kind_of(final, self._records()) if final is not None else None  # type: ignore[attr-defined]
            if not isinstance(kind, Kind) or kind.name != "Deque":
                raise Unsupported(
                    "a `deque` natively holds what its annotation says: `q: deque[int] = deque()`"
                )
            made = self._made(kind, value)
            assert made is not None
            self._bind(name, kind, made, owned=True)  # type: ignore[attr-defined]
            return True
        return super()._make_collection(name, value, declared)  # type: ignore[misc,no-any-return]

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        if isinstance(node, ast.Call) and self._stdlib_target_any(node) == "collections.deque":
            kind = self._kind_of(node)
            if kind is not None and kind.name == "Deque":
                made = self._made(kind, node)
                if made is not None:
                    return made, True
        if isinstance(node, ast.Name):
            constant = self._string_handle(node)
            if constant is not None:
                return constant
        return super()._handle(node)  # type: ignore[misc,no-any-return]

    def _string_handle(self, node: ast.expr) -> tuple[Value, bool] | None:
        """`string.ascii_lowercase` and its kin: a literal, borrowed."""
        if isinstance(node, (ast.Attribute, ast.Name)):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            if f"string.{name}" in STRING_CONSTANTS:
                lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
                if isinstance(lexical, LexicalBindings):
                    found = lexical.targets_at(node)
                    if len(found) == 1 and next(iter(found)) in STRING_CONSTANTS:
                        text = STRING_CONSTANTS[next(iter(found))]
                        return self._string_literal(text), False  # type: ignore[attr-defined]
        return super()._string_handle(node)  # type: ignore[misc,no-any-return]

    def _stdlib_constant(self, node: ast.Attribute | ast.Name) -> Value | None:
        """`math.pi`, `from math import inf`, and the rest of `math`'s constants."""
        if isinstance(node, ast.Attribute) and node.attr not in _CONSTANT_NAMES:
            return None
        if isinstance(node, ast.Name) and node.id not in _CONSTANT_NAMES:
            return None
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings):
            return None
        found = lexical.targets_at(node)
        if len(found) != 1:
            return None
        value = _CONSTANTS.get(next(iter(found)))
        return self._float(value) if value is not None else None

    def _kind_of(self, node: ast.expr) -> Kind | None:
        """An `itertools` iterator is the list it makes where it is consumed."""
        if isinstance(node, ast.Call):
            qualname = self._stdlib_target(node)
            if qualname is not None and qualname.startswith("itertools."):
                made = self._type_of(node)  # type: ignore[attr-defined]
                if isinstance(made, T.Instance) and made.name == "Iterator" and made.args:
                    found = kind_of(T.list_of(made.args[0]), self._records())  # type: ignore[attr-defined]
                    return found if isinstance(found, Kind) else None
        return super()._kind_of(node)  # type: ignore[misc,no-any-return]

    # -- collections.deque ------------------------------------------------------------

    def _is_pydeque(self, node: ast.expr) -> bool:
        found = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        return isinstance(found, T.Instance) and found.name == "collections.deque"

    def _made(self, kind: Kind, node: ast.expr) -> Value | None:
        """`deque()` and `deque(xs)`: a runtime `Deque`, filled in order."""
        if (
            kind.name == "Deque"
            and isinstance(node, ast.Call)
            and self._stdlib_target_any(node) == "collections.deque"
        ):
            if node.keywords or len(node.args) > 1:
                raise Unsupported("`deque` with `maxlen` has no native lowering")
            made = self._new(kind)  # type: ignore[attr-defined]
            if node.args:
                self._fill(kind, made, node.args[0])  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        return super()._made(kind, node)  # type: ignore[misc,no-any-return]

    def _collection_method(self, receiver: ast.expr, attr: str, node: ast.Call) -> Value:
        """A `deque`'s methods by their Python names, onto the runtime's `Deque`."""
        if self._is_pydeque(receiver):
            renamed = _DEQUE_METHODS.get(attr)
            if renamed is None:
                raise Unsupported(f"`deque.{attr}` has no native lowering")
            if attr in {"pop", "popleft"}:
                _, handle, owned = self._receiver(receiver)  # type: ignore[attr-defined]
                self._require(  # type: ignore[attr-defined]
                    core.cmp(
                        self.b,  # type: ignore[attr-defined]
                        "gt",
                        self._rt("ppy_coll_len", (handle,)),  # type: ignore[attr-defined]
                        self._word(0),  # type: ignore[attr-defined]
                    ),
                    "pop from an empty deque",
                    _text("deque_pop"),
                )
                self._done_with(handle, owned)  # type: ignore[attr-defined]
            attr = renamed
        return super()._collection_method(receiver, attr, node)  # type: ignore[misc,no-any-return]

    def _item(self, container: ast.expr, index: ast.expr, value: ast.expr | None = None) -> Value:
        """`q[i]` of a `deque`, counted from either end as CPython counts it."""
        if isinstance(index, ast.Slice) or not self._is_pydeque(container):
            return super()._item(container, index, value)  # type: ignore[misc,no-any-return]
        kind, handle, owned = self._receiver(container)  # type: ignore[attr-defined]
        shape = kind.value
        assert shape is not None
        if owned and shape.reference:
            raise Unsupported("an element read from a temporary deque outlives it")
        stored = self._value(value, shape) if value is not None else None  # type: ignore[attr-defined]
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        position = self._list_position(handle, self._int_argument(index))  # type: ignore[attr-defined]
        length = self._rt("ppy_coll_len", (handle,))  # type: ignore[attr-defined]
        inside = core.bitwise(
            b, "and", core.cmp(b, "ge", position, word(0)), core.cmp(b, "lt", position, length)
        )
        self._require(inside, "deque index out of range", _text("deque_index"))  # type: ignore[attr-defined]
        address = self._rt("ppy_seq_at", (handle, position), HANDLE)  # type: ignore[attr-defined]
        if stored is not None:
            self._put_value(address, shape, stored[0], stored[1], fresh=False)  # type: ignore[attr-defined]
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return word(0)  # type: ignore[no-any-return]
        found = self._read(address, shape)  # type: ignore[attr-defined]
        self._done_with(handle, owned)  # type: ignore[attr-defined]
        return found  # type: ignore[no-any-return]

    def _stdlib_target_any(self, node: ast.Call) -> str | None:
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings):
            return None
        found = lexical.targets_at(node.func)
        return next(iter(found)) if len(found) == 1 else None

    def _stdlib_target(self, node: ast.Call) -> str | None:
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings):
            return None
        found = lexical.targets_at(node.func)
        if len(found) != 1:
            return None
        qualname = next(iter(found))
        if qualname.startswith("math."):
            name = qualname.removeprefix("math.")
            return qualname if name in MATH_NATIVE or name in _MATH_INTRINSICS else None
        return qualname if qualname in MODELS else None

    def _stdlib_call(self, node: ast.Call, discard_result: bool) -> Value | None:
        """A call to one of the modules here, lowered; None where it is not one."""
        qualname = self._stdlib_target(node)
        if qualname is None:
            return None
        module, _, name = qualname.partition(".")
        if module == "math":
            if self.device or self.info.directive("xla.jit") is not None:  # type: ignore[attr-defined]
                # A device and XLA have their own math: the instructions only.
                return self._math_call(name, node)  # type: ignore[attr-defined,no-any-return]
            made = self._math_native(name, node) if name in MATH_NATIVE else None
            # `from math import sqrt` reaches the instructions `math.sqrt` does.
            return made if made is not None else self._math_call(name, node)  # type: ignore[attr-defined]
        if module == "itertools":
            return self._itertools_call(name, node)
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
                _text("bits"),
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
                _text("randrange1"),
            )
            return self._rt("ppy_random_below", (self._state(), stop))  # type: ignore[attr-defined,no-any-return]
        if name == "randint":
            start, last = values
            stop = self._checked_binary(last, word(1), "add")  # type: ignore[attr-defined]
            step = word(1)
            spelled = (start, stop if _randint_shows_stop() else last)
            empty = _text("randint")
        else:
            start, stop = values[0], values[1]
            step = values[2] if len(values) == 3 else word(1)
            spelled = (start, stop)
            empty = _text("randrange2")
        width = self._checked_binary(stop, start, "sub")  # type: ignore[attr-defined]
        one = core.cmp(b, "eq", step, word(1))
        require(
            core.bitwise(
                b, "or", core.cmp(b, "ne", step, word(1)), core.cmp(b, "gt", width, word(0))
            ),
            "empty range in randrange()",
            empty,
            spelled,
        )
        if len(values) == 3:
            require(
                core.bitwise(b, "or", one, core.cmp(b, "ne", step, word(0))),
                "zero step for randrange()",
                _text("zero_step"),
            )
            count = self._rt("ppy_random_range_count", (width, step))  # type: ignore[attr-defined]
            require(
                core.bitwise(b, "or", one, core.cmp(b, "gt", count, word(0))),
                "empty range in randrange()",
                _text("randrange3"),
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
            _text("choice"),
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
                _text("range_step"),
            )
            size = self._rt("ppy_random_range_len", (start, stop, step))  # type: ignore[attr-defined]
        k = self._int_argument(count)
        self._require(  # type: ignore[attr-defined]
            core.bitwise(b, "and", core.cmp(b, "ge", k, word(0)), core.cmp(b, "le", k, size)),
            "sample larger than population or negative",
            _text("sample"),
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
        weighed_owned = integers = False
        if given is not None:
            weight_kind, weighed, weighed_owned = self._plain_list(given)
            integers = weight_kind.value is not None and weight_kind.value.kind == "int"
        k = self._int_argument(keywords["k"]) if "k" in keywords else word(1)
        size = rt("ppy_coll_len", (handle,))
        state = self._state()
        if weighed is None:
            require(
                core.bitwise(
                    b, "or", core.cmp(b, "gt", size, word(0)), core.cmp(b, "le", k, word(0))
                ),
                "choices from an empty population",
                _text("choices_empty"),
            )
            positions = rt("ppy_random_choice_positions", (state, size, k), HANDLE)
        else:
            require(
                core.cmp(b, "eq", rt("ppy_coll_len", (weighed,)), size),
                "weights do not match the population",
                _text("weights_count"),
            )
            require(
                core.cmp(b, "gt", size, word(0)),
                "choices from empty weights",
                _text("weights_empty"),
            )
            flags = (word(int(cumulative is not None)), word(int(integers)))
            total = rt("ppy_random_weights_total", (weighed, *flags), F64)
            fault = rt("ppy_random_total_fault", (total,))
            require(
                core.cmp(b, "ne", fault, word(1)),
                "weights total not above zero",
                _text("weights_zero"),
            )
            require(
                core.cmp(b, "ne", fault, word(2)),
                "weights total not finite",
                _text("weights_infinite"),
            )
            positions = rt(
                "ppy_random_weighted_positions", (state, weighed, *flags, k, total), HANDLE
            )
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
            mu, sigma = floats or (self._float(0.0), self._float(1.0))
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
        if name in {"gammavariate", "betavariate"}:
            alpha, beta = floats
            zero = self._float(0.0)
            first = core.cmp(b, "gt", alpha, zero)
            if name == "gammavariate":
                first = core.bitwise(b, "and", first, core.cmp(b, "gt", beta, zero))
            self._require(  # type: ignore[attr-defined]
                first, "gammavariate: alpha and beta must be > 0.0", _text("gamma")
            )
            if name == "gammavariate":
                return rt("ppy_random_gamma", (self._state(), alpha, beta), F64)  # type: ignore[no-any-return]
            drawn = rt("ppy_random_beta", (self._state(), alpha, beta), F64)
            # The second draw's check comes after the first draw, as in Python.
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "eq", drawn, drawn),
                "gammavariate: alpha and beta must be > 0.0",
                _text("gamma"),
            )
            return drawn  # type: ignore[no-any-return]
        # Each of these draws before it divides, as Python does.
        divisor = floats[-1] if floats else self._float(1.0)
        if name == "expovariate":
            drawn = rt("ppy_random_expo", (self._state(), divisor), F64)
        elif name == "paretovariate":
            drawn = rt("ppy_random_pareto", (self._state(), divisor), F64)
        else:
            drawn = rt("ppy_random_weibull", (self._state(), floats[0], divisor), F64)
        self._require(  # type: ignore[attr-defined]
            core.cmp(b, "ne", divisor, self._float(0.0)),
            "float division by zero",
            _text("divide"),
        )
        if name == "expovariate":
            return drawn  # type: ignore[no-any-return]
        # A power past the doubles, or zero to a negative power, is an
        # exception in Python and not a number here.
        magnitude = self._math("abs", drawn)
        self._guard(  # type: ignore[attr-defined]
            core.cmp(b, "lt", magnitude, self._float(float("inf"))),
            "range",
            f"`random.{name}` past the doubles",
        )
        return drawn  # type: ignore[no-any-return]

    # -- math ----------------------------------------------------------------------

    def _math_native(self, name: str, node: ast.Call) -> Value | None:
        """`math`'s integer functions, its sums and norms, and libm's functions
        with the checks CPython's `math_1` makes of what they give."""
        args = node.args
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        require = self._require  # type: ignore[attr-defined]
        if name == "fabs" or (name == "log" and len(args) != 2):
            return None
        if any(isinstance(a, ast.Starred) for a in args):
            raise Unsupported(f"`math.{name}` with `*` arguments has no native lowering")
        if node.keywords and name != "isclose":
            raise Unsupported(f"`math.{name}` with keywords has no native lowering")
        if name in {"gcd", "lcm"}:
            values = [self._int_argument(a) for a in args]
            if not values:
                return word(0 if name == "gcd" else 1)
            # One argument is its magnitude for both: `gcd(a, 0)`.
            made = rt("ppy_math_gcd", (values[0], word(0)))
            for value in values[1:]:
                made = rt(f"ppy_math_{name}", (made, value))
            self._past_word(core.cmp(b, "ge", made, word(0)), name)
            return made  # type: ignore[no-any-return]
        if name in {"isqrt", "factorial", "comb", "perm"}:
            return self._math_integer(name, [self._int_argument(a) for a in args])
        if name in {"prod", "fsum"}:
            kind, handle, owned = self._plain_list(args[0])
            integers = kind.value is not None and kind.value.kind == "int"
            if name == "prod" and integers:
                made = rt("ppy_math_prod_ints", (handle,))
                self._past_word(core.cmp(b, "eq", rt("ppy_math_fault", ()), word(0)), name)
            elif name == "prod":
                made = rt("ppy_math_prod_floats", (handle,), F64)
            else:
                made = rt("ppy_math_fsum", (handle, word(int(integers))), F64)
                fault = rt("ppy_math_fault", ())
                require(core.cmp(b, "ne", fault, word(1)), "fsum overflow", _text("fsum_overflow"))
                require(core.cmp(b, "ne", fault, word(2)), "fsum -inf + inf", _text("fsum_nan"))
            self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        if name == "isclose":
            return self._isclose(node)
        if name == "hypot":
            floats = [self._float_argument(a) for a in args]
            return rt(f"ppy_math_hypot{len(floats)}", tuple(floats), F64)  # type: ignore[no-any-return]
        if name == "dist":
            return self._dist(args[0], args[1])
        if name in {"degrees", "radians"}:
            factor = 180.0 / math.pi if name == "degrees" else math.pi / 180.0
            return core.mul(b, self._float(factor), self._float_argument(args[0]))
        floats = [self._float_argument(a) for a in args]
        if name == "log":
            x, base = floats
            zero = self._float(0.0)
            # `x <= 0` is false for a NaN, which `log` gives back.
            positive = core.bitwise(b, "xor", core.cmp(b, "le", x, zero), core.const(b, True, BOOL))
            require(positive, "log of a number not above zero", _text("log_domain"))
            positive = core.bitwise(
                b, "xor", core.cmp(b, "le", base, zero), core.const(b, True, BOOL)
            )
            require(positive, "log of a number not above zero", _text("log_domain"))
            top = self._math("log", x)
            bottom = self._math("log", base)
            require(core.cmp(b, "ne", bottom, zero), "log base 1", _text("log_base"))
            return core.div(b, top, bottom)  # type: ignore[no-any-return]
        if name in _BINARY:
            made = rt("ppy_math_binary", (word(_BINARY[name]), *floats), F64)
        else:
            made = rt("ppy_math_unary", (word(_UNARY.index(name)), *floats), F64)
        self._math_checked(made, floats, overflows=name in _OVERFLOWS)
        return made  # type: ignore[no-any-return]

    def _math_checked(
        self,
        made: Value,
        inputs: list[Value],
        *,
        overflows: bool,
        zero_base: Value | None = None,
    ) -> None:
        """What CPython's `math_1` and `math_2` raise for: a NaN from numbers,
        and an infinity from finite numbers (a `ValueError` for `pow` of a
        zero `zero_base`, which is a division by zero there)."""
        b = self.b  # type: ignore[attr-defined]
        yes = core.const(b, True, BOOL)
        inf = self._float(math.inf)
        some_nan = core.const(b, False, BOOL)
        all_finite = yes
        for value in inputs:
            some_nan = core.bitwise(b, "or", some_nan, core.cmp(b, "ne", value, value))
            finite = core.cmp(b, "lt", self._math("abs", value), inf)
            all_finite = core.bitwise(b, "and", all_finite, finite)
        is_nan = core.cmp(b, "ne", made, made)
        fine = core.bitwise(b, "or", core.bitwise(b, "xor", is_nan, yes), some_nan)
        self._require(fine, "math domain error", _text("domain"))  # type: ignore[attr-defined]
        infinite = core.cmp(b, "eq", self._math("abs", made), inf)
        blown = core.bitwise(b, "and", infinite, all_finite)
        if zero_base is not None:
            zero = core.cmp(b, "eq", zero_base, self._float(0.0))
            self._require(  # type: ignore[attr-defined]
                core.bitwise(b, "xor", core.bitwise(b, "and", blown, zero), yes),
                "math domain error",
                _text("domain"),
            )
        self._require(  # type: ignore[attr-defined]
            core.bitwise(b, "xor", blown, yes),
            "math range error",
            _text("range") if overflows else _text("domain"),
        )

    def _past_word(self, condition: Value, name: str) -> None:
        self._guard(  # type: ignore[attr-defined]
            condition, "range", f"`math.{name}` past a word", raises=OVERFLOW
        )

    def _math_integer(self, name: str, values: list[Value]) -> Value:
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        require = self._require  # type: ignore[attr-defined]
        n = values[0]
        if name == "isqrt":
            require(core.cmp(b, "ge", n, word(0)), "isqrt of a negative", _text("isqrt"))
            return rt("ppy_math_isqrt", (n,))  # type: ignore[no-any-return]
        if name == "factorial" or (name == "perm" and len(values) == 1):
            require(core.cmp(b, "ge", n, word(0)), "factorial of a negative", _text("factorial"))
            made = rt("ppy_math_perm", (n, n))
        else:
            k = values[1]
            require(core.cmp(b, "ge", n, word(0)), "n negative", _text(f"{name}_n"))
            require(core.cmp(b, "ge", k, word(0)), "k negative", _text(f"{name}_k"))
            made = rt(f"ppy_math_{name}", (n, k))
        self._past_word(core.cmp(b, "ge", made, word(0)), name)
        return made  # type: ignore[no-any-return]

    def _isclose(self, node: ast.Call) -> Value:
        b = self.b  # type: ignore[attr-defined]
        a, other = (self._float_argument(x) for x in node.args)
        keywords = {k.arg: k.value for k in node.keywords}
        relative = (
            self._float_argument(keywords["rel_tol"])
            if "rel_tol" in keywords
            else self._float(1e-09)
        )
        absolute = (
            self._float_argument(keywords["abs_tol"]) if "abs_tol" in keywords else self._float(0.0)
        )
        zero = self._float(0.0)
        # A NaN tolerance is not below zero, as in CPython.
        below = core.bitwise(
            b, "or", core.cmp(b, "lt", relative, zero), core.cmp(b, "lt", absolute, zero)
        )
        self._require(  # type: ignore[attr-defined]
            core.bitwise(b, "xor", below, core.const(b, True, BOOL)),
            "tolerances must be non-negative",
            _text("isclose"),
        )
        found = self._rt("ppy_math_isclose", (a, other, relative, absolute))  # type: ignore[attr-defined]
        return core.cmp(b, "ne", found, self._word(0))  # type: ignore[attr-defined]

    def _dist(self, p: ast.expr, q: ast.expr) -> Value:
        b = self.b  # type: ignore[attr-defined]
        if self._builtin_of(p) is not None:  # type: ignore[attr-defined]
            _, left, left_owned = self._plain_list(p)
            _, right, right_owned = self._plain_list(q)
            self._require(  # type: ignore[attr-defined]
                core.cmp(
                    b,
                    "eq",
                    self._rt("ppy_coll_len", (left,)),  # type: ignore[attr-defined]
                    self._rt("ppy_coll_len", (right,)),  # type: ignore[attr-defined]
                ),
                "both points must have the same number of dimensions",
                _text("dist"),
            )
            made = self._rt("ppy_math_hypot_list", (left, right), F64)  # type: ignore[attr-defined]
            self._done_with(left, left_owned)  # type: ignore[attr-defined]
            self._done_with(right, right_owned)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        left, right = self._coordinates(p), self._coordinates(q)
        differences = tuple(core.sub(b, x, y) for x, y in zip(left, right, strict=True))
        return self._rt(f"ppy_math_hypot{len(differences)}", differences, F64)  # type: ignore[attr-defined,no-any-return]

    def _coordinates(self, node: ast.expr) -> list[Value]:
        """A point's coordinates as floats: a tuple display's elements, or a
        tuple value's items."""
        if isinstance(node, ast.Tuple):
            return [self._float_argument(e) for e in node.elts]
        value = self._expr(node)  # type: ignore[attr-defined]
        width = len(value.type.items)  # type: ignore[attr-defined]
        return [
            self._coerce(core.tuple_extract(self.b, value, i), "float")  # type: ignore[attr-defined]
            for i in range(width)
        ]

    # -- itertools -----------------------------------------------------------------

    def _walked_list(self, node: ast.expr) -> tuple[Value, bool]:
        """A list argument's handle, or a `range(...)` made into a list."""
        bounds = _range_bounds(node)
        if bounds is None:
            _, handle, owned = self._plain_list(node)
            return handle, owned
        word = self._word  # type: ignore[attr-defined]
        start, stop, step = (self._int_argument(a) if a is not None else None for a in bounds)
        start = start if start is not None else word(0)
        step = step if step is not None else word(1)
        self._require(  # type: ignore[attr-defined]
            core.cmp(self.b, "ne", step, word(0)),  # type: ignore[attr-defined]
            "range() arg 3 must not be zero",
            _text("range_step"),
        )
        made = self._new(kind_of(T.list_of(T.INT), {}))  # type: ignore[attr-defined]
        self._rt("ppy_iter_range", (made, start, stop, step), None)  # type: ignore[attr-defined]
        return made, True

    def _itertools_call(self, name: str, node: ast.Call) -> Value:
        """The iterator as the list of what it hands out, in CPython's order."""
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        kind = self._kind_of(node)
        if kind is None or kind.value is None:
            raise Unsupported(f"`itertools.{name}` of these arguments has no native lowering")
        args = node.args
        if name == "repeat":
            value, owned = self._value(args[0], kind.value)  # type: ignore[attr-defined]
            count = self._int_argument(args[1])
            buffer = self._alloca(kind.value.ir_type(), "value")  # type: ignore[attr-defined]
            address = core.cast(b, buffer, _pointer(buffer, HANDLE.pointee))
            self._write(address, kind.value, value)  # type: ignore[attr-defined]
            made = self._new(kind)  # type: ignore[attr-defined]
            rt("ppy_iter_repeat", (made, address, count), None)
            if owned:
                self._release(value)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        if name in {"product", "chain"}:
            sources = [self._walked_list(a) for a in args]
            made = self._new(kind)  # type: ignore[attr-defined]
            handles = [handle for handle, _ in sources]
            handles += [rt("ppy_coll_none", (), HANDLE)] * (4 - len(handles))
            rt(f"ppy_iter_{name}", (made, word(len(sources)), *handles), None)
            for handle, owned in sources:
                self._done_with(handle, owned)  # type: ignore[attr-defined]
            return made  # type: ignore[no-any-return]
        source, source_owned = self._walked_list(args[0])
        rest = [self._int_argument(a) for a in args[1:]]
        initial = None
        for keyword in node.keywords:
            initial = self._expr(keyword.value)  # type: ignore[attr-defined]
        made = self._new(kind)  # type: ignore[attr-defined]
        if name in {"permutations", "combinations", "combinations_with_replacement"}:
            choose = {"permutations": 0, "combinations": 1}.get(name, 2)
            rt("ppy_iter_choose", (made, source, rest[0], word(choose)), None)
            self._past_word(core.cmp(b, "eq", rt("ppy_math_fault", ()), word(0)), name)
        elif name == "pairwise":
            rt("ppy_iter_pairwise", (made, source), None)
        elif name == "islice":
            start, stop, step = self._islice_bounds(rest)
            rt("ppy_iter_islice", (made, source, start, stop, step), None)
        else:
            floats = kind.value.kind == "float"
            whole, part = word(0), self._float(0.0)
            if initial is not None and floats:
                part = self._coerce(initial, "float")  # type: ignore[attr-defined]
            elif initial is not None:
                whole = self._coerce(initial, "int")  # type: ignore[attr-defined]
            flags = (word(int(floats)), word(0), word(int(initial is not None)))
            rt("ppy_iter_accumulate", (made, source, *flags, whole, part), None)
            self._past_word(core.cmp(b, "eq", rt("ppy_math_fault", ()), word(0)), name)
        self._done_with(source, source_owned)  # type: ignore[attr-defined]
        return made  # type: ignore[no-any-return]

    def _islice_bounds(self, given: list[Value]) -> tuple[Value, Value, Value]:
        """`islice(xs, stop)` and `islice(xs, start, stop[, step])`, with CPython's checks."""
        b = self.b  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        if len(given) == 1:
            start, stop, step = word(0), given[0], word(1)
        else:
            start, stop = given[0], given[1]
            step = given[2] if len(given) == 3 else word(1)
        negative = core.bitwise(
            b, "and", core.cmp(b, "ge", start, word(0)), core.cmp(b, "ge", stop, word(0))
        )
        self._require(  # type: ignore[attr-defined]
            negative,
            "islice indices must be non-negative",
            _text("islice_stop" if len(given) == 1 else "islice_indices"),
        )
        self._require(  # type: ignore[attr-defined]
            core.cmp(b, "gt", step, word(0)), "islice step must be positive", _text("islice_step")
        )
        return start, stop, step

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
                _text("heap_empty"),
            )
            return self._read(rt("ppy_heapq_pop", (handle,), HANDLE), shape)  # type: ignore[attr-defined,no-any-return]
        if name == "heapreplace":
            # The item is evaluated before the check, as Python passes it.
            value, _ = self._value(args[1], shape)  # type: ignore[attr-defined]
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "gt", rt("ppy_coll_len", (handle,)), word(0)),
                "heapreplace on an empty heap",
                _text("heap_empty"),
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
            _text("lo"),
        )
        length = rt("ppy_coll_len", (handle,))
        if len(args) > 3:
            hi = self._int_argument(args[3])
            # Past the end, CPython reads an element that is not there.
            self._require(  # type: ignore[attr-defined]
                core.cmp(b, "le", hi, length),
                "bisect past the end of the list",
                _text("hi"),
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


#: What CPython raises for each check here, asked of the interpreter running
#: the compiler (3.14 says `randint(a, b)` and "division by zero" where 3.12
#: said `randrange(a, b + 1)` and "float division by zero"). Distinct
#: sentinels stand for the values a message carries, which become `{0}`...
_A, _B, _S = 1234567, 7654, 99991


def _raising() -> dict[str, Callable[[], object]]:
    import bisect  # pylint: disable=import-outside-toplevel
    import collections  # pylint: disable=import-outside-toplevel
    import heapq  # pylint: disable=import-outside-toplevel
    import itertools  # pylint: disable=import-outside-toplevel
    import random  # pylint: disable=import-outside-toplevel

    r = random.Random(0)
    return {
        "deque_pop": lambda: collections.deque().pop(),
        "deque_index": lambda: collections.deque()[0],
        "fsum_overflow": lambda: math.fsum([1e308, 1e308]),
        "fsum_nan": lambda: math.fsum([math.inf, -math.inf]),
        "log_base": lambda: math.log(2.0, 1.0),
        "isqrt": lambda: math.isqrt(-1),
        "factorial": lambda: math.factorial(-1),
        "comb_n": lambda: math.comb(-1, 1),
        "comb_k": lambda: math.comb(1, -1),
        "perm_n": lambda: math.perm(-1, 1),
        "perm_k": lambda: math.perm(1, -1),
        "isclose": lambda: math.isclose(1.0, 1.0, rel_tol=-1.0),
        "dist": lambda: math.dist([1.0], [1.0, 2.0]),
        "range": lambda: math.exp(1000.0),
        "islice_stop": lambda: itertools.islice([], -1),
        "islice_indices": lambda: itertools.islice([], -1, 2),
        "islice_step": lambda: itertools.islice([], 0, 2, 0),
        "randrange1": lambda: r.randrange(0),
        "randrange2": lambda: r.randrange(_A, _B),
        "randrange3": lambda: r.randrange(_A, _B, _S),
        "zero_step": lambda: r.randrange(1, 5, 0),
        "randint": lambda: r.randint(_A, _B),
        "choice": lambda: r.choice([]),
        "sample": lambda: r.sample([], 1),
        "choices_empty": lambda: r.choices([], k=1),
        "weights_count": lambda: r.choices([1, 2], [1], k=1),
        "weights_empty": lambda: r.choices([], [], k=1),
        "weights_zero": lambda: r.choices([1], [0], k=1),
        "weights_infinite": lambda: r.choices([1, 2], [1.0, float("inf")], k=1),
        "bits": lambda: r.getrandbits(-1),
        "divide": lambda: r.expovariate(0.0),
        "gamma": lambda: r.gammavariate(0.0, 1.0),
        "range_step": lambda: range(1, 5, 0),
        "heap_empty": lambda: heapq.heappop([]),
        "lo": lambda: bisect.bisect_left([1], 1, -1),
        "hi": lambda: bisect.bisect_left([1, 2], 5, 0, 9),
    }


#: A domain error's text where CPython's names no value (3.14 says "expected
#: a positive input, got 0.0", which a standalone binary cannot spell yet).
_DOMAIN = "ValueError: math domain error"


@cache
def _text(key: str) -> str:
    if key in {"domain", "log_domain"}:
        said_text = (
            said(lambda: math.log(0.0)) if key == "log_domain" else said(lambda: math.acos(2.0))
        )
        return said_text if "0.0" not in said_text and "2.0" not in said_text else _DOMAIN
    return _said_text(key)


def _said_text(key: str) -> str:
    """The traceback's last line for check `key`, `{0}` and on where it
    carries the values the check reports (the start, the stop, the step:
    for `randint`, `a` and whichever of `b` and `b + 1` the message shows)."""
    text = said(_raising()[key])
    for value, slot in ((_A, "{0}"), (_B + 1, "{1}"), (_B, "{1}"), (_S, "{2}")):
        text = text.replace(str(value), slot)
    return text


@cache
def _randint_shows_stop() -> bool:
    """Whether `randint(a, b)` says `randrange(a, b + 1)` (before 3.14)."""
    return str(_B + 1) in said(_raising()["randint"])
