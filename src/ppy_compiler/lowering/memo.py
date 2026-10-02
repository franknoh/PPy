"""`functools.cache` and `functools.lru_cache` natively.

A cached function lowers to two: its body, under a name of its own, and the
entry everything calls, which looks the arguments up in the function's table
first and calls the body only on a miss, keeping what it returns. A
recursive call in the body is a call to the entry, so it goes through the
table as it does in CPython: a dynamic program's subproblems are computed
once.

The table is the runtime's (`ppy_memo_*` in `stdlib.c`): plain memory that
outlives every call, with strings kept as copies of their text, so a call
that fails and falls back, which frees what its thread made, leaves it
whole. It keeps what `lru_cache` keeps: everything with no `maxsize` (or
with `cache`), and at most `maxsize` entries otherwise, letting go of the one
used longest ago. The arguments are numbers, bools, strings, and tuples of
numbers; the result is one of those.

Under `ppy run` the native entry replaces the cached function, cache and
all, so Python's callers share the native table. A function whose
`cache_info`, `cache_clear`, `cache_parameters`, or `__wrapped__` the module
reads keeps its Python cache: it stays in Python.
"""

from __future__ import annotations

import ast
import hashlib

from ..analysis import types as T
from ..backend.llvm.lowering import Unsupported
from ..ir import BOOL, F64, I64, Successor, TupleType, Value
from ..ir.dialects import core
from .collections import HANDLE, STR, _pointer

__all__ = ["CACHE_ATTRIBUTES", "MemoLowering", "cache_bound", "cached_decorator", "define_cached"]

#: The decorators that cache a function.
_CACHES = frozenset({"functools.cache", "functools.lru_cache"})

#: What the cache's wrapper has that the function does not.
CACHE_ATTRIBUTES = frozenset({"cache_info", "cache_clear", "cache_parameters", "__wrapped__"})

#: `lru_cache`'s `maxsize` when it is not given.
_DEFAULT_BOUND = 128


def cached_decorator(info) -> ast.expr | None:  # type: ignore[no-untyped-def]
    """The `@cache` or `@lru_cache(...)` of a function, if it has one."""
    for name, decorator in zip(info.decorators, info.node.decorator_list, strict=False):
        if name in _CACHES:
            return decorator  # type: ignore[no-any-return]
    return None


def cache_bound(info) -> int | None:  # type: ignore[no-untyped-def]
    """How many entries a cached function's table keeps: -1 for no bound.
    Raises `Unsupported` for a decorator written in a way this does not
    read, and for a function the table cannot cache."""
    decorator = cached_decorator(info)
    assert decorator is not None
    others = [n for n in info.decorators if n not in _CACHES and not n.startswith("ppy.")]
    if others or sum(n in _CACHES for n in info.decorators) > 1:
        raise Unsupported("a cached function with other decorators keeps its cache in Python")
    if info.owner or info.enclosing:
        raise Unsupported("a cached method or nested function keeps its cache in Python")
    canonical = info.decorators[info.node.decorator_list.index(decorator)]
    if canonical == "functools.cache":
        return -1
    if not isinstance(decorator, ast.Call):
        return _DEFAULT_BOUND
    keywords = {k.arg: k.value for k in decorator.keywords}
    if None in keywords or set(keywords) - {"maxsize", "typed"} or len(decorator.args) > 2:
        raise Unsupported("`lru_cache` with these arguments keeps its cache in Python")
    typed = keywords.get("typed", decorator.args[1] if len(decorator.args) == 2 else None)
    if typed is not None and not (
        isinstance(typed, ast.Constant) and isinstance(typed.value, bool)
    ):
        raise Unsupported("`lru_cache(typed=...)` takes a constant natively")
    bound = keywords.get("maxsize", decorator.args[0] if decorator.args else None)
    if bound is None:
        return _DEFAULT_BOUND
    if isinstance(bound, ast.Constant) and bound.value is None:
        return -1
    if isinstance(bound, ast.Constant) and type(bound.value) is int:
        return max(bound.value, 0)
    raise Unsupported("`lru_cache(maxsize=...)` takes a constant natively")


def _table_id(qualname: str) -> int:
    return int(hashlib.sha256(qualname.encode()).hexdigest()[:15], 16)


def _words(t: T.Type) -> list[str] | None:
    """The words a cached argument or result is: "int" (and a bool), "float",
    or "str"; None for what the table does not hold."""
    t = T.strip_literal(t)
    if t in (T.INT, T.BOOL):
        return ["int"]
    if t == T.FLOAT:
        return ["float"]
    if t == T.STR:
        return ["str"]
    if isinstance(t, T.Tuple_) and not t.homogeneous and t.items:
        parts = [T.strip_literal(item) for item in t.items]
        if all(part in (T.INT, T.FLOAT) for part in parts):
            return ["float" if part == T.FLOAT else "int" for part in parts]
    return None


def _mask(words: list[str], kind: str) -> int:
    return sum(1 << index for index, word in enumerate(words) if word == kind)


class MemoLowering:
    """The entry of a cached function; mixed into `_FunctionLowering`."""

    def memo_run(self, node: ast.FunctionDef, body: str, results: tuple, bound: int) -> None:
        """The entry: the table looked up, and the body called on a miss."""
        info = self.info  # type: ignore[attr-defined]
        self.entry = self.function.add_entry_block()  # type: ignore[attr-defined]
        b = self.b  # type: ignore[attr-defined]
        rt = self._rt  # type: ignore[attr-defined]
        word = self._word  # type: ignore[attr-defined]
        b.at_end(self.entry)
        self._location(node)  # type: ignore[attr-defined]
        self._bind_parameters()  # type: ignore[attr-defined]
        returned = _words(info.ret)
        if returned is None:
            raise Unsupported(f"a cached function returning `{info.ret}` keeps its cache in Python")
        keys: list[tuple[str, Value]] = []
        for parameter in info.params:
            if parameter.global_of:
                continue  # a settled global: the same at every call
            kinds = _words(parameter.type)
            if kinds is None or parameter.kind != "positional_or_keyword":
                raise Unsupported(
                    f"a cached function taking `{parameter.type}` keeps its cache in Python"
                )
            keys.extend(zip(kinds, self._argument_words(parameter.name, kinds), strict=True))
        key_kinds = [kind for kind, _ in keys]
        key_buffer = self._words_buffer("memo.key", keys)
        table = rt(
            "ppy_memo_table",
            (
                word(_table_id(info.qualname)),
                word(len(key_kinds)),
                word(len(returned)),
                word(_mask(key_kinds, "str")),
                word(_mask(returned, "str")),
                word(_mask(key_kinds, "float")),
                word(bound),
            ),
            HANDLE,
        )
        found = rt("ppy_memo_find", (table, key_buffer))
        hit = self._block("memo.hit")  # type: ignore[attr-defined]
        miss = self._block("memo.miss")  # type: ignore[attr-defined]
        core.cond_br(b, core.cmp(b, "ge", found, word(0)), Successor(hit), Successor(miss))
        b.at_end(hit)
        result_type = results[0]
        if returned == ["str"]:
            kept = rt("ppy_memo_text", (table, found, word(0)), HANDLE)
        else:
            address = rt("ppy_memo_value", (table, found), HANDLE)
            parts = [
                self._read_word(address, index, "bool" if result_type == BOOL else kind)  # type: ignore[attr-defined]
                for index, kind in enumerate(returned)
            ]
            kept = core.tuple_make(b, *parts) if isinstance(result_type, TupleType) else parts[0]
        self._memo_return(kept)
        b.at_end(miss)
        arguments = tuple(self.entry.arguments)
        made = self._call_native(body, arguments, results)  # type: ignore[attr-defined]
        value = made.results[0]  # type: ignore[attr-defined]
        if isinstance(result_type, TupleType):
            parts = [core.tuple_extract(b, value, i) for i in range(len(returned))]
        else:
            parts = [core.cast(b, value, I64) if result_type == BOOL else value]
        value_buffer = self._words_buffer("memo.value", list(zip(returned, parts, strict=True)))
        rt("ppy_memo_store", (table, key_buffer, value_buffer), None)
        self._memo_return(value)
        self._finish_exceptions()  # type: ignore[attr-defined]

    def _memo_return(self, value: Value) -> None:
        self._check_thread_failures()  # type: ignore[attr-defined]
        self._leave_for_return()  # type: ignore[attr-defined]
        self._release_collections()  # type: ignore[attr-defined]
        core.ret(self.b, value)  # type: ignore[attr-defined]

    def _argument_words(self, name: str, kinds: list[str]) -> list[Value]:
        b = self.b  # type: ignore[attr-defined]
        if name in self.tuples:  # type: ignore[attr-defined]
            whole = core.load(b, self.tuples[name])  # type: ignore[attr-defined]
            return [core.tuple_extract(b, whole, i) for i in range(len(kinds))]
        if kinds == ["str"]:
            held = self.collections.get(name)  # type: ignore[attr-defined]
            if held is None or held.kind != STR:
                raise Unsupported(f"`{name}` is not a string natively")
            return [core.load(b, held.slot)]
        slot = self.slots.get(name)  # type: ignore[attr-defined]
        if slot is None:
            raise Unsupported(f"`{name}` is not a number natively")
        return [core.load(b, slot)]

    def _words_buffer(self, label: str, words: list[tuple[str, Value]]) -> Value:
        """The words in a stack buffer, each eight bytes, and its address."""
        b = self.b  # type: ignore[attr-defined]
        types = tuple(F64 if kind == "float" else HANDLE if kind == "str" else I64 for kind, _ in words)
        buffer = self._alloca(TupleType(types or (I64,)), label)  # type: ignore[attr-defined]
        address = core.cast(b, buffer, _pointer(buffer, HANDLE.pointee))
        for index, (kind, value) in enumerate(words):
            stored = F64 if kind == "float" else HANDLE if kind == "str" else I64
            pointer = core.cast(b, address, _pointer(address, stored))
            if index:
                pointer = core.ptr_offset(b, pointer, self._word(index))  # type: ignore[attr-defined]
            if stored == I64 and value.type == BOOL:
                value = core.cast(b, value, I64)
            elif stored == F64 and value.type == I64:
                value = self._coerce(value, "float")  # type: ignore[attr-defined]
            core.store(b, value, pointer)
        return address


def define_cached(frontend, info, node: ast.FunctionDef, constants: dict) -> list[str]:  # type: ignore[no-untyped-def]
    """A cached function's body under a name of its own, and its entry over the
    table (`MemoLowering.memo_run`); the chains a proof freed of their guard."""
    from dataclasses import replace as dataclass_replace  # pylint: disable=import-outside-toplevel

    from ppy_runtime._record import replace  # pylint: disable=import-outside-toplevel

    from .ast_to_ir import _FunctionLowering  # pylint: disable=import-outside-toplevel

    bound = cache_bound(info)
    if info.is_generator or info.is_async:
        raise Unsupported("a cached generator or coroutine keeps its cache in Python")
    tree = frontend.analysis.symbols.module.tree
    for found in ast.walk(tree):
        if (
            isinstance(found, ast.Attribute)
            and found.attr in CACHE_ATTRIBUTES
            and isinstance(found.value, ast.Name)
            and found.value.id == info.name
        ):
            raise Unsupported(f"`{info.name}.{found.attr}` reads the cache Python keeps")
    function, signature = frontend.declared[info.qualname]
    if not function.results:
        raise Unsupported("a cached function that returns nothing keeps its cache in Python")
    spelled = f"{info.qualname}__cached"
    body_signature = replace(
        signature,
        symbol=f"ppy_{spelled.replace('.', '_')}",
        native=replace(signature.native, symbol=f"ppy_{spelled.replace('.', '_')}")
        if signature.native is not None
        else None,
    )
    body = frontend.declare(dataclass_replace(info, qualname=spelled), body_signature)
    # The body is called from the entry only: nothing else may find it by name.
    frontend.declared.pop(spelled, None)
    lowering = _FunctionLowering(frontend, body, body_signature, info, constants)
    lowering.run(node)
    entry = _FunctionLowering(frontend, function, signature, info, constants)
    entry.memo_run(node, body.name, tuple(function.results), bound)
    return lowering.proved
