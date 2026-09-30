"""Effects in native code under `ppy run`: `print`, `input`, and calls into Python.

A native function that fails a guard runs again as Python from the start.
That is only sound while nothing the function did can be seen: a second
run would print its lines twice. So effects come in two sorts here.

- Output is held. `print` writes into the thread's buffer (`pyio.c`), and the
  boundary writes the buffer out through `sys.stdout` or `sys.stderr` when
  the call answers, or drops it when the call falls back and Python prints
  the same lines itself. A print costs no Python and no GIL, and a function
  that only prints keeps every guard it had.
- A barrier cannot be taken back: `input()`, `print(..., flush=True)`, and a
  call into Python. It writes out what is held first, so the order of
  output is CPython's, and from it on the call must never fall back.
  `check_effects` proves that over the IR: no guard that falls back, no call
  to a function that may fall back, and no exception the boundary could not
  raise as CPython would, is reachable after a barrier, in the function or in
  any caller after the call. A check CPython raises for is an exception
  there (the module is lowered in exception mode), so it raises natively
  with CPython's text; a check that only stands for a native limit (an
  integer past 64 bits) must have been proven away, or the function stays
  in Python.

A function with effects is native but holds the GIL at its boundary, and the
boundary's C wrapper, which knows nothing of held output, is not used for it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..analysis import types as T
from ..analysis.lexical import LexicalBindings
from ..backend.llvm.lowering import Unsupported
from ..ir import F64, I64, IRFunction, IRModule, Operation, Successor, SymbolRef, Value
from ..ir.dialects import core
from .collections import HANDLE, class_tag

__all__ = ["EffectLowering", "EffectSummary", "check_effects", "wants_exceptions"]

#: The kinds of `pyio.c`.
_NONE, _INT, _FLOAT, _BOOL, _STR = 0, 1, 2, 3, 4

#: Streams, as `pyio.c` numbers them.
_STDOUT, _STDERR = 1, 2

#: Runtime calls that cannot be taken back.
BARRIERS = frozenset({"ppy_io_call", "ppy_io_flush_or_raise"})

#: Runtime calls that hold output.
HOLDS = frozenset({"ppy_io_print"})

#: Builtins whose call has no effect a later argument of `print` could show.
_QUIET = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "divmod",
        "float",
        "hex",
        "int",
        "len",
        "max",
        "min",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "list",
        "set",
        "dict",
        "isinstance",
        "enumerate",
        "zip",
        "reversed",
    }
)

#: Effects of a callee that a later argument of `print` must not have.
_LOUD = frozenset(
    {
        "io",
        "write_object",
        "write_global",
        "write_memory",
        "random",
        "python_callback",
        "python_dynamic",
        "external_unknown",
        "network",
        "process",
        "time",
        "thread",
        "sync",
    }
)


def wants_exceptions(nodes: list[ast.AST]) -> bool:
    """Whether a body reaches a barrier the lowering knows of before lowering it:
    a module that does is lowered in exception mode, so what may fail after
    the barrier raises natively instead of falling back."""
    for node in nodes:
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
                continue
            if inner.func.id == "input":
                return True
            if inner.func.id == "print" and any(k.arg == "flush" for k in inner.keywords):
                return True
    return False


def _plain(t: T.Type) -> bool:
    """Whether `str()` of a value of this type runs none of the program's code."""
    t = T.strip_literal(t)
    if isinstance(t, T.Union_):
        return all(_plain(m) for m in t.members)
    if isinstance(t, T.Tuple_):
        return all(_plain(item) for item in t.items)
    if isinstance(t, T.Instance):
        return t.name in T.BUILTIN_MRO and all(_plain(a) for a in t.args)
    return False


class EffectLowering:  # pylint: disable=too-few-public-methods
    """`print` and `input` under `ppy run`; mixed into `_FunctionLowering`."""

    def _effects_on(self) -> bool:
        return bool(getattr(self.frontend, "effects", False))  # type: ignore[attr-defined]

    def _effect_call(self, node: ast.Call, discard: bool) -> Value | None:
        """`print(...)` and `input(...)` of the builtins; None for any other call."""
        if not self._effects_on() or not isinstance(node.func, ast.Name):
            return None
        name = node.func.id
        if name not in {"print", "input"}:
            return None
        lexical = self.frontend.analysis.symbols.lexical  # type: ignore[attr-defined]
        if not isinstance(lexical, LexicalBindings) or lexical.targets_at(node.func) != {
            f"builtins.{name}"
        }:
            return None
        if name == "print":
            self._effect_print(node)
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        read = self._effect_input(node)
        if discard:
            self._release(read)  # type: ignore[attr-defined]
            return self._word(0)  # type: ignore[attr-defined,no-any-return]
        return read

    # -- print ------------------------------------------------------------------

    def _print_options(self, node: ast.Call) -> tuple[int, str, str, bool]:
        stream, separator, end, flush = _STDOUT, " ", "\n", False
        seen: set[str] = set()
        for keyword in node.keywords:
            name, value = keyword.arg, keyword.value
            if name is None or name in seen:
                raise Unsupported("`print(**options)` has no native lowering")
            seen.add(name)
            constant = value.value if isinstance(value, ast.Constant) else ...
            if name in {"sep", "end"}:
                if constant is None:
                    continue
                if not isinstance(constant, str):
                    raise Unsupported(f"`print({name}=...)` natively is a string literal or None")
                if name == "sep":
                    separator = constant
                else:
                    end = constant
            elif name == "flush":
                if not isinstance(constant, bool):
                    raise Unsupported("`print(flush=...)` natively is True or False")
                flush = constant
            elif name == "file":
                if constant is None:
                    continue
                stream = self._print_stream(value)
            else:
                raise Unsupported(f"`print` takes no keyword `{name}`")
        return stream, separator, end, flush

    def _print_stream(self, value: ast.expr) -> int:
        """`sys.stdout` or `sys.stderr`, looked up when the output is written."""
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.attr in {"stdout", "stderr"}
        ):
            binding = self.frontend.analysis.symbols.imports.get(value.value.id)  # type: ignore[attr-defined]
            if binding is not None and binding.canonical == "sys":
                return _STDOUT if value.attr == "stdout" else _STDERR
        raise Unsupported("`print(file=...)` natively is `sys.stdout` or `sys.stderr`")

    def _effect_print(self, node: ast.Call) -> None:
        """`print(a, b, sep=..., end=..., file=..., flush=...)`.

        Python evaluates every argument before it writes any, then writes each
        one's `str()` in turn. Here each argument is written as it is
        evaluated, which is the same wherever the arguments after the first
        have no effect a later one could show; `str()` of a value of the
        program's own class runs its `__str__`, which may print, so the text
        before it is held first, as CPython would have written it."""
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise Unsupported("`print(*values)` has no native lowering")
        for argument in node.args[1:]:
            reason = self._loud(argument)
            if reason:
                raise Unsupported(
                    f"`print` whose later argument {reason}: Python evaluates every "
                    "argument before it prints any"
                )
        stream, separator, end, flush = self._print_options(node)
        rt = self._rt  # type: ignore[attr-defined]
        builder = rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
        for index, argument in enumerate(node.args):
            if index:
                self._add_text(builder, separator)  # type: ignore[attr-defined]
            if isinstance(argument, ast.Constant) and (
                argument.value is None or isinstance(argument.value, str)
            ):
                self._add_text(builder, str(argument.value))  # type: ignore[attr-defined]
                continue
            if not _plain(self._type_of(argument)):  # type: ignore[attr-defined]
                self._hold(stream, builder)
                builder = rt("ppy_str_builder", (self._word(0),), HANDLE)  # type: ignore[attr-defined]
            self._add_formatted(builder, argument, -1, "")  # type: ignore[attr-defined]
        self._add_text(builder, end)  # type: ignore[attr-defined]
        self._hold(stream, builder)
        if flush:
            status = rt("ppy_io_flush_or_raise", (self._word(stream),))  # type: ignore[attr-defined]
            self._after_barrier(status)

    def _hold(self, stream: int, builder: Value) -> None:
        rt = self._rt  # type: ignore[attr-defined]
        text = rt("ppy_str_finish", (builder,), HANDLE)
        rt("ppy_io_print", (self._word(stream), text), None)  # type: ignore[attr-defined]
        self._release(text)  # type: ignore[attr-defined]

    def _loud(self, node: ast.expr) -> str:
        """What in `node` may have an effect a later argument of `print` would
        show, or "" where nothing does."""
        functions = self.frontend.analysis.functions  # type: ignore[attr-defined]
        declared = self.frontend.declared  # type: ignore[attr-defined]
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
                return "suspends or assigns"
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if isinstance(func, ast.Name):
                if func.id in _QUIET and func.id not in declared:
                    continue
                callee = next(
                    (a for q, a in functions.items() if q.rpartition(".")[2] == func.id), None
                )
                if callee is not None and not set(callee.effects.spelled()) & _LOUD:
                    continue
                return f"calls `{func.id}`"
            if isinstance(func, ast.Attribute):
                if self._string_of(func.value) is not None:  # type: ignore[attr-defined]
                    continue
                if func.attr in {"count", "index", "get", "keys", "values", "items", "copy"}:
                    continue
                return f"calls `.{func.attr}()`"
            return "makes a call"
        return ""

    # -- input and calls into Python --------------------------------------------

    def _effect_input(self, node: ast.Call) -> Value:
        """`input()` and `input(prompt)`: Python's own, called with the GIL."""
        if node.keywords or len(node.args) > 1:
            raise Unsupported("`input` takes at most one argument")
        arguments: list[tuple[int, Value, bool]] = []
        if node.args:
            if self._string_of(node.args[0]) is None:  # type: ignore[attr-defined]
                raise Unsupported("`input(prompt)` natively takes a string prompt")
            handle, owned = self._handle(node.args[0])  # type: ignore[attr-defined]
            arguments.append((_STR, handle, owned))
        return self._python_call("builtins:input", arguments, _STR)  # type: ignore[return-value]

    def _python_call(
        self, spelled: str, arguments: list[tuple[int, Value, bool]], kind: int
    ) -> Value | None:
        """Call the Python callable `module:name` with the arguments boxed, and
        unbox its result as `kind`; what it raises is raised natively."""
        rt = self._rt  # type: ignore[attr-defined]
        for argument_kind, value, _owned in arguments:
            if argument_kind == _STR:
                rt("ppy_io_push_text", (value,), None)
            elif argument_kind == _FLOAT:
                rt("ppy_io_push_float", (value,), None)
            else:
                word = value if value.type == I64 else core.cast(self.b, value, I64)  # type: ignore[attr-defined]
                rt("ppy_io_push", (self._word(argument_kind), word), None)  # type: ignore[attr-defined]
        data, length = self._text_data(spelled)  # type: ignore[attr-defined]
        status = rt("ppy_io_call", (data, length, self._word(kind), self._word(0)))  # type: ignore[attr-defined]
        for argument_kind, value, owned in arguments:
            if argument_kind == _STR and owned:
                self._release(value)  # type: ignore[attr-defined]
        self._after_barrier(status)
        if kind == _STR:
            return rt("ppy_io_result_text", (), HANDLE)  # type: ignore[no-any-return]
        if kind == _FLOAT:
            return rt("ppy_io_result_float", (), F64)  # type: ignore[no-any-return]
        if kind == _BOOL:
            word = rt("ppy_io_result", ())
            return core.cmp(self.b, "ne", word, self._word(0))  # type: ignore[attr-defined,no-any-return]
        if kind == _INT:
            return rt("ppy_io_result", ())  # type: ignore[no-any-return]
        return None

    def _after_barrier(self, status: Value) -> None:
        """A barrier's status: 0, or -1 with Python's exception pending natively."""
        answered = core.cmp(self.b, "eq", status, self._word(0))  # type: ignore[attr-defined]
        if not self._exceptions_on():  # type: ignore[attr-defined]
            # Lowered again in exception mode (`lower_module_to_ir`); this pass
            # only finds that the module needs it.
            self.frontend.wants_exceptions = True  # type: ignore[attr-defined]
            core.guard(self.b, answered, "effect", "Python raised")  # type: ignore[attr-defined]
            return
        kept = self._block("effect.ok")  # type: ignore[attr-defined]
        failed = self._block("effect.raised")  # type: ignore[attr-defined]
        core.cond_br(self.b, answered, Successor(kept), Successor(failed))  # type: ignore[attr-defined]
        self.b.at_end(failed)  # type: ignore[attr-defined]
        self._go_raise()  # type: ignore[attr-defined]
        self.b.at_end(kept)  # type: ignore[attr-defined]


# -- the barrier rule over the IR ---------------------------------------------------------


@dataclass(slots=True)
class EffectSummary:
    """What a function may do, callees included."""

    #: It holds output (prints).
    holds: bool = False
    #: It reaches a barrier.
    barrier: bool = False
    #: It may fall back to Python.
    falls_back: bool = False
    #: It may raise an exception the boundary could not raise as CPython would.
    raises_inexactly: bool = False
    #: Why, for a report: the first of each found.
    why: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple[bool, bool, bool, bool]:
        return (self.holds, self.barrier, self.falls_back, self.raises_inexactly)


#: Dialects whose operations may fall back where the backend lowers them.
_FALLING_DIALECTS = frozenset({"concurrency", "aio", "parallel", "gpu"})


def _builtin_tags() -> frozenset[int]:
    return frozenset(
        class_tag(f"builtins.{name}")
        for name, mro in T.BUILTIN_MRO.items()
        if "BaseException" in mro
    )


def _owner(value: Value) -> object:
    try:
        return value.owner
    except NotImplementedError:
        return None


def _constant(value: Value) -> object:
    owner = _owner(value)
    if isinstance(owner, Operation) and owner.name == "core.const":
        return owner.attributes.get("value")
    return None


def _callee(op: Operation) -> str | None:
    found = op.attributes.get("callee")
    return found.name if isinstance(found, SymbolRef) else None


def _extern(op: Operation) -> str | None:
    if op.name != "core.call_extern":
        return None
    return str(op.attributes.get("callee"))


def _linked_call(op: Operation) -> Operation | None:
    """The call whose status a `core.guard` asks after (`_call_native`), if any."""
    if op.name != "core.guard" or not op.operands:
        return None
    condition = _owner(op.operands[0])
    if not isinstance(condition, Operation) or condition.name != "core.cmp":
        return None
    for operand in condition.operands:
        made = _owner(operand)
        if (
            isinstance(made, Operation)
            and made.name in {"core.call", "core.call_indirect"}
            and made.attributes.get("capture_status")
            and operand is made.results[-1]
        ):
            return made
    return None


class _Checker:
    """The barrier rule over one module (see the module's docstring)."""

    def __init__(self, module: IRModule, exceptions: bool) -> None:
        self.module = module
        self.exceptions = exceptions
        self.summaries: dict[str, EffectSummary] = {}
        self.builtin_tags = _builtin_tags()

    def external(self, name: str) -> EffectSummary:
        """A callee this module only declares: assumed to fall back and raise;
        a barrier where its signature says it has effects."""
        target = self.module.functions.get(name)
        effects = bool(target is not None and target.attributes.get("ppy.effects"))
        return EffectSummary(
            holds=effects, barrier=effects, falls_back=True, raises_inexactly=True
        )

    def summary(self, name: str) -> EffectSummary:
        found = self.summaries.get(name)
        if found is not None:
            return found
        target = self.module.functions.get(name)
        if target is None or target.is_declaration:
            return self.external(name)
        return EffectSummary()

    def bad_raise(self, op: Operation) -> bool:
        """`ppy_exc_raise` of anything but a builtin exception with CPython's text."""
        if _extern(op) != "ppy_exc_raise" or not op.operands:
            return False
        made = _owner(op.operands[0])
        if not isinstance(made, Operation) or _extern(made) != "ppy_exc_make":
            return True
        tag, flags = _constant(made.operands[0]), _constant(made.operands[3])
        return not (
            isinstance(tag, int)
            and tag in self.builtin_tags
            and isinstance(flags, int)
            and flags & 1
        )

    def classify(self, op: Operation) -> tuple[str, str, str]:
        """(falls back, raises inexactly, barrier) reasons for one operation; "" for none."""
        falls = inexact = barrier = ""
        name = op.name
        if name == "core.guard":
            label = str(op.attributes.get("label") or "")
            if label != "raised" and not label.startswith("sanitize:"):
                if _linked_call(op) is None:
                    message = str(op.attributes.get("message") or op.attributes.get("kind"))
                    falls = f"a check that falls back ({message})"
        elif name == "core.call":
            callee = _callee(op) or ""
            summary = self.summary(callee)
            if summary.falls_back:
                falls = f"a call to `{callee}`, which may fall back"
            if summary.raises_inexactly:
                inexact = f"a call to `{callee}`, which may raise what Python would say otherwise"
            if not self.exceptions and not op.attributes.get("capture_status"):
                if summary.raises_inexactly or summary.falls_back:
                    falls = falls or f"a call to `{callee}`, whose exception falls back"
            if summary.barrier:
                barrier = f"`{callee}`"
        elif name == "core.call_indirect":
            falls = "a call through a function value"
            inexact = falls
        elif name == "core.call_extern":
            extern = _extern(op)
            if extern in BARRIERS:
                barrier = {
                    "ppy_io_call": "a call into Python",
                    "ppy_io_flush_or_raise": "`print(flush=True)`",
                }[extern]
            elif self.bad_raise(op):
                inexact = "an exception PPy cannot raise as Python would after it"
        elif op.dialect in _FALLING_DIALECTS:
            falls = f"`{name}`"
        return falls, inexact, barrier

    def summarize(self, function: IRFunction) -> EffectSummary:
        made = EffectSummary()
        for op in function.operations():
            falls, inexact, barrier = self.classify(op)
            if _extern(op) in HOLDS:
                made.holds = True
            if op.name == "core.call" and self.summary(_callee(op) or "").holds:
                made.holds = True
            if falls and not made.falls_back:
                made.falls_back = True
                made.why["falls_back"] = falls
            if inexact and not made.raises_inexactly:
                made.raises_inexactly = True
                made.why["raises_inexactly"] = inexact
            if barrier and not made.barrier:
                made.barrier = True
                made.why["barrier"] = barrier
        made.holds = made.holds or made.barrier
        return made

    def settle(self) -> None:
        """Every function's summary, to a fixpoint over calls (recursion included)."""
        bodies = [f for f in self.module.functions.values() if not f.is_declaration]
        for function in bodies:
            self.summaries[function.name] = EffectSummary()
        changed = True
        while changed:
            changed = False
            for function in bodies:
                made = self.summarize(function)
                if made.key() != self.summaries[function.name].key():
                    self.summaries[function.name] = made
                    changed = True

    def violation(self, function: IRFunction) -> str:
        """What in `function` may happen after a barrier that must not, or ""."""
        blocks = list(function.blocks())
        barrier_at: dict[int, tuple[int, str]] = {}
        for block in blocks:
            for index, op in enumerate(block):
                _falls, _inexact, barrier = self.classify(op)
                if barrier:
                    barrier_at[id(block)] = (index, barrier)
                    break
        if not barrier_at:
            return ""
        # Blocks entered after a barrier ran: anything a barrier's block reaches.
        after: set[int] = set()
        work = [s for b in blocks if id(b) in barrier_at for s in b.successors]
        while work:
            block = work.pop()
            if id(block) in after:
                continue
            after.add(id(block))
            work.extend(block.successors)
        first = next(iter(barrier_at.values()))[1]
        for block in blocks:
            start = 0
            if id(block) not in after:
                if id(block) not in barrier_at:
                    continue
                start = barrier_at[id(block)][0] + 1
            for op in list(block)[start:]:
                falls, inexact, _barrier = self.classify(op)
                if falls:
                    return f"{falls} can follow {first}"
                if inexact:
                    return f"{inexact} can follow {first}"
        return ""

    def referenced(self) -> dict[str, set[str]]:
        """Functions named other than by a direct call: address taken, spawned, run
        in parallel. Their callers are not known, so neither is what follows."""
        found: dict[str, set[str]] = {}
        for function in self.module.functions.values():
            if function.is_declaration:
                continue
            for op in function.operations():
                if op.name in {"core.call", "async.create"}:
                    continue
                for value in op.attributes.values():
                    if isinstance(value, SymbolRef):
                        found.setdefault(value.name, set()).add(function.name)
        return found


def check_effects(module: IRModule, exceptions: bool) -> tuple[dict[str, str], dict[str, EffectSummary]]:
    """The barrier rule over a module: the functions that break it, each with why
    (callers of one included), and every function's summary."""
    checker = _Checker(module, exceptions)
    checker.settle()
    broken: dict[str, str] = {}
    referenced = checker.referenced()
    for function in module.functions.values():
        if function.is_declaration:
            continue
        summary = checker.summaries[function.name]
        if summary.holds and function.name in referenced:
            broken[function.name] = (
                "prints or reads, and runs where its caller is not known "
                "(a function value, a thread, or a parallel loop)"
            )
            continue
        why = checker.violation(function)
        if why:
            broken[function.name] = why
    # Whatever calls or names a function that broke the rule goes with it.
    changed = True
    while changed:
        changed = False
        for function in module.functions.values():
            if function.is_declaration or function.name in broken:
                continue
            for op in function.operations():
                named = {v.name for v in op.attributes.values() if isinstance(v, SymbolRef)}
                hit = named & broken.keys()
                if hit:
                    callee = sorted(hit)[0]
                    broken[function.name] = f"calls `{callee}`, which stays in Python"
                    changed = True
                    break
    return broken, checker.summaries


def rule_of(summary: EffectSummary) -> str:
    """The rule a native function's effects run under, for `ppy explain`."""
    if summary.barrier:
        return (
            "effects: output is written at each barrier and when the call returns; "
            f"nothing falls back after the first barrier ({summary.why.get('barrier', '')})"
        )
    if summary.holds:
        return "effects: output is held until the call returns, and dropped if it falls back"
    return ""

