"""Type, effect, refinement, and validity analysis (spec 8, 9, 11, 12.3).

The checker walks each function body flow-sensitively, producing an inferred
type and a fact set for every expression node, an effect summary for every
function, and diagnostics for anything the strict mode forbids.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ..diagnostics import Diagnostic, DiagnosticBag, Severity
from ..frontend.source import span_of
from . import builtins as B
from . import stdlib
from . import types as T
from .aliasing import EXTERNAL, AliasInfo, analyze_aliases
from .annotations import AnnotationResolver, Resolved, narrow_element
from .binding import bind_call, positional_values
from .effects import Effect, EffectSet
from .env import Binding, Env
from .refinements import Facts, IntRange, width_range
from .results import FunctionAnalysis, LoweringNote, ModuleAnalysis, ProjectAnalysis
from .symbols import ClassInfo, FunctionInfo, ModuleSymbols, ProjectSymbols

if TYPE_CHECKING:
    from ..plugins.base import PluginRegistry

__all__ = ["FunctionAnalysis", "LoweringNote", "ModuleAnalysis", "ProjectAnalysis", "analyze"]


_FORBIDDEN_CALLS = {
    "eval": ("E1501", "`eval` cannot be statically analyzed"),
    "exec": ("E1501", "`exec` cannot be statically analyzed"),
    "compile": ("E1501", "runtime code-object construction cannot be statically analyzed"),
    "globals": ("E1502", "`globals()` exposes the module namespace to unanalyzable mutation"),
    "locals": ("E1502", "`locals()` exposes frame locals to unanalyzable mutation"),
    "vars": ("E1502", "`vars()` exposes an object namespace to unanalyzable mutation"),
    "__import__": ("E1503", "`__import__` is not a compile-time constant import"),
}

_DYNAMIC_ATTR_CALLS = {"setattr", "delattr"}

#: Metaclasses whose construction behavior is modeled; anything else rewrites
#: the class in ways no static analysis follows.
_KNOWN_METACLASSES = frozenset(
    {"builtins.type", "type", "abc.ABCMeta", "enum.EnumMeta", "enum.EnumType"}
)


def _static_base(node: ast.expr) -> bool:
    """A base expression that names a class rather than computing one."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _static_base(node.value)
    if isinstance(node, ast.Subscript):
        return _static_base(node.value)
    return False


#: Builtin methods that mutate their receiver in place.
_MUTATING_METHODS = frozenset(
    {
        "list.append",
        "list.extend",
        "list.insert",
        "list.pop",
        "list.clear",
        "list.sort",
        "list.reverse",
        "list.remove",
        "dict.pop",
        "dict.popitem",
        "dict.update",
        "dict.clear",
        "dict.setdefault",
        "set.add",
        "set.discard",
        "set.remove",
        "set.update",
        "set.clear",
    }
)

_COMPARE_OPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}

_ARITH_OPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.MatMult: "@",
}

_MAX_LOOP_ITERATIONS = 4

#: The set algebra, which the numeric rules below do not cover.
_SET_OPS = frozenset({ast.BitOr, ast.BitAnd, ast.Sub, ast.BitXor})

#: What a class spells to overload each operator.
_OPERATOR_DUNDERS: dict[type[ast.operator], str] = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.BitAnd: "__and__",
    ast.MatMult: "__matmul__",
}


def _display_fits(node: ast.expr, actual: T.Type, declared: T.Type) -> bool:
    """Does a container written in place fit the type declared beside it?

    Only displays and comprehensions: a value that already had a type keeps
    it. The container kinds must match, and every element type the display
    holds must be assignable to the declared element type.
    """
    kinds: dict[type[ast.expr], str] = {
        ast.List: "list",
        ast.ListComp: "list",
        ast.Set: "set",
        ast.SetComp: "set",
        ast.Dict: "dict",
        ast.DictComp: "dict",
    }
    kind = kinds.get(type(node))
    if kind is None:
        return False
    actual = T.strip_literal(actual)
    declared = T.strip_literal(declared)
    if not (isinstance(actual, T.Instance) and isinstance(declared, T.Instance)):
        return False
    if actual.name != kind or declared.name != kind or len(actual.args) != len(declared.args):
        return False
    return all(
        isinstance(held, T.NeverType) or T.is_assignable(held, wanted)
        for held, wanted in zip(actual.args, declared.args, strict=True)
    )


def _attribute_path(node: ast.expr) -> str | None:
    """`self.x.y` as the key a narrowing is kept under, or None if not a plain chain.

    A chain of names is a place in the object graph that a check can be
    about, and so is a constant subscript in it: `args[0].facts` is where
    `args[0].facts is not None` looked. A call, or a subscript by a
    variable, is a fresh value every time.
    """
    parts: list[str] = []
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            parts.append("." + node.attr)
        else:
            if not isinstance(node.slice, ast.Constant) or isinstance(node.slice.value, bool):
                return None
            if not isinstance(node.slice.value, (int, str)):
                return None
            parts.append(f"[{node.slice.value!r}]")
        node = node.value
    if not isinstance(node, ast.Name) or not parts:
        return None
    parts.append(node.id)
    return "".join(reversed(parts))


def _forget_attributes(env: Env, root: str) -> None:
    """A write to `root` ends what was known about anything under it."""
    prefixes = (root + ".", root + "[")
    for name in list(env.names()):
        if name.startswith(prefixes):
            env.remove(name)


def _is_place(node: ast.expr) -> bool:
    return isinstance(node, (ast.Attribute, ast.Subscript))


def _is_class_value(t: T.Type) -> bool:
    if isinstance(t, T.ClassObject) or t == T.NONE:
        return True
    return isinstance(t, T.Instance) and t.name == "type"


#: Lookups that may ask with a wider key than the table holds: asking
#: `table.get(type(node.op))` with a `type[operator]` of a table keyed by
#: `type[Add] | type[Sub]` is how a table is consulted, and an absent key is
#: the `None` the method promises.
_LOOKUPS = frozenset({"dict.get", "dict.pop"})


def _spread(elements: list[ast.expr]) -> bool:
    return any(isinstance(e, ast.Starred) for e in elements)


def _is_set_like(t: T.Type) -> bool:
    return isinstance(t, T.Instance) and t.name in {"set", "frozenset"}


#: Attributes every class object exposes.
_CLASS_DUNDERS: dict[str, T.Type] = {
    "__name__": T.STR,
    "__qualname__": T.STR,
    "__module__": T.STR,
    "__doc__": T.union(T.STR, T.NONE),
    "__bases__": T.Tuple_((T.instance("type"),), homogeneous=True),
    "__mro__": T.Tuple_((T.instance("type"),), homogeneous=True),
}

#: Module attributes CPython always provides.
_MODULE_DUNDERS: dict[str, T.Type] = {
    "__name__": T.STR,
    "__file__": T.STR,
    "__doc__": T.union(T.STR, T.NONE),
    "__package__": T.union(T.STR, T.NONE),
    "__spec__": T.ANY,
    "__builtins__": T.ANY,
    "__debug__": T.BOOL,
    "__annotations__": T.dict_of(T.STR, T.ANY),
}

#: Statically known attributes of plugin-owned instance types.
_PLUGIN_INSTANCE_ATTRS: dict[str, dict[str, T.Type]] = {
    "numpy.ndarray": {
        "shape": T.Tuple_((T.INT,), homogeneous=True),
        "ndim": T.INT,
        "size": T.INT,
        "itemsize": T.INT,
        "nbytes": T.INT,
        "strides": T.Tuple_((T.INT,), homogeneous=True),
        "dtype": T.Instance("numpy.dtype", (), ("numpy.dtype", "object")),
        "T": T.Instance("numpy.ndarray", (), ("numpy.ndarray", "object")),
    },
    "torch.Tensor": {
        "shape": T.Tuple_((T.INT,), homogeneous=True),
        "ndim": T.INT,
        "dtype": T.Instance("torch.dtype", (), ("torch.dtype", "object")),
        "device": T.Instance("torch.device", (), ("torch.device", "object")),
        "requires_grad": T.BOOL,
        "T": T.Instance("torch.Tensor", (), ("torch.Tensor", "object")),
    },
}


#: Builtins that read their arguments and either return a scalar or a fresh
#: copy. `enumerate`, `zip`, and `reversed` are deliberately absent: they hold
#: on to what they were given.
_INSPECTING_BUILTINS = frozenset(
    {
        "len",
        "sum",
        "min",
        "max",
        "sorted",
        "any",
        "all",
        "abs",
        "round",
        "str",
        "repr",
        "int",
        "float",
        "bool",
        "list",
        "tuple",
        "set",
        "dict",
        "print",
        "isinstance",
        "id",
        "hash",
    }
)


def _is_inspecting_builtin(node: ast.Call, env: Env) -> bool:
    """Does this call only read its arguments, never keep them?"""
    return (
        isinstance(node.func, ast.Name)
        and node.func.id in _INSPECTING_BUILTINS
        and node.func.id not in env
    )


#: Types whose `+=`/`-=`/... mutate the object rather than rebinding.
_IN_PLACE_MUTABLE = frozenset({"list", "set", "dict", "bytearray", "collections.deque"})


def _mutates_in_place(t: T.Type) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name in _IN_PLACE_MUTABLE


#: What a buffer can hold. A machine word or a byte; anything else has no
#: native storage, so the annotation would promise something untrue.
_HOLDABLE = frozenset({"int", "float", "i8", "u8"})


def _is_negative(facts: Facts) -> bool:
    """Is this count provably below zero?"""
    if facts.has_constant and isinstance(facts.constant, int):
        return facts.constant < 0
    span = facts.int_range
    return span is not None and span.high is not None and span.high < 0


def _holdable_element(t: T.Type) -> bool:
    base = T.strip_literal(t)
    return isinstance(base, T.Instance) and base.name in _HOLDABLE


def _is_fresh_allocation(node: ast.expr) -> bool:
    """Does this expression produce an object nothing else can already hold?"""
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"list", "dict", "set", "bytearray"}
    # `ppy.buffer[int](n)` and `ppy.input[Buffer[int]](n)` make the memory
    # they hand back, so nothing else can already be holding it.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
        spelled = ast.unparse(node.func.value)
        return spelled in {"ppy.buffer", "ppy.input"}
    return False


class _Checker:
    """Checks one module. Reused across the effect fixpoint and the final pass."""

    def __init__(
        self,
        symbols: ModuleSymbols,
        project: ProjectSymbols,
        analysis: ProjectAnalysis,
        diagnostics: DiagnosticBag,
        *,
        strict: bool = True,
        record: bool = True,
        dynamic_policy: str = "explicit",
        plugins: PluginRegistry | None = None,
    ) -> None:
        self.plugins = plugins
        self.symbols = symbols
        self.project = project
        self.analysis = analysis
        self.diagnostics = diagnostics
        self.strict = strict
        self.record = record
        self.dynamic_policy = dynamic_policy
        self.path: Path = symbols.path
        self.module = ModuleAnalysis(symbols=symbols)
        self.annotations = AnnotationResolver(
            project.resolver(symbols), self.path, diagnostics, strict=strict
        )
        self._effects = EffectSet()
        self._unknown: list[str] = []
        #: Errors withheld because they only restated an unresolved type.
        self.cascaded = 0
        self._calls: set[str] = set()
        self._dynamic_depth = 0
        self._dynamic_seen = False
        self._current: FunctionInfo | None = None
        self._returns: list[Binding] = []
        self._provisional_returns: list[bool] = []
        #: Locals whose type is unknown only because a recursive call fed
        #: them, so a `return` of one is provisional the same way.
        self._provisional_locals: set[str] = set()
        self._blockers: list[str] = []
        self._native_blockers: list[str] = []
        self._escaping: set[str] = set()
        self._mutated: set[str] = set()
        #: Parameters handed to a callee that writes through what it is given.
        self._delegated: set[str] = set()
        self._foreign_writes = False
        self._local_writes: set[str] = set()
        self._shared: set[str] = set()
        self._returned_names: set[str] = set()
        self._external_writes = False
        self._aliases: AliasInfo | None = None
        self._bound_methods: set[int] = set()
        #: Nodes that already produced a diagnostic, so a downstream pass does
        #: not report a second, less useful one about the same expression.
        self._reported: set[int] = set()
        self._function_locals: set[str] = set()
        self._attribute_owners: dict[int, T.Type] = {}

    def check_module(self) -> ModuleAnalysis:
        env = Env()
        self._seed_module_env(env)
        self._effects = EffectSet()
        for stmt in self.symbols.module.tree.body:
            self._stmt(stmt, env)
        self.module.module_effects = self._effects
        for info in self._all_functions():
            self.module.functions[info.qualname] = self._check_function(info)
        return self.module

    def _all_functions(self) -> list[FunctionInfo]:
        found = list(self.symbols.functions.values())
        for cls in self.symbols.classes.values():
            found.extend(cls.methods.values())
        found.extend(self._nested_functions())
        return found

    def _nested_functions(self) -> list[FunctionInfo]:
        return []

    def _check_function(self, info: FunctionInfo) -> FunctionAnalysis:
        previous = (
            self._effects,
            self._unknown,
            self._calls,
            self._current,
            self._returns,
            self._provisional_returns,
            self._provisional_locals,
            self._blockers,
            self._native_blockers,
            self._escaping,
            self._mutated,
            self._delegated,
            self._dynamic_depth,
            self._function_locals,
            self._foreign_writes,
            self._local_writes,
            self._shared,
            self._returned_names,
            self._external_writes,
            self._aliases,
        )
        self._effects = EffectSet()
        self._unknown = []
        self._calls = set()
        self._current = info
        self._returns = []
        self._provisional_returns = []
        self._provisional_locals = set()
        self._blockers = []
        self._native_blockers = []
        self._escaping = set()
        self._mutated = set()
        self._delegated = set()
        self._foreign_writes = False
        self._local_writes = set()
        self._shared = set()
        self._returned_names = set()
        self._external_writes = False
        # Names are not objects: everything below that asks "what does this
        # mutate or share?" resolves the name through the alias map first.
        # The map depends on the body and on which parameters are immutable,
        # and on nothing else, so one computed on an earlier pass still holds.
        immutable = frozenset(p.name for p in info.params if p.known and T.is_immutable(p.type))
        cached = self.project.alias_cache.get((id(info.node), immutable))
        if cached is None:
            cached = analyze_aliases(info.node, immutable)
            self.project.alias_cache[(id(info.node), immutable)] = cached
        self._aliases = cached  # type: ignore[assignment]
        if info.dynamic:
            self._dynamic_depth += 1

        env = Env()
        self._seed_module_env(env)
        # A parameter shadows a module global of the same name, so reading it
        # is not a global dependency.
        self._function_locals = {param.name for param in info.params}
        self._function_locals |= _assigned_names(info)
        for param in info.params:
            facts = param.facts
            if isinstance(param.type, T.UnknownType) and not param.annotated:
                self._implicit_any(info, param.name)
            env.set(param.name, Binding(param.type, facts))

        for stmt in info.node.body:
            self._stmt(stmt, env)

        result = self._finish_function(info, env)
        (
            self._effects,
            self._unknown,
            self._calls,
            self._current,
            self._returns,
            self._provisional_returns,
            self._provisional_locals,
            self._blockers,
            self._native_blockers,
            self._escaping,
            self._mutated,
            self._delegated,
            self._dynamic_depth,
            self._function_locals,
            self._foreign_writes,
            self._local_writes,
            self._shared,
            self._returned_names,
            self._external_writes,
            self._aliases,
        ) = previous
        return result

    def _finish_function(self, info: FunctionInfo, env: Env) -> FunctionAnalysis:
        if info.is_generator:
            inferred: T.Type = T.instance(
                "Generator", T.join(*[r.type for r in self._returns]) if self._returns else T.NONE
            )
            ret_facts = Facts()
        elif self._returns:
            # A recursive call's return type is the very thing being computed
            # here, so an unknown that came from calling into the project is
            # a placeholder, not an answer. Seeding the join from the settled
            # branches lets the effect fixpoint refine the rest -- otherwise
            # `fact(n - 1)` keeps `fact` unknown forever.
            settled = [
                r
                for r, provisional in zip(self._returns, self._provisional_returns, strict=False)
                if not provisional
            ] or self._returns
            inferred = T.join(*[r.type for r in settled])
            ret_facts = settled[0].facts
            for extra in settled[1:]:
                ret_facts = ret_facts.merge(extra.facts)
            if env.reachable:
                inferred = T.join(inferred, T.NONE)
                ret_facts = Facts()
        else:
            inferred = T.NONE
            ret_facts = Facts()

        effects = self._effects
        if info.is_async:
            effects = effects.add(Effect.SYNC)

        analysis = FunctionAnalysis(
            info=info,
            effects=effects,
            inferred_ret=inferred,
            ret_facts=ret_facts,
            locals={
                name: (env.get(name).type if env.get(name) else T.UNKNOWN)
                for name in env.names()
                if "." not in name and "[" not in name  # a narrowed path is not a local
            },
            dynamic=info.dynamic or self._dynamic_seen,
            unknown_callees=tuple(dict.fromkeys(self._unknown)),
            purity_blockers=tuple(dict.fromkeys(self._blockers)),
            native_blockers=tuple(dict.fromkeys(self._native_blockers)),
            escaping=set(self._escaping),
            mutated_params=set(self._mutated),
            delegated_writes=set(self._delegated),
            foreign_writes=self._foreign_writes,
            writes_only_locals=not self._external_writes
            and not (self._local_writes & self._shared_escapes()),
            writes_only_allocations=not self._external_writes,
            aliases=self._aliases,
            calls=set(self._calls),
        )
        info.effects = effects
        info.verified_pure = analysis.verified_pure
        if not info.ret_annotated:
            info.ret = inferred
            info.ret_facts = ret_facts
        return analysis

    def _seed_module_env(self, env: Env) -> None:
        for name, declared in _MODULE_DUNDERS.items():
            env.set(name, Binding(declared))
        for name, binding in self.symbols.imports.items():
            if binding.origin is None:
                env.set(name, Binding(T.Module_(binding.module)))
            else:
                env.set(name, Binding(self._imported_name_type(binding.module, binding.origin)))
        for name, info in self.symbols.classes.items():
            env.set(name, Binding(T.ClassObject(info.qualname, info.instance())))
        for name, info in self.symbols.functions.items():
            env.set(name, Binding(info.signature()))
        for name, declared in self.symbols.globals.items():
            env.set(name, Binding(declared, self.symbols.global_facts.get(name, Facts())))

    def _imported_name_type(self, module: str, origin: str, depth: int = 0) -> T.Type:
        qualname = f"{module}.{origin}"
        info = self.project.functions.get(qualname)
        if info is not None:
            return info.signature()
        cls = self.project.classes.get(qualname)
        if cls is not None:
            return T.ClassObject(cls.qualname, cls.instance())
        if module in self.project.modules:
            other = self.project.modules[module]
            if origin in other.globals:
                return other.globals[origin]
            # A package that imported the name itself: follow it to where
            # it is defined, as the annotation resolver does.
            onward = other.imports.get(origin)
            if onward is not None and depth < 8:
                if onward.origin is None:
                    return T.Module_(onward.module)
                return self._imported_name_type(onward.module, onward.origin, depth + 1)
        # Not the project's: the same tables `module.name` resolves through,
        # so `from pathlib import Path` and `pathlib.Path` are one thing.
        return self._library_name_type(qualname)

    def _library_name_type(self, qualname: str) -> T.Type:
        """What a library's `module.name` is, from the standard-library tables."""
        known = stdlib.MODULE_ATTRIBUTES.get(qualname)
        if known is not None:
            return known[0]
        described = stdlib.lookup(qualname)
        if described is not None:
            return described[0]
        external = self.project.external_types.get(qualname)
        if external is not None:
            instance = T.Instance(
                external, (), stdlib.EXTERNAL_MRO.get(qualname, (external, "object"))
            )
            return T.ClassObject(external, instance)
        return T.UNKNOWN

    def _stmt(self, node: ast.stmt, env: Env) -> None:
        if not env.reachable:
            return
        method = getattr(self, f"_stmt_{type(node).__name__}", None)
        if method is None:
            self._generic_stmt(node, env)
            return
        method(node, env)

    def _generic_stmt(self, node: ast.stmt, env: Env) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._expr(child, env)
            elif isinstance(child, ast.stmt):
                self._stmt(child, env)

    def _stmt_Expr(self, node: ast.Expr, env: Env) -> None:
        self._expr(node.value, env)

    def _stmt_Pass(self, node: ast.Pass, env: Env) -> None:
        return

    def _stmt_Import(self, node: ast.Import, env: Env) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.partition(".")[0]
            env.set(
                local,
                Binding(T.Module_(alias.name if alias.asname else alias.name.partition(".")[0])),
            )

    def _stmt_ImportFrom(self, node: ast.ImportFrom, env: Env) -> None:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            env.set(local, Binding(self._imported_name_type(module, alias.name)))

    def _stmt_Assign(self, node: ast.Assign, env: Env) -> None:
        if self._bind_type_alias(node.targets, node.value, env):
            return
        value = self._expr(node.value, env)
        for target in node.targets:
            self._bind_target(target, value, env, source=node.value)

    def _stmt_TypeAlias(self, node: ast.TypeAlias, env: Env) -> None:
        """`type X = ...` names a type; its right-hand side is not a value."""
        if isinstance(node.name, ast.Name):
            env.set(node.name.id, Binding(T.instance("type")))

    def _bind_type_alias(self, targets: list[ast.expr], value: ast.expr, env: Env) -> bool:
        """`X = Annotated[...]` names a type, so it is not evaluated as a value."""
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return False
        name = targets[0].id
        if self.symbols.type_aliases.get(name) is not value:
            return False
        env.set(name, Binding(T.instance("type")))
        return True

    def _stmt_AnnAssign(self, node: ast.AnnAssign, env: Env) -> None:
        if node.value is not None and self._bind_type_alias([node.target], node.value, env):
            return
        resolved = self.annotations.resolve(node.annotation)
        declared = Binding(resolved.type, resolved.facts)
        if node.value is not None:
            value = self._expr(node.value, env)
            if _display_fits(node.value, value.type, resolved.type):
                # A list, set, or dict written out in place has no type of
                # its own yet: `[stmt]` beside `list[ast.AST]` is a list of
                # AST nodes that happens to hold a statement, not a
                # `list[ast.stmt]` being narrowed. Each element is held to the
                # declared element type instead.
                value = Binding(resolved.type, value.facts)
            if not T.is_assignable(value.type, resolved.type):
                self._mismatch(
                    "E1301",
                    f"cannot assign `{value.type}` to a variable declared `{resolved.type}`",
                    node.value,
                    value.type,
                )
            else:
                declared = Binding(resolved.type, self._merge_declared(resolved.facts, value.facts))
                declared = Binding(declared.type, self._check_width(declared, node.value))
        self._bind_target(
            node.target, declared, env, declared_type=resolved.type, source=node.value
        )

    def _stmt_AugAssign(self, node: ast.AugAssign, env: Env) -> None:
        current = self._load_target(node.target, env)
        value = self._expr(node.value, env)
        if isinstance(node.target, ast.Name) and _mutates_in_place(current.type):
            # `ys += [x]` on a list is `ys.extend([x])`: the object every
            # alias of `ys` holds has changed, not just the binding.
            self._effects = self._effects.add(Effect.WRITE_OBJECT)
            self._note_mutation(node.target, env)
        result = self._binary(current, value, type(node.op), node)
        declared = None
        if isinstance(node.target, ast.Name):
            existing = env.get(node.target.id)
            if existing is not None and existing.facts.width is not None:
                declared = existing.facts
        if declared is not None:
            result = Binding(result.type, self._merge_declared(declared, result.facts))
            result = Binding(result.type, self._check_width(result, node))
        if isinstance(node.target, ast.Name) and self._is_provisional(node.value, result, env):
            self._provisional_locals.add(node.target.id)
        self._bind_target(node.target, result, env)

    def _stmt_Return(self, node: ast.Return, env: Env) -> None:
        if node.value is None:
            self._returns.append(Binding(T.NONE))
            self._provisional_returns.append(False)
        else:
            value = self._expr(node.value, env)
            self._returns.append(value)
            self._provisional_returns.append(self._is_provisional(node.value, value, env))
            if isinstance(node.value, ast.Name):
                self._returned_names.update(self._roots(node.value, node.value.id))
            self._mark_escape(node.value, env)
            info = self._current
            if info is not None and info.ret_annotated:
                if not T.is_assignable(value.type, info.ret):
                    self._mismatch(
                        "E1303",
                        f"returning `{value.type}` from a function declared `-> {info.ret}`",
                        node.value,
                        value.type,
                    )
                elif info.ret_facts.width is not None:
                    self._check_width(
                        Binding(value.type, self._merge_declared(info.ret_facts, value.facts)),
                        node.value,
                    )
        env.terminate()

    def _stmt_If(self, node: ast.If, env: Env) -> None:
        test = self._expr(node.test, env)
        then_env = self._narrow(node.test, env.fork(), True)
        else_env = self._narrow(node.test, env.fork(), False)
        constant = test.facts.constant if test.facts.has_constant else None
        for stmt in node.body:
            self._stmt(stmt, then_env)
        for stmt in node.orelse:
            self._stmt(stmt, else_env)
        if constant is True:
            merged = then_env
        elif constant is False:
            merged = else_env
        else:
            merged = then_env.merge(else_env)
        env.restore(merged.snapshot())
        env.reachable = merged.reachable

    def _stmt_While(self, node: ast.While, env: Env) -> None:
        self._loop(node, env, node.test, None)

    def _stmt_For(self, node: ast.For, env: Env) -> None:
        self._loop(node, env, None, node)

    def _stmt_AsyncFor(self, node: ast.AsyncFor, env: Env) -> None:
        self._effects = self._effects.add(Effect.SYNC)
        self._loop(node, env, None, node)

    def _loop(
        self,
        node: ast.stmt,
        env: Env,
        test: ast.expr | None,
        for_node: ast.For | ast.AsyncFor | None,
    ) -> None:
        if for_node is not None:
            iterable = self._expr(for_node.iter, env)
            element = self._iteration_element(iterable, for_node.iter)
            self._bind_target(for_node.target, element, env)

        body = getattr(node, "body", [])
        orelse = getattr(node, "orelse", [])
        for _ in range(_MAX_LOOP_ITERATIONS):
            before = env.snapshot()
            body_env = env.fork()
            if test is not None:
                self._expr(test, body_env)
                body_env = self._narrow(test, body_env, True)
            if for_node is not None:
                iterable = self._expr(for_node.iter, body_env)
                element = self._iteration_element(iterable, for_node.iter)
                self._bind_target(for_node.target, element, body_env)
            for stmt in body:
                self._stmt(stmt, body_env)
            merged = env.merge(body_env) if body_env.reachable else env
            env.restore(merged.snapshot())
            if env.equals(before):
                break
        else:
            self._widen(env)
        if test is not None:
            env.restore(self._narrow(test, env.fork(), False).snapshot())
        for stmt in orelse:
            self._stmt(stmt, env)
        env.reachable = True

    def _widen(self, env: Env) -> None:
        """Drop non-converging integer ranges rather than iterate forever."""
        for name in list(env.names()):
            binding = env.get(name)
            if binding is not None and binding.facts.int_range is not None:
                env.set(
                    name,
                    Binding(
                        binding.type,
                        binding.facts.with_(int_range=None, has_constant=False, constant=None),
                    ),
                )

    def _stmt_Break(self, node: ast.Break, env: Env) -> None:
        env.terminate()

    def _stmt_Continue(self, node: ast.Continue, env: Env) -> None:
        env.terminate()

    def _stmt_Raise(self, node: ast.Raise, env: Env) -> None:
        raised = "Exception"
        if node.exc is not None:
            binding = self._expr(node.exc, env)
            if isinstance(binding.type, T.Instance):
                raised = binding.type.name
            elif isinstance(binding.type, T.ClassObject):
                raised = binding.type.name.rpartition(".")[2]
        if node.cause is not None:
            self._expr(node.cause, env)
        self._effects = self._effects.add(raises=(raised,))
        env.terminate()

    def _stmt_Assert(self, node: ast.Assert, env: Env) -> None:
        self._expr(node.test, env)
        if node.msg is not None:
            self._expr(node.msg, env)
        self._effects = self._effects.add(raises=("AssertionError",))
        env.restore(self._narrow(node.test, env.fork(), True).snapshot())

    def _stmt_Delete(self, node: ast.Delete, env: Env) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                env.remove(target.id)
            else:
                self._expr(target, env)
                self._effects = self._effects.add(Effect.WRITE_OBJECT)
                if isinstance(target, (ast.Subscript, ast.Attribute)):
                    # `del ys[0]` shrinks the container `ys` names, wherever
                    # it came from.
                    self._note_mutation(target.value, env)
                else:
                    self._blockers.append("deletes an attribute or item")

    def _stmt_Try(self, node: ast.Try, env: Env) -> None:
        body_env = env.fork()
        for stmt in node.body:
            self._stmt(stmt, body_env)
        # A handler starts from the state before the try, because the body may
        # have raised anywhere inside it.
        merged = body_env if body_env.reachable else env.fork()
        # What follows is reachable only if the body or some handler can fall
        # out of the statement: `try: return a / except E: return b` cannot.
        reachable = body_env.reachable
        for handler in node.handlers:
            handler_env = env.fork()
            if handler.type is not None:
                bound = self._expr(handler.type, env)
                if handler.name:
                    instance = (
                        bound.type.instance_type
                        if isinstance(bound.type, T.ClassObject)
                        else T.instance("Exception")
                    )
                    handler_env.set(handler.name, Binding(instance or T.instance("Exception")))
            for stmt in handler.body:
                self._stmt(stmt, handler_env)
            if handler_env.reachable:
                merged = merged.merge(handler_env) if reachable else handler_env
                reachable = True
        for stmt in node.orelse:
            self._stmt(stmt, merged)
        env.restore(merged.snapshot())
        env.reachable = reachable
        for stmt in node.finalbody:
            self._stmt(stmt, env)

    _stmt_TryStar = _stmt_Try

    def _stmt_With(self, node: ast.With, env: Env) -> None:
        dynamic = False
        for item in node.items:
            if self._is_dynamic_marker(item.context_expr):
                dynamic = True
                continue
            value = self._expr(item.context_expr, env)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars, Binding(self._enter_type(value.type)), env)
        if dynamic:
            self._enter_dynamic(node)
        for stmt in node.body:
            self._stmt(stmt, env)
        if dynamic:
            self._dynamic_depth -= 1
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            self.module.dynamic_spans.append((node.lineno, end))

    def _stmt_AsyncWith(self, node: ast.AsyncWith, env: Env) -> None:
        self._effects = self._effects.add(Effect.SYNC)
        self._stmt_With(node, env)  # type: ignore[arg-type]

    def _stmt_FunctionDef(self, node: ast.FunctionDef, env: Env) -> None:
        self._check_decorators(node, env)
        qualname = f"{self.symbols.name}.{node.name}"
        info = self.project.functions.get(qualname)
        env.set(node.name, Binding(info.signature() if info else T.UNKNOWN))

    _stmt_AsyncFunctionDef = _stmt_FunctionDef

    def _stmt_ClassDef(self, node: ast.ClassDef, env: Env) -> None:
        self._check_class_construction(node)
        self._check_decorators(node, env)
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
                # Methods are not visited as statements, but decoration
                # rewrites them all the same.
                self._check_decorators(child, env)
        info = self.symbols.classes.get(node.name)
        env.set(
            node.name, Binding(T.ClassObject(info.qualname, info.instance()) if info else T.UNKNOWN)
        )

    def _check_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, env: Env
    ) -> None:
        """A decorator nobody vouches for may replace the decorated object.

        The symbol table would keep believing the declared signature while
        the runtime object is whatever the decorator returned -- the checker
        would then approve calls the program cannot take. Strict PPY demands
        known semantics, a plugin's word, or an explicit dynamic boundary
        (spec: strict decorator semantics).
        """
        if not self.strict or not node.decorator_list:
            return
        resolver = self.project.resolver(self.symbols)
        names = [resolver.decorator_identity(d) for d in node.decorator_list]
        if "ppy.dynamic" in names:
            return
        for decorator, name in zip(node.decorator_list, names, strict=True):
            if self._decorator_vouched(decorator, name, env, resolver):
                continue
            spelled = ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
            self._dynamic_feature(
                "E1204",
                f"decorator `@{spelled}` applies a transform nobody vouches for",
                decorator,
                help="use a vouched decorator, register this one's semantics with a "
                "plugin, or mark the decorated definition `@ppy.dynamic`",
            )

    def _decorator_vouched(  # type: ignore[no-untyped-def]
        self, decorator: ast.expr, name: str, env: Env, resolver
    ) -> bool:
        from .decorators import semantics_of

        if semantics_of(name, self.plugins) is not None:
            return True
        if name == "functools.partial" and isinstance(decorator, ast.Call) and decorator.args:
            # `@partial(D, ...)` binds arguments of `D`; the transform is D's.
            inner = resolver.decorator_identity(decorator.args[0])
            return semantics_of(inner, self.plugins) is not None
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            # `@app.get(...)`: the decorator is an attribute of a typed value,
            # and its owner's plugin may vouch for it by qualified name.
            binding = env.get(target.value.id)
            base = T.strip_literal(binding.type) if binding is not None else None
            if isinstance(base, T.Instance):
                return semantics_of(f"{base.name}.{target.attr}", self.plugins) is not None
        return False

    def _check_class_construction(self, node: ast.ClassDef) -> None:
        """Class construction must be static: PPY reorders and compiles classes.

        A base built by a call, or a metaclass nobody vouches for, runs
        arbitrary code at definition time and can hand back a class whose
        shape no analysis predicted (spec: strict class semantics).
        """
        for base in node.bases:
            if _static_base(base):
                continue
            self._dynamic_feature(
                "E1507",
                f"base `{ast.unparse(base)}` of class `{node.name}` is computed at runtime",
                base,
                help="name the base class directly, or wrap the definition "
                "in a ppy.dynamic boundary",
            )
        for keyword in node.keywords:
            if keyword.arg != "metaclass":
                continue
            canonical = self.project.resolver(self.symbols).canonical(keyword.value)
            if canonical in _KNOWN_METACLASSES:
                continue
            self._dynamic_feature(
                "E1507",
                f"metaclass `{ast.unparse(keyword.value)}` of class `{node.name}` "
                "rewrites class construction",
                keyword.value,
                help="use a plain class, a vouched metaclass such as `abc.ABCMeta`, "
                "or a ppy.dynamic boundary",
            )
        from .decorators import class_construction

        construction = class_construction(node)
        for statement in construction.executable:
            self._dynamic_feature(
                "E1507",
                f"the body of class `{node.name}` executes code at definition time",
                statement,
                help="keep the class body declarative -- fields, methods, "
                "constants -- or wrap the definition in a ppy.dynamic boundary",
            )
        for call in construction.value_calls:
            constructed = self._project_class_of(call.func)  # type: ignore[attr-defined]
            if constructed is not None and self._nontrivial_hook(constructed, "__set_name__"):
                self._dynamic_feature(
                    "E1507",
                    f"class `{node.name}` binds a `{constructed.name}`, whose "
                    "`__set_name__` runs at class creation",
                    call,
                    help="construct the descriptor through a plugin-vouched "
                    "factory, or wrap the definition in a ppy.dynamic boundary",
                )
        for base in node.bases:
            hooked = self._subclass_hook(base)
            if hooked is not None:
                self._dynamic_feature(
                    "E1507",
                    f"base `{ast.unparse(base)}` runs `__init_subclass__` "
                    f"when class `{node.name}` is created",
                    base,
                    help="make the hook trivial, vouch for it via a plugin, "
                    "or wrap the definition in a ppy.dynamic boundary",
                )

    def _project_class_of(self, expr: ast.expr):  # type: ignore[no-untyped-def]
        canonical = self.project.resolver(self.symbols).canonical(expr)
        if canonical is None:
            return None
        return self.project.classes.get(canonical)

    def _nontrivial_hook(self, info, name: str) -> bool:  # type: ignore[no-untyped-def]
        """Does `info` define `name` with a body that actually does something?"""
        method = info.methods.get(name)
        if method is None:
            return False
        return not all(
            isinstance(statement, ast.Pass)
            or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
            for statement in method.node.body
        )

    def _subclass_hook(self, base: ast.expr):  # type: ignore[no-untyped-def]
        """The project class in `base`'s chain whose `__init_subclass__` runs."""
        info = self._project_class_of(base)
        if info is None:
            return None
        for entry in (info.qualname, *info.mro):
            candidate = self.project.classes.get(entry)
            if candidate is not None and self._nontrivial_hook(candidate, "__init_subclass__"):
                return candidate
        return None

    def _stmt_Global(self, node: ast.Global, env: Env) -> None:
        self._effects = self._effects.add(Effect.WRITE_GLOBAL)
        self._blockers.append(f"declares `global {', '.join(node.names)}`")

    def _stmt_Nonlocal(self, node: ast.Nonlocal, env: Env) -> None:
        self._effects = self._effects.add(Effect.WRITE_OBJECT)

    def _stmt_Match(self, node: ast.Match, env: Env) -> None:
        subject = self._expr(node.subject, env)
        merged: Env | None = None
        for case in node.cases:
            case_env = env.fork()
            if isinstance(node.subject, ast.Name):
                case_env.set(node.subject.id, subject)
            self._bind_pattern(case.pattern, subject, case_env, node.subject)
            if case.guard is not None:
                self._expr(case.guard, case_env)
                case_env = self._narrow(case.guard, case_env, True)
            for stmt in case.body:
                self._stmt(stmt, case_env)
            merged = case_env if merged is None else merged.merge(case_env)
            # A later case is only reached when the earlier ones did not match,
            # so `case _` after `case None` sees a subject that cannot be None.
            if case.guard is None:
                subject = self._without_pattern(case.pattern, subject, env)
        if merged is not None:
            env.restore(merged.snapshot())
            env.reachable = merged.reachable

    def _without_pattern(self, pattern: ast.pattern, subject: Binding, env: Env) -> Binding:
        """The subject as it stands once `pattern` has failed to match."""
        matched = self._pattern_type(pattern, env)
        if matched is None:
            return subject
        members = list(T.members_of(subject.type))
        remaining = [m for m in members if not T.is_assignable(m, matched)]
        if not remaining or len(remaining) == len(members):
            return subject
        return Binding(T.union(*remaining), subject.facts)

    def _pattern_type(self, pattern: ast.pattern, env: Env) -> T.Type | None:
        """The type a pattern matches in full, or None when it matches partially."""
        if isinstance(pattern, ast.MatchSingleton):
            return T.type_of_constant(pattern.value)
        if isinstance(pattern, ast.MatchClass):
            if pattern.patterns or pattern.kwd_patterns:
                return None
            cls = self._expr(pattern.cls, env)
            return cls.type.instance_type if isinstance(cls.type, T.ClassObject) else None
        if isinstance(pattern, ast.MatchOr):
            parts = [self._pattern_type(sub, env) for sub in pattern.patterns]
            return T.union(*parts) if all(p is not None for p in parts) else None
        if isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
            return self._pattern_type(pattern.pattern, env)
        return None

    def _bind_target(
        self,
        target: ast.expr,
        value: Binding,
        env: Env,
        declared_type: T.Type | None = None,
        source: ast.expr | None = None,
    ) -> None:
        if isinstance(target, ast.Name):
            binding = Binding(declared_type or value.type, value.facts)
            env.set(target.id, binding)
            _forget_attributes(env, target.id)
            self._record(target, binding)
            if self._current is None:
                self.symbols.globals.setdefault(target.id, binding.type)
                self.symbols.global_facts.setdefault(target.id, binding.facts)
            elif target.id not in env or self._is_module_global(target.id):
                self._effects = self._effects.add(Effect.WRITE_GLOBAL)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._unpack(target, value, env)
        elif isinstance(target, ast.Attribute):
            owner = self._expr(target.value, env)
            self._effects = self._effects.add(Effect.WRITE_OBJECT)
            self._note_mutation(target.value, env)
            self._check_attribute_assignment(owner, target, value)
            path = _attribute_path(target)
            if path is not None:
                # The write is the new truth about the attribute, and about
                # anything reached through it.
                _forget_attributes(env, path)
                env.set(path, Binding(declared_type or value.type, value.facts))
        elif isinstance(target, ast.Subscript):
            self._expr(target.value, env)
            self._expr(target.slice, env)
            self._effects = self._effects.add(
                Effect.WRITE_OBJECT, raises=("IndexError", "KeyError", "TypeError")
            )
            self._note_mutation(target.value, env)
            container = _attribute_path(target.value) or (
                target.value.id if isinstance(target.value, ast.Name) else None
            )
            if container is not None:
                # `args[0] = ...` ends what was known about `args[0].facts`,
                # and `args[i] = ...` about every `args[...]`.
                _forget_attributes(env, container)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, Binding(T.list_of(value.type)), env)

    def _unpack(self, target: ast.Tuple | ast.List, value: Binding, env: Env) -> None:
        base = T.strip_literal(value.type)
        elements: list[T.Type] = []
        members = [T.strip_literal(m) for m in T.members_of(base)]
        tuples = [m for m in members if isinstance(m, T.Tuple_) and not m.homogeneous]
        if isinstance(base, T.Tuple_) and not base.homogeneous:
            elements = list(base.items)
        elif len(tuples) == len(members) > 1 and len({len(t.items) for t in tuples}) == 1:
            # `seen.get(key, (0, node))` is one tuple or the other; each
            # target is what its position holds in either.
            elements = [
                T.join(*[t.items[index] for t in tuples]) for index in range(len(tuples[0].items))
            ]
        else:
            element = B.element_type(base)
            elements = [element] * len(target.elts)
        starred = any(isinstance(e, ast.Starred) for e in target.elts)
        if (
            not starred
            and len(elements) != len(target.elts)
            and isinstance(base, T.Tuple_)
            and not base.homogeneous
        ):
            self._error(
                "E1301",
                f"cannot unpack {len(elements)} values into {len(target.elts)} targets",
                target,
            )
        for index, element_target in enumerate(target.elts):
            if isinstance(element_target, ast.Starred):
                self._bind_target(
                    element_target.value,
                    Binding(T.list_of(elements[index] if index < len(elements) else T.UNKNOWN)),
                    env,
                )
                continue
            element = elements[index] if index < len(elements) else T.UNKNOWN
            self._bind_target(element_target, Binding(element), env)

    def _load_target(self, target: ast.expr, env: Env) -> Binding:
        if isinstance(target, ast.Name):
            binding = env.get(target.id)
            if binding is None:
                self._error("E1101", f"`{target.id}` is used before it is bound", target)
                return Binding(T.UNKNOWN)
            return binding
        return self._expr(target, env)

    def _bind_pattern(
        self, pattern: ast.pattern, subject: Binding, env: Env, source: ast.expr
    ) -> None:
        if isinstance(pattern, ast.MatchAs):
            inner = subject
            if pattern.pattern is not None:
                self._bind_pattern(pattern.pattern, subject, env, source)
                inner = env.get(pattern.name) or subject if pattern.name else subject
            if pattern.name:
                env.set(pattern.name, inner)
        elif isinstance(pattern, ast.MatchSingleton):
            if isinstance(source, ast.Name):
                singleton = T.type_of_constant(pattern.value)
                env.set(
                    source.id, Binding(singleton, Facts(constant=pattern.value, has_constant=True))
                )
        elif isinstance(pattern, ast.MatchValue):
            value = self._expr(pattern.value, env)
            if isinstance(source, ast.Name) and value.facts.has_constant:
                env.set(source.id, Binding(value.type, value.facts))
        elif isinstance(pattern, ast.MatchClass):
            cls = self._expr(pattern.cls, env)
            narrowed = cls.type.instance_type if isinstance(cls.type, T.ClassObject) else None
            if narrowed is not None and isinstance(source, ast.Name):
                env.set(source.id, Binding(narrowed))
            for sub in pattern.patterns:
                self._bind_pattern(sub, Binding(T.UNKNOWN), env, source)
            for sub in pattern.kwd_patterns:
                self._bind_pattern(sub, Binding(T.UNKNOWN), env, source)
        elif isinstance(pattern, ast.MatchSequence):
            element = B.element_type(subject.type)
            for sub in pattern.patterns:
                self._bind_pattern(sub, Binding(element), env, source)
        elif isinstance(pattern, ast.MatchMapping):
            for sub in pattern.patterns:
                self._bind_pattern(sub, Binding(T.UNKNOWN), env, source)
            if pattern.rest:
                env.set(pattern.rest, Binding(T.dict_of(T.STR, T.ANY)))
        elif isinstance(pattern, ast.MatchOr):
            for sub in pattern.patterns:
                self._bind_pattern(sub, subject, env, source)
        elif isinstance(pattern, ast.MatchStar) and pattern.name:
            env.set(pattern.name, Binding(T.list_of(B.element_type(subject.type))))

    def _expr(self, node: ast.expr, env: Env) -> Binding:
        method = getattr(self, f"_expr_{type(node).__name__}", None)
        binding = Binding(T.UNKNOWN) if method is None else method(node, env)
        self._record(node, binding)
        return binding

    def _record(self, node: ast.AST, binding: Binding) -> None:
        if self.record:
            self.module.node_types[id(node)] = binding.type
            self.module.node_facts[id(node)] = binding.facts

    def _expr_Constant(self, node: ast.Constant, env: Env) -> Binding:
        t = T.type_of_constant(node.value)
        facts = Facts(constant=node.value, has_constant=True)
        if isinstance(node.value, bool):
            facts = facts.with_(int_range=IntRange(int(node.value), int(node.value)))
        elif isinstance(node.value, int):
            facts = facts.with_(int_range=IntRange(node.value, node.value))
        elif isinstance(node.value, str):
            facts = facts.with_(length=len(node.value))
        return Binding(t, facts)

    def _expr_Name(self, node: ast.Name, env: Env) -> Binding:
        binding = env.get(node.id)
        if binding is not None:
            if self._is_module_global(node.id) and self._current is not None:
                self._effects = self._effects.add(Effect.READ_GLOBAL)
                self._blockers.append(f"reads mutable global `{node.id}`")
            return binding
        if node.id in T.BUILTIN_MRO:
            return Binding(T.ClassObject(node.id, T.instance(node.id)))
        if node.id == "super":
            inherited = self._inherited_type()
            if inherited is not None:
                return Binding(T.Callable_((), inherited, "super"))
            if self._current is not None and self._current.owner is not None:
                # The base is not the project's -- `Exception`, a library
                # class -- so what `super()` forwards to is not modeled. It
                # is still `super()`, not an undefined name.
                return Binding(T.Callable_((), T.OBJECT, "super"))
        if B.is_builtin(node.id) or node.id in {"None", "True", "False"}:
            return Binding(T.Callable_((), T.UNKNOWN, node.id))
        if node.id in {"Ellipsis", "NotImplemented"}:
            # The two builtin singletons that are values, not callables:
            # `return NotImplemented` is how an operator method declines.
            return Binding(T.OBJECT)
        self._error("E1101", f"`{node.id}` is not defined at this point", node)
        return Binding(T.UNKNOWN)

    def _inherited_type(self) -> T.Type | None:
        """What zero-argument `super()` stands for inside the current method.

        The proxy forwards to the next class in the MRO, and with single
        inheritance that is the first base, so an instance of it describes what
        attribute lookup will find.
        """
        info = self._current
        if info is None or info.owner is None:
            return None
        owner = self.project.classes.get(info.owner)
        if owner is None:
            return None
        for base in owner.base_names:
            resolved = self.project.classes.get(base) or self.project.classes.get(
                f"{owner.module}.{base}"
            )
            if resolved is not None:
                return resolved.instance()
        return None

    def _expr_BinOp(self, node: ast.BinOp, env: Env) -> Binding:
        left = self._expr(node.left, env)
        right = self._expr(node.right, env)
        return self._binary(left, right, type(node.op), node)

    def _expr_UnaryOp(self, node: ast.UnaryOp, env: Env) -> Binding:
        operand = self._expr(node.operand, env)
        if isinstance(node.op, ast.Not):
            constant = not operand.facts.constant if operand.facts.has_constant else None
            facts = Facts(constant=constant, has_constant=operand.facts.has_constant)
            return Binding(T.BOOL, facts)
        base = T.strip_literal(operand.type)
        symbol = {ast.USub: "u-", ast.UAdd: "u+", ast.Invert: "u~"}.get(type(node.op), "")
        plugin_result = self._plugin_operator(symbol, [operand], node)
        if plugin_result is not None:
            return plugin_result
        if isinstance(node.op, ast.USub):
            facts = Facts()
            if operand.facts.int_range is not None:
                facts = facts.with_(int_range=operand.facts.int_range.negate())
            if operand.facts.has_constant and isinstance(operand.facts.constant, (int, float)):
                value = -operand.facts.constant
                facts = facts.with_(constant=value, has_constant=True)
            if base == T.BOOL:
                base = T.INT
            return Binding(base, facts)
        if isinstance(node.op, ast.UAdd):
            return Binding(T.INT if base == T.BOOL else base, operand.facts)
        if isinstance(node.op, ast.Invert):
            if not T.is_assignable(base, T.INT):
                self._error("E1302", f"`~` is not defined for `{base}`", node)
            return Binding(T.INT)
        return Binding(T.UNKNOWN)

    def _expr_BoolOp(self, node: ast.BoolOp, env: Env) -> Binding:
        # Each operand is only reached when the ones before it were truthy
        # (`and`) or falsy (`or`), which is what makes `x is None or x.f` work.
        scope = env.fork()
        keeps_going = isinstance(node.op, ast.And)
        parts: list[Binding] = []
        assigned: dict[str, Binding] = {}
        for value in node.values:
            parts.append(self._expr(value, scope))
            # A `:=` inside the chain bound a name in the fork. The name is
            # bound for whatever follows the expression, so what it assigned
            # -- before the chain narrowed it, which only the `if` body may
            # rely on -- is handed back to the enclosing scope.
            for inner in ast.walk(value):
                if isinstance(inner, ast.NamedExpr) and isinstance(inner.target, ast.Name):
                    bound = scope.get(inner.target.id)
                    if bound is not None:
                        assigned[inner.target.id] = bound
            scope = self._narrow(value, scope, keeps_going)
        for name, bound in assigned.items():
            env.set(name, bound)
        types = [p.type for p in parts]
        if isinstance(node.op, ast.Or):
            # `program or {}`: an operand before the last is the result only
            # when it is truthy, and `None` never is.
            types = [*(T.remove_none(t) for t in types[:-1]), types[-1]]
        if all(p.facts.has_constant for p in parts):
            values = [p.facts.constant for p in parts]
            result = values[0]
            for value in values[1:]:
                result = (result and value) if isinstance(node.op, ast.And) else (result or value)
            return Binding(T.join(*types), Facts(constant=result, has_constant=True))
        return Binding(T.join(*types))

    def _expr_Compare(self, node: ast.Compare, env: Env) -> Binding:
        left = self._expr(node.left, env)
        operands = [left]
        operands.extend(self._expr(comparator, env) for comparator in node.comparators)
        for op, _right in zip(node.ops, operands[1:], strict=False):
            if isinstance(op, (ast.In, ast.NotIn)):
                self._effects = self._effects.add(raises=("TypeError",))
        if len(node.ops) == 1:
            plugin_result = self._plugin_operator(
                _COMPARE_OPS.get(type(node.ops[0]), ""), operands, node
            )
            if plugin_result is not None:
                return plugin_result

        constant = self._fold_compare(node, operands)
        facts = Facts(int_range=IntRange(0, 1))
        if constant is not None:
            facts = facts.with_(constant=constant, has_constant=True)
        return Binding(T.BOOL, facts)

    def _fold_compare(self, node: ast.Compare, operands: list[Binding]) -> bool | None:
        if not all(o.facts.has_constant for o in operands):
            return None
        values = [o.facts.constant for o in operands]
        try:
            result = True
            for index, op in enumerate(node.ops):
                left, right = values[index], values[index + 1]
                match op:
                    case ast.Eq():
                        step = left == right
                    case ast.NotEq():
                        step = left != right
                    case ast.Lt():
                        step = left < right  # type: ignore[operator]
                    case ast.LtE():
                        step = left <= right  # type: ignore[operator]
                    case ast.Gt():
                        step = left > right  # type: ignore[operator]
                    case ast.GtE():
                        step = left >= right  # type: ignore[operator]
                    case ast.Is():
                        step = left is right
                    case ast.IsNot():
                        step = left is not right
                    case _:
                        return None
                result = result and bool(step)
                if not result:
                    break
            return result
        except TypeError:
            return None

    def _expr_IfExp(self, node: ast.IfExp, env: Env) -> Binding:
        test = self._expr(node.test, env)
        then_binding = self._expr(node.body, self._narrow(node.test, env.fork(), True))
        else_binding = self._expr(node.orelse, self._narrow(node.test, env.fork(), False))
        if test.facts.has_constant:
            return then_binding if test.facts.constant else else_binding
        return then_binding.merge(else_binding)

    def _expr_Call(self, node: ast.Call, env: Env) -> Binding:
        if self._check_forbidden_call(node, env):
            return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)
        if isinstance(node.func, ast.Subscript):
            checked = self._checked_conversion(node, env)
            if checked is not None:
                return checked
            typed_read = self._typed_input(node, env)
            if typed_read is not None:
                return typed_read
            allocated = self._typed_buffer(node, env)
            if allocated is not None:
                return allocated
        imported = self._constant_import(node)
        if imported is not None:
            return imported
        callee = self._expr(node.func, env)
        args = [
            self._expr(arg.value if isinstance(arg, ast.Starred) else arg, env) for arg in node.args
        ]
        keywords = {kw.arg: self._expr(kw.value, env) for kw in node.keywords if kw.arg}
        retains = not _is_inspecting_builtin(node, env)
        for argument in node.args:
            self._mark_escape(argument, env, retains=retains)
        for keyword in node.keywords:
            self._mark_escape(keyword.value, env, retains=retains)

        if isinstance(node.func, ast.Name) and node.func.id not in env:
            result = B.call_builtin(node.func.id, [(a.type, a.facts) for a in args])
            if result is not None:
                self._effects = self._effects | result.effects
                if Effect.IO in result.effects:
                    self._blockers.append(f"calls `{node.func.id}` which performs I/O")
                if Effect.PYTHON_CALLBACK in result.effects:
                    self._native_blockers.append(f"`{node.func.id}` may invoke a Python callback")
                return Binding(result.type, result.facts)

        plugin_result = self._plugin_call(node, args, keywords, env)
        if plugin_result is not None:
            return plugin_result

        if isinstance(callee.type, T.ClassObject):
            return self._construct(callee.type, node, args, keywords, env)
        if isinstance(callee.type, T.Callable_):
            refined = self._refine_builtin_method(callee.type, args)
            if refined is not None:
                return refined
            decided = stdlib.call(callee.type.qualname, [(a.type, a.facts) for a in args])
            if decided is not None:
                self._effects = self._effects | decided[1]
                return Binding(decided[0])
            described = stdlib.lookup(callee.type.qualname)
            if described is not None:
                self._effects = self._effects | described[1]
                if described[1].violations():
                    self._blockers.append(
                        f"calls `{callee.type.qualname}` with effects: {described[1]}"
                    )
                self._native_blockers.append(f"`{callee.type.qualname}` has no native lowering")
                return Binding(callee.type.ret)
            if callee.type.qualname in _MUTATING_METHODS:
                self._effects = self._effects.add(
                    Effect.WRITE_OBJECT, raises=("IndexError", "KeyError")
                )
                if isinstance(node.func, ast.Attribute):
                    self._note_mutation(node.func.value, env)
                    self._widen_empty_container(node.func, callee.type.qualname, args, env)
                    self._blockers.append(
                        f"calls `{callee.type.qualname}`, which mutates its receiver"
                    )
                return Binding(callee.type.ret)
            return self._call_signature(
                callee.type, node, args, keywords, bound=id(node.func) in self._bound_methods
            )
        if isinstance(callee.type, T.Module_):
            self._error("E1306", "a module is not callable", node)
            return Binding(T.UNKNOWN)
        return self._opaque_call(node, callee)

    def _widen_empty_container(
        self, func: ast.Attribute, qualname: str, args: list[Binding], env: Env
    ) -> None:
        """`out = []` followed by `out.append(x)` gives `out` an element type."""
        if not isinstance(func.value, ast.Name) or not args:
            return
        binding = env.get(func.value.id)
        if binding is None:
            return
        base = T.strip_literal(binding.type)
        if not isinstance(base, T.Instance) or not base.args:
            return
        added = T.strip_literal(args[0].type)
        if isinstance(added, (T.UnknownType, T.AnyType, T.NeverType)):
            return
        if qualname in {"list.append", "list.insert", "set.add"}:
            element = T.strip_literal(args[-1].type)
        elif qualname in {"list.extend", "set.update"}:
            element = T.strip_literal(B.element_type(added))
        else:
            return
        if base.name in {"list", "set"} and isinstance(base.args[0], T.NeverType):
            env.set(func.value.id, Binding(T.instance(base.name, element), binding.facts))

    def _construct(
        self,
        cls: T.ClassObject,
        node: ast.Call,
        args: list[Binding],
        keywords: dict[str | None, Binding],
        env: Env,
    ) -> Binding:
        info = self.project.classes.get(cls.name)
        self._effects = self._effects.add(Effect.ALLOC)
        if info is None:
            short = cls.name.rpartition(".")[2]
            if short in T.BUILTIN_MRO:
                return Binding(T.instance(short))
            return Binding(cls.instance_type or T.UNKNOWN)
        init = info.methods.get("__init__")
        if init is not None:
            self._effects = self._effects | init.effects
            self._calls.add(init.qualname)
            self._check_arity(init, node, len(args) + 1, set(keywords), skip_self=True)
        elif info.is_dataclass or info.is_pydantic:
            self._check_field_construction(info, node, args, keywords)
        facts = Facts(exact_class=cls.name)
        return Binding(info.instance(), facts)

    def _check_field_construction(
        self,
        info: ClassInfo,
        node: ast.Call,
        args: list[Binding],
        keywords: dict[str | None, Binding],
    ) -> None:
        fields = [name for name in info.fields if name not in info.class_vars]
        for index, argument in enumerate(args):
            if index >= len(fields):
                self._error("E1305", f"`{info.name}` takes {len(fields)} field(s)", node)
                break
            expected = info.fields[fields[index]]
            if not self._accepts(info, expected, argument.type):
                self._mismatch(
                    "E1301",
                    f"field `{fields[index]}` of `{info.name}` "
                    f"expects `{expected}`, got `{argument.type}`",
                    node.args[index],
                    argument.type,
                )
        for name, binding in keywords.items():
            if name is None:
                continue
            if name not in info.fields:
                self._error("E1305", f"`{info.name}` has no field `{name}`", node)
                continue
            expected = info.fields[name]
            if not self._accepts(info, expected, binding.type):
                self._mismatch(
                    "E1301",
                    f"field `{name}` of `{info.name}` expects `{expected}`, got `{binding.type}`",
                    node,
                    binding.type,
                )

    def _accepts(self, info: ClassInfo, expected: T.Type, actual: T.Type) -> bool:
        if T.is_assignable(actual, expected):
            return True
        # A coercing Pydantic model accepts constructor input that differs from
        # the validated output type (spec 23.2).
        return info.is_pydantic

    def _call_signature(
        self,
        signature: T.Callable_,
        node: ast.Call,
        args: list[Binding],
        keywords: dict[str | None, Binding],
        *,
        bound: bool = False,
    ) -> Binding:
        info = self.project.functions.get(signature.qualname)
        if info is not None:
            # A method reached through an instance was handed over with its
            # receiver already bound -- one parameter fewer than the
            # definition -- whether it is called on the spot or later, by
            # whatever name it was passed along under.
            bound = bound or (
                info.is_method
                and not info.is_static
                and len(signature.params) + 1 == len(info.params)
            )
            self._calls.add(info.qualname)
            self._effects = self._effects | info.effects
            self._note_callee_writes(info, node)
            if info.effects.violations():
                self._blockers.append(
                    f"calls `{info.name}` with effects: "
                    + ", ".join(sorted(str(e) for e in info.effects.violations()))
                )
            self._check_arity(
                info, node, len(args) + (1 if bound else 0), set(keywords), skip_self=bound
            )
            self._check_argument_types(info, node, args, keywords, bound=bound)
            if info.dynamic:
                self._native_blockers.append(f"`{info.name}` is a dynamic boundary")
            return Binding(info.ret, info.ret_facts if info.ret_annotated else Facts())
        for index, (param, argument) in enumerate(zip(signature.params, args, strict=False)):
            fits = T.is_assignable(argument.type, param.type)
            if not fits and signature.qualname in _LOOKUPS and index == 0:
                fits = T.is_assignable(param.type, argument.type)
            if not fits:
                self._mismatch(
                    "E1301",
                    f"argument {index + 1} expects `{param.type}`, got `{argument.type}`",
                    node.args[index] if index < len(node.args) else node,
                    argument.type,
                )
        return Binding(signature.ret)

    def _check_arity(
        self,
        info: FunctionInfo,
        node: ast.Call,
        positional: int,
        keyword_names: set[str | None],
        *,
        skip_self: bool = False,
    ) -> None:
        params = info.params[1:] if skip_self and info.params else info.params
        if any(p.kind in {"var_positional", "var_keyword"} for p in params):
            return
        supplied = positional - (1 if skip_self else 0)
        accepts_positional = [
            p for p in params if p.kind in {"positional_only", "positional_or_keyword"}
        ]
        required = [
            p
            for p in params
            if not p.has_default and p.kind not in {"var_positional", "var_keyword"}
        ]
        named = {n for n in keyword_names if n is not None}
        # `f(*pair)` or `f(**options)`: how many arrive is the value's
        # business, not the call site's. The names still have to exist.
        unpacked = None in keyword_names or any(isinstance(a, ast.Starred) for a in node.args)
        if not unpacked and supplied > len(accepts_positional):
            self._error(
                "E1305",
                f"`{info.name}` takes {len(accepts_positional)} "
                f"positional argument(s), {supplied} given",
                node,
            )
            return
        covered = supplied + len(named)
        if not unpacked and covered < len(required):
            missing = [p.name for p in required[supplied:] if p.name not in named]
            if missing:
                self._error(
                    "E1305",
                    f"`{info.name}` is missing argument(s): {', '.join(missing)}",
                    node,
                )
        valid = {p.name for p in params}
        for name in named:
            if name not in valid:
                self._error("E1305", f"`{info.name}` has no parameter `{name}`", node)

    def _check_argument_types(
        self,
        info: FunctionInfo,
        node: ast.Call,
        args: list[Binding],
        keywords: dict[str | None, Binding],
        *,
        bound: bool = False,
    ) -> None:
        offset = 1 if bound or (info.is_method and not info.is_static) else 0
        # Positional order, keywords by name, the receiver that is never
        # written down: the same rules the conversion passes use, from the
        # same code, so the two cannot drift apart.
        positional = args[: len(positional_values(node.args))]
        for reached in bind_call(info.params, positional, list(keywords.items()), offset=offset):
            param, argument = reached.param, reached.value
            if isinstance(param.type, T.UnknownType):
                continue
            where = node
            if not reached.keyword and reached.index - offset < len(node.args):
                where = node.args[reached.index - offset]
            if not T.is_assignable(argument.type, param.type):
                self._mismatch(
                    "E1301",
                    f"`{info.name}` parameter `{param.name}` "
                    f"expects `{param.type}`, got `{argument.type}`",
                    where,
                    argument.type,
                )
            elif param.facts.width is not None:
                self._check_width(
                    Binding(argument.type, self._merge_declared(param.facts, argument.facts)),
                    where,
                    what=f"parameter `{param.name}`",
                )

    def _opaque_call(self, node: ast.Call, callee: Binding) -> Binding:
        if id(node.func) in self._reported:
            # The receiver already produced the diagnostic that explains this;
            # a second one about an unknown signature would send the reader
            # after a missing stub that does not exist.
            return Binding(T.UNKNOWN)
        name = ast.unparse(node.func)
        self._unknown.append(name)
        self._effects = self._effects.add(Effect.EXTERNAL_UNKNOWN)
        self._blockers.append(f"calls `{name}` with unknown effects")
        self._native_blockers.append(f"`{name}` has no native lowering")
        if self._dynamic_depth:
            return Binding(T.ANY)
        if self.strict:
            self._error(
                "E1306",
                f"the signature of `{name}` is unknown, so this call cannot be typed",
                node,
                help="add a stub or plugin for it, or move the call inside a ppy.dynamic boundary",
            )
        return Binding(T.UNKNOWN)

    def _expr_Attribute(self, node: ast.Attribute, env: Env) -> Binding:
        owner = self._expr(node.value, env)
        self._attribute_owners[id(node)] = owner.type
        return self._attribute(owner, node, env)

    def _attribute(self, owner: Binding, node: ast.Attribute, env: Env) -> Binding:
        self._bound_methods.discard(id(node))
        if isinstance(owner.type, T.Module_):
            return self._module_attribute(owner.type.name, node)
        narrowed = env.get(_attribute_path(node) or "")
        if narrowed is not None:
            # `if self.x is not None:` was checked above; this read is what
            # the check was for.
            return narrowed
        optional = self._optional_receiver(owner, node)
        if optional is not None:
            return optional
        if isinstance(owner.type, (T.AnyType, T.UnknownType)):
            if isinstance(owner.type, T.DynamicType):
                # Half a dynamic value is still a dynamic value.
                return Binding(T.DYNAMIC)
            return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)
        if isinstance(owner.type, T.ClassObject):
            info = self.project.classes.get(owner.type.name)
            if info is not None:
                found = info.lookup(node.attr, self.project)
                if found is not None:
                    return Binding(found[0], found[1])
            dunder = _CLASS_DUNDERS.get(node.attr)
            if dunder is not None:
                return Binding(dunder)
            return Binding(T.UNKNOWN)
        base = T.strip_literal(owner.type)
        if isinstance(base, T.Instance) and base.name == "type":
            # A class held as a value, whichever class: what every class
            # object exposes is known, the rest is that class's business.
            return Binding(_CLASS_DUNDERS.get(node.attr, T.UNKNOWN))
        if (
            node.attr == "__init__"
            and isinstance(base, T.Instance)
            and base.name not in self.project.classes
        ):
            # `super().__init__(...)` past a base the project does not
            # define: what it takes is the library's business, it returns
            # nothing.
            return Binding(T.Callable_((), T.NONE, f"{base.name}.__init__"))
        if isinstance(base, T.Instance):
            info = self.project.classes.get(base.name)
            if info is not None:
                method = info.find_method(node.attr, self.project)
                if method is not None:
                    self._effects = self._effects.add(Effect.READ_OBJECT)
                    if method.is_property:
                        return Binding(method.ret, method.ret_facts)
                    if method.is_method and not method.is_static:
                        # Reached through an instance, the method has bound
                        # its receiver: what a caller sees -- and what a
                        # callback parameter receives -- starts at the next
                        # parameter. Calls bind against this same signature.
                        signature = method.signature()
                        return Binding(
                            T.Callable_(
                                signature.params[1:],
                                signature.ret,
                                signature.qualname,
                                is_async=signature.is_async,
                                is_generator=signature.is_generator,
                            )
                        )
                    return Binding(method.signature())
                found = info.lookup(node.attr, self.project)
                if found is not None:
                    self._effects = self._effects.add(Effect.READ_OBJECT)
                    return Binding(found[0], found[1])
                inherited_external = self._external_base_attribute(info, node.attr, owner.facts)
                if inherited_external is not None:
                    return inherited_external
                for entry in base.resolved_mro:
                    # `class Reached(list)`: `self.append` is the list's.
                    if entry != base.name and entry in T.BUILTIN_MRO:
                        inherited = self._builtin_method(T.instance(entry), node.attr)
                        if inherited is not None:
                            return Binding(inherited)
                if info.slots is not None and node.attr not in info.slots:
                    if not self._dynamic_depth:
                        self._error("E1202", f"`{info.name}` has no attribute `{node.attr}`", node)
                    return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)
            method = self._builtin_method(base, node.attr)
            if method is not None:
                return Binding(method)
            if info is not None and self.strict and not self._dynamic_depth:
                self._error("E1202", f"`{info.name}` has no attribute `{node.attr}`", node)
                return Binding(T.UNKNOWN)
        known = (
            stdlib.instance_attribute(base.name, node.attr)
            if isinstance(base, T.Instance)
            else None
        )
        if known is not None:
            self._effects = self._effects | known[1]
            if known[1].violations():
                self._blockers.append(f"uses `{base.name}.{node.attr}` with effects: {known[1]}")
            self._native_blockers.append(f"`{base.name}.{node.attr}` has no native lowering")
            return Binding(known[0])
        plugin_attribute = self._plugin_instance_attribute(base, node.attr, owner.facts)
        if plugin_attribute is not None:
            self._effects = self._effects.add(Effect.READ_OBJECT)
            return plugin_attribute

        method = self._builtin_method(base, node.attr)
        if method is not None:
            self._effects = self._effects.add(Effect.READ_OBJECT)
            return Binding(method)
        if self.strict and T.is_exact_builtin(base) and not self._dynamic_depth:
            self._error("E1202", f"`{base}` has no attribute `{node.attr}`", node)
            return Binding(T.UNKNOWN)
        self._effects = self._effects.add(Effect.READ_OBJECT)
        return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)

    def _external_base_attribute(self, info: ClassInfo, attr: str, facts: Facts) -> Binding | None:
        """An attribute inherited from a base only a plugin knows.

        `class Mlp(nn.Module)` gets `init` and `apply` from flax, not from
        anything in the project; the class's MRO carries the external base's
        canonical name, and the plugin that owns it says what the attribute
        is.
        """
        for entry in info.mro:
            if entry == "object" or entry in self.project.classes:
                continue
            probe = T.Instance(entry, (), (entry, "object"))
            found = self._plugin_instance_attribute(probe, attr, facts)
            if found is not None:
                self._effects = self._effects.add(Effect.READ_OBJECT)
                return found
        return None

    def _optional_receiver(self, owner: Binding, node: ast.Attribute) -> Binding | None:
        """Reading through a value that may be None, which fails at runtime.

        Without this the union falls through to the generic path and the
        reported problem is an unknown signature, which sends the reader
        looking for a missing stub instead of a missing narrowing.
        """
        if self._dynamic_depth or owner.facts.non_null:
            return None
        if not isinstance(owner.type, T.Union_) or not T.is_optional(owner.type):
            return None
        present = T.remove_none(owner.type)
        if isinstance(present, T.NeverType):
            return None
        self._error(
            "E1206",
            f"`{ast.unparse(node.value)}` may be `None`, so `.{node.attr}` is not available",
            node,
            help="narrow it first, for example with `if x is not None:`",
        )
        self._reported.add(id(node))
        return Binding(T.UNKNOWN)

    def _module_attribute(self, module: str, node: ast.Attribute) -> Binding:
        qualname = f"{module}.{node.attr}"
        info = self.project.functions.get(qualname)
        if info is not None:
            return Binding(info.signature())
        cls = self.project.classes.get(qualname)
        if cls is not None:
            return Binding(T.ClassObject(cls.qualname, cls.instance()))
        other = self.project.modules.get(module)
        if other is not None and node.attr in other.globals:
            self._effects = self._effects.add(Effect.READ_GLOBAL)
            return Binding(other.globals[node.attr], other.global_facts.get(node.attr, Facts()))
        known = stdlib.MODULE_ATTRIBUTES.get(qualname)
        if known is not None:
            return Binding(known[0], known[1])
        described = stdlib.lookup(qualname)
        if described is not None:
            return Binding(described[0])
        external = self.project.external_types.get(qualname)
        if external is not None:
            # `pathlib.Path(...)` constructs one; the class object is what the
            # attribute names.
            instance = T.Instance(
                external, (), stdlib.EXTERNAL_MRO.get(qualname, (external, "object"))
            )
            return Binding(T.ClassObject(external, instance))

        if module == "math":
            result = B.math_result(node.attr)
            if result is not None:
                return Binding(T.Callable_((), result.type, f"math.{node.attr}"))
            if node.attr in {"pi", "e", "tau", "inf", "nan"}:
                return Binding(T.FLOAT)
        if self.plugins is not None:
            plugin = self.plugins.for_module(module)
            if plugin is not None:
                found = plugin.attribute_type(qualname)
                if found is not None:
                    return Binding(found[0], found[1])
                return Binding(T.Callable_((), T.UNKNOWN, qualname))

        effects = B.MODULE_EFFECTS.get(module.partition(".")[0])
        if effects is not None:
            self._effects = self._effects | effects
            self._blockers.append(f"uses `{module}` which has effects: {effects}")
        return Binding(T.UNKNOWN)

    def _builtin_method(self, base: T.Type, attr: str) -> T.Type | None:
        if not isinstance(base, (T.Instance, T.Tuple_)):
            return None
        cache = self.project.method_cache
        key = (base, attr)
        try:
            if key in cache:
                return cache[key]
        except TypeError:  # a type holding something unhashable
            return self._builtin_method_of(base, attr)
        found = cache[key] = self._builtin_method_of(base, attr)
        return found

    def _builtin_method_of(self, base: T.Instance | T.Tuple_, attr: str) -> T.Type | None:
        name = base.name if isinstance(base, T.Instance) else "tuple"
        element = B.element_type(base)
        table: dict[tuple[str, str], T.Type] = {
            ("list", "append"): T.Callable_((T.Param("value", element),), T.NONE, "list.append"),
            ("list", "extend"): T.Callable_(
                (T.Param("values", T.instance("Iterable", element)),), T.NONE, "list.extend"
            ),
            ("list", "pop"): T.Callable_((T.Param("index", T.INT, True),), element, "list.pop"),
            ("list", "insert"): T.Callable_(
                (T.Param("index", T.INT), T.Param("value", element)), T.NONE, "list.insert"
            ),
            ("list", "index"): T.Callable_((T.Param("value", element),), T.INT, "list.index"),
            ("list", "count"): T.Callable_((T.Param("value", element),), T.INT, "list.count"),
            ("list", "clear"): T.Callable_((), T.NONE, "list.clear"),
            ("list", "sort"): T.Callable_((), T.NONE, "list.sort"),
            ("list", "reverse"): T.Callable_((), T.NONE, "list.reverse"),
            ("list", "copy"): T.Callable_((), base, "list.copy"),
            ("dict", "get"): T.Callable_(
                (
                    T.Param(
                        "key", base.args[0] if isinstance(base, T.Instance) and base.args else T.ANY
                    ),
                ),
                T.union(base.args[1], T.NONE)
                if isinstance(base, T.Instance) and len(base.args) == 2
                else T.UNKNOWN,
                "dict.get",
            ),
            # A keys view supports the set algebra and hashes its members,
            # which is what `frozenset` promises; an items view likewise.
            ("dict", "keys"): T.Callable_(
                (),
                T.instance(
                    "frozenset",
                    base.args[0] if isinstance(base, T.Instance) and base.args else T.ANY,
                ),
                "dict.keys",
            ),
            ("dict", "values"): T.Callable_(
                (),
                T.instance(
                    "Iterable",
                    base.args[1] if isinstance(base, T.Instance) and len(base.args) == 2 else T.ANY,
                ),
                "dict.values",
            ),
            ("dict", "items"): T.Callable_(
                (),
                T.instance(
                    "frozenset",
                    T.Tuple_(
                        base.args
                        if isinstance(base, T.Instance) and len(base.args) == 2
                        else (T.ANY, T.ANY)
                    ),
                ),
                "dict.items",
            ),
            ("dict", "pop"): T.Callable_(
                (),
                base.args[1] if isinstance(base, T.Instance) and len(base.args) == 2 else T.UNKNOWN,
                "dict.pop",
            ),
            ("dict", "setdefault"): T.Callable_(
                (),
                base.args[1] if isinstance(base, T.Instance) and len(base.args) == 2 else T.UNKNOWN,
                "dict.setdefault",
            ),
            ("dict", "update"): T.Callable_((), T.NONE, "dict.update"),
            ("dict", "clear"): T.Callable_((), T.NONE, "dict.clear"),
            ("set", "add"): T.Callable_((T.Param("value", element),), T.NONE, "set.add"),
            ("set", "discard"): T.Callable_((T.Param("value", element),), T.NONE, "set.discard"),
            ("set", "remove"): T.Callable_((T.Param("value", element),), T.NONE, "set.remove"),
            ("set", "union"): T.Callable_((), base, "set.union"),
            ("tuple", "count"): T.Callable_((), T.INT, "tuple.count"),
            ("tuple", "index"): T.Callable_((), T.INT, "tuple.index"),
        }
        # The widener declares read-only parameters as `Sequence`/`Mapping`
        # and promises exactly these methods; a checker that does not honor
        # the promise would reject the converter's own output.
        for owner, concrete in (("Sequence", "list"), ("Mapping", "dict")):
            for method in (
                ("count", "index") if owner == "Sequence" else ("get", "keys", "values", "items")
            ):
                modeled = table.get((concrete, method))
                if modeled is not None:
                    table[(owner, method)] = modeled
        table[("Sequence", "count")] = T.Callable_(
            (T.Param("value", element),), T.INT, "Sequence.count"
        )
        table[("Sequence", "index")] = T.Callable_(
            (T.Param("value", element),), T.INT, "Sequence.index"
        )
        table[("dict", "copy")] = T.Callable_((), base, "dict.copy")
        found = table.get((name, attr))
        if found is not None:
            return found
        if name == "str":
            return self._str_method(attr)
        if name in {"bytes", "bytearray"}:
            return self._bytes_method(name, attr)
        if name in {"array", "memoryview", "Buffer"}:
            return self._buffer_method(base, name, attr, element)
        return None

    def _buffer_method(self, base: T.Type, owner: str, attr: str, element: T.Type) -> T.Type | None:
        """Methods of a contiguous buffer: `array`, `memoryview`, `Buffer`."""
        qualname = f"{owner}.{attr}"
        if attr == "tolist":
            return T.Callable_((), T.list_of(element), qualname)
        if attr in {"tobytes", "tostring"}:
            return T.Callable_((), T.BYTES, qualname)
        if attr == "buffer_info":
            return T.Callable_((), T.Tuple_((T.INT, T.INT)), qualname)
        if attr in {"append", "extend", "insert", "remove", "reverse", "frombytes", "fromlist"}:
            return T.Callable_((), T.NONE, qualname)
        if attr == "pop":
            return T.Callable_((), element, qualname)
        if attr in {"count", "index", "itemsize", "nbytes"}:
            return T.Callable_((), T.INT, qualname) if attr in {"count", "index"} else T.INT
        if attr in {"typecode", "format"}:
            return T.STR
        if attr == "cast":
            return T.Callable_((), base, qualname)
        if attr in {"c_contiguous", "contiguous", "readonly"}:
            return T.BOOL
        if attr in {"ndim", "obj"}:
            return T.INT if attr == "ndim" else T.ANY
        if attr == "shape":
            return T.Tuple_((T.INT,), homogeneous=True)
        return None

    def _str_method(self, attr: str) -> T.Type | None:
        returns_str = {
            "upper",
            "lower",
            "strip",
            "lstrip",
            "rstrip",
            "title",
            "capitalize",
            "replace",
            "join",
            "format",
            "removeprefix",
            "removesuffix",
            "casefold",
            "zfill",
            "swapcase",
            "expandtabs",
            "center",
            "ljust",
            "rjust",
            "translate",
        }
        returns_bool = {
            "startswith",
            "endswith",
            "isdigit",
            "isalpha",
            "isalnum",
            "isspace",
            "islower",
            "isupper",
            "isnumeric",
            "isdecimal",
            "isidentifier",
            "isascii",
            "isprintable",
            "istitle",
        }
        if attr in returns_str:
            return T.Callable_((), T.STR, f"str.{attr}")
        if attr in returns_bool:
            return T.Callable_((), T.BOOL, f"str.{attr}")
        if attr in {"find", "rfind", "count", "index", "rindex"}:
            return T.Callable_((), T.INT, f"str.{attr}")
        if attr in {"partition", "rpartition"}:
            return T.Callable_((), T.Tuple_((T.STR, T.STR, T.STR)), f"str.{attr}")
        if attr in {"split", "rsplit", "splitlines"}:
            return T.Callable_((), T.list_of(T.STR), f"str.{attr}")
        if attr == "encode":
            return T.Callable_((), T.BYTES, "str.encode")
        return None

    def _bytes_method(self, owner: str, attr: str) -> T.Type | None:
        if attr == "decode":
            return T.Callable_((), T.STR, f"{owner}.decode")
        if attr == "hex":
            return T.Callable_((), T.STR, f"{owner}.hex")
        if attr in {"startswith", "endswith", "isdigit", "isalpha", "isascii"}:
            return T.Callable_((), T.BOOL, f"{owner}.{attr}")
        if attr in {"find", "rfind", "count", "index"}:
            return T.Callable_((), T.INT, f"{owner}.{attr}")
        if attr in {"split", "rsplit", "splitlines"}:
            return T.Callable_((), T.list_of(T.BYTES), f"{owner}.{attr}")
        if attr in {
            "strip",
            "lstrip",
            "rstrip",
            "upper",
            "lower",
            "replace",
            "join",
            "removeprefix",
            "removesuffix",
            "zfill",
            "title",
            "capitalize",
        }:
            return T.Callable_((), T.BYTES, f"{owner}.{attr}")
        return None

    def _expr_Subscript(self, node: ast.Subscript, env: Env) -> Binding:
        owner = self._expr(node.value, env)
        index = self._expr(node.slice, env)
        narrowed = env.get(_attribute_path(node) or "")
        if narrowed is not None:
            return narrowed
        base = T.strip_literal(owner.type)
        is_slice = isinstance(node.slice, ast.Slice)

        if isinstance(base, T.Tuple_):
            self._effects = self._effects.add(raises=("IndexError",))
            if is_slice:
                return Binding(T.Tuple_((B.element_type(base),), homogeneous=True))
            if (
                index.facts.has_constant
                and isinstance(index.facts.constant, int)
                and not base.homogeneous
            ):
                position = index.facts.constant
                if -len(base.items) <= position < len(base.items):
                    return Binding(base.items[position])
                self._error("E1301", f"index {position} is out of range for `{base}`", node)
                return Binding(T.UNKNOWN)
            return Binding(B.element_type(base))
        if isinstance(base, T.Instance):
            if base.name in {"list", "Sequence", "Buffer", "memoryview", "array"}:
                self._effects = self._effects.add(raises=("IndexError",))
                return Binding(base if is_slice else B.element_type(base))
            if base.name == "dict":
                self._effects = self._effects.add(raises=("KeyError",))
                return Binding(base.args[1] if len(base.args) == 2 else T.UNKNOWN)
            if base.name == "str":
                self._effects = self._effects.add(raises=("IndexError",))
                return Binding(T.STR)
            if base.name == "bytes":
                self._effects = self._effects.add(raises=("IndexError",))
                return Binding(T.BYTES if is_slice else T.INT)
            indexed = self._plugin_subscript(base, node, is_slice)
            if indexed is not None:
                return indexed
        if isinstance(base, (T.AnyType, T.UnknownType)):
            if isinstance(base, T.DynamicType):
                return Binding(T.DYNAMIC)
            return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)
        self._effects = self._effects.add(Effect.READ_OBJECT, raises=("TypeError",))
        return Binding(T.UNKNOWN)

    def _expr_Slice(self, node: ast.Slice, env: Env) -> Binding:
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                self._expr(part, env)
        return Binding(T.instance("slice"))

    def _expr_List(self, node: ast.List, env: Env) -> Binding:
        elements = [self._expr(e, env) for e in node.elts]
        self._effects = self._effects.add(Effect.ALLOC)
        element = T.join(*[e.type for e in elements]) if elements else T.NEVER
        facts = Facts() if _spread(node.elts) else Facts(length=len(elements))
        return Binding(T.list_of(T.strip_literal(element)), facts)

    def _expr_Tuple(self, node: ast.Tuple, env: Env) -> Binding:
        elements = [self._expr(e, env) for e in node.elts]
        self._effects = self._effects.add(Effect.ALLOC)
        if _spread(node.elts):
            # `(first, *rest)` is as long as `rest` makes it.
            element = T.join(*[e.type for e in elements])
            return Binding(T.Tuple_((T.strip_literal(element),), homogeneous=True))
        constants = [e.facts.constant for e in elements]
        facts = Facts(length=len(elements))
        if elements and all(e.facts.has_constant for e in elements):
            facts = facts.with_(constant=tuple(constants), has_constant=True)
        return Binding(T.Tuple_(tuple(e.type for e in elements)), facts)

    def _expr_Set(self, node: ast.Set, env: Env) -> Binding:
        elements = [self._expr(e, env) for e in node.elts]
        self._effects = self._effects.add(Effect.ALLOC)
        element = T.join(*[e.type for e in elements]) if elements else T.NEVER
        return Binding(T.set_of(T.strip_literal(element)))

    def _expr_Dict(self, node: ast.Dict, env: Env) -> Binding:
        keys = [self._expr(k, env) for k in node.keys if k is not None]
        values = [self._expr(v, env) for v in node.values]
        self._effects = self._effects.add(Effect.ALLOC)
        key = T.strip_literal(T.join(*[k.type for k in keys])) if keys else T.NEVER
        value = T.strip_literal(T.join(*[v.type for v in values])) if values else T.NEVER
        return Binding(T.dict_of(key, value), Facts(length=len(values)))

    def _expr_JoinedStr(self, node: ast.JoinedStr, env: Env) -> Binding:
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self._expr(value.value, env)
        self._effects = self._effects.add(Effect.ALLOC)
        return Binding(T.STR)

    def _expr_FormattedValue(self, node: ast.FormattedValue, env: Env) -> Binding:
        self._expr(node.value, env)
        return Binding(T.STR)

    def _expr_Starred(self, node: ast.Starred, env: Env) -> Binding:
        # Only reached inside a display: `[a, *rest]` holds what `rest`
        # holds. A call unwraps its starred arguments before asking.
        spread = self._expr(node.value, env)
        return Binding(B.element_type(spread.type))

    def _expr_NamedExpr(self, node: ast.NamedExpr, env: Env) -> Binding:
        value = self._expr(node.value, env)
        self._bind_target(node.target, value, env)
        return value

    def _expr_Lambda(self, node: ast.Lambda, env: Env) -> Binding:
        inner = env.fork()
        for arg in node.args.args:
            inner.set(arg.arg, Binding(T.UNKNOWN))
        body = self._expr(node.body, inner)
        self._native_blockers.append("contains a lambda")
        return Binding(
            T.Callable_(tuple(T.Param(a.arg, T.UNKNOWN) for a in node.args.args), body.type)
        )

    def _expr_Await(self, node: ast.Await, env: Env) -> Binding:
        value = self._expr(node.value, env)
        self._effects = self._effects.add(Effect.SYNC)
        self._native_blockers.append("awaits a coroutine")
        base = T.strip_literal(value.type)
        if isinstance(base, T.Instance) and base.args and base.name in {"Coroutine", "Awaitable"}:
            return Binding(base.args[0])
        if isinstance(base, (T.AnyType, T.UnknownType)):
            if isinstance(base, T.DynamicType):
                return Binding(T.DYNAMIC)
            return Binding(T.ANY if isinstance(base, T.AnyType) else T.UNKNOWN)
        return Binding(T.UNKNOWN if self.strict else T.ANY)

    def _expr_Yield(self, node: ast.Yield, env: Env) -> Binding:
        if node.value is not None:
            value = self._expr(node.value, env)
            self._returns.append(value)
        self._native_blockers.append("is a generator")
        return Binding(T.UNKNOWN)

    def _expr_YieldFrom(self, node: ast.YieldFrom, env: Env) -> Binding:
        value = self._expr(node.value, env)
        self._returns.append(Binding(B.element_type(value.type)))
        self._native_blockers.append("is a generator")
        return Binding(T.UNKNOWN)

    def _expr_ListComp(self, node: ast.ListComp, env: Env) -> Binding:
        element = self._comprehension(node, node.elt, env)
        return Binding(T.list_of(T.strip_literal(element)))

    def _expr_SetComp(self, node: ast.SetComp, env: Env) -> Binding:
        element = self._comprehension(node, node.elt, env)
        return Binding(T.set_of(T.strip_literal(element)))

    def _expr_GeneratorExp(self, node: ast.GeneratorExp, env: Env) -> Binding:
        element = self._comprehension(node, node.elt, env)
        return Binding(T.instance("Iterator", T.strip_literal(element)))

    def _expr_DictComp(self, node: ast.DictComp, env: Env) -> Binding:
        inner = env.fork()
        self._bind_generators(node.generators, inner)
        key = self._expr(node.key, inner)
        value = self._expr(node.value, inner)
        self._effects = self._effects.add(Effect.ALLOC)
        return Binding(T.dict_of(T.strip_literal(key.type), T.strip_literal(value.type)))

    def _comprehension(self, node: ast.expr, element: ast.expr, env: Env) -> T.Type:
        inner = env.fork()
        self._bind_generators(node.generators, inner)  # type: ignore[attr-defined]
        self._effects = self._effects.add(Effect.ALLOC)
        return self._expr(element, inner).type

    def _bind_generators(self, generators: list[ast.comprehension], env: Env) -> None:
        for generator in generators:
            iterable = self._expr(generator.iter, env)
            self._bind_target(
                generator.target, self._iteration_element(iterable, generator.iter), env
            )
            for condition in generator.ifs:
                self._expr(condition, env)
                env.restore(self._narrow(condition, env.fork(), True).snapshot())
            if generator.is_async:
                self._effects = self._effects.add(Effect.SYNC)

    def _binary(
        self, left: Binding, right: Binding, op: type[ast.operator], node: ast.AST
    ) -> Binding:
        left_base = T.strip_literal(left.type)
        right_base = T.strip_literal(right.type)

        folded = self._fold_binary(left, right, op)
        if folded is not None:
            return folded

        plugin_result = self._plugin_operator(_ARITH_OPS.get(op, ""), [left, right], node)
        if plugin_result is not None:
            return plugin_result

        if isinstance(left_base, (T.AnyType, T.UnknownType)) or isinstance(
            right_base, (T.AnyType, T.UnknownType)
        ):
            if isinstance(left_base, T.DynamicType) or isinstance(right_base, T.DynamicType):
                return Binding(T.DYNAMIC)
            return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)

        if op is ast.BitOr and _is_class_value(left_base) and _is_class_value(right_base):
            # `list | dict` in an `isinstance` is a type made of types.
            return Binding(T.instance("type"))

        if op is ast.Add:
            for container in ("list", "str", "bytes"):
                if (
                    isinstance(left_base, T.Instance)
                    and left_base.name == container
                    and (T.is_assignable(right_base, left_base) or right_base == left_base)
                ):
                    self._effects = self._effects.add(Effect.ALLOC)
                    return Binding(left_base)
            if isinstance(left_base, T.Tuple_) and isinstance(right_base, T.Tuple_):
                self._effects = self._effects.add(Effect.ALLOC)
                if not left_base.homogeneous and not right_base.homogeneous:
                    return Binding(T.Tuple_(left_base.items + right_base.items))
                return Binding(
                    T.Tuple_(
                        (T.join(B.element_type(left_base), B.element_type(right_base)),),
                        homogeneous=True,
                    )
                )
        if op is ast.Mult:
            for sequence, count in ((left_base, right_base), (right_base, left_base)):
                if isinstance(sequence, (T.Tuple_,)) and count == T.INT:
                    self._effects = self._effects.add(Effect.ALLOC)
                    return Binding(T.Tuple_((B.element_type(sequence),), homogeneous=True))
                if (
                    isinstance(sequence, T.Instance)
                    and sequence.name in {"list", "str", "bytes"}
                    and count in (T.INT, T.BOOL)
                ):
                    self._effects = self._effects.add(Effect.ALLOC)
                    return Binding(sequence)
        if op is ast.Mod and left_base == T.STR:
            self._effects = self._effects.add(Effect.ALLOC)
            return Binding(T.STR)
        if op in _SET_OPS and _is_set_like(left_base) and _is_set_like(right_base):
            # `a | b`, `a & b`, `a - b`, `a ^ b` on sets are new sets of the
            # left operand's kind -- `frozenset | set` is a frozenset, and the
            # other way round a set -- holding elements of either.
            self._effects = self._effects.add(Effect.ALLOC)
            element = T.join(B.element_type(left_base), B.element_type(right_base))
            if op in {ast.BitAnd, ast.Sub}:
                element = B.element_type(left_base)
            return Binding(T.instance(left_base.name, T.strip_literal(element)))  # type: ignore[union-attr]

        overloaded = self._operator_method(left_base, right_base, op, node)
        if overloaded is not None:
            return overloaded
        if (
            op is ast.Div
            and isinstance(left_base, T.Instance)
            and left_base.name.startswith("pathlib.")
            and (right_base == T.STR or T.is_assignable(right_base, left_base))
        ):
            # `Path / "name"` is the way a path is spelled.
            return Binding(left_base)

        if not (T.is_numeric(left_base) and T.is_numeric(right_base)):
            self._error(
                "E1302",
                f"`{_ARITH_OPS.get(op, '?')}` is not defined for `{left.type}` and `{right.type}`",
                node,
            )
            return Binding(T.UNKNOWN)

        if (
            left_base == T.BOOL
            and right_base == T.BOOL
            and op in {ast.Add, ast.Sub, ast.Mult, ast.Pow}
        ):
            self._warn("W2002", "arithmetic on `bool` values is legal but usually unintended", node)

        return self._numeric_result(left, right, left_base, right_base, op, node)

    def _operator_method(
        self, left_base: T.Type, right_base: T.Type, op: type[ast.operator], node: ast.AST
    ) -> Binding | None:
        """A project class's own `__add__`, `__or__`, and the rest.

        `EffectSet | EffectSet` is whatever `EffectSet.__or__` returns, with
        whatever effects it has; the reflected form on the right operand is
        the fallback Python itself takes. Only project classes are looked
        at: a builtin's operators are modeled above, by hand.
        """
        dunder = _OPERATOR_DUNDERS.get(op)
        if dunder is None:
            return None
        for owner, name in ((left_base, dunder), (right_base, f"__r{dunder[2:]}")):
            if not isinstance(owner, T.Instance):
                continue
            cls = self.project.classes.get(owner.name)
            if cls is None:
                continue
            method = cls.find_method(name, self.project)
            if method is None:
                continue
            self._effects = self._effects | method.effects
            self._calls.add(method.qualname)
            # An unknown return is a summary the fixpoint has not settled
            # yet; the next pass reads the settled one.
            return Binding(method.ret)
        return None

    def _numeric_result(
        self,
        left: Binding,
        right: Binding,
        left_base: T.Type,
        right_base: T.Type,
        op: type[ast.operator],
        node: ast.AST,
    ) -> Binding:
        rank = max(T.numeric_rank(left_base) or 0, T.numeric_rank(right_base) or 0)
        result_type: T.Type = [T.INT, T.INT, T.FLOAT, T.COMPLEX][rank]

        if op in {ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor}:
            if rank > 1:
                self._error("E1302", f"`{_ARITH_OPS[op]}` requires integer operands", node)
                return Binding(T.UNKNOWN)
            return Binding(T.INT)
        if op is ast.Div:
            self._effects = self._effects.add(raises=("ZeroDivisionError",))
            return Binding(T.COMPLEX if rank == 3 else T.FLOAT)
        if op in {ast.FloorDiv, ast.Mod}:
            self._effects = self._effects.add(raises=("ZeroDivisionError",))
            return Binding(result_type)
        if op is ast.MatMult:
            self._error("E1302", "`@` is not defined for builtin numeric types", node)
            return Binding(T.UNKNOWN)
        if op is ast.Pow:
            return Binding(result_type if rank != 1 else T.INT)

        facts = Facts()
        if (
            result_type == T.INT
            and left.facts.int_range is not None
            and right.facts.int_range is not None
        ):
            lr, rr = left.facts.int_range, right.facts.int_range
            if op is ast.Add:
                facts = Facts(int_range=lr + rr)
            elif op is ast.Sub:
                facts = Facts(int_range=lr - rr)
            elif op is ast.Mult:
                facts = Facts(int_range=lr * rr)
        return Binding(result_type, facts)

    def _fold_binary(self, left: Binding, right: Binding, op: type[ast.operator]) -> Binding | None:
        if not (left.facts.has_constant and right.facts.has_constant):
            return None
        operations = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a**b,
            ast.LShift: lambda a, b: a << b,
            ast.RShift: lambda a, b: a >> b,
            ast.BitOr: lambda a, b: a | b,
            ast.BitAnd: lambda a, b: a & b,
            ast.BitXor: lambda a, b: a ^ b,
        }
        operation = operations.get(op)
        if operation is None:
            return None
        try:
            value = operation(left.facts.constant, right.facts.constant)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        if isinstance(value, (int, float, complex, str, bytes, bool)) and not isinstance(
            value, bool
        ):
            facts = Facts(constant=value, has_constant=True)
            if isinstance(value, int):
                facts = facts.with_(int_range=IntRange(value, value))
            return Binding(T.type_of_constant(value), facts)
        return None

    def _narrow(self, test: ast.expr, env: Env, positive: bool) -> Env:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._narrow(test.operand, env, not positive)
        if isinstance(test, ast.BoolOp):
            sequential = (isinstance(test.op, ast.And) and positive) or (
                isinstance(test.op, ast.Or) and not positive
            )
            if sequential:
                for value in test.values:
                    env = self._narrow(value, env, positive)
            return env
        if isinstance(test, ast.Compare):
            return self._narrow_compare(test, env, positive)
        if isinstance(test, ast.Call):
            return self._narrow_call(test, env, positive)
        if isinstance(test, ast.NamedExpr):
            test = test.target
        if isinstance(test, (ast.Name, ast.Attribute, ast.Subscript)) and positive:
            key = test.id if isinstance(test, ast.Name) else _attribute_path(test)
            binding = env.get(key or "")
            if binding is None and isinstance(test, ast.Attribute) and key is not None:
                binding = self._expr(test, env)
            if key is not None and binding is not None and T.is_optional(binding.type):
                env.set(
                    key,
                    Binding(T.remove_none(binding.type), binding.facts.with_(non_null=True)),
                )
        return env

    def _narrow_compare(self, test: ast.Compare, env: Env, positive: bool) -> Env:
        if len(test.ops) != 1 or len(test.comparators) != 1:
            return env
        op, right = test.ops[0], test.comparators[0]
        left = test.left
        if isinstance(left, ast.NamedExpr):
            left = left.target

        if (
            isinstance(op, (ast.Is, ast.IsNot))
            and isinstance(right, ast.Constant)
            and right.value is None
        ):
            wants_none = isinstance(op, ast.Is) == positive
            key = left.id if isinstance(left, ast.Name) else _attribute_path(left)
            if key is not None:
                binding = env.get(key)
                if binding is None and _is_place(left):
                    # Not narrowed before: what the attribute is now, so the
                    # narrowing has something to remove `None` from.
                    binding = self._expr(left, env)
                if binding is not None:
                    if wants_none:
                        env.set(key, Binding(T.NONE, Facts()))
                    else:
                        env.set(
                            key,
                            Binding(
                                T.remove_none(binding.type), binding.facts.with_(non_null=True)
                            ),
                        )
            return env

        if (
            isinstance(op, (ast.In, ast.NotIn))
            and isinstance(right, (ast.Set, ast.Tuple, ast.List))
            and right.elts
            and all(isinstance(e, ast.Constant) for e in right.elts)
            and isinstance(op, ast.In) == positive
        ):
            # Being one of the constants is being of their type: the members
            # of `x`'s type none of them could be are gone.
            key = left.id if isinstance(left, ast.Name) else _attribute_path(left)
            binding = env.get(key or "")
            if binding is None and key is not None and _is_place(left):
                binding = self._expr(left, env)
            if key is not None and binding is not None:
                offered = T.strip_literal(
                    T.join(*[T.type_of_constant(e.value) for e in right.elts])  # type: ignore[attr-defined]
                )
                kept = [
                    m
                    for m in T.members_of(binding.type)
                    if T.is_assignable(offered, m) or T.is_assignable(m, offered)
                ]
                if kept and len(kept) < len(T.members_of(binding.type)):
                    env.set(key, Binding(T.union(*kept), binding.facts))
            return env

        constant = self._constant_of(right, env)
        if constant is not None and isinstance(left, ast.Name):
            env = self._narrow_by_constant(left.id, op, constant, env, positive)
        elif constant is not None and isinstance(left, ast.Call) and self._is_len_call(left):
            env = self._narrow_by_length(left, op, constant, env, positive)
        return env

    def _narrow_by_constant(
        self, name: str, op: ast.cmpop, constant: object, env: Env, positive: bool
    ) -> Env:
        binding = env.get(name)
        if binding is None:
            return env
        if isinstance(op, (ast.Eq, ast.NotEq)):
            wants_equal = isinstance(op, ast.Eq) == positive
            if wants_equal:
                literal = T.type_of_constant(constant)
                facts = binding.facts.with_(constant=constant, has_constant=True)
                if isinstance(constant, int) and not isinstance(constant, bool):
                    facts = facts.with_(int_range=IntRange(constant, constant))
                if isinstance(binding.type, T.Union_):
                    env.set(name, Binding(literal, facts))
                else:
                    env.set(name, Binding(binding.type, facts))
            return env
        if not isinstance(constant, int) or isinstance(constant, bool):
            return env
        current = binding.facts.int_range or IntRange()
        effective = op if positive else _invert_order(op)
        match effective:
            case ast.Lt():
                bound = IntRange(None, constant - 1)
            case ast.LtE():
                bound = IntRange(None, constant)
            case ast.Gt():
                bound = IntRange(constant + 1, None)
            case ast.GtE():
                bound = IntRange(constant, None)
            case _:
                return env
        env.set(name, Binding(binding.type, binding.facts.with_(int_range=current.meet(bound))))
        return env

    def _narrow_by_length(
        self, call: ast.Call, op: ast.cmpop, constant: object, env: Env, positive: bool
    ) -> Env:
        if not (isinstance(op, ast.Eq) and positive and isinstance(constant, int)):
            return env
        target = call.args[0] if call.args else None
        if isinstance(target, ast.Name):
            binding = env.get(target.id)
            if binding is not None:
                env.set(target.id, Binding(binding.type, binding.facts.with_(length=constant)))
        return env

    def _narrow_call(self, call: ast.Call, env: Env, positive: bool) -> Env:
        if not isinstance(call.func, ast.Name):
            return self._narrow_typeguard(call, env, positive)
        if call.func.id not in {"isinstance", "issubclass"} or len(call.args) != 2:
            return self._narrow_typeguard(call, env, positive)
        subject, classes = call.args
        if isinstance(subject, ast.NamedExpr):
            subject = subject.target
        key = subject.id if isinstance(subject, ast.Name) else _attribute_path(subject)
        if key is None:
            return env
        binding = env.get(key)
        if binding is None and _is_place(subject):
            binding = self._expr(subject, env)
        if binding is None:
            return env
        candidates = self._class_types(classes, env)
        if not candidates:
            return env
        if positive:
            narrowed = T.union(*candidates)
            facts = binding.facts
            if len(candidates) == 1 and isinstance(candidates[0], T.Instance):
                facts = facts.with_(exact_class=candidates[0].name)
            env.set(key, Binding(narrowed, facts))
        else:
            members = list(T.members_of(binding.type))
            remaining = [m for m in members if not any(T.is_assignable(m, c) for c in candidates)]
            if remaining and len(remaining) < len(members):
                env.set(key, Binding(T.union(*remaining), binding.facts))
        return env

    def _narrow_typeguard(self, call: ast.Call, env: Env, positive: bool) -> Env:
        if not positive or not call.args:
            return env
        callee = self._callee_info(call, env)
        if callee is None or not callee.ret_annotated:
            return env
        guard = _typeguard_target(callee.node)
        if guard is None:
            return env
        subject = call.args[0]
        if not isinstance(subject, ast.Name):
            return env
        resolved = self.annotations.resolve(guard)
        binding = env.get(subject.id)
        if binding is not None and not isinstance(resolved.type, T.UnknownType):
            env.set(subject.id, Binding(resolved.type, resolved.facts))
        return env

    def _class_types(self, expr: ast.expr, env: Env) -> list[T.Type]:
        nodes = expr.elts if isinstance(expr, ast.Tuple) else [expr]
        found: list[T.Type] = []
        for node in nodes:
            binding = self._expr(node, env)
            if isinstance(binding.type, T.ClassObject) and binding.type.instance_type is not None:
                found.append(binding.type.instance_type)
            elif isinstance(node, ast.Name) and node.id in T.BUILTIN_MRO:
                found.append(T.instance(node.id))
        return found

    def _typed_buffer(self, node: ast.Call, env: Env) -> Binding | None:
        """`ppy.buffer[T](n)`: `n` elements of `T`, all zero.

        A native build allocates it natively and a standalone one has no
        `array.array` to make, so the allocation is spelled once and both
        paths understand it.
        """
        func = node.func
        assert isinstance(func, ast.Subscript)
        if self.project.resolver(self.symbols).canonical(func.value) != "ppy.buffer":
            return None
        if len(node.args) != 1 or node.keywords:
            self._error("E1305", "`ppy.buffer[T]` takes how many elements to make", node)
        resolved = self.annotations.resolve(func.slice)
        # `ppy.buffer[ppy.i8]` holds bytes, exactly as `Buffer[ppy.i8]` does.
        element = Resolved(narrow_element(resolved) or resolved.type, resolved.facts)
        if not _holdable_element(element.type):
            self._error(
                "E1306",
                f"a buffer holds `int`, `float`, `ppy.i8`, or `ppy.u8`, not `{element.type}`",
                node,
            )
        for argument in node.args:
            count = self._expr(argument, env)
            if T.strip_literal(count.type) not in (T.INT, T.UNKNOWN):
                self._error(
                    "E1301",
                    f"a buffer is made with how many elements it holds, not `{count.type}`",
                    argument,
                )
            elif _is_negative(count.facts):
                self._error("E1401", "a buffer cannot hold fewer than no elements", argument)
        self._effects = self._effects | EffectSet.of(
            Effect.ALLOC, raises=("TypeError", "ValueError")
        )
        return Binding(T.instance("Buffer", element.type))

    def _typed_input(self, node: ast.Call, env: Env) -> Binding | None:
        """`ppy.input[T](...)`: read the next value, typed by what was asked for.

        The subscript is read as a type, the way an annotation is, so
        `ppy.input[tuple[int, int]]()` types as that tuple and
        `ppy.input[Buffer[int]](n)` as that buffer. The call takes a prompt
        for a scalar, or how many values to read for a buffer.
        """
        func = node.func
        assert isinstance(func, ast.Subscript)
        if self.project.resolver(self.symbols).canonical(func.value) != "ppy.input":
            return None
        if len(node.args) > 1 or node.keywords:
            self._error("E1305", "`ppy.input[T]` takes a prompt or a count, and nothing else", node)
        resolved = self.annotations.resolve(func.slice)
        for argument in node.args:
            self._expr(argument, env)
        self._effects = self._effects | EffectSet.of(
            Effect.IO, Effect.ALLOC, raises=("EOFError", "TypeError", "ValueError")
        )
        return Binding(resolved.type, resolved.facts)

    def _checked_conversion(self, node: ast.Call, env: Env) -> Binding | None:
        """`ppy.check[T](value)`: the sanctioned crossing out of dynamic code.

        The runtime validates the value against `T` (raising `TypeError`) and
        hands it back; the checker takes that word and types the result `T`.
        This is the inverse of `typing.cast`, which asserts and checks
        nothing -- the name is different because the behavior is.
        """
        func = node.func
        assert isinstance(func, ast.Subscript)
        if self.project.resolver(self.symbols).canonical(func.value) != "ppy.check":
            return None
        resolved = self.annotations.resolve(func.slice)
        if len(node.args) != 1 or node.keywords:
            self._error("E1305", "`ppy.check[T]` takes exactly one positional value", node)
        for argument in node.args:
            self._expr(argument, env)
            self._mark_escape(argument, env)
        self._effects = self._effects | EffectSet.of(raises=("TypeError",))
        return Binding(resolved.type, resolved.facts)

    def _check_forbidden_call(self, node: ast.Call, env: Env) -> bool:
        if not isinstance(node.func, ast.Name):
            return self._check_dynamic_import(node, env)
        name = node.func.id
        if name in env:
            # A bound name is not a builtin -- but it may still be an alias
            # of importlib's importer, which the lexical layer resolves.
            return self._check_dynamic_import(node, env)
        if name in _FORBIDDEN_CALLS:
            code, message = _FORBIDDEN_CALLS[name]
            self._dynamic_feature(code, message, node)
            return True
        if name == "getattr" and len(node.args) >= 2:
            if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                return False
            self._dynamic_feature(
                "E1504",
                "`getattr` with a computed attribute name cannot be resolved statically",
                node,
                help="use a constant attribute name, or wrap the call in a ppy.dynamic boundary",
            )
            return True
        if name in _DYNAMIC_ATTR_CALLS:
            constant_name = (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            )
            if not constant_name:
                self._dynamic_feature(
                    "E1506",
                    f"`{name}` with a computed attribute name "
                    "mutates an unresolvable attribute set",
                    node,
                )
                return True
            self._effects = self._effects.add(Effect.WRITE_OBJECT)
        return False

    def _constant_import(self, node: ast.Call) -> Binding | None:
        """`importlib.import_module("known.literal")` is a static import.

        Typing the result as the module keeps the rest of the checker honest
        about it: attributes resolve like any import's, and assigning through
        it is the monkey-patch it always was.
        """
        if self._lexical_target(node.func) != "importlib.import_module":
            return None
        if len(node.args) != 1 or node.keywords:
            return None
        argument = node.args[0]
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            return None
        name = argument.value
        if not name or name.startswith("."):
            return None
        return Binding(T.Module_(name))

    def _check_dynamic_import(self, node: ast.Call, env: Env) -> bool:
        """Is this call `importlib.import_module` under any of its spellings?

        Resolution goes through the lexical bindings, not the source text: a
        user method that happens to be named `import_module` is nobody's
        import, and `from importlib import import_module as imp` is exactly
        importlib's.
        """
        canonical = self._lexical_target(node.func)
        if canonical not in {"importlib.import_module", "builtins.__import__"}:
            return False
        spelled = canonical.rpartition(".")[2]
        constant = (
            len(node.args) >= 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
        if not constant:
            self._dynamic_feature("E1503", f"`{spelled}` requires a constant module name", node)
            return True
        return False

    def _lexical_target(self, expr: ast.expr) -> str | None:
        """The one canonical thing `expr` means here, resolved point-sensitively."""
        lexical = self.symbols.lexical
        if lexical is not None:
            found = lexical.targets_at(expr)
            if len(found) == 1:
                return next(iter(found))
            if found:
                return None
        return self.project.resolver(self.symbols).canonical(expr)

    def _mismatch(self, code: str, message: str, node: ast.AST, source: T.Type) -> None:
        """A type mismatch, unless the value is dynamic -- then it is an escape.

        `Dynamic -> int` is not a wrong type so much as a missing runtime
        check; the diagnostic should hand the reader the conversion, not a
        shrug.
        """
        if isinstance(T.strip_literal(source), T.DynamicType):
            self._error(
                "E1508",
                f"{message}; a dynamic value is crossing into typed code",
                node,
                help="validate it explicitly: `ppy.check[T](value)` checks at "
                "runtime and hands back a typed value",
            )
            return
        self._error(code, message, node)

    def _dynamic_feature(
        self, code: str, message: str, node: ast.AST, help: str | None = None
    ) -> None:
        if self._dynamic_depth:
            self._dynamic_seen = True
            self._effects = self._effects.add(Effect.EXTERNAL_UNKNOWN)
            self._native_blockers.append(message)
            if self.dynamic_policy == "deny":
                self._error(
                    "E1505",
                    f'{message}; dynamic boundaries are disabled by `dynamic-boundaries = "deny"`',
                    node,
                )
            return
        self._error(
            code,
            message,
            node,
            help=help
            or "wrap the region in `with ppy.dynamic:` or mark the function `@ppy.dynamic`",
        )

    def _check_attribute_assignment(
        self, owner: Binding, target: ast.Attribute, value: Binding
    ) -> None:
        if isinstance(owner.type, (T.Module_, T.ClassObject)):
            self._dynamic_feature(
                "E1506",
                f"assigning to `{ast.unparse(target)}` "
                "monkey-patches a module or class after analysis",
                target,
            )
            return
        base = T.strip_literal(owner.type)
        if isinstance(base, T.Instance):
            info = self.project.classes.get(base.name)
            if info is None:
                return
            declared = info.lookup(target.attr, self.project)
            if declared is None:
                if info.slots is not None:
                    self._error("E1202", f"`{info.name}` has no attribute `{target.attr}`", target)
                return
            if not isinstance(declared[0], T.Callable_) and not T.is_assignable(
                value.type, declared[0]
            ):
                self._mismatch(
                    "E1301",
                    f"attribute `{target.attr}` of `{info.name}` "
                    f"expects `{declared[0]}`, got `{value.type}`",
                    target,
                    value.type,
                )

    def _check_width(self, binding: Binding, node: ast.AST, what: str = "value") -> Facts:
        """Prove a fixed-width contract, or insert a checked conversion (spec 12.4).

        Returns the facts that hold afterwards: once a check is in place, the
        value is known to be inside the marker's range.
        """
        facts = binding.facts
        if facts.width is None:
            return facts
        bits, signed = facts.width
        allowed = width_range(bits, signed)
        actual = facts.int_range
        marker = f"{'i' if signed else 'u'}{bits}"
        if actual is None:
            return facts.with_(int_range=allowed)
        if allowed.contains(actual):
            return facts
        if facts.has_constant and isinstance(facts.constant, int):
            self._error(
                "E1401",
                f"{facts.constant} does not fit in `ppy.{marker}` (allowed {allowed})",
                node,
            )
            return facts
        if self._contract_mode_forbids_checks():
            self._error(
                "E1402",
                f"{what} may leave the range of `ppy.{marker}` "
                "and the contract mode forbids a runtime check",
                node,
            )
            return facts
        self._remark(f"inserted a checked `{marker}` conversion for {what} (range {actual})", node)
        return facts.with_(
            int_range=allowed.meet(actual) if allowed.meet(actual).low is not None else allowed
        )

    def _contract_mode_forbids_checks(self) -> bool:
        return False

    def _implicit_any(self, info: FunctionInfo, name: str) -> None:
        if not self.strict:
            return
        self._error_at(
            "E1201",
            f"parameter `{name}` has no annotation and no inferable type",
            info.node,
            help="annotate the parameter, or run `ppy convert` to insert inferred annotations",
        )

    def _iteration_element(self, iterable: Binding, node: ast.expr) -> Binding:
        base = T.strip_literal(iterable.type)
        if isinstance(base, (T.AnyType, T.UnknownType)):
            if isinstance(base, T.DynamicType):
                return Binding(T.DYNAMIC)
            return Binding(T.DYNAMIC if self._dynamic_depth else T.UNKNOWN)
        element = B.element_type(base)
        if isinstance(element, T.UnknownType):
            if isinstance(base, T.Instance) and base.name == "range":
                return Binding(T.INT, self._range_facts(node))
            if self.strict and T.is_exact_builtin(base):
                self._error("E1302", f"`{iterable.type}` is not iterable", node)
            return Binding(T.UNKNOWN)
        if isinstance(base, T.Instance) and base.name == "range":
            return Binding(T.INT, self._range_facts(node))
        return Binding(element)

    def _range_facts(self, node: ast.expr) -> Facts:
        """Bound a `range` induction variable when its arguments are constant."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        ):
            return Facts(int_range=IntRange())
        values: list[int] = []
        for arg in node.args:
            facts = self.module.facts_of(arg)
            if facts.has_constant and isinstance(facts.constant, int):
                values.append(facts.constant)
            else:
                return Facts(int_range=IntRange())
        match values:
            case [stop]:
                return Facts(int_range=IntRange(0, max(0, stop - 1)))
            case [start, stop]:
                return Facts(int_range=IntRange(start, max(start, stop - 1)))
            case [start, stop, step] if step > 0:
                return Facts(int_range=IntRange(start, max(start, stop - 1)))
            case [start, stop, step] if step < 0:
                return Facts(int_range=IntRange(min(start, stop + 1), start))
        return Facts(int_range=IntRange())

    def _enter_type(self, t: T.Type) -> T.Type:
        base = T.strip_literal(t)
        if isinstance(base, T.Instance):
            info = self.project.classes.get(base.name)
            if info is not None:
                enter = info.methods.get("__enter__") or info.methods.get("__aenter__")
                if enter is not None:
                    return enter.ret
        return T.UNKNOWN

    def _is_dynamic_marker(self, expr: ast.expr) -> bool:
        # The runtime marker works both bare and called, so both open a boundary.
        target = expr.func if isinstance(expr, ast.Call) and not expr.args else expr
        return self.project.resolver(self.symbols).canonical(target) == "ppy.dynamic"

    def _enter_dynamic(self, node: ast.AST) -> None:
        self._dynamic_depth += 1
        self._dynamic_seen = True
        if self.dynamic_policy == "deny":
            self._error(
                "E1505", 'dynamic boundaries are disabled by `dynamic-boundaries = "deny"`', node
            )

    def _is_module_global(self, name: str) -> bool:
        """A mutable module-level binding, which reading is an effect."""
        if name in self._function_locals or name in self.symbols.constant_globals:
            return False
        return (
            name in self.symbols.globals
            and name not in self.symbols.functions
            and name not in self.symbols.classes
        )

    def _is_len_call(self, node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"
        )

    def _constant_of(self, expr: ast.expr, env: Env) -> object | None:
        try:
            return ast.literal_eval(expr)
        except (ValueError, SyntaxError, TypeError):
            return None

    def _callee_info(self, call: ast.Call, env: Env) -> FunctionInfo | None:
        if isinstance(call.func, ast.Name):
            binding = env.get(call.func.id)
            if binding is not None and isinstance(binding.type, T.Callable_):
                return self.project.functions.get(binding.type.qualname)
        return None

    def _roots(self, node: ast.expr, name: str) -> frozenset[str]:
        if self._aliases is None:
            return frozenset({name})
        return self._aliases.roots_at(node, name)

    def _is_provisional(self, node: ast.expr, value: Binding, env: Env) -> bool:
        """An unknown whose source is a call back into the project.

        `n * fact(n - 1)` is unknown only because `fact` is still being
        computed; any expression over such a call is provisional the same way
        the bare call is.
        """
        if not isinstance(value.type, (T.UnknownType, T.AnyType)):
            return False
        for child in ast.walk(node):
            # A local that a recursive call fed carries the placeholder on.
            if isinstance(child, ast.Name) and child.id in self._provisional_locals:
                return True
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Name):
                continue
            binding = env.get(child.func.id)
            if (
                binding is not None
                and isinstance(binding.type, T.Callable_)
                and binding.type.qualname in self.project.functions
            ):
                return True
        return False

    def _mark_escape(self, node: ast.expr, env: Env, *, retains: bool = True) -> None:
        if isinstance(node, ast.Name) and node.id in env:
            roots = self._roots(node, node.id)
            self._escaping.update(roots - {EXTERNAL})
            if retains:
                self._shared.update(roots - {EXTERNAL})

    def _note_mutation(self, node: ast.expr, env: Env) -> None:
        if isinstance(node, ast.Name):
            roots = self._roots(node, node.id)
            params = (
                self._aliases.param_roots(roots)
                if self._aliases is not None
                else roots & {p.name for p in (self._current.params if self._current else [])}
            )
            self._mutated.update(params)
            if params:
                for name in sorted(params):
                    self._blockers.append(f"mutates parameter `{name}`")
                self._external_writes = True
            elif self._aliases is not None and self._aliases.only_local(roots):
                self._local_writes.update(roots)
            else:
                self._external_writes = True
            return
        # The target is an expression, so which object it reached is unknown.
        self._foreign_writes = True
        self._external_writes = True
        self._blockers.append(f"mutates `{ast.unparse(node)}`")

    def _note_callee_writes(self, info: FunctionInfo, node: ast.Call) -> None:
        """A callee that mutates writes through whatever it was handed.

        The write only stays local if every argument was allocated here, so
        anything else makes this function's own writes externally visible.
        """
        if Effect.WRITE_OBJECT not in info.effects:
            return
        positional = [a.value if isinstance(a, ast.Starred) else a for a in node.args]
        named = [(k.arg, k.value) for k in node.keywords]
        for index, argument in enumerate(positional):
            declared = info.params[index].type if index < len(info.params) else None
            if self._argument_is_safe(argument, declared):
                continue
            self._note_delegated_write(argument)
            self._external_writes = True
        for name, argument in named:
            declared = next((p.type for p in info.params if p.name == name), None)
            if self._argument_is_safe(argument, declared):
                continue
            self._note_delegated_write(argument)
            self._external_writes = True

    def _note_delegated_write(self, argument: ast.expr) -> None:
        """One of our parameters was handed to a callee that writes.

        The write lands in memory this function was lent, exactly as a direct
        `xs[i] = v` would, so a function that delegates its writes is not
        thereby stuck on the Python side. Whether the parameter can actually
        carry a write is the backend's question, not this one's.
        """
        if not isinstance(argument, ast.Name):
            return
        roots = self._roots(argument, argument.id)
        own = {p.name for p in (self._current.params if self._current else [])}
        params = self._aliases.param_roots(roots) if self._aliases is not None else roots & own
        self._delegated.update(params)

    def _argument_is_safe(self, argument: ast.expr, declared: T.Type | None) -> bool:
        """Could a mutating callee reach anything this function does not own?"""
        if declared is not None and T.is_immutable(declared):
            return True
        if (
            isinstance(argument, ast.Name)
            and self._aliases is not None
            and self._aliases.only_local(self._roots(argument, argument.id))
        ):
            return True
        return _is_fresh_allocation(argument)

    def _shared_escapes(self) -> set[str]:
        """Locals handed to something else, which is what `observably shared` means.

        Returning is not sharing: spec 11.2 only rules out sharing *before*
        return, so `_mark_escape` from a return statement is not counted here.
        """
        return self._shared - self._returned_names

    def _merge_declared(self, declared: Facts, value: Facts) -> Facts:
        """Combine a declaration's contract with a value's proven facts.

        Only the contract survives from `declared`: carrying a previous value's
        constant or range forward would let a loop-carried variable keep a stale
        fact it no longer satisfies.
        """
        merged = Facts(
            width=declared.width,
            float_bits=declared.float_bits,
            no_alias=declared.no_alias,
            contiguous=declared.contiguous,
            shape=declared.shape,
            exact_class=declared.exact_class,
        )
        if value.int_range is not None:
            merged = merged.with_(int_range=value.int_range)
        elif declared.int_range is not None:
            merged = merged.with_(int_range=declared.int_range)
        if value.has_constant:
            merged = merged.with_(constant=value.constant, has_constant=True)
        if value.length is not None:
            merged = merged.with_(length=value.length)
        return merged

    def _refine_builtin_method(self, signature: T.Callable_, args: list[Binding]) -> Binding | None:
        """Some builtin methods have a result the argument count decides."""
        if signature.qualname in {"dict.get", "dict.pop"} and len(args) == 2:
            self._effects = self._effects.add(Effect.READ_OBJECT)
            return Binding(T.join(T.remove_none(signature.ret), args[1].type))
        return None

    def _plugin_qualname(self, func: ast.expr, env: Env) -> str | None:
        """Resolve a call target to a plugin-owned qualified name."""
        if self.plugins is None:
            return None
        direct = self.project.resolver(self.symbols).canonical(func)
        if direct is not None and self.plugins.for_qualname(direct) is not None:
            return direct
        if isinstance(func, ast.Attribute):
            owner = self._attribute_owners.get(id(func), T.UNKNOWN)
            base = T.strip_literal(owner)
            if isinstance(base, T.Instance) and "." in base.name:
                root = base.name.rpartition(".")[0]
                candidate = f"{root}.{func.attr}"
                if self.plugins.for_qualname(candidate) is not None:
                    return candidate
        return None

    def _plugin_call(
        self,
        node: ast.Call,
        args: list[Binding],
        keywords: dict[str | None, Binding],
        env: Env,
    ) -> Binding | None:
        qualname = self._plugin_qualname(node.func, env)
        if qualname is None or self.plugins is None:
            return None
        plugin = self.plugins.for_qualname(qualname)
        if plugin is None:
            return None
        result = plugin.call(
            qualname,
            [(a.type, a.facts) for a in args],
            {k: (v.type, v.facts) for k, v in keywords.items() if k is not None},
        )
        if result is None:
            return None
        self._effects = self._effects | result.effects
        if result.lowering == "Reject":
            self._error("E1802", f"`{qualname}` is not supported under the current PPY mode", node)
        if result.lowering == "PythonFallback":
            self._native_blockers.append(f"`{qualname}` stays on the Python path: {result.reason}")
        if self.record:
            self.module.lowerings[id(node)] = LoweringNote(
                qualname=qualname,
                lowering=str(result.lowering),
                reason=result.reason,
                guards=result.guards,
                line=getattr(node, "lineno", 0),
            )
        return Binding(result.type, result.facts)

    def _plugin_operator(
        self, symbol: str, operands: list[Binding], node: ast.AST
    ) -> Binding | None:
        """Let a plugin type an operator applied to its own values."""
        if self.plugins is None or not symbol:
            return None
        plugin = root = None
        for operand in operands:
            base = T.strip_literal(operand.type)
            if isinstance(base, T.Instance) and "." in base.name:
                found = self.plugins.for_qualname(base.name)
                if found is not None:
                    plugin, root = found, base.name.rpartition(".")[0]
                    break
        if plugin is None or root is None:
            return None
        translate = getattr(plugin, "operator", None)
        if translate is None:
            return None
        operation = translate(symbol)
        if operation is None:
            return None
        qualname = f"{root}.{operation}"
        result = plugin.call(qualname, [(o.type, o.facts) for o in operands], {})
        if result is None:
            return None
        self._effects = self._effects | result.effects
        if result.lowering == "PythonFallback":
            self._native_blockers.append(f"`{symbol}` on `{root}` stays on the Python path")
        if self.record:
            self.module.lowerings[id(node)] = LoweringNote(
                qualname=qualname,
                lowering=str(result.lowering),
                reason=result.reason,
                guards=result.guards,
                line=getattr(node, "lineno", 0),
            )
        return Binding(result.type, result.facts)

    def _plugin_subscript(
        self, base: T.Instance, node: ast.Subscript, is_slice: bool
    ) -> Binding | None:
        """`a[i]` and `a[i:j]` on a library array type.

        A slice or a multi-dimensional index yields another array; a single
        index into a one-dimensional array yields a scalar, which the plugin
        names because only it knows the element type.
        """
        if self.plugins is None:
            return None
        plugin = self.plugins.for_qualname(base.name)
        describe = getattr(plugin, "subscript", None) if plugin is not None else None
        if describe is None:
            return None
        described = describe(base.name, is_slice=is_slice, tupled=isinstance(node.slice, ast.Tuple))
        if described is None:
            return None
        self._effects = self._effects.add(Effect.READ_OBJECT, raises=("IndexError",))
        self._native_blockers.append(f"`{base.name}` indexing has no native lowering")
        return Binding(described[0], described[1])

    def _plugin_instance_attribute(
        self, base: T.Type, attr: str, facts: Facts | None = None
    ) -> Binding | None:
        if self.plugins is None or not isinstance(base, T.Instance) or "." not in base.name:
            return None
        plugin = self.plugins.for_qualname(base.name)
        if plugin is None:
            return None
        describe = getattr(plugin, "instance_attribute", None)
        if describe is not None:
            # A plugin that reads the receiver's refinements can answer exactly
            # where a dtype was declared, instead of guessing an element type.
            described = describe(base.name, attr, facts or Facts())
            if described is not None:
                return Binding(described[0], described[1])
        table = _PLUGIN_INSTANCE_ATTRS.get(base.name, {})
        found = table.get(attr)
        if found is not None:
            return Binding(found, Facts())
        root = base.name.rpartition(".")[0]
        candidate = f"{root}.{attr}"
        if plugin.call(candidate, [], {}) is not None:
            return Binding(T.Callable_((), T.UNKNOWN, candidate))
        return Binding(T.UNKNOWN)

    def _error(self, code: str, message: str, node: ast.AST, help: str | None = None) -> None:
        if "<unknown>" in message:
            # An error that has to spell `<unknown>` is a consequence of a
            # type that was never resolved, and that unresolved type is the
            # finding: the untyped parameter (E1201, E1304) or the call with
            # no signature (E1306). Reporting every place the unknown flowed
            # to buried the one line that said why, two hundred times on a
            # large module. The count is reported once, with the origins.
            self.cascaded += 1
            return
        self.diagnostics.add(
            Diagnostic(code, Severity.ERROR, message, span_of(self.path, node), help=help)
        )

    def _error_at(self, code: str, message: str, node: ast.AST, help: str | None = None) -> None:
        self._error(code, message, node, help)

    def _warn(self, code: str, message: str, node: ast.AST) -> None:
        self.diagnostics.add(Diagnostic(code, Severity.WARNING, message, span_of(self.path, node)))

    def _remark(self, message: str, node: ast.AST) -> None:
        self.diagnostics.add(
            Diagnostic("R3001", Severity.REMARK, message, span_of(self.path, node))
        )


def _assigned_names(info: FunctionInfo) -> set[str]:
    """Names this function binds itself, which therefore shadow any global."""
    found: set[str] = set()
    for child in info.nodes:
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            found.add(child.id)
        elif isinstance(child, ast.arg):
            found.add(child.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(child.name)
        elif isinstance(child, ast.Global):
            found.difference_update(child.names)
    return found


def _invert_order(op: ast.cmpop) -> ast.cmpop:
    match op:
        case ast.Lt():
            return ast.GtE()
        case ast.LtE():
            return ast.Gt()
        case ast.Gt():
            return ast.LtE()
        case ast.GtE():
            return ast.Lt()
    return op


def _typeguard_target(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.expr | None:
    returns = node.returns
    if isinstance(returns, ast.Subscript):
        head = returns.value
        name = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", None)
        if name in {"TypeGuard", "TypeIs"}:
            return returns.slice
    return None


#: How many recording passes may run before the summaries are taken as settled.
_FIXPOINT_ROUNDS = 3


def analyze(
    symbols: ProjectSymbols,
    diagnostics: DiagnosticBag,
    *,
    strict: bool = True,
    dynamic_policy: str = "explicit",
    plugins: PluginRegistry | None = None,
    previous: ProjectAnalysis | None = None,
) -> ProjectAnalysis:
    """Run the effect fixpoint, then a final recording pass (spec 11.5).

    With `previous`, a module whose inputs -- its own signatures and fields,
    and those of everything it imports -- have not changed since it was last
    checked is not checked again: the earlier `ModuleAnalysis` is the answer.
    Inference asks this once per round and most rounds move a few modules,
    so most of a round used to be spent confirming what nothing had touched.
    A reused module contributes no diagnostics, so `previous` is only for
    callers that discard them; the analyses that report run without it.

    The fields a class assigns without annotating settle here too:
    `self.width = width` is typed from what the seed pass saw assigned, and
    a field assigned from another class's field in the round after. Every
    caller -- `check`, `run`, `build`, and the conversion fixpoint -- then
    starts from the same fields, and the reporting pass reads fields that
    are known.
    """
    from .inference import infer_fields, modules_with_unannotated_fields

    analysis = ProjectAnalysis(symbols=symbols, diagnostics=diagnostics)
    ordered = [symbols.modules[m.name] for m in symbols.graph.order() if m.name in symbols.modules]
    reusable: dict[str, ModuleAnalysis] = dict(previous.modules) if previous is not None else {}

    def summaries() -> dict[str, str]:
        return {
            qualname: f"{info.effects}|{info.ret}" for qualname, info in symbols.functions.items()
        }

    # One silent pass seeds every summary; a second pass is what confirms the
    # first reached a fixed point. Rather than confirm it and then traverse a
    # third time to record, the confirming pass does the recording, and its
    # diagnostics are kept only if it changed nothing -- which is exactly the
    # condition under which they were computed from final summaries.
    #
    # The seed is for the first analysis of a project. Inference calls this
    # once per round, and by then every function carries the summary the
    # previous round left; the confirming loop below finds what moved and
    # runs until nothing does, exactly as it would after a seed.
    if not symbols.seeded:
        silent = DiagnosticBag()
        recording = modules_with_unannotated_fields(symbols)
        seeded: dict[str, ModuleAnalysis] = {}
        for module_symbols in ordered:
            seeded[module_symbols.name] = _Checker(
                module_symbols,
                symbols,
                analysis,
                silent,
                strict=False,
                record=module_symbols.name in recording,
                dynamic_policy=dynamic_policy,
                plugins=plugins,
            ).check_module()
        infer_fields(symbols, seeded)
        symbols.seeded = True

    for attempt in range(_FIXPOINT_ROUNDS):
        before = summaries()
        final = attempt == _FIXPOINT_ROUNDS - 1
        bag = diagnostics if final else DiagnosticBag()
        modules: dict[str, ModuleAnalysis] = {}
        cascaded = 0
        digests = _module_digests(symbols, ordered, strict, dynamic_policy)
        for module_symbols in ordered:
            name = module_symbols.name
            kept = reusable.get(name)
            if kept is not None and symbols.module_digests.get(name) == digests[name]:
                modules[name] = kept
                continue
            checker = _Checker(
                module_symbols,
                symbols,
                analysis,
                bag,
                strict=strict,
                record=True,
                dynamic_policy=dynamic_policy,
                plugins=plugins,
            )
            modules[name] = checker.check_module()
            symbols.module_digests[name] = digests[name]
            cascaded += checker.cascaded
        # What this pass produced is what the next one may keep.
        reusable = modules
        # A field that moved changes the digest of every module that reads
        # it, so the next round rechecks exactly those.
        moved = infer_fields(symbols, modules)
        if final or (summaries() == before and not moved):
            analysis.modules.update(modules)
            if cascaded:
                bag.add(_unresolved_summary(cascaded, modules))
            if bag is not diagnostics:
                diagnostics.extend(bag)
            return analysis
    return analysis


def _module_digests(
    symbols: ProjectSymbols, ordered: list[ModuleSymbols], strict: bool, dynamic_policy: str
) -> dict[str, str]:
    """A digest per module of everything its checking can read.

    A module's functions see their own module's signatures and classes, and
    reach any function or class of a module they import through attribute
    access, so the digest is the module's own summary joined with the
    summaries of every module its imports name -- and of every module under
    those, since `import pkg` reaches `pkg.sub.f`. The summary of a module is
    its functions' signatures, effects, and returns, and its classes' fields
    and methods, which are exactly the things inference moves between passes.
    Module globals and imports are fixed at symbol time and need no digest.
    """
    own: dict[str, str] = {}
    for module_symbols in ordered:
        parts = [_function_summary(info) for info in module_symbols.functions.values()]
        for cls in module_symbols.classes.values():
            parts.append(f"{cls.qualname}|{cls.fields}|{cls.field_facts}|{sorted(cls.class_vars)}")
            parts.extend(_function_summary(method) for method in cls.methods.values())
        own[module_symbols.name] = hashlib.blake2b(
            "\n".join(parts).encode("utf-8"), digest_size=16
        ).hexdigest()
    names = sorted(own)
    digests: dict[str, str] = {}
    for module_symbols in ordered:
        seen: list[str] = [f"{strict}|{dynamic_policy}", own[module_symbols.name]]
        for edge in module_symbols.module.imports:
            target = edge.target
            seen.extend(
                f"{name}:{own[name]}"
                for name in names
                if name == target or name.startswith(target + ".")
            )
        digests[module_symbols.name] = hashlib.blake2b(
            "\n".join(seen).encode("utf-8"), digest_size=16
        ).hexdigest()
    return digests


def _function_summary(info: FunctionInfo) -> str:
    params = ";".join(
        f"{p.name}:{p.type}:{p.facts}:{p.has_default}:{p.kind}:{p.annotated}" for p in info.params
    )
    return (
        f"{info.qualname}({params})->{info.ret}|{info.ret_facts}|{info.ret_annotated}"
        f"|{info.effects}|{info.verified_pure}|{info.dynamic}|{info.directives}"
    )


def _unresolved_summary(cascaded: int, modules: dict[str, ModuleAnalysis]) -> Diagnostic:
    """One line for every error that only restated an unresolved type.

    The origins it names are the calls whose signatures the analysis does
    not know; an untyped parameter already has its own line (E1201 under
    strict analysis, E1304 from conversion).
    """
    callees: dict[str, int] = {}
    for module in modules.values():
        for function in module.functions.values():
            for name in function.unknown_callees:
                callees[name] = callees.get(name, 0) + 1
    ranked = sorted(callees, key=lambda name: (-callees[name], name))
    shown = ", ".join(f"`{name}`" for name in ranked[:5])
    if len(ranked) > 5:
        shown += f", and {len(ranked) - 5} more"
    where = f"; unknown signatures: {shown}" if shown else ""
    plural = "s" if cascaded != 1 else ""
    return Diagnostic(
        "W2006",
        Severity.WARNING,
        f"{cascaded} further error{plural} only restated a type that could not be resolved"
        f" and {'are' if cascaded != 1 else 'is'} not shown{where}",
        help="resolve the origins above -- annotate the parameter, or add a stub or plugin "
        "for the call -- and the rest follow",
    )
