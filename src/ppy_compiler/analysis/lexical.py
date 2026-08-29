"""Point-sensitive lexical name resolution for one module.

"What does this name mean *here*?" is a question about a place in the file,
not about the file: `@cache` above a local `def cache` is that function even
if `from functools import cache` appears further down, and was `functools.cache`
only until the local `def` rebound it. Decorator identity, the reflection
scan, and the cross-module write index all need this same answer, so it is
computed once, here, instead of three slightly different ways.

The scanner walks a module top to bottom keeping an environment of
`name -> canonical targets` ("functools.cache", "mymodule.helper",
"builtins.staticmethod", or a bare module like "store"), snapshotting it at
every statement. Reassignment kills, branches join by union, class bodies get
their own scope, and a function body is resolved against the environment at
its `def` merged with everything module scope may ever hold -- the body runs
later, when any module-level rebinding may have happened. A name a function
rebinds under `global` feeds that module-level union too.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field

__all__ = ["LexicalBindings", "scan_module"]

_Env = dict[str, frozenset[str]]

_BUILTINS = frozenset(dir(builtins))


@dataclass(slots=True)
class LexicalBindings:
    """Snapshots of the binding environment across one module."""

    module: str
    #: id(node) -> environment before the enclosing statement runs.
    _at: dict[int, _Env] = field(default_factory=dict)
    #: Every binding module scope may ever hold, `global` rebindings included.
    anywhere: _Env = field(default_factory=dict)

    def targets_at(self, node: ast.expr) -> frozenset[str]:
        """The canonical things a Name/Attribute chain may mean at `node`.

        Empty means "a value this pass cannot name" -- consumers must treat
        that as unknown, not as safe.
        """
        parts: list[str] = []
        probe: ast.expr = node
        while isinstance(probe, ast.Attribute):
            parts.append(probe.attr)
            probe = probe.value
        if not isinstance(probe, ast.Name):
            return frozenset()
        env = self._at.get(id(node)) or self._at.get(id(probe)) or {}
        bases = env.get(probe.id)
        if bases is None:
            bases = self.anywhere.get(probe.id)
        if bases is None:
            if probe.id in _BUILTINS:
                bases = frozenset({f"builtins.{probe.id}"})
            else:
                return frozenset()
        if not parts:
            return bases
        tail = ".".join(reversed(parts))
        return frozenset(f"{base}.{tail}" for base in bases)


class _Scanner:
    def __init__(self, module: str, package: str) -> None:
        self.module = module
        self.package = package
        self.at: dict[int, _Env] = {}
        self.anywhere: _Env = {}

    def run(self, tree: ast.Module) -> LexicalBindings:
        # Pass 1 collects `anywhere`: every module-scope binding, plus every
        # assignment made under a `global` declaration inside any function.
        # Two rounds, because `global s; s = other` needs `other` resolved.
        for _ in range(2):
            self._collect_anywhere(tree.body, dict(self.anywhere), globals_only=False)
        # Pass 2 records precise, in-order snapshots.
        functions: list[tuple[list[ast.stmt], _Env]] = []
        self._flow(tree.body, {}, functions, snapshot=True)
        while functions:
            body, inherited = functions.pop()
            # The body runs later: any module-scope name may have been
            # rebound by then -- `global s` in some other function included --
            # so the def-point view joins with everything module scope may
            # ever hold rather than shadowing it.
            self._flow(body, _join(self.anywhere, inherited), functions, snapshot=True)
        return LexicalBindings(self.module, self.at, self.anywhere)

    # -- pass 1 --------------------------------------------------------------

    def _collect_anywhere(self, body: list[ast.stmt], env: _Env, *, globals_only: bool) -> _Env:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared: set[str] = set()
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Global):
                        declared.update(inner.names)
                if declared:
                    # Only the global-declared bindings leak out; everything
                    # else in the body is the function's own business.
                    inner_env = dict(self.anywhere)
                    inner_env.update(env)
                    self._collect_globals(stmt.body, inner_env, declared)
                if not globals_only:
                    env = self._bind_anywhere(env, stmt.name, self._local(stmt.name))
                continue
            if isinstance(stmt, ast.ClassDef):
                if not globals_only:
                    env = self._bind_anywhere(env, stmt.name, self._local(stmt.name))
                continue
            env = self._stmt(stmt, env, functions=None, anywhere=not globals_only)
        return env

    def _collect_globals(self, body: list[ast.stmt], env: _Env, declared: set[str]) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            env = self._stmt(stmt, env, functions=None, anywhere=False)
            # A binding to a global-declared name lands at module scope.
            for name in declared:
                bound = env.get(name)
                if bound:
                    self.anywhere[name] = self.anywhere.get(name, frozenset()) | bound

    # -- pass 2 --------------------------------------------------------------

    def _flow(self, body: list[ast.stmt], env: _Env, functions: list, *, snapshot: bool) -> _Env:
        for stmt in body:
            if snapshot:
                self._snapshot(stmt, env)
            env = self._stmt(stmt, env, functions=functions, anywhere=False)
        return env

    def _snapshot(self, stmt: ast.stmt, env: _Env) -> None:
        frozen = dict(env)
        stack: list[ast.AST] = [stmt]
        while stack:
            node = stack.pop()
            existing = self.at.get(id(node))
            if existing is None:
                self.at[id(node)] = frozen
            elif existing is not frozen:
                self.at[id(node)] = _join(existing, frozen)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.stmt, ast.Lambda)):
                    continue
                stack.append(child)

    # -- shared statement semantics ------------------------------------------

    def _stmt(self, node: ast.stmt, env: _Env, functions, *, anywhere: bool) -> _Env:  # type: ignore[no-untyped-def]
        bind = self._bind_anywhere if anywhere else _bind
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    env = bind(env, alias.asname, frozenset({alias.name}))
                else:
                    root = alias.name.partition(".")[0]
                    env = bind(env, root, frozenset({root}))
            return env
        if isinstance(node, ast.ImportFrom):
            base = self._from_target(node)
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = frozenset({f"{base}.{alias.name}" if base else alias.name})
                env = bind(env, alias.asname or alias.name, bound)
            return env
        if isinstance(node, ast.Assign):
            value = self._eval(node.value, env)
            for target in node.targets:
                env = self._bind_target(target, value, env, bind)
            return env
        if isinstance(node, ast.AnnAssign):
            value = self._eval(node.value, env) if node.value is not None else frozenset()
            return self._bind_target(node.target, value, env, bind)
        if isinstance(node, ast.AugAssign):
            # Whatever it was, it is not simply that thing any more.
            return self._bind_target(node.target, frozenset(), env, bind)
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env = bind(env, target.id, frozenset())
            return env
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if functions is not None:
                functions.append((node.body, dict(env)))
            return bind(env, node.name, self._local(node.name))
        if isinstance(node, ast.ClassDef):
            if functions is not None:
                self._flow(node.body, dict(env), functions, snapshot=True)
            return bind(env, node.name, self._local(node.name))
        if isinstance(node, ast.If):
            then = self._branch(node.body, env, functions)
            other = self._branch(node.orelse, env, functions)
            return _join(then, other)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            after = self._branch(node.body, env, functions)
            after = self._branch(node.orelse, after, functions)
            return _join(env, after)
        if isinstance(node, ast.While):
            after = self._branch(node.body, env, functions)
            after = self._branch(node.orelse, after, functions)
            return _join(env, after)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    env = bind(env, item.optional_vars.id, frozenset())
            return self._branch(node.body, env, functions)
        if isinstance(node, (ast.Try, ast.TryStar)):
            merged = self._branch(node.body, env, functions)
            for handler in node.handlers:
                inner = dict(env)
                if handler.name:
                    inner = _bind(inner, handler.name, frozenset())
                merged = _join(merged, self._branch(handler.body, inner, functions))
            merged = self._branch(node.orelse, merged, functions)
            return self._branch(node.finalbody, merged, functions)
        if isinstance(node, ast.Match):
            merged: _Env | None = None
            for case in node.cases:
                after = self._branch(case.body, env, functions)
                merged = after if merged is None else _join(merged, after)
            return merged if merged is not None else env
        return env

    def _branch(self, body: list[ast.stmt], env: _Env, functions) -> _Env:  # type: ignore[no-untyped-def]
        if functions is None:
            return self._collect_anywhere(body, dict(env), globals_only=True)
        return self._flow(body, dict(env), functions, snapshot=True)

    def _bind_target(self, target: ast.expr, value: frozenset[str], env: _Env, bind) -> _Env:  # type: ignore[no-untyped-def]
        if isinstance(target, ast.Name):
            return bind(env, target.id, value)
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                env = self._bind_target(
                    element.value if isinstance(element, ast.Starred) else element,
                    frozenset(),
                    env,
                    bind,
                )
        return env

    def _eval(self, node: ast.expr, env: _Env) -> frozenset[str]:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return frozenset()
        bases = env.get(node.id)
        if bases is None and node.id in _BUILTINS:
            bases = frozenset({f"builtins.{node.id}"})
        if not bases:
            return frozenset()
        if not parts:
            return bases
        tail = ".".join(reversed(parts))
        return frozenset(f"{base}.{tail}" for base in bases)

    def _local(self, name: str) -> frozenset[str]:
        return frozenset({f"{self.module}.{name}"})

    def _bind_anywhere(self, env: _Env, name: str, value: frozenset[str]) -> _Env:
        env = _bind(env, name, value)
        self.anywhere[name] = self.anywhere.get(name, frozenset()) | value
        return env

    def _from_target(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        parts = self.package.split(".") if self.package else []
        drop = node.level - 1
        if drop:
            parts = parts[:-drop] if drop <= len(parts) else []
        if node.module:
            parts.append(node.module)
        return ".".join(p for p in parts if p)


def _bind(env: _Env, name: str, value: frozenset[str]) -> _Env:
    env = dict(env)
    env[name] = value
    return env


def _join(left: _Env, right: _Env) -> _Env:
    merged: _Env = {}
    for name in left.keys() | right.keys():
        merged[name] = left.get(name, frozenset()) | right.get(name, frozenset())
    return merged


def scan_module(tree: ast.Module, module: str, *, is_package: bool = False) -> LexicalBindings:
    package = module if is_package else module.rpartition(".")[0]
    return _Scanner(module, package).run(tree)
