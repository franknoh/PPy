"""Which module-level names the rest of the project assigns to.

`Final` is a promise about the whole program, so proving it takes the whole
program: `config.LIMIT = 5` in a file that was not being converted rebinds the
name just as surely as a second assignment at home would. This index is built
once over every source under the project root -- including files outside the
current conversion -- and answers the one question the converter asks.

The scan is scope-aware, because import aliases are: `import other as s`
inside a function shadows nothing at module level, `s = store` makes `s` the
module for as long as that binding lasts, and `from . import store` resolves
against the file's own package. When resolution is uncertain -- a name bound
to two modules on different branches -- every candidate gets the write; when
a file cannot be parsed at all, the index fails closed and vouches for
nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..frontend.modules import resolve_module_name

__all__ = ["GlobalWriteIndex", "build_write_index"]

#: Directories whose sources are not the project's own.
_SKIP = frozenset(
    {".venv", "venv", ".git", "__pycache__", "build", "dist", ".ppy-cache", ".tox", "node_modules"}
)

#: name -> the modules that name may currently refer to.
_Env = dict[str, frozenset[str]]


@dataclass(slots=True)
class GlobalWriteIndex:
    """`module -> names` for every cross-module attribute write."""

    writes: dict[str, set[str]] = field(default_factory=dict)
    #: Modules hit by a `setattr` whose name is not a literal. No name in
    #: them is provably unbound.
    dynamic: set[str] = field(default_factory=set)
    #: A project file could not be read or parsed, so the scan is incomplete
    #: and the index must vouch for nothing.
    tainted: bool = False

    def can_emit_final(self, module: str, name: str) -> bool:
        """Is `module.name` free of assignments anywhere else in the project?

        `module` is the converter's qualified name for the file; imports may
        reach it under a shorter spelling, so both directions of suffix match
        count as the same module. Missing evidence is a no.
        """
        if self.tainted:
            return False
        for target, names in self.writes.items():
            if _same_module(module, target) and name in names:
                return False
        return not any(_same_module(module, target) for target in self.dynamic)


def _same_module(module: str, target: str) -> bool:
    return module == target or module.endswith("." + target) or target.endswith("." + module)


def build_write_index(root: Path, source_roots: tuple[str, ...] = ("src", ".")) -> GlobalWriteIndex:
    index = GlobalWriteIndex()
    search_paths = [root / entry for entry in source_roots if (root / entry).is_dir()]
    if not search_paths:
        search_paths = [root]
    for path in _sources(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            # An unreadable file could hold any write; claiming `Final`
            # anyway would be promising something the scan never saw.
            index.tainted = True
            continue
        module_name = resolve_module_name(path, search_paths)
        _FileScan(index, module_name, path.name.startswith("__init__.")).run(tree)
    return index


def _sources(root: Path):  # type: ignore[no-untyped-def]
    for suffix in ("*.py", "*.ppy"):
        for path in root.rglob(suffix):
            if not any(part in _SKIP for part in path.parts):
                yield path


class _FileScan:
    """Lexical, scope-aware walk of one file."""

    def __init__(self, index: GlobalWriteIndex, module_name: str, is_package: bool) -> None:
        self.index = index
        self.module = module_name
        self.package = module_name if is_package else module_name.rpartition(".")[0]

    def run(self, tree: ast.Module) -> None:
        # Pass 1: every module-scope binding a name ever has. A function body
        # runs at an unknown time, so inside one, a module-level name may hold
        # any of these -- resolving it to all of them records every write the
        # program could make, which is the safe direction.
        anywhere: _Env = {}
        self._block(tree.body, {}, anywhere, functions=[])
        # Pass 2: module scope again, precisely and in order, descending into
        # the definitions collected on the way with `anywhere` as their base.
        functions: list[tuple[list[ast.stmt], _Env]] = []
        self._block(tree.body, {}, None, functions=functions)
        while functions:
            body, inherited = functions.pop()
            base = dict(anywhere)
            base.update(inherited)
            self._block(body, base, None, functions=functions)

    # -- statement flow -----------------------------------------------------

    def _block(
        self,
        body: list[ast.stmt],
        env: _Env,
        anywhere: _Env | None,
        functions: list,
    ) -> _Env:
        for stmt in body:
            env = self._stmt(stmt, env, anywhere, functions)
        return env

    def _stmt(self, node: ast.stmt, env: _Env, anywhere, functions) -> _Env:  # type: ignore[no-untyped-def]
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    env = self._bind(env, anywhere, alias.asname, frozenset({alias.name}))
                else:
                    root = alias.name.partition(".")[0]
                    env = self._bind(env, anywhere, root, frozenset({root}))
            return env
        if isinstance(node, ast.ImportFrom):
            base = self._from_target(node)
            for alias in node.names:
                if alias.name == "*":
                    continue
                # `from pkg import store` may bind the module `pkg.store`;
                # recording it costs nothing when it turns out to be a value.
                bound = frozenset({f"{base}.{alias.name}" if base else alias.name})
                env = self._bind(env, anywhere, alias.asname or alias.name, bound)
            return env
        if isinstance(node, ast.Assign):
            self._note_write_targets(node.targets, env)
            value = self._modules_of(node.value, env)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env = self._bind(env, anywhere, target.id, value)
            self._scan_calls(node.value, env)
            return env
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            self._note_write_targets([node.target], env)
            if node.value is not None:
                self._scan_calls(node.value, env)
            if isinstance(node.target, ast.Name) and isinstance(node, ast.AnnAssign):
                env = self._bind(env, anywhere, node.target.id, self._modules_of(node.value, env))
            return env
        if isinstance(node, ast.Delete):
            self._note_write_targets(node.targets, env)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env = self._bind(env, anywhere, target.id, frozenset())
            return env
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                self._scan_calls(decorator, env)
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    self._scan_calls(default, env)
            functions.append((node.body, dict(env)))
            return self._bind(env, anywhere, node.name, frozenset())
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                self._scan_calls(decorator, env)
            # A class body runs here and now, in order, in its own scope.
            self._block(node.body, dict(env), anywhere, functions)
            return self._bind(env, anywhere, node.name, frozenset())
        if isinstance(node, ast.If):
            self._scan_calls(node.test, env)
            then = self._block(node.body, dict(env), anywhere, functions)
            other = self._block(node.orelse, dict(env), anywhere, functions)
            return _join(then, other)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self._scan_calls(node.iter, env)
            after = self._block(node.body, dict(env), anywhere, functions)
            after = self._block(node.orelse, after, anywhere, functions)
            return _join(env, after)
        if isinstance(node, ast.While):
            self._scan_calls(node.test, env)
            after = self._block(node.body, dict(env), anywhere, functions)
            after = self._block(node.orelse, after, anywhere, functions)
            return _join(env, after)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self._scan_calls(item.context_expr, env)
                if isinstance(item.optional_vars, ast.Name):
                    env = self._bind(env, anywhere, item.optional_vars.id, frozenset())
            return self._block(node.body, env, anywhere, functions)
        if isinstance(node, (ast.Try, ast.TryStar)):
            merged = self._block(node.body, dict(env), anywhere, functions)
            for handler in node.handlers:
                merged = _join(merged, self._block(handler.body, dict(env), anywhere, functions))
            merged = self._block(node.orelse, merged, anywhere, functions)
            return self._block(node.finalbody, merged, anywhere, functions)
        if isinstance(node, ast.Match):
            self._scan_calls(node.subject, env)
            merged: _Env | None = None
            for case in node.cases:
                after = self._block(case.body, dict(env), anywhere, functions)
                merged = after if merged is None else _join(merged, after)
            return merged if merged is not None else env
        if isinstance(node, (ast.Expr, ast.Return, ast.Raise, ast.Assert)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._scan_calls(child, env)
            return env
        if isinstance(node, ast.Global):
            # `global s` then `s = ...` rebinding inside a function: the
            # binding lands at module scope, which pass 1 already saw.
            return env
        return env

    def _bind(self, env: _Env, anywhere: _Env | None, name: str, value: frozenset[str]) -> _Env:
        env = dict(env)
        env[name] = value
        if anywhere is not None:
            anywhere[name] = anywhere.get(name, frozenset()) | value
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

    # -- writes -------------------------------------------------------------

    def _note_write_targets(self, targets: list[ast.expr], env: _Env) -> None:
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            for module in self._modules_of(target.value, env):
                self.index.writes.setdefault(module, set()).add(target.attr)

    def _scan_calls(self, node: ast.expr, env: _Env) -> None:
        """`setattr`/`delattr` anywhere inside an expression."""
        for child in ast.walk(node):
            if (
                not isinstance(child, ast.Call)
                or not isinstance(child.func, ast.Name)
                or child.func.id not in {"setattr", "delattr"}
                or not child.args
            ):
                continue
            modules = self._modules_of(child.args[0], env)
            if not modules:
                continue
            written = child.args[1] if len(child.args) > 1 else None
            for module in modules:
                if isinstance(written, ast.Constant) and isinstance(written.value, str):
                    self.index.writes.setdefault(module, set()).add(written.value)
                else:
                    self.index.dynamic.add(module)

    def _modules_of(self, value: ast.expr | None, env: _Env) -> frozenset[str]:
        """The modules an expression may name, through any alias chain."""
        parts: list[str] = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if not isinstance(value, ast.Name):
            return frozenset()
        bases = env.get(value.id, frozenset())
        if not parts:
            return bases
        tail = ".".join(reversed(parts))
        return frozenset(f"{base}.{tail}" for base in bases)


def _join(left: _Env, right: _Env) -> _Env:
    merged: _Env = {}
    for name in left.keys() | right.keys():
        merged[name] = left.get(name, frozenset()) | right.get(name, frozenset())
    return merged
