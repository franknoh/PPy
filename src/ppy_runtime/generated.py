"""Generated-module identity and the hooks a backend injects (spec 15.4).

A generated module is source the compiler wrote plus the names it binds at
load time. Executing one is a runtime activity -- this module keeps the
dataclass and the hook names free of any compiler dependency, so a built
artifact can load its generated code with the compiler uninstalled.
"""

from __future__ import annotations

from pathlib import Path

from ._record import field
from ._record import record as dataclass

TYPE_CHECKING = False
if TYPE_CHECKING:
    import ast

__all__ = [
    "BINDER_NAME",
    "EXPORTED_BINDER",
    "FUSED_BINDER",
    "REGION_BINDER",
    "GeneratedModule",
]

#: Names injected into a generated module namespace by the backends.
BINDER_NAME = "__ppy_bind_native__"
EXPORTED_BINDER = "__ppy_bind_exported__"
REGION_BINDER = "__ppy_bind_region__"
FUSED_BINDER = "__ppy_bind_fused__"


@dataclass(slots=True)
class GeneratedModule:
    name: str
    source_path: Path
    code: str
    artifact: Path
    key: str
    line_map: dict[int, int]
    fused_symbols: tuple[str, ...] = ()
    #: Compiled code by the names bound into it, filled as the module is
    #: compiled and carried in the launcher's cache.
    compiled: dict = field(default_factory=dict)  # type: ignore[type-arg]

    @property
    def needs_fused_binder(self) -> bool:
        return bool(self.fused_symbols)

    def compile(
        self,
        native_names: frozenset[str] = frozenset(),
        exported_names: frozenset[str] = frozenset(),
        region_names: frozenset[str] = frozenset(),
    ) -> object:
        """Compile with the original `.ppy` filename so tracebacks map back."""
        key = (
            tuple(sorted(native_names)),
            tuple(sorted(exported_names)),
            tuple(sorted(region_names)),
        )
        found = self.compiled.get(key)
        if found is not None:
            return found
        import ast  # pylint: disable=import-outside-toplevel

        tree = ast.parse(self.code, filename=str(self.source_path))
        _restore_lines(tree, self.line_map)
        bindings = [
            (BINDER_NAME, native_names),
            (EXPORTED_BINDER, exported_names),
            (REGION_BINDER, region_names),
        ]
        if any(names for _binder, names in bindings):
            _insert_definition_bindings(tree, bindings)
        made = compile(tree, str(self.source_path), "exec", dont_inherit=True)
        self.compiled[key] = made
        return made


def _restore_lines(tree: ast.Module, line_map: dict[int, int]) -> None:
    import ast  # pylint: disable=import-outside-toplevel

    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        original = line_map.get(node.lineno)
        if original is None:
            continue
        node.lineno = original
        if getattr(node, "end_lineno", None) is not None:
            node.end_lineno = max(original, line_map.get(node.end_lineno, original))


def _insert_definition_bindings(
    tree: ast.Module, bindings: list[tuple[str, frozenset[str]]]
) -> None:
    """Bind an entry point immediately after its `def`, so every later
    reference -- a top-level `main()`, or a method looked up on its class --
    uses it."""
    tree.body = _bind_in(tree.body, bindings, prefix="")


def _bind_in(
    body: list[ast.stmt], bindings: list[tuple[str, frozenset[str]]], prefix: str
) -> list[ast.stmt]:
    import ast  # pylint: disable=import-outside-toplevel

    rebuilt: list[ast.stmt] = []
    for statement in body:
        if isinstance(statement, ast.ClassDef):
            statement.body = _bind_in(statement.body, bindings, f"{prefix}{statement.name}.")
            rebuilt.append(statement)
            continue
        rebuilt.append(statement)
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        key = f"{prefix}{statement.name}"
        for binder, names in bindings:
            if key not in names:
                continue
            binding = ast.Assign(
                targets=[ast.Name(id=statement.name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=binder, ctx=ast.Load()),
                    args=[
                        ast.Constant(value=key),
                        ast.Name(id=statement.name, ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
            ast.copy_location(binding, statement)
            ast.fix_missing_locations(binding)
            rebuilt.append(binding)
    return rebuilt


#: Names injected into a generated module namespace by the backends.
BINDER_NAME = "__ppy_bind_native__"
EXPORTED_BINDER = "__ppy_bind_exported__"
REGION_BINDER = "__ppy_bind_region__"
