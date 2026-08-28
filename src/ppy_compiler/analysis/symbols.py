"""Symbol tables, class hierarchy, and function signatures (spec 7.4, 8.5)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..diagnostics import DiagnosticBag
from ..frontend.modules import Module, ModuleGraph
from . import types as T
from .annotations import AnnotationResolver
from .effects import EffectSet
from .refinements import Facts, IntRange

__all__ = [
    "ClassInfo",
    "Directive",
    "FunctionInfo",
    "ImportBinding",
    "ModuleSymbols",
    "ParamInfo",
    "ProjectSymbols",
    "directives_from",
]

_BUILTIN_NAMES = {
    "int",
    "float",
    "bool",
    "str",
    "bytes",
    "complex",
    "object",
    "bytearray",
    "memoryview",
    "list",
    "dict",
    "set",
    "frozenset",
    "tuple",
    "range",
    "slice",
    "type",
    "BaseException",
    "Exception",
    "ValueError",
    "TypeError",
    "IndexError",
    "KeyError",
    "LookupError",
    "ZeroDivisionError",
    "ArithmeticError",
    "OverflowError",
    "StopIteration",
    "AttributeError",
    "None",
    "True",
    "False",
    "OSError",
    "RuntimeError",
    "NotImplementedError",
    "AssertionError",
    "StopAsyncIteration",
    "GeneratorExit",
    "KeyboardInterrupt",
    "SystemExit",
}


@dataclass(frozen=True, slots=True)
class Directive:
    name: str
    options: dict[str, object] = field(default_factory=dict)
    node: ast.expr | None = None

    @property
    def level(self) -> int | None:
        value = self.options.get("level")
        return value if isinstance(value, int) else None

    @property
    def require(self) -> bool:
        return bool(self.options.get("require", False))


@dataclass(frozen=True, slots=True)
class ImportBinding:
    """A module-level name introduced by an import statement."""

    local: str
    module: str
    origin: str | None = None
    external: bool = False

    @property
    def canonical(self) -> str:
        return f"{self.module}.{self.origin}" if self.origin else self.module


@dataclass(slots=True)
class ParamInfo:
    name: str
    type: T.Type
    facts: Facts = field(default_factory=Facts)
    has_default: bool = False
    default: ast.expr | None = None
    kind: str = "positional_or_keyword"
    #: Whether the source declares this type. Inference may not overrule it.
    annotated: bool = False
    #: Whether a pass filled this type in. Usable, but still open to new
    #: evidence -- which is why it is not the same flag as `annotated`.
    inferred: bool = False

    @property
    def known(self) -> bool:
        """Whether the type is settled enough to rely on."""
        return self.annotated or self.inferred


@dataclass(slots=True)
class FunctionInfo:
    name: str
    qualname: str
    module: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    path: Path
    params: list[ParamInfo] = field(default_factory=list)
    ret: T.Type = T.UNKNOWN
    ret_facts: Facts = field(default_factory=Facts)
    ret_annotated: bool = False
    directives: tuple[Directive, ...] = ()
    decorators: tuple[str, ...] = ()
    is_method: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    owner: str | None = None
    effects: EffectSet = field(default_factory=EffectSet)
    verified_pure: bool = False
    dynamic: bool = False
    #: Whether the body yields. The answer needs a walk of the whole function,
    #: and `signature()` asks for it once per function per checked function, so
    #: it is computed once and kept.
    _generator: bool | None = field(default=None, repr=False, compare=False)

    @property
    def is_async(self) -> bool:
        return isinstance(self.node, ast.AsyncFunctionDef)

    @property
    def is_generator(self) -> bool:
        if self._generator is None:
            self._generator = _contains_yield(self.node)
        return self._generator

    def directive(self, name: str) -> Directive | None:
        for d in self.directives:
            if d.name == name:
                return d
        return None

    @property
    def opt_level(self) -> int | None:
        d = self.directive("opt")
        return d.level if d else None

    @property
    def declared_pure(self) -> bool:
        return self.directive("pure") is not None

    def signature(self) -> T.Callable_:
        return T.Callable_(
            tuple(T.Param(p.name, p.type, p.has_default, p.kind) for p in self.params),
            self.ret,
            self.qualname,
            is_async=self.is_async,
            is_generator=self.is_generator,
        )


@dataclass(slots=True)
class ClassInfo:
    name: str
    qualname: str
    module: str
    node: ast.ClassDef
    path: Path
    base_names: tuple[str, ...] = ()
    mro: tuple[str, ...] = ()
    fields: dict[str, T.Type] = field(default_factory=dict)
    field_facts: dict[str, Facts] = field(default_factory=dict)
    class_vars: set[str] = field(default_factory=set)
    methods: dict[str, FunctionInfo] = field(default_factory=dict)
    decorators: tuple[str, ...] = ()
    directives: tuple[Directive, ...] = ()
    slots: tuple[str, ...] | None = None
    is_dataclass: bool = False
    is_enum: bool = False
    is_pydantic: bool = False
    is_protocol: bool = False

    def instance(self, args: tuple[T.Type, ...] = ()) -> T.Instance:
        return T.Instance(self.qualname, args, self.mro or (self.qualname, "object"))

    def find_method(self, name: str, project: ProjectSymbols) -> FunctionInfo | None:
        for entry in self.mro or (self.qualname,):
            info = project.classes.get(entry)
            if info is not None and name in info.methods:
                return info.methods[name]
        return None

    def lookup(self, name: str, project: ProjectSymbols) -> tuple[T.Type, Facts] | None:
        for entry in self.mro or (self.qualname,):
            info = project.classes.get(entry)
            if info is None:
                continue
            if name in info.fields:
                return info.fields[name], info.field_facts.get(name, Facts())
            if name in info.methods:
                return info.methods[name].signature(), Facts()
        return None


@dataclass(slots=True)
class ModuleSymbols:
    module: Module
    imports: dict[str, ImportBinding] = field(default_factory=dict)
    classes: dict[str, ClassInfo] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    globals: dict[str, T.Type] = field(default_factory=dict)
    global_facts: dict[str, Facts] = field(default_factory=dict)
    constant_globals: dict[str, object] = field(default_factory=dict)
    type_aliases: dict[str, ast.expr] = field(default_factory=dict)
    all_exports: tuple[str, ...] | None = None

    @property
    def name(self) -> str:
        return self.module.name

    @property
    def path(self) -> Path:
        return self.module.path


def _contains_yield(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.Yield, ast.YieldFrom)) and _enclosing_function(node, child):
            return True
    return False


def _enclosing_function(root: ast.AST, target: ast.AST) -> bool:
    """True when `target` belongs to `root` rather than a nested function."""
    stack: list[tuple[ast.AST, bool]] = [(root, True)]
    while stack:
        node, is_root = stack.pop()
        for child in ast.iter_child_nodes(node):
            if child is target:
                return True
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                and not is_root
            ):
                continue
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                and not is_root
            ):
                continue
            stack.append((child, False))
    return False


#: The decorators `ppy` actually exports. A name outside this set is a typo,
#: and would raise AttributeError the moment plain CPython ran the file.
DIRECTIVE_NAMES = frozenset(
    {
        "pure",
        "opt",
        "jit",
        "parallel",
        "native",
        "inline",
        "noinline",
        "specialize",
        "fastmath",
        "dynamic",
        "jax",
    }
)


def directives_from(decorators: list[ast.expr], resolver: NameResolver) -> tuple[Directive, ...]:
    """Read PPY directives from decorator syntax (spec 6.1)."""
    found: list[Directive] = []
    for decorator in decorators:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        qualname = resolver.canonical(target)
        if qualname is None or not qualname.startswith("ppy."):
            continue
        name = qualname.removeprefix("ppy.")
        options: dict[str, object] = {}
        if isinstance(decorator, ast.Call):
            for index, arg in enumerate(decorator.args):
                try:
                    value = ast.literal_eval(arg)
                except (ValueError, SyntaxError):
                    continue
                options["level" if name == "opt" and index == 0 else f"arg{index}"] = value
            for keyword in decorator.keywords:
                if keyword.arg is None:
                    continue
                try:
                    options[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    continue
        found.append(Directive(name, options, decorator))
    return tuple(found)


class NameResolver:
    """Resolves annotation names for one module against the whole project."""

    def __init__(self, symbols: ModuleSymbols, project: ProjectSymbols) -> None:
        self.symbols = symbols
        self.project = project

    def canonical(self, expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return self._canonical_name(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self.canonical(expr.value)
            return f"{base}.{expr.attr}" if base else None
        if isinstance(expr, ast.Subscript):
            return self.canonical(expr.value)
        return None

    def _canonical_name(self, name: str) -> str | None:
        binding = self.symbols.imports.get(name)
        if binding is not None:
            return binding.canonical
        local = f"{self.symbols.name}.{name}"
        if local in self.project.classes:
            return local
        if local in self.project.functions:
            return local
        if name in _BUILTIN_NAMES:
            return f"builtins.{name}"
        if name in self.symbols.classes:
            return self.symbols.classes[name].qualname
        return None

    def type_alias(self, name: str) -> ast.expr | None:
        return self.symbols.type_aliases.get(name)

    def class_instance(self, qualname: str, args: tuple[T.Type, ...]) -> T.Instance | None:
        info = self.project.classes.get(qualname)
        if info is not None:
            return info.instance(args)
        external = self.project.external_types.get(qualname)
        if external is not None:
            return T.Instance(external, args, (external, "object"))
        return None


class ProjectSymbols:
    """Project-wide symbol, class, and signature tables."""

    def __init__(
        self, graph: ModuleGraph, diagnostics: DiagnosticBag, *, strict: bool = True
    ) -> None:
        self.graph = graph
        self.diagnostics = diagnostics
        self.strict = strict
        self.modules: dict[str, ModuleSymbols] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.functions: dict[str, FunctionInfo] = {}
        self.external_types: dict[str, str] = {}

    def build(self) -> ProjectSymbols:
        ordered = self.graph.order()
        for module in ordered:
            self.modules[module.name] = ModuleSymbols(module=module)
        for module in ordered:
            self._collect_imports(self.modules[module.name])
        for module in ordered:
            self._collect_declarations(self.modules[module.name])
        for module in ordered:
            self._collect_type_aliases(self.modules[module.name])
        for module in ordered:
            self._compute_mro(self.modules[module.name])
        for module in ordered:
            self._resolve_signatures(self.modules[module.name])
        self._mark_constant_globals()
        return self

    def _mark_constant_globals(self) -> None:
        """Find module globals bound once, at module level, to a literal.

        Such a name is effectively final, so reading it is neither a global
        dependency for purity nor a barrier to constant propagation.
        """
        rebound: set[tuple[str, str]] = set()
        for symbols in self.modules.values():
            for node in ast.walk(symbols.module.tree):
                if isinstance(node, ast.Global):
                    rebound.update((symbols.name, name) for name in node.names)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    owner = node.value
                    if isinstance(owner, ast.Name):
                        binding = symbols.imports.get(owner.id)
                        if binding is not None:
                            rebound.add((binding.module, node.attr))

        for symbols in self.modules.values():
            counts: dict[str, int] = {}
            literals: dict[str, object] = {}
            for node in ast.walk(symbols.module.tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    counts[node.id] = counts.get(node.id, 0) + 1
            for node in symbols.module.tree.body:
                target = value = None
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    target, value = node.target.id, node.value
                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    target, value = node.targets[0].id, node.value
                if target is None or value is None:
                    continue
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if not isinstance(literal, (int, float, complex, str, bytes, bool, type(None))):
                    continue
                literals[target] = literal
            for name, literal in literals.items():
                if counts.get(name, 0) != 1 or (symbols.name, name) in rebound:
                    continue
                symbols.constant_globals[name] = literal
                existing = symbols.global_facts.get(name, Facts())
                facts = existing.with_(constant=literal, has_constant=True)
                if isinstance(literal, int) and not isinstance(literal, bool):
                    facts = facts.with_(int_range=IntRange(literal, literal))
                symbols.global_facts[name] = facts
                symbols.globals.setdefault(name, T.type_of_constant(literal))

    def resolver(self, symbols: ModuleSymbols) -> NameResolver:
        return NameResolver(symbols, self)

    def annotation_resolver(self, symbols: ModuleSymbols) -> AnnotationResolver:
        return AnnotationResolver(
            self.resolver(symbols), symbols.path, self.diagnostics, strict=self.strict
        )

    def register_external_type(self, qualname: str, display: str | None = None) -> None:
        self.external_types[qualname] = display or qualname.rpartition(".")[2]

    def _collect_imports(self, symbols: ModuleSymbols) -> None:
        for edge in symbols.module.imports:
            if edge.star:
                continue
            if not edge.is_from:
                for name, asname in edge.names:
                    local = asname or name.partition(".")[0]
                    target = name if asname else name.partition(".")[0]
                    symbols.imports[local] = ImportBinding(local, target, None, edge.external)
            else:
                for name, asname in edge.names:
                    local = asname or name
                    submodule = f"{edge.target}.{name}" if edge.target else name
                    if submodule in self.graph.modules:
                        symbols.imports[local] = ImportBinding(local, submodule, None, False)
                    else:
                        symbols.imports[local] = ImportBinding(
                            local, edge.target, name, edge.external
                        )

    def _collect_type_aliases(self, symbols: ModuleSymbols) -> None:
        """Record module-level type aliases so annotations can name them.

        `X = Annotated[...]` and the 3.12 `type X = ...` form are both ways of
        giving a type a name, and an annotation that uses one has to resolve
        through it (spec 8.1).
        """
        for node in symbols.module.tree.body:
            if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
                symbols.type_aliases[node.name.id] = node.value
                continue
            target = value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and _is_type_alias_annotation(node.annotation)
            ):
                target, value = node.target, node.value
            if not isinstance(target, ast.Name) or value is None:
                continue
            if self._is_type_expression(symbols, value):
                symbols.type_aliases[target.id] = value

    def _is_type_expression(self, symbols: ModuleSymbols, node: ast.expr) -> bool:
        """Does this expression name a type rather than compute a value?"""
        resolver = self.resolver(symbols)
        root = node
        while isinstance(root, ast.Subscript):
            root = root.value
        if isinstance(root, ast.BinOp) and isinstance(root.op, ast.BitOr):
            return self._is_type_expression(symbols, root.left) and self._is_type_expression(
                symbols, root.right
            )
        if isinstance(root, ast.Constant) and root.value is None:
            return True
        if not isinstance(root, (ast.Name, ast.Attribute)):
            return False
        qualname = resolver.canonical(root)
        if qualname is None:
            return False
        return (
            qualname.startswith(("typing.", "ppy.", "collections.abc."))
            or qualname in self.classes
            or qualname in self.external_types
            or (qualname.startswith("builtins.") and qualname.rpartition(".")[2] in T.BUILTIN_MRO)
        )

    def _collect_declarations(self, symbols: ModuleSymbols) -> None:
        for node in symbols.module.tree.body:
            if isinstance(node, ast.ClassDef):
                self._declare_class(symbols, node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._declare_function(symbols, node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        try:
                            values = ast.literal_eval(node.value)
                        except (ValueError, SyntaxError):
                            continue
                        if isinstance(values, (list, tuple)):
                            symbols.all_exports = tuple(str(v) for v in values)

    def _declare_class(self, symbols: ModuleSymbols, node: ast.ClassDef) -> ClassInfo:
        resolver = self.resolver(symbols)
        qualname = f"{symbols.name}.{node.name}"
        decorators = tuple(
            ast.unparse(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list
        )
        info = ClassInfo(
            name=node.name,
            qualname=qualname,
            module=symbols.name,
            node=node,
            path=symbols.path,
            base_names=tuple(ast.unparse(b) for b in node.bases),
            decorators=decorators,
            directives=directives_from(node.decorator_list, resolver),
            is_dataclass=any("dataclass" in d for d in decorators),
        )
        symbols.classes[node.name] = info
        self.classes[qualname] = info
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._declare_function(symbols, child, owner=info)
                info.methods[child.name] = method
        return info

    def _declare_function(
        self,
        symbols: ModuleSymbols,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: ClassInfo | None = None,
    ) -> FunctionInfo:
        resolver = self.resolver(symbols)
        qualname = f"{owner.qualname}.{node.name}" if owner else f"{symbols.name}.{node.name}"
        decorators = tuple(
            ast.unparse(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list
        )
        info = FunctionInfo(
            name=node.name,
            qualname=qualname,
            module=symbols.name,
            node=node,
            path=symbols.path,
            directives=directives_from(node.decorator_list, resolver),
            decorators=decorators,
            is_method=owner is not None,
            is_static="staticmethod" in decorators,
            is_classmethod="classmethod" in decorators,
            is_property="property" in decorators,
            owner=owner.qualname if owner else None,
        )
        info.dynamic = info.directive("dynamic") is not None
        if owner is None:
            symbols.functions[node.name] = info
        self.functions[qualname] = info
        return info

    def _compute_mro(self, symbols: ModuleSymbols) -> None:
        for info in symbols.classes.values():
            info.mro = self._linearize(info, symbols, set())
            self._classify(info)

    def _linearize(
        self, info: ClassInfo, symbols: ModuleSymbols, seen: set[str]
    ) -> tuple[str, ...]:
        if info.qualname in seen:
            return (info.qualname, "object")
        seen = seen | {info.qualname}
        order = [info.qualname]
        resolver = self.resolver(symbols)
        for base in info.node.bases:
            qualname = resolver.canonical(base)
            if qualname is None:
                continue
            parent = self.classes.get(qualname)
            if parent is not None:
                parent_symbols = self.modules.get(parent.module)
                inherited = parent.mro or (
                    self._linearize(parent, parent_symbols, seen)
                    if parent_symbols
                    else (parent.qualname, "object")
                )
                for entry in inherited:
                    if entry not in order and entry != "object":
                        order.append(entry)
            else:
                short = qualname.rpartition(".")[2]
                builtin = T.BUILTIN_MRO.get(short)
                if builtin:
                    for entry in builtin:
                        if entry not in order and entry != "object":
                            order.append(entry)
                elif qualname not in order:
                    order.append(qualname)
        order.append("object")
        return tuple(order)

    def _classify(self, info: ClassInfo) -> None:
        joined = " ".join(info.base_names)
        info.is_enum = any(part in joined for part in ("Enum", "IntEnum", "StrEnum", "Flag"))
        info.is_pydantic = "BaseModel" in joined or any("BaseModel" in e for e in info.mro)
        info.is_protocol = "Protocol" in joined

    def _resolve_signatures(self, symbols: ModuleSymbols) -> None:
        annotations = self.annotation_resolver(symbols)
        for info in symbols.classes.values():
            self._resolve_class_fields(symbols, info, annotations)
        for info in list(symbols.functions.values()):
            self._resolve_function(symbols, info, annotations)
        for info in symbols.classes.values():
            for method in info.methods.values():
                self._resolve_function(symbols, method, annotations, owner=info)
        self._resolve_module_globals(symbols, annotations)

    def _resolve_class_fields(
        self, symbols: ModuleSymbols, info: ClassInfo, annotations: AnnotationResolver
    ) -> None:
        for child in info.node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                resolved = annotations.resolve(child.annotation)
                info.fields[child.target.id] = resolved.type
                facts = resolved.facts
                if child.value is not None:
                    # `count: int = Field(ge=0)` states the same bound that
                    # `Annotated[int, Field(ge=0)]` does, and is the commoner
                    # way to write it.
                    facts = _with_field_bounds(facts, child.value)
                info.field_facts[child.target.id] = facts
                if _is_class_var(child.annotation):
                    info.class_vars.add(child.target.id)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        try:
                            values = ast.literal_eval(child.value)
                        except (ValueError, SyntaxError):
                            continue
                        if isinstance(values, (list, tuple)):
                            info.slots = tuple(str(v) for v in values)
        self._collect_self_assignments(info, annotations)

    def _collect_self_assignments(self, info: ClassInfo, annotations: AnnotationResolver) -> None:
        init = info.methods.get("__init__")
        if init is None:
            return
        for node in ast.walk(init.node):
            if isinstance(node, ast.AnnAssign) and _is_self_attribute(node.target):
                attr = node.target.attr  # type: ignore[union-attr]
                resolved = annotations.resolve(node.annotation)
                info.fields.setdefault(attr, resolved.type)
                info.field_facts.setdefault(attr, resolved.facts)

    def _resolve_function(
        self,
        symbols: ModuleSymbols,
        info: FunctionInfo,
        annotations: AnnotationResolver,
        owner: ClassInfo | None = None,
    ) -> None:
        if info.params:
            return
        args = info.node.args
        entries: list[tuple[ast.arg, str, ast.expr | None]] = [
            (arg, "positional_only", None) for arg in args.posonlyargs
        ]
        defaults_start = len(args.posonlyargs) + len(args.args) - len(args.defaults)
        for index, arg in enumerate(args.args):
            position = len(args.posonlyargs) + index
            default = (
                args.defaults[position - defaults_start] if position >= defaults_start else None
            )
            entries.append((arg, "positional_or_keyword", default))
        if args.vararg:
            entries.append((args.vararg, "var_positional", None))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
            entries.append((arg, "keyword_only", default))
        if args.kwarg:
            entries.append((args.kwarg, "var_keyword", None))

        for index, (arg, kind, default) in enumerate(entries):
            if arg.annotation is None:
                implicit = self._implicit_param(info, arg, index, owner, kind)
                info.params.append(implicit)
                continue
            resolved = annotations.resolve(arg.annotation)
            declared_type = resolved.type
            if kind == "var_positional":
                declared_type = T.Tuple_((declared_type,), homogeneous=True)
            elif kind == "var_keyword":
                declared_type = T.dict_of(T.STR, declared_type)
            info.params.append(
                ParamInfo(
                    name=arg.arg,
                    type=declared_type,
                    facts=resolved.facts,
                    has_default=default is not None,
                    default=default,
                    kind=kind,
                    annotated=True,
                )
            )
        if info.node.returns is not None:
            resolved = annotations.resolve(info.node.returns)
            info.ret = resolved.type
            info.ret_facts = resolved.facts
            info.ret_annotated = True

    def _implicit_param(
        self,
        info: FunctionInfo,
        arg: ast.arg,
        index: int,
        owner: ClassInfo | None,
        kind: str,
    ) -> ParamInfo:
        if index == 0 and owner is not None and not info.is_static:
            if info.is_classmethod:
                return ParamInfo(
                    arg.arg,
                    T.ClassObject(owner.qualname, owner.instance()),
                    kind=kind,
                    annotated=True,
                )
            return ParamInfo(arg.arg, owner.instance(), kind=kind, annotated=True)
        return ParamInfo(arg.arg, T.UNKNOWN, kind=kind, annotated=False)

    def _resolve_module_globals(
        self, symbols: ModuleSymbols, annotations: AnnotationResolver
    ) -> None:
        for node in symbols.module.tree.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                resolved = annotations.resolve(node.annotation)
                symbols.globals[node.target.id] = resolved.type
                symbols.global_facts[node.target.id] = resolved.facts


def _is_type_alias_annotation(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    return text.endswith("TypeAlias")


#: Keyword names a constraint helper uses for an inclusive or exclusive bound.
_LOWER_BOUNDS = {"ge": 0, "gt": 1}
_UPPER_BOUNDS = {"le": 0, "lt": -1}


def _integer_literal(node: ast.expr) -> int | None:
    """An integer written in the source, negative sign included."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _integer_literal(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return None if isinstance(node.value, bool) else node.value
    return None


def _with_field_bounds(facts: Facts, value: ast.expr) -> Facts:
    """Read `Field(ge=..., lt=...)` as the integer range it describes."""
    if not isinstance(value, ast.Call):
        return facts
    name = (
        value.func.attr if isinstance(value.func, ast.Attribute) else getattr(value.func, "id", "")
    )
    if name not in {"Field", "conint", "condecimal"}:
        return facts
    low: int | None = None
    high: int | None = None
    for keyword in value.keywords:
        if keyword.arg is None:
            continue
        bound = _integer_literal(keyword.value)
        if bound is None:
            continue
        if keyword.arg in _LOWER_BOUNDS:
            low = bound + _LOWER_BOUNDS[keyword.arg]
        elif keyword.arg in _UPPER_BOUNDS:
            high = bound + _UPPER_BOUNDS[keyword.arg]
    if low is None and high is None:
        return facts
    return facts.with_(int_range=IntRange(low, high))


def _is_class_var(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    return text.startswith("ClassVar") or ".ClassVar" in text


def _is_self_attribute(target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    )
