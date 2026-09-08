"""The IR linker: one program from a project's module IRs (spec 10, 11).

Every module lowers on its own to an `IRModule` whose calls into another
module go through a declaration -- a function with no body, carrying the
callee's symbol. Linking puts the modules together: a definition answers
the declarations of its symbol, which go away; two definitions of one
symbol are one when both are the same generic instance (a specialization
each module made for itself) and an error otherwise; a private symbol --
a string constant, a helper -- that two modules both spell is renamed after
its module, and every reference follows. The dialects a module needs the
program needs, at the newest version any asked for; the libraries a module
links the program links. What is left unresolved is reported by name, so
the driver can keep its callers on CPython the way the frontend keeps a
caller of a function that did not lower.

The linked module is what whole-program optimization runs on and what
`ppy emit ir --linked` prints; a backend sees one module either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Global, IRFunction, IRModule, Operation, Symbol, SymbolRef

__all__ = ["LinkError", "Linked", "link", "unresolved"]


class LinkError(ValueError):
    """Two modules define one public symbol differently, or a module names a dialect twice."""


@dataclass(slots=True)
class Linked:
    module: IRModule
    #: Symbols renamed on the way in: (module, old name) -> new name.
    renamed: dict[tuple[str, str], str] = field(default_factory=dict)
    #: Declarations no module defines, by symbol name.
    unresolved: tuple[str, ...] = ()
    #: Generic instances two or more modules had made; one copy was kept.
    shared: tuple[str, ...] = ()


def link(modules: list[IRModule], name: str = "program") -> Linked:
    """Link `modules`, in order, into one program named `name`."""
    program = IRModule(name)
    program.dialects = {}
    renamed: dict[tuple[str, str], str] = {}
    shared: list[str] = []
    definitions: dict[str, IRFunction] = {}
    pending: dict[str, list[IRFunction]] = {}
    libraries: list[str] = []
    for module in modules:
        for dialect, version in module.dialects.items():
            program.require(dialect, version)
        for library in module.attributes.get("ppy.libraries", ()):  # type: ignore[union-attr]
            if library not in libraries:
                libraries.append(str(library))
        for key, value in module.attributes.items():
            if key != "ppy.libraries" and key not in program.attributes:
                program.attributes[key] = value
        _link_globals(program, module, renamed)
        for function in module.functions.values():
            _link_function(program, module, function, definitions, pending, renamed, shared)
    for symbol, declarations in list(pending.items()):
        definition = definitions.get(symbol)
        if definition is None:
            # One declaration stands for all: the symbol is external to the program.
            first = declarations[0]
            if symbol not in program.functions:
                _adopt(program, first, symbol)
        # Every call to the symbol already names it by symbol; nothing to rewrite.
    if libraries:
        program.attributes["ppy.libraries"] = tuple(libraries)
    for function in program.functions.values():
        _retarget(function, renamed, modules)
    return Linked(program, renamed, unresolved(program), tuple(shared))


def unresolved(program: IRModule) -> tuple[str, ...]:
    """The functions `program` declares and no one defines, that its code calls."""
    called: set[str] = set()
    for function in program.functions.values():
        for op in function.operations():
            callee = op.attributes.get("callee")
            if isinstance(callee, SymbolRef):
                called.add(callee.name)
    return tuple(
        sorted(name for name, f in program.functions.items() if f.is_declaration and name in called)
    )


def _adopt(program: IRModule, function: IRFunction, name: str) -> None:
    """Move `function` into `program` under `name`."""
    function.symbol = Symbol(name, function.symbol.visibility)
    function.module = program
    program.functions[name] = function
    for block in function.body.blocks:
        block.region = function.body


def _link_function(
    program: IRModule,
    module: IRModule,
    function: IRFunction,
    definitions: dict[str, IRFunction],
    pending: dict[str, list[IRFunction]],
    renamed: dict[tuple[str, str], str],
    shared: list[str],
) -> None:
    name = function.name
    if function.is_declaration:
        pending.setdefault(name, []).append(function)
        return
    existing = definitions.get(name)
    if existing is not None:
        if function.attributes.get("ppy.generic") and existing.attributes.get("ppy.generic"):
            # Both modules instantiated the same generic the same way; one serves.
            if name not in shared:
                shared.append(name)
            return
        if function.symbol.visibility == "private":  # type: ignore[attr-defined]
            fresh = _fresh(program, f"{_stem(module.name)}__{name}")
            renamed[(module.name, name)] = fresh
            _adopt(program, function, fresh)
            return
        raise LinkError(f"@{name} is defined by two modules; one of them is {module.name}")
    if name in program.functions:
        # A declaration got here first; the definition takes its place.
        del program.functions[name]
    definitions[name] = function
    _adopt(program, function, name)


def _link_globals(program: IRModule, module: IRModule, renamed: dict[tuple[str, str], str]) -> None:
    for name, item in module.globals.items():
        target = name
        if name in program.globals or name in program.functions:
            if item.symbol.visibility != "private":
                raise LinkError(f"@{name} is defined by two modules; one of them is {module.name}")
            target = _fresh(program, f"{_stem(module.name)}__{name}")
            renamed[(module.name, name)] = target
        program.globals[target] = Global(
            Symbol(target, item.symbol.visibility),
            item.type,
            item.value,
            item.constant,
            item.location,
        )


def _retarget(
    function: IRFunction, renamed: dict[tuple[str, str], str], modules: list[IRModule]
) -> None:
    """Calls and string references follow a renamed private symbol."""
    if not renamed:
        return
    origin = _origin_of(function, modules)
    for op in function.operations():
        _retarget_op(op, origin, renamed)


def _retarget_op(op: Operation, origin: str | None, renamed: dict[tuple[str, str], str]) -> None:
    callee = op.attributes.get("callee")
    if isinstance(callee, SymbolRef) and origin is not None:
        fresh = renamed.get((origin, callee.name))
        if fresh is not None:
            op.attributes["callee"] = SymbolRef(fresh)
    symbol = op.attributes.get("symbol")
    if isinstance(symbol, str) and origin is not None:
        fresh = renamed.get((origin, symbol))
        if fresh is not None:
            op.attributes["symbol"] = fresh
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                _retarget_op(inner, origin, renamed)


def _origin_of(function: IRFunction, modules: list[IRModule]) -> str | None:
    qualname = function.attributes.get("ppy.qualname")
    if isinstance(qualname, str):
        for module in modules:
            if qualname == module.name or qualname.startswith(module.name + "."):
                return module.name
    origin = function.attributes.get("ppy.module")
    return str(origin) if isinstance(origin, str) else None


def _fresh(program: IRModule, wanted: str) -> str:
    candidate = wanted
    counter = 1
    while candidate in program.functions or candidate in program.globals:
        counter += 1
        candidate = f"{wanted}{counter}"
    return candidate


def _stem(module_name: str) -> str:
    return module_name.replace(".", "_")
