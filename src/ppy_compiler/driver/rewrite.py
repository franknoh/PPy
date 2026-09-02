"""Rewriting one module's concrete syntax tree (spec 9.4).

The planning step decides what a module should say; this writes it, through
a concrete syntax tree so comments, spacing, and string quoting survive.
Everything here is a transformation of source into source: nothing decides
what the source should mean.
"""

from __future__ import annotations

import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

from .formatting import normalize_source
from .plan import ConversionPlan, mentions

__all__ = ["convert_source"]

#: Names PEP 585 moved out of `typing`.
_COLLECTIONS_ABC = frozenset(
    {
        "Sequence",
        "Iterable",
        "Iterator",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "Callable",
        "Generator",
        "Coroutine",
        "Awaitable",
        "AsyncIterable",
        "AsyncIterator",
        "Container",
        "Collection",
        "Set",
    }
)


class _Annotator(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, plan: ConversionPlan) -> None:
        self.plan = plan
        self._function_lines: list[int] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._function_lines.append(self.get_metadata(PositionProvider, node).start.line)
        return True

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        line = self._function_lines.pop()
        returns = self.plan.returns.get(line)
        if returns is not None and updated.returns is None:
            text = _quote_if_forward(returns, line, self.plan.forward)
            updated = updated.with_changes(returns=cst.Annotation(cst.parse_expression(text)))
        wanted = self.plan.decorators.get(line, ())
        if wanted:
            present = {_dotted(d.decorator) for d in updated.decorators}
            added = [
                cst.Decorator(decorator=cst.parse_expression(text))
                for text in wanted
                if text.partition("(")[0] not in present
            ]
            if added:
                updated = updated.with_changes(decorators=[*added, *updated.decorators])
        return updated

    def leave_Param(self, original: cst.Param, updated: cst.Param) -> cst.Param:
        if updated.annotation is not None or not self._function_lines:
            return updated
        line = self._function_lines[-1]
        annotation = self.plan.params.get((line, updated.name.value))
        if annotation is None:
            return updated
        text = _quote_if_forward(annotation, line, self.plan.forward)
        return updated.with_changes(annotation=cst.Annotation(cst.parse_expression(text)))

    def leave_SimpleStatementLine(
        self, original: cst.SimpleStatementLine, updated: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        if len(updated.body) != 1:
            return updated
        statement = updated.body[0]
        if not isinstance(statement, cst.Assign) or len(statement.targets) != 1:
            return updated
        target = statement.targets[0].target
        line = self.get_metadata(PositionProvider, original).start.line

        code = self.plan.buffers.get(line)
        if code is not None and isinstance(target, cst.Name):
            wrapped = cst.parse_expression(
                f'array.array("{code}", {cst.Module(body=[]).code_for_node(statement.value)})'
            )
            return updated.with_changes(
                body=[cst.Assign(targets=list(statement.targets), value=wrapped)]
            )

        if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
            annotation = self.plan.fields.get((line, target.attr.value))
            if annotation is None:
                return updated
            text = _quote_if_forward(annotation, line, self.plan.forward)
            return updated.with_changes(
                body=[
                    cst.AnnAssign(
                        target=target,
                        annotation=cst.Annotation(cst.parse_expression(text)),
                        value=statement.value,
                    )
                ]
            )

        # Both module-level bindings and the locals that start empty are
        # planned, so being inside a function is no longer a reason to skip.
        if not isinstance(target, cst.Name):
            return updated
        annotation = self.plan.assignments.get((line, target.value))
        if annotation is None:
            return updated
        return updated.with_changes(
            body=[
                cst.AnnAssign(
                    target=target,
                    annotation=cst.Annotation(cst.parse_expression(annotation)),
                    value=statement.value,
                )
            ]
        )


#: What `T(input())` reads, by the name of `T`.
_READ_AS = {"int": "int", "float": "float", "str": "str"}


def _is_input_call(node: cst.BaseExpression) -> cst.Call | None:
    """`input()` or `input(prompt)`, or None for anything else."""
    if isinstance(node, cst.Call) and isinstance(node.func, cst.Name):
        return node if node.func.value == "input" else None
    return None


def _typed_read(spec: str, arguments: list[cst.Arg]) -> cst.BaseExpression:
    """`ppy.input[spec](...)`, carrying the prompt the original had."""
    prompt = "".join(cst.Module(body=()).code_for_node(a.value) for a in arguments)
    return cst.parse_expression(f"ppy.input[{spec}]({prompt})")


class _TypedReads(cst.CSTTransformer):
    """Rewrite the `input()` idioms into `ppy.input[T]`.

    `int(input())` is what a submission writes and what it means is `read an
    integer`; saying so lets the value be read without a Python object per
    field. The originals are matched, not the rewritten children, so the
    inner `input()` of `int(input())` is not rewritten twice.
    """

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        wrapper = isinstance(original.func, cst.Name) and original.func.value in _READ_AS
        if wrapper and len(original.args) == 1:
            inner = _is_input_call(original.args[0].value)
            if inner is not None:
                return _typed_read(_READ_AS[original.func.value], list(inner.args))
        if _is_input_call(original) is not None:
            return _typed_read("str", list(original.args))
        return updated

    def leave_For(self, original: cst.For, updated: cst.For) -> cst.BaseStatement:
        bulk = _filling_loop(original)
        return bulk if bulk is not None else updated

    def leave_Assign(self, original: cst.Assign, updated: cst.Assign) -> cst.Assign:
        """`a, b = map(int, input().split())` reads a tuple of that width."""
        if len(original.targets) != 1:
            return updated
        target = original.targets[0].target
        if not isinstance(target, (cst.Tuple, cst.List)):
            return updated
        spec = _mapped_read(original.value, len(target.elements))
        if spec is None:
            return updated
        return updated.with_changes(value=spec)


def _filling_loop(node: cst.For) -> cst.BaseStatement | None:
    """`for i in range(n): xs[i] = int(input())` is one bulk read.

    Filling a buffer one value at a time pays the boundary per element; the
    same values land in the same slots in one call. The slice keeps it exact:
    the loop wrote `n` entries, so the read fills `n` entries.
    """
    index = node.target
    if not isinstance(index, cst.Name) or node.orelse is not None:
        return None
    iterated = node.iter
    if not isinstance(iterated, cst.Call) or not isinstance(iterated.func, cst.Name):
        return None
    if iterated.func.value != "range" or len(iterated.args) != 1:
        return None
    body = node.body
    if not isinstance(body, cst.IndentedBlock) or len(body.body) != 1:
        return None
    line = body.body[0]
    if not isinstance(line, cst.SimpleStatementLine) or len(line.body) != 1:
        return None
    assign = line.body[0]
    if not isinstance(assign, cst.Assign) or len(assign.targets) != 1:
        return None
    written = assign.targets[0].target
    if not isinstance(written, cst.Subscript) or len(written.slice) != 1:
        return None
    subscript = written.slice[0].slice
    if not isinstance(subscript, cst.Index) or not isinstance(subscript.value, cst.Name):
        return None
    if subscript.value.value != index.value:
        return None
    read = assign.value
    if not isinstance(read, cst.Call) or not isinstance(read.func, cst.Name):
        return None
    if read.func.value != "int" or len(read.args) != 1:
        return None
    if _is_input_call(read.args[0].value) is None:
        return None

    code = cst.Module(body=())
    buffer = code.code_for_node(written.value)
    bound = code.code_for_node(iterated.args[0].value)
    filled = buffer if bound == f"len({buffer})" else f"memoryview({buffer})[:{bound}]"
    return cst.parse_statement(f"ppy.read_ints({filled})")


def _mapped_read(value: cst.BaseExpression, width: int) -> cst.BaseExpression | None:
    """`map(T, input().split())` over a fixed number of targets."""
    if not isinstance(value, cst.Call) or not isinstance(value.func, cst.Name):
        return None
    if value.func.value != "map" or len(value.args) != 2:
        return None
    caster = value.args[0].value
    if not isinstance(caster, cst.Name) or caster.value not in _READ_AS:
        return None
    split = value.args[1].value
    if not isinstance(split, cst.Call) or not isinstance(split.func, cst.Attribute):
        return None
    if split.func.attr.value != "split" or split.args:
        return None
    if _is_input_call(split.func.value) is None:
        return None
    element = _READ_AS[caster.value]
    return cst.parse_expression(f"ppy.input[tuple[{', '.join([element] * width)}]]()")


def convert_source(source: str, plan: ConversionPlan) -> str:
    """Rewrite `source` through a concrete syntax tree, preserving trivia.

    Inserting annotations and imports disturbs the blank lines around what it
    touches, so the result is normalized before it is written. The built-in
    normalizer is used rather than an installed formatter, because the output
    has to be the same on every machine.
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    annotated = wrapper.visit(_Annotator(plan))
    if plan.rewrite_input:
        annotated = annotated.visit(_TypedReads())
    imported = _insert_imports(annotated, plan)
    # A quoted annotation only exists because the class was not bound yet.
    # Moving the class above its first use removes the reason for the quotes.
    reordered = _unquote_resolved(_hoist_classes(imported, plan.hoistable, plan.reorder_safe))
    return normalize_source(reordered.code, frozenset(plan.local_imports))


def _insert_imports(module: cst.Module, plan: ConversionPlan) -> cst.Module:
    """Insert `import ppy` after the docstring and `__future__` imports."""
    existing = _existing_imports(module)
    # PEP 8 groups imports, and pylint checks the grouping, so a standard
    # library addition goes at the top of the block and the rest at the end.
    standard: list[cst.SimpleStatementLine] = []
    trailing: list[cst.SimpleStatementLine] = []

    if plan.needs_array and "array" not in existing:
        standard.append(cst.parse_statement("import array"))
    wanted = plan.typing_imports - existing
    # PEP 585 moved the container protocols to `collections.abc`; importing
    # them from `typing` is deprecated.
    abc_names = sorted(wanted & _COLLECTIONS_ABC)
    if abc_names:
        standard.append(cst.parse_statement(f"from collections.abc import {', '.join(abc_names)}"))
    typing_names = sorted(wanted - _COLLECTIONS_ABC)
    if typing_names:
        standard.append(cst.parse_statement(f"from typing import {', '.join(typing_names)}"))
    if plan.needs_ppy and "ppy" not in existing:
        trailing.append(cst.parse_statement("import ppy"))
    ppy_names = sorted(plan.ppy_imports - existing)
    if ppy_names:
        trailing.append(cst.parse_statement(f"from ppy import {', '.join(ppy_names)}"))

    if not standard and not trailing:
        return module
    body = list(module.body)
    head = _insert_index(module)
    tail = _import_block_end(module, head, plan.local_imports)
    return module.with_changes(
        body=[*body[:head], *standard, *body[head:tail], *trailing, *body[tail:]]
    )


def _import_block_end(module: cst.Module, start: int, local: set[str]) -> int:
    """Where a `ppy` import belongs: after the third-party ones, before the local.

    `import ppy` installs the loader that a sibling `.ppy` module needs, so it
    cannot follow one. That is also where PEP 8 puts it, third-party coming
    before first-party.
    """
    end = start
    for position in range(start, len(module.body)):
        statement = module.body[position]
        if not isinstance(statement, cst.SimpleStatementLine):
            break
        first = statement.body[0]
        if not isinstance(first, (cst.Import, cst.ImportFrom)):
            break
        if _imports_any(first, local):
            break
        end = position + 1
    return end


def _imports_any(statement: cst.BaseSmallStatement, local: set[str]) -> bool:
    """Does this import statement bring in one of the project's own modules?"""
    if isinstance(statement, cst.ImportFrom):
        module = statement.module
        return module is not None and _dotted(module).partition(".")[0] in local
    if isinstance(statement, cst.Import):
        return any(_dotted(alias.name).partition(".")[0] in local for alias in statement.names)
    return False


def _existing_imports(module: cst.Module) -> set[str]:
    found: set[str] = set()
    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if isinstance(small, cst.Import):
                for alias in small.names:
                    found.add(_dotted(alias.name))
            elif isinstance(small, cst.ImportFrom) and small.module is not None:
                found.add(_dotted(small.module))
                if not isinstance(small.names, cst.ImportStar):
                    for alias in small.names:
                        found.add(_dotted(alias.name))
    return found


def _dotted(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted(node.value)}.{node.attr.value}"
    return ""


def _insert_index(module: cst.Module) -> int:
    index = 0
    for position, statement in enumerate(module.body):
        if not isinstance(statement, cst.SimpleStatementLine):
            break
        first = statement.body[0]
        is_docstring = (
            position == 0
            and isinstance(first, cst.Expr)
            and isinstance(first.value, cst.SimpleString)
        )
        is_future = (
            isinstance(first, cst.ImportFrom)
            and _dotted(first.module or cst.Name("")) == "__future__"
        )
        if is_docstring or is_future:
            index = position + 1
            continue
        break
    return index


def _definition_time_names(node: cst.CSTNode) -> set[str]:
    """Names a class needs bound the moment its `class` statement executes.

    The body of a method does not run at class creation, but its decorators,
    defaults, and annotations do, and so do the bases and the class-level
    statements. Anything reachable that way has to already exist.
    """
    found: set[str] = set()

    def walk(target: cst.CSTNode) -> None:
        for child in target.children:
            if isinstance(child, cst.Name):
                found.add(child.value)
            walk(child)

    if isinstance(node, cst.ClassDef):
        for decorator in node.decorators:
            walk(decorator)
        for base in node.bases:
            walk(base)
        for keyword in node.keywords:
            walk(keyword)
        body = node.body.body if isinstance(node.body, cst.IndentedBlock) else []
        for statement in body:
            if isinstance(statement, cst.FunctionDef):
                for decorator in statement.decorators:
                    walk(decorator)
                walk(statement.params)
                if statement.returns is not None:
                    walk(statement.returns)
                continue
            walk(statement)
    else:
        walk(node)
    return found


def _hoist_classes(
    module: cst.Module,
    hoistable: frozenset[str] | None,
    reorder_safe: frozenset[str] | None = None,
) -> cst.Module:
    """Move a class above the definitions that annotate against it.

    A quoted annotation is only needed because the class is not bound yet.
    Moving the class up removes the reason for the quotes. Which classes may
    move at all was decided against the analysis (`hoistable`): a definition
    with observable effects stays put, and the annotation stays quoted.
    """
    body = list(module.body)
    moved = True
    passes = 0
    while moved and passes < len(body):
        moved = False
        passes += 1
        for index, statement in enumerate(body):
            if not isinstance(statement, cst.ClassDef):
                continue
            if hoistable is not None and statement.name.value not in hoistable:
                continue
            target = _earliest_position(body, index, statement, reorder_safe)
            if target is None or target >= index:
                continue
            body.insert(target, body.pop(index))
            moved = True
            break
    return module.with_changes(body=body)


def _earliest_position(
    body: list[cst.BaseStatement],
    index: int,
    statement: cst.ClassDef,
    reorder_safe: frozenset[str] | None = None,
) -> int | None:
    """The first slot this class can occupy without breaking a dependency."""
    needed = _definition_time_names(statement)
    position = index
    while position > 0:
        previous = body[position - 1]
        if not isinstance(previous, (cst.FunctionDef, cst.ClassDef)):
            break
        if isinstance(previous, cst.ClassDef) and previous.name.value in needed:
            break
        if isinstance(previous, cst.FunctionDef) and previous.name.value in needed:
            break
        if reorder_safe is not None and previous.name.value not in reorder_safe:
            # Crossing this definition would run its decorators and defaults
            # with the moved class already bound -- observable.
            break
        position -= 1
    if position == index:
        return None
    # Only worth moving when something ahead of it actually names this class.
    name = statement.name.value
    crossed = body[position:index]
    if not any(name in _annotation_names(other) for other in crossed):
        return None
    return position


def _annotation_names(node: cst.CSTNode) -> set[str]:
    """Every name appearing in an annotation, quoted or not."""
    found: set[str] = set()
    for annotation in m.findall(node, m.Annotation()):
        expression = annotation.annotation
        if isinstance(expression, cst.SimpleString):
            text = expression.raw_value
            try:
                expression = cst.parse_expression(text)
            except cst.ParserSyntaxError:
                continue
        for name in m.findall(expression, m.Name()):
            found.add(name.value)
    return found


def _unquote_resolved(module: cst.Module) -> cst.Module:
    """Drop quotes from an annotation whose names are all bound before it.

    Only a top-level statement can be judged this way: a method annotating
    against its own enclosing class still needs the quotes, because the class
    is not bound until its body has finished executing.
    """
    body = list(module.body)
    defined: set[str] = set()
    rewritten: list[cst.BaseStatement] = []
    for original in body:
        statement = original
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            inner = statement.name.value
            statement = statement.with_changes(
                **_unquoted_signature(
                    statement, defined - {inner} if isinstance(statement, cst.ClassDef) else defined
                )
            )
            defined.add(inner)
        rewritten.append(statement)
    return module.with_changes(body=rewritten)


def _unquoted_signature(statement: cst.BaseStatement, defined: set[str]) -> dict:
    """Replacement fields for a definition whose annotations can lose quotes."""
    if not isinstance(statement, cst.FunctionDef):
        return {}
    changes: dict = {}
    params = statement.params
    updated = [
        param.with_changes(annotation=_unquote(param.annotation, defined))
        for param in params.params
    ]
    if any(new is not old for new, old in zip(updated, params.params, strict=False)):
        changes["params"] = params.with_changes(params=updated)
    returns = _unquote(statement.returns, defined)
    if returns is not statement.returns:
        changes["returns"] = returns
    return changes


def _unquote(annotation, defined: set[str]):  # type: ignore[no-untyped-def]
    if annotation is None or not isinstance(annotation.annotation, cst.SimpleString):
        return annotation
    text = annotation.annotation.raw_value
    try:
        expression = cst.parse_expression(text)
    except cst.ParserSyntaxError:
        return annotation
    names = {name.value for name in m.findall(expression, m.Name())}
    if not names <= defined | _ALWAYS_BOUND:
        return annotation
    return annotation.with_changes(annotation=expression)


#: Names an annotation may use that are never module-level definitions here.
_ALWAYS_BOUND = {
    "int",
    "float",
    "str",
    "bytes",
    "bool",
    "complex",
    "None",
    "object",
    "list",
    "dict",
    "set",
    "tuple",
    "frozenset",
    "type",
    "Optional",
    "Union",
    "Any",
    "Annotated",
    "Literal",
    "Callable",
    "Sequence",
    "Iterable",
    "Mapping",
    "Buffer",
    "ppy",
}


def _quote_if_forward(text: str, line: int, forward: dict[str, int]) -> str:
    """Quote an annotation that names a class not yet defined at `line`.

    Annotations are evaluated eagerly before Python 3.14, so a bare reference
    to the enclosing class would raise at import.
    """
    for name, defined_at in forward.items():
        if line <= defined_at and mentions(text, name):
            return repr(text)
    return text
