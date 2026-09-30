"""`str.format` and `%` of a string literal, as the f-string they mean.

`"{} has {:>4}".format(name, n)` and `"%s has %4d" % (name, n)` format each
value as `format(value, spec)` does, which is what an f-string field does;
rewritten into one, they lower as f-strings lower. Only a literal format
string is rewritten, and only where the rewrite keeps what CPython does: the
arguments are evaluated once each, in order, and every field is one native
code formats. Anything else stays in Python.
"""

from __future__ import annotations

import ast
import re
import string
from collections.abc import Callable

from ..backend.llvm.lowering import Unsupported

#: A `%` conversion: flags, width, precision, type.
_PERCENT = re.compile(r"%(?P<flags>[-+ 0#]*)(?P<width>\d*)(?:\.(?P<precision>\d+))?(?P<type>.)")
_NUMBER_TYPES = frozenset("dixXofFeEgG")


def _simple(node: ast.expr) -> bool:
    """An argument that may be read more than once, or out of order."""
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _simple(node.value)
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    )


def _field(value: ast.expr, conversion: str | None, spec: str) -> ast.FormattedValue:
    format_spec = ast.JoinedStr([ast.Constant(spec)]) if spec else None
    return ast.FormattedValue(
        value=value, conversion=ord(conversion) if conversion else -1, format_spec=format_spec
    )


def _joined(parts: list[ast.expr], where: ast.expr) -> ast.JoinedStr:
    joined = ast.JoinedStr([part for part in parts if not _empty(part)])
    ast.copy_location(joined, where)
    for node in ast.walk(joined):
        if not hasattr(node, "lineno"):
            ast.copy_location(node, where)
    return joined


def _empty(part: ast.expr) -> bool:
    return isinstance(part, ast.Constant) and part.value == ""


def _in_order(used: list[ast.expr], given: list[ast.expr]) -> bool:
    """Whether each argument is used once, in the order it was given."""
    return len(used) == len(given) and all(a is b for a, b in zip(used, given, strict=True))


def format_call(node: ast.Call) -> ast.JoinedStr | None:
    """`"...".format(...)` as an f-string; None where `node` is not one."""
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "format"
        and isinstance(func.value, ast.Constant)
        and isinstance(func.value.value, str)
    ):
        return None
    if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
        keyword.arg is None for keyword in node.keywords
    ):
        raise Unsupported("`str.format` with `*` or `**` stays in Python")
    named = {keyword.arg: keyword.value for keyword in node.keywords}
    try:
        pieces = list(string.Formatter().parse(func.value.value))
    except ValueError as exc:
        raise Unsupported(f"`str.format` raises here: {exc}") from exc
    parts: list[ast.expr] = []
    used: list[ast.expr] = []
    automatic = 0
    numbered = False
    for literal, name, spec, conversion in pieces:
        if literal:
            parts.append(ast.Constant(literal))
        if name is None:
            continue
        if spec and ("{" in spec or "}" in spec):
            raise Unsupported("a `str.format` spec with fields stays in Python")
        if name == "":
            if numbered:
                raise Unsupported("`str.format` mixes numbered and automatic fields")
            index: int | str = automatic
            automatic += 1
        elif name.isdigit():
            if automatic:
                raise Unsupported("`str.format` mixes numbered and automatic fields")
            numbered = True
            index = int(name)
        elif name.isidentifier():
            index = name
        else:
            raise Unsupported("a `str.format` field reaching into its value stays in Python")
        if isinstance(index, int):
            if index >= len(node.args):
                raise Unsupported("`str.format` names an argument it was not given")
            value = node.args[index]
        else:
            if index not in named:
                raise Unsupported(f"`str.format` names `{index}`, which it was not given")
            value = named[index]
        used.append(value)
        parts.append(_field(value, conversion, spec or ""))
    given = list(node.args) + [keyword.value for keyword in node.keywords]
    if not _in_order(used, given) and not all(_simple(arg) for arg in given):
        raise Unsupported("`str.format` reads its arguments out of order, which Python does once")
    return _joined(parts, node)


def percent(node: ast.BinOp, kind_of: Callable[[ast.expr], str]) -> ast.JoinedStr | None:
    """`"..." % values` as an f-string; None where `node` is not one. `kind_of`
    names a value's type: "int", "float", "bool", "str", or "" for another."""
    if not (
        isinstance(node.op, ast.Mod)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
    ):
        return None
    text = node.left.value
    if isinstance(node.right, ast.Tuple):
        values = list(node.right.elts)
    elif kind_of(node.right) in {"int", "float", "bool", "str"}:
        values = [node.right]
    else:
        raise Unsupported("`%` of a value that may be a tuple or a mapping stays in Python")
    parts: list[ast.expr] = []
    taken = 0
    at = 0
    while at < len(text):
        found = text.find("%", at)
        if found < 0:
            parts.append(ast.Constant(text[at:]))
            break
        parts.append(ast.Constant(text[at:found]))
        match = _PERCENT.match(text, found)
        if match is None:
            raise Unsupported("an incomplete `%` conversion stays in Python")
        at = match.end()
        kind = match.group("type")
        if kind == "%" and match.group(0) == "%%":
            parts.append(ast.Constant("%"))
            continue
        if taken >= len(values):
            raise Unsupported("`%` has fewer values than conversions")
        value = values[taken]
        taken += 1
        parts.append(_conversion(match, value, kind_of(value)))
    if taken != len(values):
        raise Unsupported("`%` has more values than conversions")
    return _joined(parts, node)


def _conversion(match: re.Match[str], value: ast.expr, kind: str) -> ast.FormattedValue:
    flags, width, code = (str(match.group(part)) for part in ("flags", "width", "type"))
    given = match.group("precision")
    precision = None if given is None else str(given)
    left = "-" in flags
    if code in {"s", "r", "a"}:
        if "0" in flags and not left:
            raise Unsupported("`%0s` stays in Python")
        spec = ("<" if left else ">") + width if width else ""
        if precision is not None:
            spec += f".{precision}"
        return _field(value, code, spec)
    if code == "c":
        one = isinstance(value, ast.Constant) and isinstance(value.value, str)
        if not one or len(value.value) != 1 or width or precision is not None:
            raise Unsupported("`%c` of this value stays in Python")
        return _field(value, None, "")
    if code not in _NUMBER_TYPES:
        raise Unsupported(f"`%{code}` stays in Python")
    if code in "dixXo" and kind not in {"int", "bool"}:
        raise Unsupported(f"`%{code}` of a `{kind or 'value'}` stays in Python")
    if code in "fFeEgG" and kind not in {"int", "float"}:
        raise Unsupported(f"`%{code}` of a `{kind or 'value'}` stays in Python")
    if code == "i":
        code = "d"
    if kind == "bool" and code in "xXo":
        raise Unsupported(f"`%{code}` of a `bool` stays in Python")
    spec = ""
    if left and width:
        spec += "<"
    if "+" in flags:
        spec += "+"
    elif " " in flags:
        spec += " "
    if "#" in flags:
        if code in "fFeEgG":
            raise Unsupported(f"`%#{code}` stays in Python")
        spec += "#"
    if "0" in flags and not left and width:
        spec += "0"
    spec += width
    if precision is not None:
        if code in "dxXo":
            raise Unsupported(f"`%.{precision}{code}` stays in Python")
        spec += f".{precision}"
    spec += code
    return _field(value, None, spec)
