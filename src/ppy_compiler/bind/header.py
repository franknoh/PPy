"""`ppy bind header foo.h`: a C header read by Clang, written as PPY bindings.

libclang parses the header -- the real parser, so a macro-heavy or
attribute-laden header reads exactly as the C compiler reads it -- and the
importer walks the declarations the header itself makes: functions,
typedefs, enums, structs, and constants. Each becomes the PPY spelling a
program can call through `ppy.ffi`: a function is a `@ffi.bind` stub with
typed parameters, a typedef of a scalar is an alias, an enum is its
constants, a struct of scalars is a dataclass, and a `#define` of
one number is a typed constant. What has no PPY spelling yet -- a function
pointer, an array parameter, a struct passed by value -- is left out and
listed by name, never guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..target import TargetInfo, host_target

__all__ = ["BindError", "Bindings", "bind_header", "libclang_status"]


class BindError(RuntimeError):
    """The header could not be read, or libclang is not installed."""


@dataclass(slots=True)
class Bindings:
    """What the importer made of a header."""

    source: str
    functions: list[str] = field(default_factory=list)
    typedefs: list[str] = field(default_factory=list)
    enums: list[str] = field(default_factory=list)
    structs: list[str] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    #: Declarations left out, with the reason each was.
    skipped: list[tuple[str, str]] = field(default_factory=list)


def libclang_status() -> tuple[bool, str]:
    """Whether Clang's Python bindings can load a libclang."""
    try:
        from clang import cindex
    except ImportError:
        return False, "the `libclang` package is not installed (`pip install ppy-lang[bind]`)"
    try:
        cindex.Index.create()
    except Exception as error:  # noqa: BLE001 - the loader's message is the reason
        return False, f"libclang could not be loaded: {error}"
    return True, "libclang"


def bind_header(
    header: Path,
    *,
    library: str | None = None,
    include_dirs: tuple[Path, ...] = (),
    target: TargetInfo | None = None,
) -> Bindings:
    """Read `header` and write the bindings module for it."""
    ready, detail = libclang_status()
    if not ready:
        raise BindError(detail)
    from clang import cindex

    target = target or host_target()
    index = cindex.Index.create()
    arguments = ["-x", "c", "-std=c11", f"--target={target.triple}", *_builtin_includes()]
    arguments.extend(f"-I{directory}" for directory in include_dirs)
    options = (
        cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        | cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
    )
    try:
        unit = index.parse(str(header), args=arguments, options=options)
    except cindex.TranslationUnitLoadError as error:
        raise BindError(f"{header}: clang could not read the header ({error})") from error
    errors = [d for d in unit.diagnostics if d.severity >= cindex.Diagnostic.Error]
    if errors:
        first = errors[0]
        raise BindError(f"{header}:{first.location.line}: {first.spelling}")
    return _Importer(unit, header, library or header.stem, target).run()


class _Importer:
    def __init__(self, unit, header: Path, library: str, target: TargetInfo) -> None:  # type: ignore[no-untyped-def]
        from clang import cindex

        self.cindex = cindex
        self.unit = unit
        self.header = header.resolve()
        self.library = library
        self.target = target
        self.bindings = Bindings(source="")
        self.structs: dict[str, str] = {}
        self.aliases: dict[str, str] = {}
        self.enums: set[str] = set()
        self.opaque: set[str] = set()

    def run(self) -> Bindings:
        cindex = self.cindex
        seen: set[str] = set()
        for cursor in self.unit.cursor.get_children():
            if not self._in_header(cursor):
                continue
            kind = cursor.kind
            if kind == cindex.CursorKind.MACRO_DEFINITION:
                self._macro(cursor)
            elif kind == cindex.CursorKind.ENUM_DECL:
                self._enum(cursor)
            elif kind == cindex.CursorKind.STRUCT_DECL:
                self._struct(cursor)
            elif kind == cindex.CursorKind.TYPEDEF_DECL:
                self._typedef(cursor)
            elif kind == cindex.CursorKind.FUNCTION_DECL:
                if cursor.spelling in seen:
                    continue
                seen.add(cursor.spelling)
                self._function(cursor)
            elif kind == cindex.CursorKind.VAR_DECL:
                self.bindings.skipped.append((cursor.spelling, "a variable has no binding yet"))
        self.bindings.source = self._render()
        return self.bindings

    def _in_header(self, cursor) -> bool:  # type: ignore[no-untyped-def]
        location = cursor.location
        return location.file is not None and Path(location.file.name).resolve() == self.header

    # -- declarations ---------------------------------------------------------

    def _macro(self, cursor) -> None:  # type: ignore[no-untyped-def]
        spelled = list(cursor.get_tokens())
        tokens = [t.spelling for t in spelled]
        if len(tokens) < 2 or tokens[0] != cursor.spelling:
            return
        if tokens[1] == "(" and spelled[1].extent.start.offset == spelled[0].extent.end.offset:
            return  # `NAME(` with nothing between: a function-like macro
        body = tokens[1:]
        if body and body[0] == "(" and body[-1] == ")":
            body = body[1:-1]
        literal = _number(body)
        if literal is None:
            self.bindings.skipped.append((cursor.spelling, "a macro that is not one number"))
            return
        kind, text = literal
        self.bindings.constants.append(f"{cursor.spelling}: {kind} = {text}")

    def _enum(self, cursor) -> None:  # type: ignore[no-untyped-def]
        name = cursor.spelling if not cursor.is_anonymous() else ""
        lines = [
            f"{member.spelling}: int = {member.enum_value}"
            for member in cursor.get_children()
            if member.kind == self.cindex.CursorKind.ENUM_CONSTANT_DECL
        ]
        if name:
            self.enums.add(name)
            self.aliases[name] = "int"
            lines.insert(0, f"{name} = int")
        self.bindings.enums.extend(lines)

    def _struct(self, cursor) -> None:  # type: ignore[no-untyped-def]
        if not cursor.is_definition():
            name = cursor.spelling
            if name and cursor.get_definition() is None and name not in self.opaque:
                self.opaque.add(name)
                self.bindings.skipped.append((name, "an opaque struct"))
            return
        name = cursor.spelling
        if not name or cursor.is_anonymous():
            return
        fields: list[str] = []
        for member in cursor.get_children():
            if member.kind != self.cindex.CursorKind.FIELD_DECL:
                continue
            spelled = self._scalar(member.type)
            if spelled is None:
                self.bindings.skipped.append(
                    (name, f"field `{member.spelling}` is {member.type.spelling}, not a scalar")
                )
                return
            fields.append(f"    {member.spelling}: {spelled}")
        if not fields:
            self.bindings.skipped.append((name, "a struct with no fields"))
            return
        self.structs[name] = name
        self.bindings.structs.append("@dataclass\nclass " + name + ":\n" + "\n".join(fields))

    def _typedef(self, cursor) -> None:  # type: ignore[no-untyped-def]
        underlying = cursor.underlying_typedef_type
        name = cursor.spelling
        canonical = underlying.get_canonical()
        if canonical.kind == self.cindex.TypeKind.RECORD:
            declared = canonical.get_declaration()
            if declared.spelling in self.structs:
                if declared.spelling != name:
                    self.aliases[name] = declared.spelling
                    self.bindings.typedefs.append(f"{name} = {declared.spelling}")
                return
            if declared.spelling and declared.is_definition() and not self._in_header(declared):
                self.bindings.skipped.append((name, "a struct declared outside the header"))
                return
            if declared.spelling and declared.spelling not in self.structs:
                if declared.is_definition():
                    self._struct(declared)
                    if declared.spelling in self.structs and declared.spelling != name:
                        self.aliases[name] = declared.spelling
                        self.bindings.typedefs.append(f"{name} = {declared.spelling}")
                    return
                self.bindings.skipped.append((name, "an opaque struct"))
                return
            # typedef struct { ... } Name;
            fields: list[str] = []
            for member in declared.get_children():
                if member.kind != self.cindex.CursorKind.FIELD_DECL:
                    continue
                spelled = self._scalar(member.type)
                if spelled is None:
                    self.bindings.skipped.append(
                        (name, f"field `{member.spelling}` is {member.type.spelling}, not a scalar")
                    )
                    return
                fields.append(f"    {member.spelling}: {spelled}")
            if not fields:
                self.bindings.skipped.append((name, "a struct with no fields"))
                return
            self.structs[name] = name
            self.bindings.structs.append("@dataclass\nclass " + name + ":\n" + "\n".join(fields))
            return
        if canonical.kind == self.cindex.TypeKind.ENUM:
            self.aliases[name] = "int"
            self.bindings.typedefs.append(f"{name} = int")
            return
        spelled = self._scalar(canonical)
        if spelled is None:
            self.bindings.skipped.append((name, f"a typedef of {underlying.spelling}"))
            return
        self.aliases[name] = spelled
        self.bindings.typedefs.append(f"{name} = {spelled}")

    def _function(self, cursor) -> None:  # type: ignore[no-untyped-def]
        name = cursor.spelling
        if cursor.type.is_function_variadic():
            self.bindings.skipped.append((name, "a variadic function"))
            return
        parameters: list[str] = []
        for argument in cursor.get_arguments():
            spelled = self._parameter(argument.type)
            if spelled is None:
                self.bindings.skipped.append(
                    (name, f"parameter `{argument.spelling}` is {argument.type.spelling}")
                )
                return
            parameter_name = argument.spelling or f"arg{len(parameters)}"
            parameters.append(f"{parameter_name}: {spelled}")
        result = cursor.result_type
        if result.get_canonical().kind == self.cindex.TypeKind.VOID:
            returns = "None"
        else:
            returns = self._parameter(result)
            if returns is None:
                self.bindings.skipped.append((name, f"returns {result.spelling}"))
                return
        self.bindings.functions.append(
            f'@ffi.bind(lib, symbol="{name}")\n'
            f"def {name}({', '.join(parameters)}) -> {returns}: ..."
        )

    # -- types ----------------------------------------------------------------

    def _scalar(self, t) -> str | None:  # type: ignore[no-untyped-def]
        """The PPY spelling of a C scalar type, or None."""
        TypeKind = self.cindex.TypeKind
        canonical = t.get_canonical()
        kind = canonical.kind
        if kind == TypeKind.BOOL:
            return "bool"
        if kind in {TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LONGDOUBLE}:
            if kind == TypeKind.FLOAT:
                return "ppy.f32"
            if kind == TypeKind.LONGDOUBLE:
                return None
            return "float"
        widths = {
            TypeKind.CHAR_S: ("i", 8),
            TypeKind.SCHAR: ("i", 8),
            TypeKind.CHAR_U: ("u", 8),
            TypeKind.UCHAR: ("u", 8),
            TypeKind.SHORT: ("i", 16),
            TypeKind.USHORT: ("u", 16),
            TypeKind.INT: ("i", 32),
            TypeKind.UINT: ("u", 32),
            TypeKind.LONG: ("i", self.target.long_width),
            TypeKind.ULONG: ("u", self.target.long_width),
            TypeKind.LONGLONG: ("i", 64),
            TypeKind.ULONGLONG: ("u", 64),
        }
        described = widths.get(kind)
        if described is None:
            return None
        sign, width = described
        if sign == "i" and width == 64:
            return "int"
        return f"ppy.{sign}{width}"

    def _parameter(self, t) -> str | None:  # type: ignore[no-untyped-def]
        """The PPY spelling of a parameter or result type, or None."""
        TypeKind = self.cindex.TypeKind
        canonical = t.get_canonical()
        if canonical.kind == TypeKind.POINTER:
            pointee = canonical.get_pointee()
            const = pointee.is_const_qualified()
            base = pointee.get_canonical()
            if base.kind == TypeKind.VOID:
                element = "ppy.u8"
            elif base.kind == TypeKind.RECORD:
                return None
            else:
                element = self._scalar(base)
                if element is None:
                    return None
            return f"native.{'const_ptr' if const else 'ptr'}[{element}]"
        if canonical.kind in {TypeKind.RECORD, TypeKind.CONSTANTARRAY, TypeKind.INCOMPLETEARRAY}:
            return None
        if canonical.kind == TypeKind.ENUM:
            return self._scalar(canonical.get_declaration().enum_type)
        return self._scalar(canonical)

    # -- the module -----------------------------------------------------------

    def _render(self) -> str:
        b = self.bindings
        lines = [
            f'"""Bindings for `{self.header.name}`, written by `ppy bind header`."""',
            "",
            "from dataclasses import dataclass",
            "",
            "import ppy",
            "from ppy import ffi, native",
            "",
            f'lib = ffi.library("{self.library}")',
        ]
        for title, items in (
            ("constants", b.constants),
            ("enums", b.enums),
            ("typedefs", b.typedefs),
        ):
            if items:
                lines += ["", f"# {title}", *items]
        for struct in b.structs:
            lines += ["", "", struct]
        for function in b.functions:
            lines += ["", "", function]
        if b.skipped:
            lines += ["", "# left out (no PPY spelling yet):"]
            lines += [f"#   {name}: {reason}" for name, reason in b.skipped]
        return "\n".join(lines) + "\n"


def _builtin_includes() -> list[str]:
    """Where `<stdint.h>` and friends are: libclang alone carries no headers.

    A clang on the path lends its resource directory; failing that, GCC's
    own include directory holds the same builtin headers.
    """
    import shutil
    import subprocess

    clang = shutil.which("clang")
    if clang is not None:
        found = subprocess.run(
            [clang, "-print-resource-dir"], capture_output=True, text=True, check=False
        )
        if found.returncode == 0 and found.stdout.strip():
            return [f"-resource-dir={found.stdout.strip()}"]
    gcc = shutil.which("gcc") or shutil.which("cc")
    if gcc is not None:
        found = subprocess.run(
            [gcc, "-print-file-name=include"], capture_output=True, text=True, check=False
        )
        if found.returncode == 0 and Path(found.stdout.strip()).is_dir():
            return [f"-isystem{found.stdout.strip()}"]
    return []


def _number(tokens: list[str]) -> tuple[str, str] | None:
    """A numeric literal from macro tokens: (`int` | `float`, the Python text)."""
    if not tokens:
        return None
    sign = ""
    if tokens[0] in {"-", "+"} and len(tokens) > 1:
        sign = tokens[0]
        tokens = tokens[1:]
    if len(tokens) != 1:
        return None
    text = tokens[0].rstrip("uUlL")
    lowered = text.lower()
    try:
        if lowered.startswith("0x"):
            return "int", f"{sign}{int(lowered, 16)}"
        if any(c in lowered for c in ".e") and not lowered.startswith("0x"):
            return "float", f"{sign}{float(text.rstrip('fF'))!r}"
        if lowered.startswith("0") and len(lowered) > 1 and lowered.isdigit():
            return "int", f"{sign}{int(lowered, 8)}"
        return "int", f"{sign}{int(text)}"
    except ValueError:
        return None
