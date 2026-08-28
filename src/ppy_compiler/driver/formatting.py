"""`ppy fmt`: source-preserving formatting for PPY sources (spec 4.3, 7.2)."""

from __future__ import annotations

import argparse
import shutil
import sys
import subprocess
from pathlib import Path

import libcst as cst

from ..diagnostics import Diagnostic, Severity, Span
from .pipeline import collect_sources
from .reporting import Reporter

__all__ = [
    "normalize_source",
    "run_fmt",
    "format_source",
]

_EXTERNAL = ("ruff", "black")


def run_fmt(options: argparse.Namespace, reporter: Reporter) -> int:
    target: Path = options.path
    if not target.exists():
        reporter.emit(Diagnostic("E1002", Severity.ERROR, f"{target} does not exist"))
        return 2

    sources = collect_sources(target, ppy_only=target.is_dir())
    if not sources:
        reporter.note(f"no PPY sources found under {target}")
        return 0

    changed: list[Path] = []
    failed = 0
    for path in sources:
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            reporter.emit(Diagnostic("E1002", Severity.ERROR, f"cannot read {path}: {exc}"))
            failed += 1
            continue
        try:
            formatted = format_source(original, path)
        except cst.ParserSyntaxError as exc:
            reporter.emit(
                Diagnostic("E1001", Severity.ERROR, str(exc), Span(path, getattr(exc, "raw_line", 1), 0))
            )
            failed += 1
            continue
        if formatted == original:
            continue
        changed.append(path)
        if not options.check:
            path.write_text(formatted, encoding="utf-8")

    if options.check:
        for path in changed:
            reporter.note(f"would reformat {path}")
        if changed:
            reporter.note(f"{len(changed)} file(s) would be reformatted")
            return 1
        reporter.note(f"{len(sources)} file(s) already formatted")
        return 0 if not failed else 1

    reporter.note(f"formatted {len(changed)} of {len(sources)} file(s)")
    return 1 if failed else 0


def format_source(source: str, path: Path | None = None) -> str:
    """Format PPY source, delegating to an installed formatter when present."""
    external = _external_format(source, path)
    if external is not None:
        return external
    return normalize_source(source, _siblings(path))


def _siblings(path: Path | None) -> frozenset[str]:
    """Modules sitting next to this file, which are first-party to it.

    Without this a sibling sorts among the third-party imports, and `import ppy`
    can end up after the import whose loader it installs.
    """
    if path is None or not path.parent.is_dir():
        return frozenset()
    return frozenset(
        entry.stem
        for entry in path.parent.iterdir()
        if entry.suffix in {".py", ".ppy"} and entry.stem != path.stem
    ) | frozenset(
        entry.name
        for entry in path.parent.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists()
    )


def _external_format(source: str, path: Path | None) -> str | None:
    """`.ppy` is ordinary Python, so an existing formatter can be reused."""
    for tool in _EXTERNAL:
        executable = shutil.which(tool)
        if executable is None:
            continue
        command = (
            [executable, "format", "--stdin-filename", str(path or "source.py"), "-"]
            if tool == "ruff"
            else [executable, "-q", "-"]
        )
        try:
            completed = subprocess.run(  # noqa: S603 - explicit executable path
                command,
                input=source,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0 and completed.stdout:
            return completed.stdout
    return None


def normalize_source(source: str, local: "frozenset[str]" = frozenset()) -> str:
    """Conservative normalization through the concrete syntax tree.

    Deterministic and self-contained, so a converted file is byte-identical
    wherever it is produced.
    """
    module = cst.parse_module(source)
    module = _group_imports(module, local)
    module = module.with_changes(body=_space_top_level(list(module.body)))
    module = module.visit(_WrapSignatures())
    code = module.code
    code = "\n".join(line.rstrip() for line in code.splitlines())
    return code + "\n" if code and not code.endswith("\n") else code


#: Annotating a signature can push it past a line limit that the source kept.
_LINE_LIMIT = 100


def _import_rank(statement: cst.BaseStatement, local: "frozenset[str]") -> int | None:
    """Which PEP 8 group an import belongs to, or None if it is not one."""
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    first = statement.body[0]
    if isinstance(first, cst.ImportFrom):
        root = _dotted_name(first.module).partition(".")[0] if first.module else ""
        if root == "__future__":
            return 0
    elif isinstance(first, cst.Import):
        root = _dotted_name(first.names[0].name).partition(".")[0]
    else:
        return None
    if not root:
        return 3
    if root in local:
        return 3
    return 1 if root in sys.stdlib_module_names else 2


def _sort_key(statement: cst.BaseStatement) -> tuple[int, str]:
    """`import x` before `from x import y`, alphabetical within each."""
    first = statement.body[0]  # type: ignore[union-attr]
    if isinstance(first, cst.Import):
        return (0, _dotted_name(first.names[0].name))
    module = _dotted_name(first.module) if first.module else ""  # type: ignore[union-attr]
    return (1, module)


def _dotted_name(node: object) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr.value}"
    return ""


def _group_imports(module: cst.Module, local: "frozenset[str]") -> cst.Module:
    """Sort the leading imports into PEP 8 groups, one blank line apart.

    Only the run of imports at the top is touched, so an import placed later on
    purpose -- after a `ppy.install()`, for instance -- keeps its position.
    """
    body = list(module.body)
    start = 0
    for index, statement in enumerate(body):
        rank = _import_rank(statement, local)
        if rank is not None:
            start = index
            break
        if isinstance(statement, cst.SimpleStatementLine) and isinstance(
            statement.body[0], cst.Expr
        ):
            continue
        return module
    else:
        return module

    end = start
    while end < len(body) and _import_rank(body[end], local) is not None:
        end += 1
    block = body[start:end]
    if len(block) < 2:
        return module

    grouped: list[cst.BaseStatement] = []
    previous: int | None = None
    for rank in sorted({_import_rank(s, local) for s in block}):  # type: ignore[type-var]
        members = sorted((s for s in block if _import_rank(s, local) == rank), key=_sort_key)
        for position, statement in enumerate(members):
            blanks = 1 if previous is not None and position == 0 else 0
            grouped.append(statement.with_changes(leading_lines=_blank_lines((), blanks)))
        previous = rank
    grouped[0] = grouped[0].with_changes(leading_lines=block[0].leading_lines)
    return module.with_changes(body=[*body[:start], *grouped, *body[end:]])


class _WrapSignatures(cst.CSTTransformer):
    """Put each parameter on its own line when a signature grows too long."""

    def leave_FunctionDef(
        self, original: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        params = updated.params
        # `star_arg` is a sentinel rather than None when absent, so it has to be
        # tested by type.
        variadic = isinstance(params.star_arg, cst.Param) or params.star_kwarg is not None
        if not params.params or variadic or params.kwonly_params:
            return updated
        header = _header_width(updated)
        if header <= _LINE_LIMIT:
            return updated
        wrapped = [
            param.with_changes(comma=_break_after(last=index == len(params.params) - 1))
            for index, param in enumerate(params.params)
        ]
        return updated.with_changes(
            params=params.with_changes(params=wrapped),
            whitespace_before_params=cst.ParenthesizedWhitespace(
                first_line=cst.TrailingWhitespace(),
                indent=True,
                last_line=cst.SimpleWhitespace("    "),
            ),
        )


def _break_after(*, last: bool) -> cst.Comma:
    """A trailing comma that starts the next line, closing at column 0 on the last."""
    return cst.Comma(
        whitespace_after=cst.ParenthesizedWhitespace(
            first_line=cst.TrailingWhitespace(),
            indent=True,
            last_line=cst.SimpleWhitespace("" if last else "    "),
        )
    )


def _header_width(node: cst.FunctionDef) -> int:
    """How wide the `def` line is once its body is set aside."""
    stripped = node.with_changes(
        body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
        leading_lines=(),
    )
    rendered = cst.Module(body=[stripped]).code.splitlines()
    return max((len(line) for line in rendered[:-1]), default=0)


def _space_top_level(body: list[cst.BaseStatement]) -> list[cst.BaseStatement]:
    """Two blank lines before each top-level `def` or `class`."""
    spaced: list[cst.BaseStatement] = []
    for index, statement in enumerate(body):
        if (
            index > 0
            and isinstance(statement, (cst.FunctionDef, cst.ClassDef))
            and not _decorated_continuation(body[index - 1])
        ):
            statement = statement.with_changes(
                leading_lines=_blank_lines(statement.leading_lines, 2)
            )
        spaced.append(statement)
    return spaced


def _decorated_continuation(previous: cst.BaseStatement) -> bool:
    return isinstance(previous, cst.Decorator)


def _blank_lines(existing: tuple[cst.EmptyLine, ...], count: int) -> list[cst.EmptyLine]:
    comments = [line for line in existing if line.comment is not None]
    blanks = [cst.EmptyLine(indent=False) for _ in range(count)]
    return [*blanks, *comments]
