"""`clang-format` over an emitted unit, for `ppy emit --format`.

The emitter writes C the way a careful person would; the formatter settles
line breaks and spacing. A project's own `.clang-format` wins where it has
one; otherwise the style is LLVM's with four-space indentation and lines of
a hundred columns, the emitter's own habits.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = ["ClangFormatError", "clang_format"]

#: The style used where the project declares none.
STYLE = (
    "{BasedOnStyle: LLVM, IndentWidth: 4, ColumnLimit: 100, PointerAlignment: Right, "
    "AllowShortIfStatementsOnASingleLine: WithoutElse, AllowShortBlocksOnASingleLine: Never, "
    "AllowShortFunctionsOnASingleLine: None, AllowShortLoopsOnASingleLine: false, "
    "AlignTrailingComments: false, SpaceAfterCStyleCast: false, "
    "BreakBeforeBinaryOperators: None, AlwaysBreakTemplateDeclarations: Yes}"
)


class ClangFormatError(Exception):
    """`clang-format` is missing or refused the text."""

    def __init__(self, message: str, help: str | None = None) -> None:
        super().__init__(message)
        self.help = help


def clang_format(text: str, suffix: str, root: Path | None) -> str:
    """`text` as clang-format lays it out; `suffix` names the language, `root`
    the project whose `.clang-format` applies."""
    tool = shutil.which("clang-format")
    if tool is None:
        raise ClangFormatError(
            "`--format` needs `clang-format` on PATH",
            help="install LLVM's clang-format, or leave `--format` off: the text is C as emitted",
        )
    assumed = (root or Path.cwd()) / f"ppy_emitted{suffix}"
    style = "file" if root is not None and (root / ".clang-format").exists() else STYLE
    done = subprocess.run(
        [tool, f"-style={style}", f"-assume-filename={assumed}"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise ClangFormatError(
            "`clang-format` failed on the emitted text", help=done.stderr.strip() or None
        )
    return done.stdout
