"""A MkDocs hook: the example counts on the landing pages come from the tree.

`@@EXAMPLE_FOLDERS@@` and `@@EXAMPLE_PROGRAMS@@` in a page are the number of
example folders the gallery lists and the number of programs
`examples/run_all.py` runs, counted the way those two count them,
`@@COMPARED_FOLDERS@@` how many of the folders carry a `compare/` directory,
`@@TEST_FUNCTIONS@@` the test functions under `tests/`, and
`@@DIAGNOSTIC_CODES@@` the codes the compiler can emit, so no page carries a
number the tree has moved past. `examples/README.md`
is not a page, so a test holds its own count to the same functions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = ROOT / "examples"


def _folders() -> int:
    sys.path.insert(0, str(ROOT / "docs" / "gen"))
    from folders import FOLDERS  # pylint: disable=import-outside-toplevel

    return len(FOLDERS)


def _compared() -> int:
    """The folders the gallery lists that carry counterparts in `compare/`."""
    sys.path.insert(0, str(ROOT / "docs" / "gen"))
    from folders import FOLDERS  # pylint: disable=import-outside-toplevel

    return sum((EXAMPLES / folder / "compare").is_dir() for folder in FOLDERS)


def _programs() -> int:
    entries = (
        sorted(EXAMPLES.glob("[0-9]*/[a-z]*.ppy"))
        + sorted(EXAMPLES.glob("[0-9]*/[0-9]*/[a-z]*.ppy"))
        + sorted(EXAMPLES.glob("*/src/app.ppy"))
    )
    return len({entry for entry in entries if ".ppy-cache" not in entry.parts})


_TEST_FUNCTION = re.compile(r"^\s*(?:async )?def test_\w+", re.MULTILINE)


def _test_functions() -> int:
    """The `test_` functions under `tests/`: what the suite is made of, not the
    parametrized cases pytest expands them to."""
    return sum(
        len(_TEST_FUNCTION.findall(path.read_text(encoding="utf-8")))
        for path in (ROOT / "tests").rglob("*.py")
    )


def _diagnostic_codes() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from ppy_compiler.diagnostics import CODES  # pylint: disable=import-outside-toplevel

    return len(CODES)


_COUNTS = {
    "@@EXAMPLE_FOLDERS@@": _folders,
    "@@EXAMPLE_PROGRAMS@@": _programs,
    "@@COMPARED_FOLDERS@@": _compared,
    "@@TEST_FUNCTIONS@@": _test_functions,
    "@@DIAGNOSTIC_CODES@@": _diagnostic_codes,
}


def on_page_markdown(markdown: str, **_kwargs: object) -> str:
    for marker, count in _COUNTS.items():
        if marker in markdown:
            markdown = markdown.replace(marker, str(count()))
    return markdown


def on_post_build(**_kwargs: object) -> None:
    leftover = re.compile(r"@@[A-Z_]+@@")
    for page in (ROOT / "site").rglob("*.html"):
        if leftover.search(page.read_text(encoding="utf-8", errors="replace")):
            raise RuntimeError(f"a count was not filled in: {page}")
