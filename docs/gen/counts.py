"""A MkDocs hook: the example counts on the landing pages come from the tree.

`@@EXAMPLE_FOLDERS@@` and `@@EXAMPLE_PROGRAMS@@` in a page are the number of
example folders the gallery lists and the number of programs
`examples/run_all.py` runs, counted the way those two count them, so no page
carries a number the tree has moved past.
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


def _programs() -> int:
    entries = (
        sorted(EXAMPLES.glob("[0-9]*/[a-z]*.ppy"))
        + sorted(EXAMPLES.glob("[0-9]*/[0-9]*/[a-z]*.ppy"))
        + sorted(EXAMPLES.glob("*/src/app.ppy"))
    )
    return len({entry for entry in entries if ".ppy-cache" not in entry.parts})


_COUNTS = {"@@EXAMPLE_FOLDERS@@": _folders, "@@EXAMPLE_PROGRAMS@@": _programs}


def on_page_markdown(markdown: str, **_kwargs: object) -> str:
    for marker, count in _COUNTS.items():
        if marker in markdown:
            markdown = markdown.replace(marker, str(count()))
    return markdown


def on_post_build(**_kwargs: object) -> None:
    leftover = re.compile(r"@@EXAMPLE_[A-Z]+@@")
    for page in (ROOT / "site").rglob("*.html"):
        if leftover.search(page.read_text(encoding="utf-8", errors="replace")):
            raise RuntimeError(f"an example count was not filled in: {page}")
