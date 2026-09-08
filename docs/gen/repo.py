"""Contributing and the changelog, from the repository's own files.

`CONTRIBUTING.md` and `CHANGELOG.md` are written for a reader on GitHub, so
their links name paths from the repository root; the site pages are the
same text with those links pointed at the site's pages.
"""

from __future__ import annotations

import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent.parent
REPO = "https://github.com/franknoh/PPy/blob/main"

_LINK = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def _resolve(target: str) -> str:
    if target.startswith("docs/"):
        return target[len("docs/") :]
    return f"{REPO}/{target}"


def _page(source: str, title: str | None) -> str:
    text = (ROOT / source).read_text(encoding="utf-8")
    text = _LINK.sub(lambda m: f"]({_resolve(m.group(1))})", text)
    if title is not None:
        text = re.sub(r"\A# .*\n", f"# {title}\n", text, count=1)
    return text


for name, source, title in [
    ("contributing.md", "CONTRIBUTING.md", None),
    ("changelog.md", "CHANGELOG.md", None),
]:
    with mkdocs_gen_files.open(name, "w") as handle:
        handle.write(_page(source, title))
    mkdocs_gen_files.set_edit_path(name, f"../{source}")
