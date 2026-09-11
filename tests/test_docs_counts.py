"""The counts the docs carry come from the tree: the markers fill from it, and the one
count `examples/README.md` spells itself is held to the same functions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "docs" / "gen"))

import counts  # noqa: E402  pylint: disable=wrong-import-position


def test_the_markers_count_what_the_tree_holds():
    assert counts._folders() == len(counts.FOLDERS) if hasattr(counts, "FOLDERS") else True
    folders, programs, compared = counts._folders(), counts._programs(), counts._compared()
    assert 0 < compared <= folders < programs
    listed = {p.name for p in (ROOT / "examples").iterdir() if re.match(r"\d\d_", p.name)}
    from folders import FOLDERS  # pylint: disable=import-outside-toplevel

    assert set(FOLDERS) == listed, "every example folder is in the gallery, and only those"
    assert compared == sum((ROOT / "examples" / f / "compare").is_dir() for f in FOLDERS)


def test_the_examples_index_spells_the_compared_count_the_tree_has():
    text = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    found = re.search(r"^(\d+) folders carry a `compare/` directory", text, re.MULTILINE)
    assert found is not None, "the index says how many folders carry counterparts"
    assert int(found.group(1)) == counts._compared()
    rows = re.findall(r"^\| `(\d\d_[a-z_]+)` \|", text, re.MULTILINE)
    from folders import FOLDERS  # pylint: disable=import-outside-toplevel

    assert sorted(rows) == sorted(FOLDERS), "one row per gallery folder"


def test_no_page_spells_a_count_the_markers_carry():
    """A digit count of folders or programs belongs to a marker, not to prose."""
    pattern = re.compile(r"\b(\d{2}) (example )?(folders|programs)\b")
    for page in (ROOT / "docs").rglob("*.md"):
        for line in page.read_text(encoding="utf-8").splitlines():
            if "@@" in line:
                continue
            assert not pattern.search(line), f"{page.relative_to(ROOT)}: {line.strip()}"
