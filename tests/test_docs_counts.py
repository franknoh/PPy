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


#: Markdown emphasis and code spans, which a count may be dressed in: `**42** example folders`.
_DECORATION = re.compile(r"[*_`]+")
#: A count of folders or programs written as digits: the markers' business, not prose's.
_SPELLED = re.compile(
    r"\b\d[\d,]* (?:(?:example |comparison |compared )?(?:folders|programs)"
    r"|test functions|diagnostic codes)\b"
)


def test_no_page_spells_a_count_the_markers_carry():
    """A digit count of folders or programs belongs to a marker, not to prose, however
    the Markdown dresses it."""
    for page in (ROOT / "docs").rglob("*.md"):
        for line in page.read_text(encoding="utf-8").splitlines():
            bare = _DECORATION.sub("", line)
            found = _SPELLED.search(bare)
            assert found is None or "@@" in line, f"{page.relative_to(ROOT)}: {line.strip()}"


def test_the_guard_sees_through_markdown_emphasis():
    for dressed in (
        "**42** example folders",
        "_52_ programs",
        "`45` folders",
        "42 example folders",
        "**1,110** test functions",
        "1,058 test functions",
        "76 diagnostic codes",
    ):
        assert _SPELLED.search(_DECORATION.sub("", dressed)), dressed
    assert not _SPELLED.search("@@EXAMPLE_FOLDERS@@ example folders")


def test_the_markers_fill_inside_emphasis():
    filled = counts.on_page_markdown(
        "**@@EXAMPLE_FOLDERS@@** folders, `@@EXAMPLE_PROGRAMS@@` programs"
    )
    assert filled == f"**{counts._folders()}** folders, `{counts._programs()}` programs"


def test_the_test_and_code_counts_are_the_tree_s():
    from ppy_compiler.diagnostics import CODES  # pylint: disable=import-outside-toplevel

    assert counts._diagnostic_codes() == len(CODES) > 0
    functions = counts._test_functions()
    here = (ROOT / "tests" / "test_docs_counts.py").read_text(encoding="utf-8")
    assert functions >= here.count("\ndef test_") > 0
    filled = counts.on_page_markdown("**@@TEST_FUNCTIONS@@** tests, @@DIAGNOSTIC_CODES@@ codes")
    assert filled == f"**{functions}** tests, {len(CODES)} codes"
