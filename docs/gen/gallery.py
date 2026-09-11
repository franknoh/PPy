"""The example gallery: one page per folder under `examples/`, built from its README.

Each page is the folder's README with its links resolved for the site, the
folder's `.ppy` sources appended, and a link back to the repository. The
README is the one description of an example, so the page follows it.
"""

from __future__ import annotations

import posixpath
import re
import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
from folders import FOLDERS, GROUPS  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = ROOT / "examples"
REPO = "https://github.com/franknoh/PPy/blob/main"

_LINK = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def _resolve(folder: str, target: str) -> str:
    """A README link rewritten for the site: sibling folders are pages, the rest the repository."""
    path, _, fragment = target.partition("#")
    suffix = f"#{fragment}" if fragment else ""
    normalized = posixpath.normpath(posixpath.join("examples", folder, path))
    parts = normalized.split("/")
    if parts == ["examples", "README.md"]:
        return f"index.md{suffix}"
    if parts[0] == "docs" and normalized.endswith(".md"):
        return "../" + "/".join(parts[1:]) + suffix
    sibling = len(parts) >= 2 and parts[0] == "examples" and parts[1] in FOLDERS
    if sibling and (len(parts) == 2 or (len(parts) == 3 and parts[2] == "README.md")):
        return f"{parts[1]}.md{suffix}"
    return f"{REPO}/{normalized}{suffix}"


_FILED_OUTPUT = re.compile(
    r"^\*(\d+) lines: \[outputs/([^\]]+)\]\(outputs/[^)]+\)\*$", re.MULTILINE
)
_FOLDED_OUTPUT = re.compile(
    r'^<details markdown="1">\n<summary>(\d+) lines</summary>\n\n'
    r"```text\n(.*?)\n```\n\n</details>$",
    re.MULTILINE | re.DOTALL,
)


def _admonition(count: str, whole: str) -> str:
    body = "\n".join("    " + line for line in whole.rstrip("\n").splitlines())
    return f'??? note "{count} lines"\n\n    ```text\n{body}\n    ```'


def _embed(folder: str, text: str) -> str:
    """A folded or filed output, as the site's own collapsible block."""
    text = _FOLDED_OUTPUT.sub(lambda m: _admonition(m.group(1), m.group(2)), text)

    def filed(match: re.Match[str]) -> str:
        whole = (EXAMPLES / folder / "outputs" / match.group(2)).read_text(encoding="utf-8")
        return _admonition(match.group(1), whole)

    return _FILED_OUTPUT.sub(filed, text)


def _links(folder: str, text: str) -> str:
    """The README's links resolved for the site -- in prose only. A fenced block
    or a code span keeps its text: `ppy.check[int](eval(source))` is code, not
    a link to `eval(source)`."""
    out: list[str] = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        pieces = re.split(r"(`[^`]*`)", line)
        for index, piece in enumerate(pieces):
            if index % 2 == 0:
                pieces[index] = _LINK.sub(lambda m: f"]({_resolve(folder, m.group(1))})", piece)
        out.append("".join(pieces))
    return "\n".join(out)


def _readme(folder: str) -> tuple[str, str, str]:
    """The README's title, its first paragraph, and its body with links resolved."""
    text = (EXAMPLES / folder / "README.md").read_text(encoding="utf-8")
    text = _embed(folder, text)
    text = _links(folder, text)
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()
    body = lines[1:]
    summary: list[str] = []
    for line in body:
        if line.strip():
            summary.append(line.strip())
        elif summary:
            break
    paragraph = " ".join(summary)
    first = re.match(r"(.+?[.!?])(?:\s|$)", paragraph)
    return title, first.group(1) if first else paragraph, "\n".join(body).strip("\n")


def _sources(folder: Path) -> list[Path]:
    found = sorted(folder.glob("*.ppy"))
    if not found:
        found = sorted(folder.glob("src/*.ppy"))
    return found


def _page(folder: str) -> str:
    title, _, body = _readme(folder)
    directory = EXAMPLES / folder
    lines = [f"# {title}", "", body, ""]
    subfolders = sorted(p.name for p in directory.iterdir() if p.is_dir() and p.name[:1].isdigit())
    if subfolders:
        lines.append("## Problems")
        lines.append("")
        lines.extend(
            f"- [`{name}`]({REPO}/examples/{folder}/{name}/README.md)" for name in subfolders
        )
        lines.append("")
    for source in _sources(directory):
        lines.append(f"## `{source.relative_to(EXAMPLES)}`")
        lines.append("")
        lines.append("```python")
        lines.append(source.read_text(encoding="utf-8").rstrip("\n"))
        lines.append("```")
        lines.append("")
    counterparts = _counterparts(directory)
    if counterparts:
        lines.append("## The counterparts, side by side")
        lines.append("")
        lines.append(
            "The programs the comparison above was measured with, each written the way "
            "its tool wants it; the PPY one is first."
        )
        lines.append("")
        for source in counterparts:
            body = source.read_text(encoding="utf-8").rstrip("\n")
            lines.append(f'??? example "`{source.name}` -- {_TOOLS[source.suffix]}"')
            lines.append("")
            lines.append(f"    ```{_FENCES[source.suffix]}")
            lines.extend("    " + line if line else "" for line in body.splitlines())
            lines.append("    ```")
            lines.append("")
    lines.append(f"The folder in the repository: [`examples/{folder}`]({REPO}/examples/{folder}).")
    lines.append("")
    return "\n".join(lines)


#: How a counterpart is fenced and named on the page, by its suffix.
_FENCES = {
    ".ppy": "python",
    ".py": "python",
    ".pyx": "cython",
    ".mojo": "mojo",
    ".rs": "rust",
    ".c": "c",
    ".cu": "cuda",
}
_TOOLS = {
    ".ppy": "PPY",
    ".py": "Python",
    ".pyx": "Cython",
    ".mojo": "Mojo",
    ".rs": "Rust",
    ".c": "C",
    ".cu": "CUDA C",
}


def _counterparts(directory: Path) -> list[Path]:
    """The `compare/` programs, the PPY one first, then the rest by name."""
    compare = directory / "compare"
    if not compare.is_dir():
        return []
    found = [p for p in sorted(compare.iterdir()) if p.suffix in _FENCES and p.is_file()]
    return sorted(found, key=lambda p: (p.suffix != ".ppy", p.name))


_COMPARED = re.compile(
    r"^## (?:Compared with|The .* against) .*?(?=^## |\Z)", re.MULTILINE | re.DOTALL
)


def _comparisons() -> str:
    """One page of every comparison section, so the tables can be read together."""
    lines = [
        "# Comparisons",
        "",
        "Every example that measures itself against other tools, collected. Each section",
        "is the example's own, with its table and what each port asked for; the",
        "counterpart programs are on the example's page and in its `compare/` folder,",
        "and `examples/compare.py` is the harness that held them to one answer before",
        "timing them.",
        "",
    ]
    for folder in FOLDERS:
        title, _, body = _readme(folder)
        for section in _COMPARED.findall(body):
            heading, _, rest = section.partition("\n")
            lines.append(f"## {title}: {heading.removeprefix('## ').strip()}")
            lines.append("")
            lines.append(f"From [{title}]({folder}.md).")
            lines.append(rest.strip("\n"))
            lines.append("")
    return "\n".join(lines)


def _index() -> str:
    lines = [
        "# Examples",
        "",
        "One folder per program, and one thing each program shows. `examples/run_all.py`",
        "runs every one of them on all three paths — plain CPython, the Python backend,",
        "LLVM native — and diffs the output; a disagreement is a compiler bug. Where a",
        "folder holds both `<name>.py` and `<name>.ppy`, the `.ppy` is exactly what",
        "`ppy convert` wrote, and `examples/verify_conversions.py` regenerates it to prove it.",
        "",
    ]
    for group, folders in GROUPS:
        lines.append(f"## {group}")
        lines.append("")
        lines.append('<div class="grid cards" markdown>')
        lines.append("")
        for folder in folders:
            title, summary, _ = _readme(folder)
            lines.append(f"-   **[{title}]({folder}.md)**")
            lines.append("")
            lines.append("    ---")
            lines.append("")
            lines.append(f"    {summary}")
            lines.append("")
        lines.append("</div>")
        lines.append("")
    return "\n".join(lines)


with mkdocs_gen_files.open("howto/index.md", "w") as handle:
    handle.write(_index())

for folder in FOLDERS:
    with mkdocs_gen_files.open(f"howto/{folder}.md", "w") as handle:
        handle.write(_page(folder))
    mkdocs_gen_files.set_edit_path(f"howto/{folder}.md", f"../examples/{folder}/README.md")

with mkdocs_gen_files.open("howto/comparisons.md", "w") as handle:
    handle.write(_comparisons())
