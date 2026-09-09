"""The example gallery: one page per folder under `examples/`, built from its README.

Each page is the folder's README with its links resolved for the site, the
folder's `.ppy` sources appended, and a link back to the repository. The
README is the one description of an example, so the page follows it.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = ROOT / "examples"
REPO = "https://github.com/franknoh/PPy/blob/main"

GROUPS = [
    (
        "Basics",
        [
            "01_basics",
            "02_arbitrary_precision",
            "03_effects_and_contracts",
            "04_classes",
            "08_native_data",
            "10_narrowing",
            "11_numerics",
            "13_value_classes",
            "14_tuples",
            "16_dynamic",
            "17_containers",
            "18_errors",
            "19_strings",
        ],
    ),
    (
        "Native code",
        [
            "12_buffers_and_jit",
            "15_algorithms",
            "28_threads",
            "32_native_memory",
            "33_simd_and_cpu",
            "34_atomics_and_threads",
            "35_parallel_range",
            "36_autodiff",
            "37_aio",
            "40_generics",
            "42_toolbox",
        ],
    ),
    ("Accelerators", ["38_cuda", "39_xla"]),
    (
        "Libraries",
        [
            "05_numpy",
            "06_pydantic",
            "07_parallel",
            "09_torch",
            "21_training_torch",
            "22_training_jax",
            "25_jax_export",
            "27_uvicorn",
            "29_flax",
            "31_torchrun",
            "41_columnar",
        ],
    ),
    (
        "Conversion and projects",
        ["20_inventory", "23_inference", "24_interop", "26_project", "30_migrate"],
    ),
]
FOLDERS = [folder for _, folders in GROUPS for folder in folders]

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


_FULL_OUTPUT = re.compile(
    r"^\*(\d+) lines in all — \[full output\]\(outputs/([^)]+)\)\.\*$", re.MULTILINE
)


def _unfold(folder: str, text: str) -> str:
    """A clipped output's whole text, folded into the page where the README links to it."""

    def block(match: re.Match[str]) -> str:
        whole = (EXAMPLES / folder / "outputs" / match.group(2)).read_text(encoding="utf-8")
        body = "\n".join("    " + line for line in whole.rstrip("\n").splitlines())
        return f'??? note "All {match.group(1)} lines"\n\n    ```text\n{body}\n    ```'

    return _FULL_OUTPUT.sub(block, text)


def _readme(folder: str) -> tuple[str, str, str]:
    """The README's title, its first paragraph, and its body with links resolved."""
    text = (EXAMPLES / folder / "README.md").read_text(encoding="utf-8")
    text = _unfold(folder, text)
    text = _LINK.sub(lambda m: f"]({_resolve(folder, m.group(1))})", text)
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
    lines.append(f"The folder in the repository: [`examples/{folder}`]({REPO}/examples/{folder}).")
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
