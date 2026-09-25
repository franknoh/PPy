"""The performance page, from the recorded measurements. No number is retyped.

`examples/15_algorithms/measurements.json` is the record for the six judge
problems; the collatz table comes from the marked section of `README.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDED = ROOT / "examples" / "15_algorithms" / "measurements.json"

PATHS = ["plain", "ppy run", "ppy build", "standalone", "C gcc", "C clang"]
LABELS = {
    "plain": "CPython",
    "ppy run": "`ppy run`",
    "ppy build": "`ppy build --unsafe`",
    "standalone": "`--standalone`",
    "C gcc": "C (`gcc -O3`)",
    "C clang": "C (`clang -O3`)",
}
PROBLEMS = {
    "15a_nqueens": "N-Queens",
    "15b_dijkstra": "Shortest paths",
    "15c_kmp": "Substring search",
    "15d_segment_tree": "Range sums",
    "15e_lis": "Longest increasing subsequence",
    "15f_input": "Counting inversions",
}


def _cell(record: dict, path: str) -> str:
    entry = record.get(path)
    if not entry:
        return "—"
    return f"{entry['mean']:.1f} ± {entry['stdev']:.1f} ms"


def _readme_table() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("| compiler | kernel | integer semantics |")
    end = text.index("\n\n", start)
    return text[start:end]


def _page() -> str:
    data = json.loads(RECORDED.read_text(encoding="utf-8"))
    environment = data["environment"]
    lines = [
        "# Performance",
        "",
        "This page shows recorded timings. None of the numbers are typed by hand:",
        "when the site is built, the six judge problems are read from",
        "`examples/15_algorithms/measurements.json` and the collatz table from the README.",
        "",
        "`scripts/refresh.py` re-measures and reports what has drifted past a tolerance,",
        "and `--write` records it.",
        "",
        "## Collatz kernel compared with other compilers",
        "",
        "This is the collatz kernel from the README. It was run on one machine, ten runs",
        "each in a fresh process, and the table shows the kernel's wall time (mean ±",
        "standard deviation).",
        "",
        "Of the PPy rows, `ppy build --unsafe` is the wrap-semantics artifact. `ppy run`",
        "keeps Python-integer semantics: overflow is guarded and falls back to arbitrary",
        "precision.",
        "",
        _readme_table(),
        "",
        "## Six judge problems, whole process",
        "",
        "These are wall times of the whole process, including input reading and",
        "interpreter startup. Nothing inside the programs is instrumented. Each problem",
        "reads its input from standard input.",
        "",
        "| problem | " + " | ".join(LABELS[p] for p in PATHS) + " |",
        "|---|" + "---:|" * len(PATHS),
    ]
    for name, title in PROBLEMS.items():
        record = data["problems"].get(name)
        if record is None:
            continue
        cells = " | ".join(_cell(record, p) for p in PATHS)
        lines.append(f"| [{title}](../howto/15_algorithms.md) `{name}` | {cells} |")
    lines.extend(
        [
            "",
            (
                "Recorded on "
                f"{environment.get('processor', '?')}, {environment.get('cores', '?')} cores, "
                f"{environment.get('implementation', 'CPython')} {environment.get('python', '?')}, "
                f"{environment.get('platform', '?')}, at {data.get('recorded', '?')}."
            ),
            "",
            "The `ppy run` column includes the first run's build into the cache. That is",
            "why its deviation is wide. From the second run on, it is the launcher alone.",
            "",
            "`--standalone` is a native executable with no interpreter inside.",
            "",
        ]
    )
    return "\n".join(lines)


with mkdocs_gen_files.open("reference/performance.md", "w") as handle:
    handle.write(_page())
mkdocs_gen_files.set_edit_path(
    "reference/performance.md", "../examples/15_algorithms/measurements.json"
)
