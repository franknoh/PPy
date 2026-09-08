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

PATHS = ["plain", "ppy run", "ppy build", "standalone", "C scanf"]
LABELS = {
    "plain": "CPython",
    "ppy run": "`ppy run`",
    "ppy build": "`ppy build`",
    "standalone": "`--standalone`",
    "C scanf": "C (`gcc -O3`)",
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
        "Nothing on this page is retyped: the six judge problems are read from",
        "`examples/15_algorithms/measurements.json` and the collatz table from the README",
        "when the site is built. `scripts/refresh.py` re-measures and reports what has",
        "drifted past a tolerance, and `--write` records it.",
        "",
        "## The same kernel through the neighbours",
        "",
        "The collatz kernel from the README: one machine, ten runs each in a fresh",
        "process, the kernel's wall time (mean ± standard deviation). Of the PPY rows,",
        "`ppy build` is the wrap-semantics artifact and `ppy run` keeps Python-integer",
        "semantics — overflow is guarded and falls back to arbitrary precision.",
        "",
        _readme_table(),
        "",
        "## Six judge problems, whole process",
        "",
        "Nothing inside the programs is instrumented. These are wall times of the whole",
        "process, input reading and interpreter startup included; each problem reads its",
        "input from standard input.",
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
            "The `ppy run` column includes the first run's build into the cache, which is",
            "why its deviation is wide; from the second run on it is the launcher alone.",
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
