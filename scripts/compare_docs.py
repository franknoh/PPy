"""Measure every example comparison and keep the READMEs' tables to the numbers.

Each comparison in `MANIFEST` names an example folder, the programs in its
`compare/` directory with the command that runs each, and how the README
labels the columns. This script builds what needs building, runs every
program through `examples/compare.py` -- which refuses to time programs that
do not agree on the answer -- and writes the table between the
`<!-- compare:start -->` and `<!-- compare:end -->` markers of the README,
recording the run in `compare/measurements.json` beside the programs.

    python scripts/compare_docs.py            # report what drifted
    python scripts/compare_docs.py --write    # measure and rewrite the tables
    python scripts/compare_docs.py 38_cuda    # one comparison only

The counterparts want toolchains this repository does not carry; where they
live is read from the environment, with this machine's defaults:
`PPY_COMPARE_PYTHON` (a Python with Numba, CuPy, Taichi, Triton, numexpr,
Cython), `PPY_JAX_PYTHON` (one with JAX and PyTorch), `PPY_CODON`,
`PPY_MOJO`, `PPY_NVCC`, `PPY_CARGO`. A comparison whose tools are missing is
skipped and said so; a table is never written from a partial run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
import compare  # noqa: E402  pylint: disable=wrong-import-position

#: A measurement this far from the recorded one is drift rather than noise.
TOLERANCE = 0.25
RUNS = 5
START = "<!-- compare:start -->"
END = "<!-- compare:end -->"

TOOLS = {
    "CV": os.environ.get("PPY_COMPARE_PYTHON", "/tmp/compare-venv/bin/python"),
    "JV": os.environ.get("PPY_JAX_PYTHON", "/tmp/ppy-bench/.venv313/bin/python"),
    "CODON": os.environ.get("PPY_CODON", str(Path.home() / ".codon/bin/codon")),
    "MOJO": os.environ.get("PPY_MOJO", "/tmp/compare-venv/bin/mojo"),
    "NVCC": os.environ.get("PPY_NVCC", "/usr/local/cuda/bin/nvcc"),
    "CARGO": os.environ.get("PPY_CARGO", str(Path.home() / ".cargo/bin/cargo")),
    "PPY": f"{sys.executable} -m ppy_compiler",
}
#: Where WSL keeps the CUDA driver library, which Taichi needs on the path.
DRIVER_LIBS = "/usr/lib/wsl/lib"


@dataclass
class Comparison:
    """One README table: the programs behind it and how it names them."""

    folder: str
    programs: list[tuple[str, str]]
    labels: dict[str, str]
    reference: str | None = None
    builds: list[str] = field(default_factory=list)
    rows: dict[str, str] = field(default_factory=dict)
    needs: tuple[str, ...] = ()


MANIFEST: dict[str, Comparison] = {
    "07_parallel": Comparison(
        "07_parallel",
        [
            ("ppy", "{PPY} run compare/fused_bench.ppy"),
            ("numpy", "{CV} compare/fused_numpy.py"),
            ("numexpr", "{CV} compare/fused_numexpr.py"),
            ("numba", "{CV} compare/fused_numba.py"),
            ("jax", "{JV} compare/fused_jax.py"),
        ],
        {
            "ppy": "PPY",
            "numpy": "NumPy",
            "numexpr": "numexpr",
            "numba": "Numba `prange`",
            "jax": "JAX `jit`",
        },
        reference="numpy",
        needs=("CV", "JV"),
    ),
    "12_buffers_and_jit": Comparison(
        "12_buffers_and_jit",
        [
            ("ppy", "{PPY} run compare/kernels_bench.ppy"),
            ("numba", "{CV} compare/kernels_numba.py"),
            (
                "cython",
                (
                    '{CV} -c \'import sys; sys.path.insert(0, "compare"); '
                    "import kernels_cython; kernels_cython.main()'"
                ),
            ),
            ("numpy", "{CV} compare/kernels_numpy.py"),
            ("c", "./compare/build/kernels_c"),
        ],
        {
            "ppy": "PPY `ppy run`",
            "numba": "Numba `@njit`",
            "cython": "Cython",
            "numpy": "NumPy",
            "c": "C (the loop alone)",
        },
        builds=[
            "mkdir -p compare/build",
            (
                'cd compare && {CV} -c "import sys; '
                "sys.argv = ['cythonize', '-q', '-i', '-3', 'kernels_cython.pyx']; "
                'from Cython.Build.Cythonize import main; main()"'
            ),
            "gcc -O3 compare/kernels.c -o compare/build/kernels_c",
        ],
        needs=("CV",),
    ),
    "15_algorithms": Comparison(
        "15_algorithms",
        [
            ("ppy run", "{PPY} run algorithms.ppy"),
            ("ppy build", "./compare/build/dist/algorithms"),
            ("numba", "{CV} compare/algorithms_numba.py"),
            ("mojo", "./compare/build/algorithms_mojo"),
            ("codon", "{CODON} run -release compare/algorithms_codon.py"),
        ],
        {
            "ppy run": "`ppy run`",
            "ppy build": "`ppy build --unsafe`",
            "numba": "Numba `@njit`",
            "mojo": "Mojo",
            "codon": "Codon",
        },
        builds=[
            "mkdir -p compare/build",
            "{PPY} build --unsafe algorithms.ppy -o compare/build/dist >/dev/null",
            "{MOJO} build -O3 compare/algorithms.mojo -o compare/build/algorithms_mojo 2>/dev/null",
        ],
        needs=("CV", "MOJO", "CODON"),
    ),
    "35_parallel_range": Comparison(
        "35_parallel_range",
        [
            ("ppy", "{PPY} run compare/ranges_bench.ppy"),
            ("numba", "{CV} compare/ranges_numba.py"),
            ("taichi", "{CV} compare/ranges_taichi.py"),
            ("mojo", "./compare/build/ranges_mojo"),
            ("numpy", "{CV} compare/ranges_numpy.py"),
        ],
        {
            "ppy": "PPY `ppy run`",
            "numba": "Numba `prange`",
            "taichi": "Taichi",
            "mojo": "Mojo",
            "numpy": "NumPy",
        },
        reference="numpy",
        builds=[
            "mkdir -p compare/build",
            "{MOJO} build -O3 compare/ranges.mojo -o compare/build/ranges_mojo 2>/dev/null",
        ],
        rows={"dot": "dot, in order", "dot_relaxed": "dot, reassociated", "fill": "fill, serial"},
        needs=("CV", "MOJO"),
    ),
    "36_autodiff": Comparison(
        "36_autodiff",
        [
            ("ppy", "{PPY} run compare/gradients_bench.ppy"),
            ("jax", "{JV} compare/gradients_jax.py"),
            ("torch", "{JV} compare/gradients_torch.py"),
        ],
        {"ppy": "PPY", "jax": "JAX `vmap` + `jit`", "torch": "PyTorch `torch.func`"},
        needs=("JV",),
    ),
    "38_cuda": Comparison(
        "38_cuda",
        [
            ("ppy", "{PPY} run compare/saxpy_bench.ppy"),
            ("cupy", "{CV} compare/saxpy_cupy.py"),
            ("numba", "{CV} compare/saxpy_numba.py"),
            ("mojo", "./compare/build/saxpy_mojo"),
            ("cudac", "./compare/build/saxpy_cu"),
        ],
        {
            "ppy": "PPY `cuda.launch`",
            "cupy": "CuPy",
            "numba": "Numba CUDA",
            "mojo": "Mojo",
            "cudac": "CUDA C",
        },
        builds=[
            "mkdir -p compare/build",
            "{NVCC} -O3 -o compare/build/saxpy_cu compare/saxpy.cu",
            "{MOJO} build -O3 compare/saxpy.mojo -o compare/build/saxpy_mojo 2>/dev/null",
        ],
        rows={
            "saxpy": "saxpy, arrays on the device",
            "block_max": "block max, arrays on the device",
            "saxpy with copies": "saxpy, arrays copied in and out per launch",
            "block_max with copies": "block max, array copied in per launch",
        },
        needs=("CV", "MOJO", "NVCC"),
    ),
    "44_tile": Comparison(
        "44_tile",
        [
            ("ppy", "{PPY} run compare/tiles_bench.ppy"),
            ("triton", "{CV} compare/tiles_triton.py"),
            ("taichi", "{CV} compare/tiles_taichi.py"),
        ],
        {"ppy": "PPY `tile.launch`", "triton": "Triton", "taichi": "Taichi"},
        rows={
            "saxpy": "saxpy, arrays on the device",
            "block_max": "block max, arrays on the device",
            "saxpy with copies": "saxpy, arrays copied in and out per launch",
            "block_max with copies": "block max, array copied in per launch",
        },
        needs=("CV",),
    ),
    "43_regex": Comparison(
        "43_regex",
        [
            ("ppy", "{PPY} run compare/patterns_bench.ppy"),
            ("re", "{CV} compare/patterns_re.py"),
            ("rust", "./compare/target/release/patterns"),
        ],
        {"ppy": "PPY `ppy run`", "re": "CPython `re`", "rust": "Rust `regex`"},
        builds=["{CARGO} build -q --release --manifest-path compare/Cargo.toml"],
        needs=("CV", "CARGO"),
    ),
}


def _missing(comparison: Comparison) -> list[str]:
    return [
        f"{name} ({TOOLS[name]})"
        for name in comparison.needs
        if shutil.which(TOOLS[name]) is None and not Path(TOOLS[name]).is_file()
    ]


def _environment() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return {
        "commit": head.stdout.strip(),
        "processor": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": RUNS,
    }


def measure(comparison: Comparison) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    """Every tool's mean and spread per kernel, or the problems that forbid a table."""
    folder = EXAMPLES / comparison.folder
    env = {**os.environ}
    env["LD_LIBRARY_PATH"] = ":".join(p for p in (DRIVER_LIBS, env.get("LD_LIBRARY_PATH")) if p)
    for step in comparison.builds:
        done = subprocess.run(
            step.format(**TOOLS),
            shell=True,
            cwd=folder,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if done.returncode != 0:
            tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else step
            return {}, [f"build failed: {tail}"]
    names = [name for name, _ in comparison.programs]
    outcomes: dict[str, list[compare.Outcome]] = {name: [] for name in names}
    previous = os.getcwd()
    os.chdir(folder)
    try:
        os.environ["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
        for name, command in comparison.programs:
            for _ in range(RUNS):
                outcomes[name].append(compare.run(command.format(**TOOLS)))
    finally:
        os.chdir(previous)
    problems = compare.check(names, outcomes, comparison.reference or names[0])
    if problems:
        return {}, problems
    measured: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        measured[name] = {}
        labels = [label for outcome in outcomes[name] for label in outcome.timings]
        for label in dict.fromkeys(labels):
            values = [o.timings[label] for o in outcomes[name] if label in o.timings]
            measured[name][label] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return measured, []


def render(comparison: Comparison, measured: dict) -> str:
    """The README's table: the rows in the order the programs print them."""
    names = [name for name, _ in comparison.programs]
    rows: list[str] = []
    for name in names:
        for label in measured[name]:
            if label not in rows:
                rows.append(label)
    smallest = min((m["mean"] for name in names for m in measured[name].values()), default=1.0)
    places = 2 if smallest >= 0.1 else 4
    lines = [
        "| | " + " | ".join(comparison.labels[name] for name in names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for label in rows:
        cells = []
        means = [measured[name][label]["mean"] for name in names if label in measured[name]]
        low = min(means) if len(means) > 1 else None
        for name in names:
            found = measured[name].get(label)
            if found is None:
                cells.append("—")
                continue
            text = f"{found['mean']:.{places}f} ± {found['stdev']:.{places}f}"
            cells.append(f"**{text}**" if low is not None and found["mean"] == low else text)
        lines.append(f"| {comparison.rows.get(label, label)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _drift(recorded: dict, measured: dict) -> list[str]:
    out = []
    for name, kernels in measured.items():
        for label, found in kernels.items():
            was = recorded.get(name, {}).get(label, {}).get("mean")
            if was is None:
                out.append(f"{name} / {label}: new, {found['mean']:.2f} ms")
            elif abs(found["mean"] - was) > TOLERANCE * max(was, 1e-9):
                out.append(f"{name} / {label}: {was:.2f} ms -> {found['mean']:.2f} ms")
    return out


def _place(readme: Path, table: str, write: bool) -> str | None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return f"{readme.relative_to(ROOT)}: no {START} / {END} markers"
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    wanted = f"{head}{START}\n{table}\n{END}{tail}"
    if wanted == text:
        return None
    if write:
        readme.write_text(wanted, encoding="utf-8")
    return f"{readme.relative_to(ROOT)}: the table is behind the measurements"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("names", nargs="*", help="which comparisons; every one by default")
    parser.add_argument("--write", action="store_true", help="rewrite the tables and records")
    options = parser.parse_args()
    chosen = options.names or list(MANIFEST)
    unknown = [name for name in chosen if name not in MANIFEST]
    if unknown:
        parser.error(f"no such comparison: {', '.join(unknown)}; one of {', '.join(MANIFEST)}")
    status = 0
    for name in chosen:
        comparison = MANIFEST[name]
        missing = _missing(comparison)
        if missing:
            print(f"[SKIP] {name}: missing {', '.join(missing)}")
            continue
        measured, problems = measure(comparison)
        if problems:
            status = 1
            print(f"[FAIL] {name}: no table from this run")
            for problem in problems:
                print(f"       {problem}")
            continue
        record = EXAMPLES / comparison.folder / "compare" / "measurements.json"
        stored = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
        drifted = _drift(stored.get("timings", {}), measured)
        moved = _place(
            EXAMPLES / comparison.folder / "README.md", render(comparison, measured), options.write
        )
        if options.write:
            record.write_text(
                json.dumps({"environment": _environment(), "timings": measured}, indent=1) + "\n",
                encoding="utf-8",
            )
        verdict = "written" if options.write else ("drifted" if drifted or moved else "ok")
        print(f"[{'OK' if verdict != 'drifted' else 'DRIFT'}] {name}: {verdict}")
        for line in [*drifted, *([moved] if moved else [])]:
            print(f"       {line}")
        if verdict == "drifted":
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
