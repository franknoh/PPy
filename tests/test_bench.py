"""The benchmark harness and the tables it renders (examples/15_algorithms).

These numbers are a claim the READMEs make, so what turns measurements into
a table -- and what refuses to record one -- is held to the same standard as
the compiler that produced them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ROOT / "examples" / "15_algorithms"


def _load(path: Path, name: str) -> ModuleType:
    """Import a script that lives outside any package, by its path."""
    sys.path.insert(0, str(ALGORITHMS))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ALGORITHMS))


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    return _load(ALGORITHMS / "bench.py", "bench_under_test")


@pytest.fixture(scope="module")
def refresh() -> ModuleType:
    return _load(ROOT / "scripts" / "refresh.py", "refresh_under_test")


def _row(**paths: float) -> dict:
    return {name: {"mean": mean, "stdev": 0.5} for name, mean in paths.items()}


def test_a_path_that_answers_differently_between_runs_is_reported(bench):
    """Keeping the last answer would hide a path that is right most of the time."""
    answers, by_path = bench.collapse(
        {"plain": ["14200", "14200"], "ppy build": ["14200", "0", "14200"]}
    )
    assert answers == ["0", "14200"], "the disagreement survives into the verdict"
    assert by_path == {"plain": "14200", "ppy build": ["0", "14200"]}


def test_paths_that_always_agree_collapse_to_the_answer(bench):
    answers, by_path = bench.collapse({"plain": ["7"] * 5, "C scanf": ["7"] * 5})
    assert answers == ["7"]
    assert by_path == {"plain": "7", "C scanf": "7"}


def test_the_standalone_column_appears_only_where_there_is_one(bench):
    with_variant = dict(bench._commands("nqueens", "nqueens"))
    without = dict(bench._commands("kmp", None))
    assert "standalone" in with_variant
    assert "standalone" not in without
    assert list(without) == ["plain", "ppy run", "ppy build", "C scanf"]


def test_a_missing_executable_says_so_rather_than_failing_to_open_it(bench, tmp_path: Path):
    data = tmp_path / "input.txt"
    data.write_text("1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no executable"):
        bench._wall(["./dist/absent"], tmp_path, data, {})


def test_the_overview_renders_without_a_c_reference(refresh):
    """No C column is a machine without gcc, not a table that cannot be drawn."""
    problems = {"15a_nqueens": _row(plain=300.0, **{"ppy build": 45.0, "standalone": 7.0})}
    table = refresh._overview(problems)
    assert "| 300.0 ms | 45.0 ms | 7.0 ms | — |" in table
    assert "**" not in table, "nothing is bold without a reference to beat"


def test_the_overview_bolds_only_what_beat_the_c_reference(refresh):
    problems = {
        "15a_nqueens": _row(plain=300.0, **{"ppy build": 45.0, "standalone": 7.0, "C scanf": 5.0}),
        "15d_segment_tree": _row(
            plain=710.0, **{"ppy build": 115.0, "standalone": 26.0, "C scanf": 55.0}
        ),
    }
    lines = refresh._overview(problems).splitlines()
    assert "**" not in lines[2], "N-Queens does not beat C"
    assert "**26.0 ms**" in lines[3] and "**115.0 ms**" not in lines[3]


def test_a_problem_table_marks_the_fastest_path(refresh):
    row = _row(plain=300.0, **{"ppy run": 1700.0, "ppy build": 45.0, "C scanf": 5.0})
    table = refresh._table(row)
    assert "| C (`gcc -O3`, `scanf`) | **5.0 ± 0.5 ms** |" in table
    assert "| `ppy build --standalone` |" not in table, "an unmeasured path gets no row"


def test_an_incomplete_machine_does_not_overwrite_a_whole_record(refresh, monkeypatch, tmp_path):
    """A record missing a column would read as a result rather than a gap."""
    recorded = tmp_path / "measurements.json"
    whole = {
        "environment": {"processor": "some cpu"},
        "problems": {"15a_nqueens": {**_row(plain=1.0, **{"C scanf": 2.0}), "answers": ["7"]}},
    }
    recorded.write_text(json.dumps(whole), encoding="utf-8")
    monkeypatch.setattr(refresh, "RECORDED", recorded)

    stub = ModuleType("bench")
    stub.available = lambda: {"ppy build": "no shared libpython to embed"}
    stub.measure = lambda: {"15a_nqueens": {**_row(plain=1.0), "answers": ["7"]}}
    stub.environment = lambda: {"processor": "some cpu"}
    monkeypatch.setitem(sys.modules, "bench", stub)

    lines, fatal = refresh.refresh_measurements(write=True)
    assert fatal, "a partial run is not a baseline"
    assert any("no shared libpython" in line for line in lines)
    assert json.loads(recorded.read_text()) == whole, "the whole record is still there"


def test_a_partial_run_still_writes_the_raw_numbers_it_took(refresh, monkeypatch, tmp_path):
    """`--record` is this run's evidence, kept whether or not it is a baseline."""
    monkeypatch.setattr(refresh, "RECORDED", tmp_path / "absent.json")
    stub = ModuleType("bench")
    stub.available = lambda: {"C scanf": "gcc is not on PATH"}
    stub.measure = lambda: {"15a_nqueens": {**_row(plain=1.0), "answers": ["7"]}}
    stub.environment = lambda: {"processor": "some cpu"}
    monkeypatch.setitem(sys.modules, "bench", stub)

    raw = tmp_path / "run.json"
    refresh.refresh_measurements(write=True, record=raw)
    assert json.loads(raw.read_text())["problems"]["15a_nqueens"]["plain"]["mean"] == 1.0


def test_a_disagreement_fails_even_when_asked_to_record(refresh, monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "RECORDED", tmp_path / "absent.json")
    stub = ModuleType("bench")
    stub.available = dict  # nothing missing
    stub.measure = lambda: {"15a_nqueens": {**_row(plain=1.0), "answers": ["7", "8"]}}
    stub.environment = lambda: {"processor": "some cpu"}
    monkeypatch.setitem(sys.modules, "bench", stub)

    lines, fatal = refresh.refresh_measurements(write=True)
    assert fatal
    assert any("disagree" in line for line in lines)
    assert not (tmp_path / "absent.json").exists()


def test_an_incomplete_machine_fails_even_when_only_reporting(refresh, monkeypatch, tmp_path):
    """Nothing to record and nothing to compare: a partial run is not a result."""
    monkeypatch.setattr(refresh, "RECORDED", tmp_path / "absent.json")
    stub = ModuleType("bench")
    stub.available = lambda: {"standalone": "no C compiler (cc, gcc, or clang) is on PATH"}
    stub.measure = lambda: {"15a_nqueens": {**_row(plain=1.0), "answers": ["7"]}}
    stub.environment = lambda: {"processor": "some cpu"}
    monkeypatch.setitem(sys.modules, "bench", stub)

    lines, fatal = refresh.refresh_measurements(write=False)
    assert fatal, "a scheduled run must not report success on a partial machine"
    assert any("no C compiler" in line for line in lines)


def test_the_raw_record_may_not_be_the_baseline(refresh, tmp_path):
    """`--record` writes what was measured; the baseline writes what was judged."""
    assert refresh._same_file(refresh.RECORDED, Path(str(refresh.RECORDED)))
    assert not refresh._same_file(tmp_path / "run.json", refresh.RECORDED)
    # Neither existing yet is the ordinary case, and still resolves.
    assert refresh._same_file(tmp_path / "run.json", tmp_path / "sub" / ".." / "run.json")
