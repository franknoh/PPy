"""Randomized differential testing: three paths, one answer (spec 33.1).

Each seed is a deterministic program over the subset the compiler makes hard
promises about; plain CPython is the oracle and both backends must match it
byte for byte. A seed that ever fails goes straight into the fixed list below
and stays there.

Set PPY_FUZZ_SEEDS to sweep a wider range locally:

    PPY_FUZZ_SEEDS=0:500 pytest tests/test_differential_fuzz.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ppy_compiler.testing.differential import compare, run_paths
from ppy_compiler.testing.generate import generate_program

#: Always-on seeds: a spread of shapes, plus every seed that ever regressed.
#: 5 caught a CSE temporary reused across pass reruns; 23 caught the same as
#: a silent wrong answer; 37 caught copy propagation forwarding a name into a
#: reassigned parameter; 126 caught an unresolved copy chain whose middle
#: binding was deleted.
_FIXED_SEEDS = (*range(24), 37, 126)


def _seeds() -> tuple[int, ...]:
    wanted = os.environ.get("PPY_FUZZ_SEEDS")
    if not wanted:
        return _FIXED_SEEDS
    start, _, stop = wanted.partition(":")
    return tuple(range(int(start), int(stop or int(start) + 1)))


@pytest.mark.parametrize("seed", _seeds())
def test_generated_program_agrees_on_all_paths(seed: int, tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    program = tmp_path / f"fuzz_{seed}.ppy"
    program.write_text(generate_program(seed), encoding="utf-8")

    outcomes = run_paths(program)
    verdict = compare(program, outcomes)
    baseline = outcomes["python"]
    assert baseline.exit_code == 0, f"seed {seed} generated an invalid program:\n{baseline.stderr}"
    assert not verdict.mismatches, (
        f"seed {seed} disagrees: {verdict.mismatches}\n--- program ---\n{generate_program(seed)}"
    )


def test_the_generator_is_deterministic():
    assert generate_program(7) == generate_program(7)
    assert generate_program(7) != generate_program(8)
