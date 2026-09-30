"""`ppy explain --summary`: how much of a project goes native, and what keeps the rest out.

The categories are what a user acts on, so they are tested as text: an effect
list becomes one category per effect in plain words, a type the checker could
not tell is a missing annotation, an unknown shape keeps its words with the
names taken out, and a generic function is not a blocker. The command itself
is run over a small project: every function lands in one tier, the reasons are
grouped with where they come from, a file that does not parse is reported
rather than fatal, and `--json` carries all of it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.driver.summary import FunctionOutcome, Summary, categorize, render_summary

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")


def test_an_effect_list_is_one_category_per_effect_in_plain_words():
    found = categorize("has effects that must run on CPython: IO, ReadGlobal")
    assert [category for category, _, _ in found] == [
        "does I/O natively out of reach (sockets, `os`, binary files)",
        "reads a module global that can change",
    ]
    assert all(page.endswith("/guide/effects/") for _, _, page in found)
    assert all(hint for _, hint, _ in found)


def test_a_type_the_checker_could_not_tell_is_a_missing_annotation():
    ((category, hint, _),) = categorize("parameter `xs` is `<unknown>`, which has no native ABI")
    assert category == "a parameter or result with no annotation the checker could infer"
    assert "ppy convert" in hint
    ((category, _, _),) = categorize(
        "parameter `xs` is `dict[float, int]`, which has no native ABI"
    )
    assert category == "a parameter of type `dict[float, int]`"


def test_an_unknown_reason_keeps_its_words_without_the_names():
    ((category, hint, page),) = categorize("`grid` is not filled from natively")
    assert category == "`…` is not filled from natively"
    assert (hint, page) == ("", "")


def test_opaque_operator_reasons_name_the_code_they_come_from():
    assert categorize("a chained comparison's operand is a name or a number here")[0][0].startswith(
        "a chained comparison"
    )
    assert categorize("integer operator has no native lowering")[0][0].startswith(
        "an integer operator"
    )
    assert categorize("`isinstance` of a `int` depends on the value")[0][0].startswith(
        "`isinstance`"
    )
    assert categorize("a generator holds a buffer or a vector, which its frame cannot")[0][
        2
    ].endswith("/guide/exceptions-and-generators/")


def test_what_keeps_a_closure_in_python_has_a_hint_and_the_closures_page():
    for reason in (
        "a lambda whose type native code does not know",
        "a lambda with defaults or special parameters",
        "a nested `a` with decorators, defaults, or special parameters has no native lowering",
        "a function value takes positional arguments natively",
        "`self.step` is not a function native code can call",
    ):
        ((category, hint, page),) = categorize(reason)
        assert hint and page.endswith("/guide/closures/"), (reason, category)


def test_generics_are_not_blockers_and_unknown_calls_are_named():
    def outcome(reason: str, unknown: tuple[str, ...] = ()) -> FunctionOutcome:
        return FunctionOutcome("m.f", "m", "m.py", 1, 5, "python", reason, unknown)

    summary = Summary(
        [
            outcome("a generic function is specialized where it is called; no entry point"),
            outcome("has effects that must run on CPython: ExternalUnknown", ("requests.get",)),
            outcome(
                "has effects that must run on CPython: ExternalUnknown",
                ("requests.get", "self.cache.lookup"),
            ),
        ],
        {},
    )
    (blocker,) = summary.blockers()
    assert blocker.functions == 2 and blocker.statements == 10
    assert blocker.callees == {"requests.get": 2, ".lookup": 1}
    text = render_summary(summary)
    assert "1 of the Python functions are generic" in text
    assert "most often: `requests.get` (2), `.lookup` (1)" in text


PROJECT = {
    "fast.ppy": """
        def total(n: int) -> int:
            s = 0
            for i in range(n):
                s += i * i
            return s


        def shout(n: int) -> None:
            print(total(n))
    """,
    "counts.ppy": """
        import random


        def roll(n: int) -> int:
            return sum(random.randint(1, 6) for _ in range(n))


        def reseed(text: str) -> float:
            random.seed(text)
            return random.random()
    """,
    "broken.ppy": "def oops(:\n    pass\n",
}


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    for name, text in PROJECT.items():
        (tmp_path / name).write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return tmp_path


def _explain(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "explain", "--summary", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@requires_llvm
def test_the_summary_places_every_function_and_says_why(tmp_path: Path):
    _project(tmp_path)
    done = _explain(tmp_path, "--json", ".")
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    tiers = {f["qualname"].rpartition(".")[2]: f["tier"] for f in report["functions"]}
    assert tiers["total"] == "native"
    assert tiers["shout"] in {"python", "internal"}
    # Drawing random numbers is native since 0.6.0: CPython's generator. A
    # string seed is still Python's.
    assert tiers["roll"] != "python"
    assert tiers["reseed"] == "python"
    categories = {b["category"] for b in report["blockers"]}
    assert "does I/O (`print`, `input`, files)" in categories or tiers["shout"] == "internal"
    assert any("broken.ppy" in path for path in report["failed"])
    counted = sum(t["functions"] for t in report["totals"].values())
    assert counted == len(report["functions"])


@requires_llvm
def test_the_text_summary_names_places_and_is_limited(tmp_path: Path):
    _project(tmp_path)
    done = _explain(tmp_path, "--limit", "1", "--modules")
    assert done.returncode == 0, done.stderr
    text = done.stdout
    assert "native, called from Python" in text
    assert "counts.ppy:" in text or "fast.ppy:" in text
    assert "by module" in text
    assert "could not be analyzed" in text


def test_explain_without_summary_still_takes_one_location(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "explain", "a", "b"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert done.returncode == 2
    assert "one location" in done.stdout + done.stderr


def test_a_bare_callable_is_a_gradual_callable_not_an_unknown_type(write, codes):
    """TheAlgorithms other/pipeline.py: `f: Callable` was E1101 ("not a type
    the project can analyze") in either mode; it is `Callable[..., Any]`,
    reported as a bare generic is."""
    path = write(
        "calls.ppy",
        """
        from collections.abc import Callable


        def twice(f: Callable, x: int) -> int:
            return f(f(x))
        """,
    )
    assert "E1101" not in codes(path, strict=False)
    assert "E1101" not in codes(path)


def test_an_empty_display_takes_the_type_of_what_fills_it(write, codes):
    """TheAlgorithms strings/min_window_substring.py and ciphers/des_ecb.py:
    `seen = {}` filled by `seen[ch] = 1` in one branch refused `seen[ch] += 1`
    in the other as `Never + 1`, and `[] + names` as `list[Never] + list[str]`."""
    path = write(
        "fills.ppy",
        """
        def counts(text: str) -> int:
            seen = {}
            for ch in text:
                if ch not in seen:
                    seen[ch] = 1
                else:
                    seen[ch] += 1
            return len(seen)


        def padded(names: list[str]) -> list[str]:
            return [] + names
        """,
    )
    assert "E1302" not in codes(path, strict=False)


@requires_llvm
def test_a_nested_function_runs_where_the_function_around_it_runs(tmp_path: Path):
    """A closure lowers with the function it is defined in, so it has no entry
    of its own; its statements count once, under itself."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    (tmp_path / "nest.ppy").write_text(
        textwrap.dedent(
            """
            def scaled(n: int, k: int) -> int:
                def times(x: int) -> int:
                    return x * k

                total = 0
                for i in range(n):
                    total += times(i)
                return total


            def shown(n: int) -> None:
                def show(x: int) -> None:
                    print(x)

                show(n)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    done = _explain(tmp_path, "--json", ".")
    assert done.returncode == 0, done.stderr
    functions = {f["qualname"].rpartition(".")[2]: f for f in json.loads(done.stdout)["functions"]}
    assert functions["times"]["tier"] in {"native", "internal"}
    assert functions["show"]["tier"] == "python"
    assert functions["show"]["reason"] == "the function around it stays in Python"
    assert functions["scaled"]["statements"] == 5
    assert functions["times"]["statements"] == 1
