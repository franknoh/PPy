"""CLI surface: convert, check, fmt, explain, inspect, cache (spec 4.3)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.driver.cli import main


def _ppy(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    return tmp_path


def test_version_and_help(capsys):
    assert main(["--version"]) == 0
    assert "ppy" in capsys.readouterr().out
    assert main([]) == 0


def test_check_reports_success(workspace: Path):
    (workspace / "ok.ppy").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    result = _ppy(["check", "ok.ppy"], workspace)
    assert result.returncode == 0
    assert "no errors" in result.stderr


def test_check_reports_failure_with_a_code_and_span(workspace: Path):
    (workspace / "bad.ppy").write_text("def f(x):\n    return x\n", encoding="utf-8")
    result = _ppy(["check", "bad.ppy"], workspace)
    assert result.returncode == 1
    assert "error[E1201]" in result.stderr
    assert "bad.ppy:1:1" in result.stderr
    assert "= help:" in result.stderr


def test_check_scans_a_directory(workspace: Path):
    (workspace / "a.ppy").write_text("def a() -> int:\n    return 1\n", encoding="utf-8")
    (workspace / "b.ppy").write_text("def b() -> int:\n    return 2\n", encoding="utf-8")
    result = _ppy(["check", "."], workspace)
    assert result.returncode == 0
    assert "2 module(s)" in result.stderr


def test_convert_writes_a_ppy_file_and_preserves_the_source(workspace: Path):
    source = workspace / "mod.py"
    source.write_text(
        textwrap.dedent(
            '''
            """Docstring."""
            from __future__ import annotations


            def square(x):
                # keep me
                return x * x


            answer = square(7)
            '''
        ).lstrip("\n"),
        encoding="utf-8",
    )
    original = source.read_text(encoding="utf-8")
    result = _ppy(["convert", "mod.py"], workspace)
    assert result.returncode == 0

    converted = (workspace / "mod.ppy").read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == original
    assert "def square(x: int) -> int:" in converted
    assert "# keep me" in converted
    assert '"""Docstring."""' in converted
    assert "answer: int = square(7)" in converted


def test_convert_inserts_ppy_after_docstring_and_future(workspace: Path):
    source = workspace / "order.py"
    source.write_text(
        '"""Doc."""\n'
        "from __future__ import annotations\n"
        "\n"
        "import math\n"
        "\n"
        "def area(r):\n"
        "    return math.pi * r * r\n"
        "\n"
        "X = area(2.0)\n",
        encoding="utf-8",
    )
    _ppy(["convert", "order.py"], workspace)
    lines = (workspace / "order.ppy").read_text(encoding="utf-8").splitlines()
    assert lines[0] == '"""Doc."""'
    assert lines[1] == "from __future__ import annotations"
    assert "import ppy" in lines
    # Standard library first, then `ppy`, and all of it ahead of any statement.
    assert lines.index("import math") < lines.index("import ppy")
    assert lines.index("import ppy") < lines.index("def area(r: float) -> float:")


def test_convert_omits_ppy_when_nothing_would_use_it(workspace: Path):
    """An unused import is a lint finding the source did not have."""
    (workspace / "plain.py").write_text("X = 1\nprint(X)\n", encoding="utf-8")
    _ppy(["convert", "plain.py"], workspace)
    converted = (workspace / "plain.ppy").read_text(encoding="utf-8")
    assert "import ppy" not in converted
    assert "X: Final[int] = 1" in converted


def test_convert_refuses_to_clobber_without_force(workspace: Path):
    (workspace / "exists.py").write_text("X = 1\n", encoding="utf-8")
    (workspace / "exists.ppy").write_text("# handwritten\n", encoding="utf-8")
    result = _ppy(["convert", "exists.py"], workspace)
    assert result.returncode == 1
    assert "--force" in result.stderr
    assert (workspace / "exists.ppy").read_text(encoding="utf-8") == "# handwritten\n"


def test_convert_in_place_replaces_the_source(workspace: Path):
    """The module becomes `.ppy` and the `.py` it came from is gone."""
    source = workspace / "inplace.py"
    source.write_text("def f(x):\n    return x + 1\n\n\nf(1)\n", encoding="utf-8")
    result = _ppy(["convert", "inplace.py", "--in-place"], workspace)
    assert result.returncode == 0
    assert not source.exists()
    converted = workspace / "inplace.ppy"
    assert "def f(x: int) -> int:" in converted.read_text(encoding="utf-8")


def test_convert_reports_uninferable_parameters(workspace: Path):
    (workspace / "vague.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    result = _ppy(["convert", "vague.py", "--dry-run"], workspace)
    assert "E1304" in result.stderr
    assert "Any" not in (workspace / "vague.py").read_text(encoding="utf-8")


def test_convert_rejects_dynamic_features(workspace: Path):
    """Strict conversion refuses to produce what `ppy check` would reject."""
    (workspace / "dyn.py").write_text("def f(src: str):\n    return eval(src)\n", encoding="utf-8")
    result = _ppy(["convert", "dyn.py"], workspace)
    assert result.returncode == 1
    assert "E1501" in result.stderr
    assert not (workspace / "dyn.ppy").exists()


def test_migrate_flags_dynamic_features(workspace: Path):
    """Migration converts the dynamic feature faithfully and says what remains."""
    (workspace / "dyn.py").write_text("def f(src: str):\n    return eval(src)\n", encoding="utf-8")
    result = _ppy(["migrate", "dyn.py", "--dry-run"], workspace)
    assert "E1504" in result.stderr
    assert "ppy.dynamic" in result.stderr


def test_converted_output_runs_on_all_paths(workspace: Path):
    (workspace / "roundtrip.py").write_text(
        textwrap.dedent(
            """
            def scale(values, factor):
                return [v * factor for v in values]


            def main():
                print(scale([1.0, 2.0], 2.0))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "roundtrip.py"], workspace).returncode == 0
    converted = workspace / "roundtrip.ppy"

    plain = subprocess.run(
        [sys.executable, converted.name], cwd=workspace, capture_output=True, text=True, check=False
    )
    optimized = _ppy([converted.name], workspace)
    assert plain.returncode == 0, plain.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert optimized.stdout == plain.stdout == "[2.0, 4.0]\n"


def test_fmt_normalizes_and_check_mode_reports(workspace: Path):
    target = workspace / "messy.ppy"
    target.write_text(
        "def a() -> int:\n    return 1\ndef b() -> int:\n    return 2\n", encoding="utf-8"
    )

    check = _ppy(["fmt", "messy.ppy", "--check"], workspace)
    assert check.returncode == 1
    assert "would reformat" in check.stderr

    assert _ppy(["fmt", "messy.ppy"], workspace).returncode == 0
    assert _ppy(["fmt", "messy.ppy", "--check"], workspace).returncode == 0


def test_fmt_reports_a_syntax_error(workspace: Path):
    (workspace / "broken.ppy").write_text("def f(:\n", encoding="utf-8")
    result = _ppy(["fmt", "broken.ppy"], workspace)
    assert result.returncode == 1
    assert "E1001" in result.stderr


def test_explain_reports_optimization_decisions(workspace: Path):
    (workspace / "ex.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def total(xs: list[float]) -> float:
                result: float = 0.0
                for x in xs:
                    result += x
                return result
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["explain", "ex.ppy:6"], workspace)
    assert result.returncode == 0
    assert "function: total" in result.stdout
    assert "optimization: O3" in result.stdout
    assert "effects:" in result.stdout
    assert "parallel:" in result.stdout
    assert "representation:" in result.stdout


def test_explain_reports_why_parallelization_was_rejected(workspace: Path):
    (workspace / "why.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            def log_and_total(xs: list[int]) -> int:
                result: int = 0
                for x in xs:
                    print(x)
                    result += x
                return result
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["explain", "log_and_total"], workspace)
    assert "parallel: rejected" in result.stdout
    assert "reason:" in result.stdout
    assert "I/O" in result.stdout


def test_explain_describes_a_diagnostic_code(workspace: Path):
    result = _ppy(["explain", "E1304"], workspace)
    assert result.returncode == 0
    assert "E1304" in result.stdout


def test_inspect_prints_generated_python(workspace: Path):
    (workspace / "gen.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\ndef f() -> int:\n    return 2 + 3\n", encoding="utf-8"
    )
    result = _ppy(["inspect", "gen.ppy"], workspace)
    assert result.returncode == 0
    assert "return 5" in result.stdout
    assert "@ppy.pure" not in result.stdout


def test_cache_status_clean_and_gc(workspace: Path):
    (workspace / "c.ppy").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    assert _ppy(["c.ppy"], workspace).returncode == 0

    status = _ppy(["cache", "status"], workspace)
    assert status.returncode == 0
    assert "entries:" in status.stdout

    assert _ppy(["cache", "gc"], workspace).returncode == 0
    assert _ppy(["cache", "clean"], workspace).returncode == 0
    assert _ppy(["clean"], workspace).returncode == 0
    assert not (workspace / ".ppy-cache").exists()


def test_doctor_reports_the_toolchain(workspace: Path):
    result = _ppy(["doctor"], workspace)
    assert result.returncode == 0
    assert "llvm backend" in result.stdout
    assert "plugins:" in result.stdout
    assert "numpy" in result.stdout


def test_program_arguments_reach_the_script(workspace: Path):
    (workspace / "args.ppy").write_text("import sys\n\nprint(sys.argv[1:])\n", encoding="utf-8")
    result = _ppy(["args.ppy", "--", "one", "two"], workspace)
    assert result.returncode == 0
    assert result.stdout.strip() == "['one', 'two']"


def test_program_arguments_may_look_like_compiler_flags(workspace: Path):
    """Past `--`, an argument belongs to the program even if it starts with `-`.

    `ppy` has no `--verbose`, so re-scanning the whole command line for flags
    makes the compiler reject an argument that was never addressed to it.
    """
    (workspace / "args.ppy").write_text("import sys\n\nprint(sys.argv[1:])\n", encoding="utf-8")
    result = _ppy(["args.ppy", "--", "--verbose", "-x"], workspace)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['--verbose', '-x']"

    both = _ppy(["-O", "0", "args.ppy", "--", "--verbose"], workspace)
    assert both.returncode == 0, both.stderr
    assert both.stdout.strip() == "['--verbose']"


def test_opt_level_flag_overrides_the_project_setting(workspace: Path):
    (workspace / "lvl.ppy").write_text(
        "import ppy\n\n\ndef f() -> int:\n    total: int = 0\n    for i in range(3):\n        total += i\n    return total\n",
        encoding="utf-8",
    )
    at_o3 = _ppy(["-O", "3", "inspect", "lvl.ppy"], workspace)
    at_o1 = _ppy(["-O", "1", "inspect", "lvl.ppy"], workspace)
    assert "for i in range" not in at_o3.stdout
    assert "for i in range" in at_o1.stdout


def test_differential_test_command(workspace: Path):
    (workspace / "same.ppy").write_text("print(1 + 1)\n", encoding="utf-8")
    result = _ppy(["test", "."], workspace)
    assert result.returncode == 0
    assert "matched plain CPython" in result.stderr


def test_missing_target_is_reported(workspace: Path):
    result = _ppy(["check", "nope.ppy"], workspace)
    assert result.returncode == 2
    assert "E1002" in result.stderr


def test_lsp_command_serves_a_session(workspace: Path):
    from ppy_compiler.lsp.protocol import encode

    (workspace / "a.ppy").write_text("def f(x: int) -> int:\n    return x\n", encoding="utf-8")
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"rootPath": str(workspace)}},
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
        {"jsonrpc": "2.0", "method": "exit"},
    ]
    result = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "lsp", "--root", "."],
        cwd=workspace,
        input=b"".join(encode(message) for message in messages),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert b'"serverInfo"' in result.stdout
    assert b'"hoverProvider":true' in result.stdout


def test_explain_reports_which_boundary_a_function_crosses(workspace: Path):
    (workspace / "boundary.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def square(x: int) -> int:
                return x * x


            @ppy.jit
            @ppy.pure
            def cube(x: int) -> int:
                return x * x * x
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _ppy(["explain", "square"], workspace)
    assert plain.returncode == 0
    assert "python boundary: a generated CPython-ABI wrapper" in plain.stdout

    jitted = _ppy(["explain", "cube"], workspace)
    assert "selects the specialization" in jitted.stdout
    assert "specializes on x" in jitted.stdout


def test_convert_propagates_types_through_a_call_chain(workspace: Path):
    (workspace / "chain.py").write_text(
        textwrap.dedent(
            """
            def scale(x):
                return x * 2.0


            def middle(x):
                return scale(x)


            def outer(x):
                return middle(x)


            print(outer(1.5))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "chain.py"], workspace).returncode == 0
    converted = (workspace / "chain.ppy").read_text(encoding="utf-8")
    assert "def outer(x: float) -> float:" in converted
    assert "def middle(x: float) -> float:" in converted
    assert "def scale(x: float) -> float:" in converted
    assert _ppy(["check", "chain.ppy"], workspace).returncode == 0


def test_convert_infers_instance_fields_and_leaves_self_alone(workspace: Path):
    (workspace / "box.py").write_text(
        textwrap.dedent(
            """
            class Box:
                def __init__(self, width, height):
                    self.width = width
                    self.height = height

                def area(self):
                    return self.width * self.height


            print(Box(2.0, 3.0).area())
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "box.py"], workspace).returncode == 0
    converted = (workspace / "box.ppy").read_text(encoding="utf-8")
    assert "def __init__(self, width: float, height: float) -> None:" in converted
    assert "self.width: float = width" in converted
    assert "self.height: float = height" in converted
    assert "def area(self) -> float:" in converted
    assert _ppy(["check", "box.ppy"], workspace).returncode == 0


def test_convert_moves_a_class_above_the_function_that_names_it(workspace: Path):
    """Quotes exist only because the class was not bound yet; move it and they go."""
    (workspace / "fwd.py").write_text(
        textwrap.dedent(
            """
            def first(node):
                return node


            class Node:
                def __init__(self, value):
                    self.value = value


            print(first(Node(1)).value)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "fwd.py"], workspace).returncode == 0
    converted = (workspace / "fwd.ppy").read_text(encoding="utf-8")
    assert "def first(node: Node) -> Node:" in converted
    assert converted.index("class Node:") < converted.index("def first(")
    assert _ppy(["check", "fwd.ppy"], workspace).returncode == 0


def test_convert_keeps_quotes_a_class_cannot_avoid(workspace: Path):
    """A method naming its own class cannot be reordered out of the problem."""
    (workspace / "self_ref.py").write_text(
        textwrap.dedent(
            """
            class Node:
                def __init__(self, value):
                    self.value = value

                def grown(self):
                    return Node(self.value + 1)


            print(Node(1).grown().value)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "self_ref.py"], workspace).returncode == 0
    converted = (workspace / "self_ref.ppy").read_text(encoding="utf-8")
    assert "def grown(self) -> 'Node':" in converted
    assert _ppy(["check", "self_ref.ppy"], workspace).returncode == 0


def test_convert_does_not_move_a_class_past_a_statement(workspace: Path):
    """Only `def` and `class` are safe to cross; a statement may have run."""
    (workspace / "guarded.py").write_text(
        textwrap.dedent(
            """
            def first(node):
                return node


            LIMIT = 3


            class Node:
                def __init__(self, value):
                    self.value = value


            print(first(Node(LIMIT)).value)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "guarded.py"], workspace).returncode == 0
    converted = (workspace / "guarded.ppy").read_text(encoding="utf-8")
    assert converted.index("LIMIT") < converted.index("class Node:")
    assert _ppy(["check", "guarded.ppy"], workspace).returncode == 0


def test_convert_attaches_purity_the_checker_proved(workspace: Path):
    (workspace / "pure.py").write_text(
        textwrap.dedent(
            """
            def double(x):
                return x * 2


            def shout(name):
                print(name)
                return name


            print(double(3), shout("hi"))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "pure.py"], workspace).returncode == 0
    converted = (workspace / "pure.ppy").read_text(encoding="utf-8")
    assert "@ppy.pure\ndef double" in converted
    assert "@ppy.pure\ndef shout" not in converted, "printing is not pure"
    assert _ppy(["check", "pure.ppy"], workspace).returncode == 0


def test_convert_promotes_an_indexed_list_parameter_to_a_buffer(workspace: Path):
    (workspace / "sums.py").write_text(
        textwrap.dedent(
            """
            def total(xs):
                out = 0.0
                for i in range(len(xs)):
                    out += xs[i]
                return out


            values = [float(i) for i in range(8)]
            print(total(values))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["convert", "sums.py", "--promote-buffers"], workspace)
    assert result.returncode == 0
    converted = (workspace / "sums.ppy").read_text(encoding="utf-8")
    assert "def total(xs: Buffer[float])" in converted
    assert "import array" in converted
    assert 'array.array("d", [float(i) for i in range(8)])' in converted
    assert _ppy(["check", "sums.ppy"], workspace).returncode == 0

    plain = subprocess.run(
        [sys.executable, "sums.py"], cwd=workspace, capture_output=True, text=True, check=False
    )
    built = _ppy(["sums.ppy"], workspace)
    assert built.stdout == plain.stdout


def test_convert_leaves_a_list_alone_without_the_flag(workspace: Path):
    (workspace / "keep.py").write_text(
        textwrap.dedent(
            """
            def total(xs):
                out = 0.0
                for i in range(len(xs)):
                    out += xs[i]
                return out


            values = [float(i) for i in range(8)]
            print(total(values))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "keep.py"], workspace).returncode == 0
    converted = (workspace / "keep.ppy").read_text(encoding="utf-8")
    # Read-only, so it widens to `Sequence`; without the flag it is still a
    # Python container rather than borrowed memory.
    assert "def total(xs: Sequence[float])" in converted
    assert "array.array" not in converted
    assert "Buffer" not in converted


def test_convert_says_what_blocks_a_promotion(workspace: Path):
    (workspace / "sliced.py").write_text(
        textwrap.dedent(
            """
            def total(xs, n):
                out = 0.0
                for i in range(n):
                    row = xs[i * 2:(i + 1) * 2]
                    for v in row:
                        out += v
                return out


            values = [float(i) for i in range(8)]
            print(total(values, 4))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["convert", "sliced.py", "--promote-buffers", "--dry-run"], workspace)
    assert "R3003" in result.stderr
    assert "is sliced" in result.stderr
    assert "list[float]" in result.stdout


def test_a_promoted_buffer_lowers_natively(workspace: Path):
    (workspace / "native.py").write_text(
        textwrap.dedent(
            """
            def total(xs):
                out = 0.0
                for i in range(len(xs)):
                    out += xs[i]
                return out


            values = [float(i) for i in range(8)]
            print(total(values))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "native.py", "--promote-buffers"], workspace).returncode == 0
    explained = _ppy(["explain", "total"], workspace)
    assert "llvm backend: native" in explained.stdout


def test_convert_formats_what_it_writes(workspace: Path):
    """Inserting annotations disturbs the blank lines around what it touches."""
    (workspace / "tight.py").write_text(
        "import math\ndef a(x):\n    return math.sqrt(x)\ndef b(x):\n    return a(x) + 1.0\nprint(b(4.0))\n",
        encoding="utf-8",
    )
    assert _ppy(["convert", "tight.py"], workspace).returncode == 0
    converted = (workspace / "tight.ppy").read_text(encoding="utf-8")
    assert "\n\n\n@ppy.pure\ndef a(" in converted
    assert "\n\n\n@ppy.pure\ndef b(" in converted


def test_convert_wraps_a_signature_that_grew_too_long(workspace: Path):
    (workspace / "wide.py").write_text(
        textwrap.dedent(
            """
            def combine(alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel):
                return alpha + bravo + charlie + delta + echo + foxtrot + golf + hotel


            print(combine(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "wide.py"], workspace).returncode == 0
    converted = (workspace / "wide.ppy").read_text(encoding="utf-8")
    assert all(len(line) <= 100 for line in converted.splitlines())
    assert "def combine(\n    alpha: float," in converted
    assert ") -> float:" in converted
    assert _ppy(["check", "wide.ppy"], workspace).returncode == 0


def test_conversion_adds_no_pylint_finding(workspace: Path):
    """Whatever the source scores, the converted file must score the same."""
    pytest.importorskip("pylint")
    (workspace / "linted.py").write_text(
        textwrap.dedent(
            '''
            """A module."""
            import math


            def area(radius):
                """Area of a circle."""
                return math.pi * radius * radius


            def total(radii):
                """Sum of areas."""
                out = 0.0
                for radius in radii:
                    out += area(radius)
                return out


            print(total([1.0, 2.0]))
            '''
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "linted.py"], workspace).returncode == 0
    (workspace / "converted.py").write_text(
        (workspace / "linted.ppy").read_text(encoding="utf-8"), encoding="utf-8"
    )

    def _findings(name: str) -> set[str]:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                "--enable=all",
                "--score=n",
                "--disable=import-error",
                name,
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            line.rsplit("(", 1)[-1].rstrip(")")
            for line in done.stdout.splitlines()
            if line.startswith(name)
        }

    assert _findings("converted.py") <= _findings("linted.py")


def test_inspect_shows_the_c_it_generated(workspace: Path):
    """The native path is IR plus the C boundary back into CPython."""
    (workspace / "hot.ppy").write_text(
        textwrap.dedent(
            """
            import ppy

            @ppy.pure
            @ppy.opt(3)
            def square(x: int) -> int:
                return x * x
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["inspect", "hot.ppy", "--backend", "llvm", "--ir"], workspace)
    assert result.returncode == 0, result.stderr
    assert "; ---- hot ----" in result.stdout
    assert "CPython ABI wrappers, C" in result.stdout
    assert "PyObject" in result.stdout


def test_a_per_function_opt_level_outranks_the_flag(workspace: Path):
    """`-O` sets the project default; `@ppy.opt` is a contract on one function."""
    (workspace / "pin.ppy").write_text(
        textwrap.dedent(
            """
            import ppy

            @ppy.opt(3)
            def hot(x: int) -> int:
                return x * x
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    lowered = _ppy(["-O0", "explain", "hot"], workspace)
    assert "optimization: O3" in lowered.stdout


def _project(workspace: Path) -> Path:
    """A three-module untyped project whose types only exist across files."""
    src = workspace / "src"
    src.mkdir(exist_ok=True)
    (workspace / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\nsource-roots = ["src"]\n', encoding="utf-8"
    )
    (src / "geometry.py").write_text(
        textwrap.dedent(
            """
            import math


            def distance(x1, y1, x2, y2):
                return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (src / "shapes.py").write_text(
        textwrap.dedent(
            """
            import geometry


            def perimeter(points):
                total = 0.0
                for i in range(len(points)):
                    a = points[i]
                    b = points[(i + 1) % len(points)]
                    total += geometry.distance(a[0], a[1], b[0], b[1])
                return total
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (src / "app.py").write_text(
        textwrap.dedent(
            """
            import shapes


            def main():
                square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                print(round(shapes.perimeter(square), 3))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    return src


def test_convert_types_a_project_across_module_boundaries(workspace: Path):
    """`distance` has no call site in its own file; the types come from another."""
    src = _project(workspace)
    assert _ppy(["convert", "src"], workspace).returncode == 0

    geometry = (src / "geometry.ppy").read_text(encoding="utf-8")
    assert "def distance(x1: float, y1: float, x2: float, y2: float) -> float:" in geometry
    shapes = (src / "shapes.ppy").read_text(encoding="utf-8")
    assert "def perimeter(points: Sequence[tuple[float, float]]) -> float:" in shapes


def test_convert_warns_when_both_sources_survive(workspace: Path):
    src = _project(workspace)
    result = _ppy(["convert", "src"], workspace)
    assert "W2005" in result.stderr
    assert (src / "app.py").exists() and (src / "app.ppy").exists()
    # The ambiguity the warning is about is a real error for the next command.
    assert _ppy(["check", "src"], workspace).returncode == 1


def test_in_place_migrates_a_project_to_ppy(workspace: Path):
    src = _project(workspace)
    result = _ppy(["convert", "src", "--in-place"], workspace)
    assert result.returncode == 0
    assert "W2005" not in result.stderr
    assert sorted(p.name for p in src.glob("*.py")) == []
    assert sorted(p.name for p in src.glob("*.ppy")) == ["app.ppy", "geometry.ppy", "shapes.ppy"]


def test_a_migrated_project_still_runs_on_plain_cpython(workspace: Path):
    """`import ppy` installs the loader, so it cannot follow a sibling import."""
    src = _project(workspace)
    assert _ppy(["convert", "src", "--in-place"], workspace).returncode == 0

    app = (src / "app.ppy").read_text(encoding="utf-8").splitlines()
    assert app.index("import ppy") < app.index("import shapes")

    plain = subprocess.run(
        [sys.executable, "app.ppy"],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert plain.returncode == 0, plain.stderr
    assert plain.stdout.strip() == "4.0"


def test_fmt_keeps_ppy_ahead_of_a_sibling_import(workspace: Path):
    """Sorting `ppy` after the module whose loader it installs breaks the import."""
    (workspace / "geometry.ppy").write_text("VALUE: int = 1\n", encoding="utf-8")
    consumer = workspace / "consumer.py"
    consumer.write_text(
        "import ppy\n\nimport geometry\n\nprint(geometry.VALUE)\n", encoding="utf-8"
    )
    assert _ppy(["fmt", "consumer.py"], workspace).returncode == 0
    lines = consumer.read_text(encoding="utf-8").splitlines()
    assert lines.index("import ppy") < lines.index("import geometry")

    plain = subprocess.run(
        [sys.executable, "consumer.py"], cwd=workspace, capture_output=True, text=True, check=False
    )
    assert plain.returncode == 0, plain.stderr


def test_fmt_groups_imports_by_kind(workspace: Path):
    source = workspace / "messy.ppy"
    source.write_text(
        "import ppy\nimport math\nfrom ppy import Buffer\nimport time\n\n\nX: int = 1\n",
        encoding="utf-8",
    )
    assert _ppy(["fmt", "messy.ppy"], workspace).returncode == 0
    assert source.read_text(encoding="utf-8").startswith(
        "import math\nimport time\n\nimport ppy\nfrom ppy import Buffer\n"
    )


def test_convert_keeps_a_closed_return_set_but_not_a_parameter_one(workspace: Path):
    """A body is complete evidence; call sites are a sample."""
    (workspace / "modes.py").write_text(
        textwrap.dedent(
            """
            def kind(n):
                if n < 0:
                    return "negative"
                if n == 0:
                    return "zero"
                return "positive"


            def scale(value, mode):
                if mode == "double":
                    return value * 2
                return value * 3


            print(kind(-1), kind(0), kind(5))
            print(scale(10, "double"), scale(10, "triple"))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "modes.py"], workspace).returncode == 0
    converted = (workspace / "modes.ppy").read_text(encoding="utf-8")
    assert "def kind(n: int) -> Literal['negative', 'zero', 'positive']:" in converted
    assert "from typing import Literal" in converted
    # `mode` is only ever seen with two values here, but a third is legal.
    assert "def scale(value: int, mode: str) -> int:" in converted
    assert _ppy(["check", "modes.ppy"], workspace).returncode == 0

    plain = subprocess.run(
        [sys.executable, "modes.py"], cwd=workspace, capture_output=True, text=True, check=False
    )
    assert _ppy(["modes.ppy"], workspace).stdout == plain.stdout


def test_convert_widens_a_read_only_container_parameter(workspace: Path):
    """A function that never mutates its argument should not demand a `list`."""
    (workspace / "ro.py").write_text(
        textwrap.dedent(
            """
            def total(items):
                out = 0.0
                for item in items:
                    out += item
                return out


            def grow(items):
                items.append(1.0)
                return len(items)


            def passthrough(items):
                return items


            DATA = [1.0, 2.0]
            print(total(DATA), grow(DATA), len(passthrough(DATA)))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "ro.py"], workspace).returncode == 0
    converted = (workspace / "ro.ppy").read_text(encoding="utf-8")
    assert "def total(items: Sequence[float]) -> float:" in converted
    # Mutating it, or handing it back, keeps the concrete type.
    assert "def grow(items: list[float]) -> int:" in converted
    assert "def passthrough(items: list[float]) -> list[float]:" in converted
    # PEP 585 moved the protocols out of `typing`.
    assert "from collections.abc import Sequence" in converted
    assert _ppy(["check", "ro.ppy"], workspace).returncode == 0


def test_convert_leaves_no_unused_import_behind(workspace: Path):
    """A later decision can replace an annotation the import was added for."""
    (workspace / "promoted.py").write_text(
        textwrap.dedent(
            """
            def total(values):
                out = 0.0
                for i in range(len(values)):
                    out += values[i]
                return out


            def main():
                data = [float(i) for i in range(8)]
                print(total(data))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "promoted.py", "--promote-buffers"], workspace).returncode == 0
    converted = (workspace / "promoted.ppy").read_text(encoding="utf-8")
    # The parameter is read-only, so it was widened before promotion replaced it.
    assert "Buffer[float]" in converted
    assert "Sequence" not in converted
    assert _ppy(["check", "promoted.ppy"], workspace).returncode == 0


def test_lint_runs_an_external_checker_over_ppy_sources(workspace: Path):
    """`.ppy` is staged as `.py` so a tool that keys off the extension can see it."""
    pytest.importorskip("pyright")
    (workspace / "typed.ppy").write_text(
        "def double(x: int) -> int:\n    return x * 2\n", encoding="utf-8"
    )
    done = _ppy(["lint", "--backend", "pyright", "."], workspace)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "typed.ppy" not in done.stdout or "error" not in done.stdout


def test_lint_reports_findings_against_the_real_ppy_path(workspace: Path):
    pytest.importorskip("pyright")
    (workspace / "wrong.ppy").write_text(
        "def double(x: int) -> str:\n    return x * 2\n", encoding="utf-8"
    )
    done = _ppy(["lint", "--backend", "pyright", "."], workspace)
    assert done.returncode == 1
    assert "wrong.ppy" in done.stdout
    assert ".py:" not in done.stdout.replace(".ppy:", "")


def test_lint_says_so_when_the_backend_is_missing(workspace: Path):
    (workspace / "x.ppy").write_text("VALUE: int = 1\n", encoding="utf-8")
    done = _ppy(["lint", "--backend", "mypy", "."], workspace)
    if done.returncode == 2:
        assert "not installed" in done.stderr


def test_test_backend_pytest_runs_a_suite_against_ppy_modules(workspace: Path):
    (workspace / "lib.ppy").write_text(
        "def area(w: float, h: float) -> float:\n    return w * h\n", encoding="utf-8"
    )
    (workspace / "test_lib.py").write_text(
        "import lib\n\n\ndef test_area():\n    assert lib.area(3.0, 4.0) == 12.0\n",
        encoding="utf-8",
    )
    (workspace / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\nsource-roots = ["."]\n', encoding="utf-8"
    )
    done = _ppy(["test", "--backend", "pytest", "."], workspace)
    assert done.returncode == 0, done.stdout + done.stderr

    # Without the hook the same suite cannot even import the module.
    plain = subprocess.run(
        [sys.executable, "-m", "pytest", ".", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.returncode != 0


def test_fmt_runs_the_builtin_pass_before_an_external_formatter(workspace: Path):
    """Import grouping is settled by PPY; the project's formatter styles the rest."""
    (workspace / "ruff.toml").write_text("", encoding="utf-8")
    source = workspace / "messy.ppy"
    source.write_text(
        "import ppy\nimport math\nfrom ppy import Buffer\nimport time\ndef f( x:int )->int:\n    return  x+1\n",
        encoding="utf-8",
    )
    assert _ppy(["fmt", "messy.ppy"], workspace).returncode == 0
    text = source.read_text(encoding="utf-8")
    assert text.startswith("import math\nimport time\n\nimport ppy\nfrom ppy import Buffer\n")
    assert "def f(x: int) -> int:" in text


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("out = 0.0\n    for v in xs:\n        out += v\n    return out", "Sequence[float]"),
        ("return len(xs)", "Sequence[float]"),
        ("return xs[0]", "Sequence[float]"),
        ("return xs.count(1.0)", "Sequence[float]"),
        ("return xs.index(1.0)", "Sequence[float]"),
        ("return 1.0 in xs", "Sequence[float]"),
        ("return sorted(xs)", "Sequence[float]"),
        # Uses that a `Sequence` either lacks or answers differently.
        ("return xs.copy()", "list[float]"),
        ("return xs[:]", "list[float]"),
        ("return xs + xs", "list[float]"),
        ("return xs * 2", "list[float]"),
        ("return xs", "list[float]"),
        ("xs.append(1.0)\n    return len(xs)", "list[float]"),
    ],
)
def test_a_parameter_is_widened_only_to_what_its_body_needs(
    workspace: Path, body: str, expected: str
):
    """Read-only is not the test; the protocol offering every use is."""
    (workspace / "uses.py").write_text(
        f"def f(xs):\n    {body}\n\n\nDATA = [1.0, 2.0]\nprint(f(DATA))\n", encoding="utf-8"
    )
    result = _ppy(["convert", "uses.py", "--dry-run"], workspace)
    assert result.returncode == 0, result.stderr
    assert f"def f(xs: {expected})" in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("return values[key]", "Mapping[str, float]"),
        ("return len(values)", "Mapping[str, float]"),
        ("return key in values", "Mapping[str, float]"),
        ("return values.get(key, 0.0)", "Mapping[str, float]"),
        ("return sum(values.values())", "Mapping[str, float]"),
        ("return values.pop(key)", "dict[str, float]"),
        ("values[key] = 1.0\n    return len(values)", "dict[str, float]"),
        ("return values.copy()", "dict[str, float]"),
    ],
)
def test_a_mapping_parameter_follows_the_same_rule(workspace: Path, body: str, expected: str):
    (workspace / "maps.py").write_text(
        f'def f(values, key):\n    {body}\n\n\nDATA = {{"a": 1.0}}\nprint(f(DATA, "a"))\n',
        encoding="utf-8",
    )
    result = _ppy(["convert", "maps.py", "--dry-run"], workspace)
    assert result.returncode == 0, result.stderr
    assert f"def f(values: {expected}" in result.stdout, result.stdout


def test_iterating_a_union_of_containers_yields_the_joined_element(workspace: Path):
    """`list[float]` here, `tuple[float, float]` there: the loop still knows.

    This is the README's own quickstart shape; evidence from both call sites
    joins into a union of containers, and what the union hands out is the
    join of what its members hand out.
    """
    (workspace / "mixed.py").write_text(
        textwrap.dedent(
            """
            def clamp(value):
                return min(value, 3.0)


            def spread(samples):
                total = 0.0
                for sample in samples:
                    total += clamp(sample)
                return total


            print(spread([1.0, 2.0]), spread((4.0, 5.0)))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "mixed.py"], workspace).returncode == 0
    converted = (workspace / "mixed.ppy").read_text(encoding="utf-8")
    assert "def clamp(value: float) -> float:" in converted
    assert "def spread(samples: Sequence[float]) -> float:" in converted


def test_a_widened_signature_accepts_what_it_promises(workspace: Path):
    """A `Sequence` parameter must really take a tuple, on every path."""
    (workspace / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    (workspace / "widened.py").write_text(
        textwrap.dedent(
            """
            def total(xs):
                out = 0.0
                for x in xs:
                    out += x
                return out


            DATA = [1.0, 2.0, 3.0]
            print(total(DATA), total((4.0, 5.0)))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "widened.py"], workspace).returncode == 0
    assert "def total(xs: Sequence[float])" in (workspace / "widened.ppy").read_text(
        encoding="utf-8"
    )
    plain = subprocess.run(
        [sys.executable, "widened.py"], cwd=workspace, capture_output=True, text=True, check=False
    )
    assert _ppy(["widened.ppy"], workspace).stdout == plain.stdout
    assert _ppy(["run", "widened.ppy"], workspace).stdout.splitlines()[-1] == plain.stdout.strip()


def test_convert_is_deterministic_unless_formatting_is_asked_for(workspace: Path):
    (workspace / "ruff.toml").write_text("", encoding="utf-8")
    (workspace / "styled.py").write_text(
        "def f(x):\n    return  x+1\n\n\nprint(f(1))\n", encoding="utf-8"
    )
    plain = _ppy(["convert", "styled.py", "--dry-run"], workspace).stdout
    formatted = _ppy(["convert", "styled.py", "--dry-run", "--format"], workspace).stdout
    assert "return  x+1" in plain, "the deterministic pass restyled the body"
    assert "return x + 1" in formatted


def test_convert_formatting_can_come_from_configuration(workspace: Path):
    (workspace / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\n\n[tool.ppy.convert]\nformat = true\n\n[tool.ruff]\n",
        encoding="utf-8",
    )
    (workspace / "cfg.py").write_text(
        "def f(x):\n    return  x+1\n\n\nprint(f(1))\n", encoding="utf-8"
    )
    assert "return x + 1" in _ppy(["convert", "cfg.py", "--dry-run"], workspace).stdout


def test_a_module_constant_becomes_final(workspace: Path):
    (workspace / "consts.py").write_text(
        textwrap.dedent(
            """
            LIMIT = 100
            counter = 0


            def bump():
                global counter
                counter += 1
                return counter


            total = 0
            total = total + LIMIT
            print(LIMIT, bump(), total)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "consts.py"], workspace).returncode == 0
    converted = (workspace / "consts.ppy").read_text(encoding="utf-8")
    assert "LIMIT: Final[int] = 100" in converted
    assert "from typing import Final" in converted
    # Rebound through `global`, and rebound at module level.
    assert "counter: int = 0" in converted
    assert "total: int = 0" in converted
    assert converted.count("total:") == 1, "a later assignment was re-annotated"
    assert _ppy(["check", "consts.ppy"], workspace).returncode == 0


def test_a_union_of_containers_becomes_the_protocol_they_share(workspace: Path):
    """Passing a list from one site and a tuple from another names the protocol."""
    (workspace / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    (workspace / "both.py").write_text(
        textwrap.dedent(
            """
            def total(xs):
                out = 0.0
                for x in xs:
                    out += x
                return out


            DATA = [1.0, 2.0, 3.0]
            print(total(DATA), total((4.0, 5.0)))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "both.py"], workspace).returncode == 0
    converted = (workspace / "both.ppy").read_text(encoding="utf-8")
    assert "def total(xs: Sequence[float]) -> float:" in converted
    assert _ppy(["check", "both.ppy"], workspace).returncode == 0

    plain = subprocess.run(
        [sys.executable, "both.py"], cwd=workspace, capture_output=True, text=True, check=False
    )
    assert _ppy(["both.ppy"], workspace).stdout == plain.stdout
    assert _ppy(["run", "both.ppy"], workspace).stdout.splitlines()[-1] == plain.stdout.strip()


def test_containers_with_different_elements_are_not_merged(workspace: Path):
    (workspace / "mixed.py").write_text(
        textwrap.dedent(
            """
            def sized(xs):
                return len(xs)


            print(sized([1.0]), sized(["a"]))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["convert", "mixed.py", "--dry-run"], workspace)
    assert "Sequence[float]" not in result.stdout
    assert "Sequence[str]" not in result.stdout


def test_convert_joins_call_site_evidence_across_rounds(workspace: Path):
    """A caller typed on a later round still gets to widen the signature.

    `relay` cannot be typed until the module-level call to it is seen, so the
    dict it forwards reaches `sink` one round after the list does. Freezing
    the parameter on the first round would describe only the easy caller.
    """
    (workspace / "freeze.py").write_text(
        textwrap.dedent(
            """
            def sink(x):
                return len(x)


            def direct():
                return sink([1])


            def relay(y):
                return sink(y)


            print(direct())
            print(relay({"a": 1}))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "freeze.py"], workspace).returncode == 0
    converted = (workspace / "freeze.ppy").read_text(encoding="utf-8")
    assert "def sink(x: list[int] | dict[str, int]) -> int:" in converted
    assert "def relay(y: dict[str, int]) -> int:" in converted


def test_convert_reads_arguments_passed_by_keyword(workspace: Path):
    (workspace / "kw.py").write_text(
        textwrap.dedent(
            """
            def consume(values):
                return len(values)


            print(consume([1, 2]))
            print(consume(values={"a": 1}))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "kw.py"], workspace).returncode == 0
    converted = (workspace / "kw.ppy").read_text(encoding="utf-8")
    assert "def consume(values: list[int] | dict[str, int]) -> int:" in converted


def test_convert_will_not_widen_past_a_callee_named_by_keyword(workspace: Path):
    """`concrete` wants a list, so forwarding to it blocks the widening.

    The forwarding is written as a keyword argument, which the analysis has to
    bind exactly as Python would -- reading only positional arguments would
    miss the constraint and emit a signature its own callee rejects.
    """
    (workspace / "fwd.py").write_text(
        textwrap.dedent(
            """
            def concrete(values: list[int]) -> list[int]:
                return values.copy()


            def wrapper(values):
                return concrete(values=values)


            print(wrapper([1, 2]))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "fwd.py"], workspace).returncode == 0
    converted = (workspace / "fwd.ppy").read_text(encoding="utf-8")
    assert "def wrapper(values: list[int]) -> list[int]:" in converted
    assert "Sequence" not in converted


def test_final_is_written_only_where_it_states_something(workspace: Path):
    """`Final` is an interface contract, not a note about assignment counts.

    A lowercase global that happens to be bound once is not announcing itself
    as a constant, and freezing it commits the module's callers to something
    its author never said.
    """
    (workspace / "consts.py").write_text(
        textwrap.dedent(
            """
            LIMIT = 3.0
            scratch = 1.0
            RETRIES = 2
            RETRIES = 3


            def f(x: float) -> float:
                return x * LIMIT + scratch + RETRIES
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "consts.py"], workspace).returncode == 0
    converted = (workspace / "consts.ppy").read_text(encoding="utf-8")
    assert "LIMIT: Final[float] = 3.0" in converted
    assert "scratch: float = 1.0" in converted
    assert "RETRIES: int = 2" in converted


def test_final_accounts_for_another_module_rebinding_the_name(workspace: Path):
    (workspace / "store.py").write_text("REGISTRY = {}\nSTABLE = 1\n", encoding="utf-8")
    (workspace / "user.py").write_text(
        textwrap.dedent(
            """
            import store

            store.REGISTRY = {"b": 2}
            print(store.STABLE)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "."], workspace).returncode == 0
    converted = (workspace / "store.ppy").read_text(encoding="utf-8")
    assert "REGISTRY: Final" not in converted
    assert "STABLE: Final[int] = 1" in converted


def test_final_sees_a_write_through_an_import_module_alias(workspace: Path):
    """`s = import_module("store"); s.LIMIT = 2` rebinds `store.LIMIT`."""
    (workspace / "store.py").write_text("LIMIT = 1\nSTABLE = 2\n", encoding="utf-8")
    (workspace / "user.py").write_text(
        textwrap.dedent(
            """
            import importlib

            s = importlib.import_module("store")
            s.LIMIT = 2
            print(s.LIMIT, s.STABLE)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "."], workspace).returncode == 0
    converted = (workspace / "store.ppy").read_text(encoding="utf-8")
    assert "LIMIT: Final" not in converted
    assert "STABLE: Final[int] = 2" in converted


def test_final_sees_the_other_ways_python_binds_a_name(workspace: Path):
    (workspace / "binds.py").write_text(
        textwrap.dedent(
            """
            import contextlib

            OPENED = 1
            HELD = 2
            CAUGHT = 3

            with contextlib.suppress(ValueError) as OPENED:
                pass

            try:
                pass
            except ValueError as CAUGHT:
                pass

            print(OPENED, HELD, CAUGHT)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "binds.py"], workspace).returncode == 0
    converted = (workspace / "binds.ppy").read_text(encoding="utf-8")
    assert "OPENED: Final" not in converted
    assert "CAUGHT: Final" not in converted
    assert "HELD: Final[int] = 2" in converted


def test_convert_refuses_to_write_over_broken_analysis(workspace: Path):
    """An error anywhere means no file is written anywhere.

    Half-converting a project -- some modules `.ppy`, the broken ones left
    `.py` -- is the one outcome with nothing to recommend it: the tree does
    not even import. `--dry-run` still shows what conversion would say.
    """
    (workspace / "good.py").write_text(
        "def double(x):\n    return x * 2\n\n\nprint(double(2))\n", encoding="utf-8"
    )
    (workspace / "bad.py").write_text(
        "def broken(x: int) -> int:\n    return x + 'no'\n\n\nprint(broken(1))\n",
        encoding="utf-8",
    )
    result = _ppy(["convert", "."], workspace)
    assert result.returncode == 1
    assert not (workspace / "good.ppy").exists()
    assert not (workspace / "bad.ppy").exists()

    kept = _ppy(["convert", "--in-place", "."], workspace)
    assert kept.returncode == 1
    assert (workspace / "good.py").exists()
    assert (workspace / "bad.py").exists()

    shown = _ppy(["convert", "--dry-run", "."], workspace)
    assert shown.returncode == 1
    assert "def double" in shown.stdout
    assert not (workspace / "good.ppy").exists()


def test_widening_sees_a_mutation_through_an_alias(workspace: Path):
    """`zs = ys = xs; zs.append(1)` mutates `xs`, whatever name did it.

    Declaring `Sequence` here would emit a signature whose own body needs a
    method the protocol does not offer -- an output any type checker rejects.
    """
    (workspace / "alias.py").write_text(
        textwrap.dedent(
            """
            def push(xs):
                ys = xs
                zs = ys
                zs.append(4)
                return len(xs)


            def read(xs):
                ys = xs
                return len(ys)


            def killed(xs):
                ys = xs
                ys = []
                ys.append(1)
                return len(xs)


            print(push([1]), read([2]), killed([3]))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "alias.py"], workspace).returncode == 0
    converted = (workspace / "alias.ppy").read_text(encoding="utf-8")
    assert "def push(xs: list[int]) -> int:" in converted
    assert "def read(xs: Sequence[int]) -> int:" in converted
    assert "def killed(xs: Sequence[int]) -> int:" in converted


def test_final_proof_is_project_wide_even_for_one_file(workspace: Path):
    """Converting one file still consults every file in the project.

    The bundle holds the conversion target and its imports; a *reverse*
    dependency assigning `store.NAME` is in neither, and is exactly what
    `Final` promises cannot happen.
    """
    (workspace / "store.py").write_text(
        "LIMIT = 5\nREGISTRY = {}\nSTABLE = 1\nOPAQUE = 2\n", encoding="utf-8"
    )
    (workspace / "user.py").write_text(
        textwrap.dedent(
            """
            import store

            store.REGISTRY = {"a": 1}
            setattr(store, "LIMIT", 9)
            print(store.STABLE)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (workspace / "dynamic_user.py").write_text(
        textwrap.dedent(
            """
            import store as st

            def poke(name):
                setattr(st, name, 0)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "store.py"], workspace).returncode == 0
    converted = (workspace / "store.ppy").read_text(encoding="utf-8")
    # A computed setattr through an alias taints the whole module.
    assert "Final" not in converted

    (workspace / "dynamic_user.py").unlink()
    (workspace / "store.ppy").unlink()
    assert _ppy(["convert", "store.py"], workspace).returncode == 0
    converted = (workspace / "store.ppy").read_text(encoding="utf-8")
    assert "STABLE: Final[int] = 1" in converted
    assert "LIMIT: Final" not in converted
    assert "REGISTRY: Final" not in converted


def test_annotations_stay_off_functions_with_unknown_decorators(workspace: Path):
    """An unknown decorator saw an untyped function; it must keep getting one.

    The analysis still infers the types -- it simply does not write them,
    because `inspect.signature` or `__annotations__` inside the decorator
    would see input the original program never gave it.
    """
    (workspace / "wrapped.py").write_text(
        textwrap.dedent(
            """
            import functools


            def registry(fn):
                return fn


            @registry
            def opaque(x):
                return x * 2


            @functools.cache
            def known(x):
                return x * 3


            print(opaque(1), known(2))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "wrapped.py"], workspace).returncode == 0
    converted = (workspace / "wrapped.ppy").read_text(encoding="utf-8")
    assert "def opaque(x):" in converted
    assert "def known(x: int) -> int:" in converted


def test_hoisting_is_conservative_by_default(workspace: Path):
    """A class whose definition has effects moves only under `aggressive`."""
    source = textwrap.dedent(
        """
        def tag(cls):
            print("tagged", cls.__name__)
            return cls


        def widest(items: 'list[Loud]') -> int:
            return len(items)


        @tag
        class Loud:
            pass


        print(widest([]))
        """
    ).lstrip("\n")
    (workspace / "noisy.py").write_text(source, encoding="utf-8")
    assert _ppy(["migrate", "noisy.py"], workspace).returncode == 0
    converted = (workspace / "noisy.ppy").read_text(encoding="utf-8")
    # Not moved: the quoted forward reference is the price of the decorator.
    assert converted.index("def widest") < converted.index("class Loud")
    assert "'list[Loud]'" in converted

    (workspace / "noisy.ppy").unlink()
    assert _ppy(["migrate", "--hoist-classes", "aggressive", "noisy.py"], workspace).returncode == 0
    converted = (workspace / "noisy.ppy").read_text(encoding="utf-8")
    assert converted.index("class Loud") < converted.index("def widest")
    assert "list[Loud]" in converted and "'list[Loud]'" not in converted


def test_lint_pyright_carries_the_projects_own_configuration(workspace: Path):
    """`[tool.pyright]`, existing configs, and stubs all reach the staging tree.

    A type-check that dropped the project's `reportMissingImports` choice or
    its `.pyi` stubs would answer questions about a project that does not
    exist.
    """
    import importlib.util

    if importlib.util.find_spec("pyright") is None:
        pytest.skip("pyright is not installed")
    (workspace / "pyproject.toml").write_text(
        '[tool.ppy]\n\n[tool.pyright]\nreportMissingImports = "none"\n', encoding="utf-8"
    )
    (workspace / "vendored.pyi").write_text("VALUE: int\n", encoding="utf-8")
    (workspace / "vendored.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "uses.ppy").write_text(
        "import missing_third_party\n\nfrom vendored import VALUE\n\n"
        "print(VALUE, missing_third_party)\n",
        encoding="utf-8",
    )
    result = _ppy(["lint", "--backend", "pyright", "."], workspace)
    # The project turned reportMissingImports off, so the unresolvable
    # third-party import is not a finding.
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_declared_formatter_that_is_missing_fails_the_conversion(workspace: Path):
    (workspace / "pyproject.toml").write_text(
        '[tool.ppy]\n\n[tool.ppy.format]\nbackend = "black"\n', encoding="utf-8"
    )
    (workspace / "plain.py").write_text(
        "def f(x):\n    return x\n\n\nprint(f(1))\n", encoding="utf-8"
    )
    import importlib.util

    if importlib.util.find_spec("black") is not None:
        pytest.skip("black is installed here, so the declared formatter works")
    result = _ppy(["convert", "plain.py", "--format"], workspace)
    assert result.returncode == 1
    assert "E1802" in result.stdout + result.stderr
    assert not (workspace / "plain.ppy").exists()


def test_an_undeclared_formatter_means_builtin_only(workspace: Path):
    """No config, no external formatter -- whatever happens to be installed."""
    (workspace / "styled.py").write_text(
        "def f(x):\n    return  x+1\n\n\nprint(f(1))\n", encoding="utf-8"
    )
    formatted = _ppy(["convert", "styled.py", "--dry-run", "--format"], workspace).stdout
    assert "return  x+1" in formatted


def test_buffer_promotion_respects_aliases_and_keywords(workspace: Path):
    """The promotion planner uses the same alias map and call binder.

    `view = values; view.append(...)` grows the parameter, so it may not be
    borrowed; a call written `f(data=xs)` traces the same list `f(xs)` would;
    and a keyword call into a function that stayed a list abandons the group.
    """
    (workspace / "grow.py").write_text(
        textwrap.dedent(
            """
            def total(values):
                view = values
                view.append(0.0)
                out = 0.0
                for v in values:
                    out += v
                return out


            data = [1.0, 2.0, 3.0]
            print(total(data))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "grow.py", "--promote-buffers"], workspace).returncode == 0
    converted = (workspace / "grow.ppy").read_text(encoding="utf-8")
    assert "Buffer" not in converted

    (workspace / "kwcall.py").write_text(
        textwrap.dedent(
            """
            def norm(scale, values):
                out = 0.0
                for v in values:
                    out += v * scale
                return out


            data = [1.0, 2.0]
            print(norm(2.0, values=data))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "kwcall.py", "--promote-buffers"], workspace).returncode == 0
    converted = (workspace / "kwcall.ppy").read_text(encoding="utf-8")
    assert "values: Buffer[float]" in converted
    assert 'array.array("d", [1.0, 2.0])' in converted

    (workspace / "kwreader.py").write_text(
        textwrap.dedent(
            """
            def summed(values):
                out = 0.0
                for v in values:
                    out += v
                return out


            def grows(values):
                values.append(9.0)
                return len(values)


            data = [1.0, 2.0]
            print(summed(data), grows(values=data))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "kwreader.py", "--promote-buffers"], workspace).returncode == 0
    converted = (workspace / "kwreader.ppy").read_text(encoding="utf-8")
    # The keyword call reaches `grows`, which needs a list, so nothing is
    # rewritten -- otherwise `grows` would receive an array it did not declare.
    assert "array.array" not in converted
    assert "Buffer" not in converted


def test_the_write_index_is_scope_aware(workspace: Path):
    """Aliases live in scopes and in time, and writes follow both.

    A function-scope `import other as s` must not launder the module-scope
    write on `store`; `s = store` makes `s` the module for as long as the
    binding lasts; `from . import store` resolves against the package; and a
    file that does not parse fails the whole proof closed.
    """
    (workspace / "store.py").write_text("LIMIT = 5\nSAFE = 1\n", encoding="utf-8")
    (workspace / "other.py").write_text("LIMIT = 7\n", encoding="utf-8")
    (workspace / "shadow.py").write_text(
        textwrap.dedent(
            """
            import store as s

            s.LIMIT = 9


            def f():
                import other as s

                return s
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (workspace / "chain.py").write_text(
        'import store\n\nalias = store\nsetattr(alias, "SAFE", 2)\n', encoding="utf-8"
    )
    pkg = workspace / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "inner.py").write_text("DEPTH = 1\n", encoding="utf-8")
    (pkg / "user.py").write_text("from . import inner\n\ninner.DEPTH = 2\n", encoding="utf-8")

    assert _ppy(["convert", "store.py"], workspace).returncode == 0
    converted = (workspace / "store.ppy").read_text(encoding="utf-8")
    assert "LIMIT: Final" not in converted
    assert "SAFE: Final" not in converted
    (workspace / "store.ppy").unlink()

    assert _ppy(["convert", str(pkg / "inner.py")], workspace).returncode == 0
    assert "DEPTH: Final" not in (pkg / "inner.ppy").read_text(encoding="utf-8")

    # `other` was only ever rebound inside a function's own alias.
    assert _ppy(["convert", "other.py"], workspace).returncode == 0
    assert "LIMIT: Final[int] = 7" in (workspace / "other.ppy").read_text(encoding="utf-8")


def test_an_unparsable_project_file_fails_the_final_proof_closed(workspace: Path):
    (workspace / "store.py").write_text("STEADY = 1\n", encoding="utf-8")
    (workspace / "mystery.py").write_text("def (broken\n", encoding="utf-8")
    assert _ppy(["convert", "store.py"], workspace).returncode == 0
    assert "STEADY: Final" not in (workspace / "store.ppy").read_text(encoding="utf-8")


def test_reflective_keeps_annotations_exactly_as_written(workspace: Path):
    """`@ppy.reflective` pins the source: no annotations are added or relied on."""
    (workspace / "pinned.py").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.reflective
            def observed(x):
                return x * 2


            def plain(x):
                return x + 1


            print(observed(3), plain(4))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "pinned.py"], workspace).returncode == 0
    converted = (workspace / "pinned.ppy").read_text(encoding="utf-8")
    assert "def observed(x):" in converted
    assert "def plain(x: int) -> int:" in converted


def test_strict_convert_rejects_what_check_would_reject(workspace: Path):
    """`ppy convert` output must be valid strict PPY, or nothing is written.

    The same program migrates fine: `ppy migrate` is the command that writes
    work-in-progress code on purpose.
    """
    (workspace / "homemade.py").write_text(
        textwrap.dedent(
            """
            def cache(fn):
                print(sorted(fn.__annotations__))
                return fn


            @cache
            def f(x):
                return x


            print(f(1))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["convert", "homemade.py"], workspace)
    assert result.returncode == 1
    assert "E1201" in result.stderr
    assert "ppy migrate" in result.stderr
    assert "@ppy.reflective" in result.stderr
    assert not (workspace / "homemade.ppy").exists()

    assert _ppy(["migrate", "homemade.py"], workspace).returncode == 0
    assert (workspace / "homemade.ppy").exists()


def test_convert_no_strict_downgrades_the_gate(workspace: Path):
    """`--no-strict` writes the permissive result without the migrate framing."""
    (workspace / "wrapped.py").write_text(
        textwrap.dedent(
            """
            def registry(fn):
                return fn


            @registry
            def opaque(x):
                return x * 2


            print(opaque(1))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "--no-strict", "wrapped.py"], workspace).returncode == 0
    assert "def opaque(x):" in (workspace / "wrapped.ppy").read_text(encoding="utf-8")


def test_a_user_defined_cache_is_not_functools_cache(workspace: Path):
    """Decorator identity is resolved, not spelled.

    A local `def cache` that prints `fn.__annotations__` must keep seeing the
    empty dict it always saw; only the real `functools.cache` is known to
    tolerate annotations.
    """
    (workspace / "homemade.py").write_text(
        textwrap.dedent(
            """
            def cache(fn):
                print(sorted(fn.__annotations__))
                return fn


            @cache
            def f(x):
                return x


            print(f(1))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "homemade.py"], workspace).returncode == 0
    converted = (workspace / "homemade.ppy").read_text(encoding="utf-8")
    assert "def f(x):" in converted
    assert "def f(x: int)" not in converted

    (workspace / "genuine.py").write_text(
        textwrap.dedent(
            """
            from functools import cache


            @cache
            def f(x):
                return x


            print(f(1))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["convert", "genuine.py"], workspace).returncode == 0
    converted = (workspace / "genuine.ppy").read_text(encoding="utf-8")
    assert "def f(x: int) -> int:" in converted


def test_safe_hoisting_will_not_cross_an_unsafe_definition(workspace: Path):
    """A crossed decorator observing `globals()` pins the class in place."""
    (workspace / "probe.py").write_text(
        textwrap.dedent(
            """
            def probe(fn):
                print("Node" in globals())
                return fn


            @probe
            def use(x: "Node"):
                return x


            class Node:
                pass


            print(use(Node()).__class__.__name__)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "probe.py"], workspace).returncode == 0
    converted = (workspace / "probe.ppy").read_text(encoding="utf-8")
    assert converted.index("def use") < converted.index("class Node")
    assert '"Node"' in converted or "'Node'" in converted


def test_reflection_readers_freeze_what_they_observe(workspace: Path):
    """Whoever reads annotations at runtime must keep reading the original.

    The observation may live in a different file than the function; the scan
    is project-wide, like the one behind `Final`.
    """
    (workspace / "lib.py").write_text(
        textwrap.dedent(
            """
            def observed(x):
                return x


            def free(x):
                return x + 1


            X = 1
            print(free(1), observed(2), X)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (workspace / "inspector.py").write_text(
        textwrap.dedent(
            """
            import inspect

            import lib

            print(inspect.signature(lib.observed))
            print(lib.__annotations__)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "lib.py"], workspace).returncode == 0
    converted = (workspace / "lib.ppy").read_text(encoding="utf-8")
    assert "def observed(x):" in converted
    assert "def free(x: int) -> int:" in converted
    # `lib.__annotations__` is printed, so the module gets no new entries.
    assert "X: " not in converted


def test_pyright_config_may_be_jsonc_with_extends(workspace: Path):
    """Real pyright configs carry comments, trailing commas, and `extends`."""
    import importlib.util

    if importlib.util.find_spec("pyright") is None:
        pytest.skip("pyright is not installed")
    (workspace / "config").mkdir()
    (workspace / "config" / "base.json").write_text(
        '{\n  // the base\n  "reportMissingImports": "none",\n}\n', encoding="utf-8"
    )
    (workspace / "pyrightconfig.json").write_text(
        '{\n  /* local */\n  "extends": "./config/base.json",\n}\n', encoding="utf-8"
    )
    (workspace / "uses.ppy").write_text(
        "import missing_third_party\n\nprint(missing_third_party)\n", encoding="utf-8"
    )
    result = _ppy(["lint", "--backend", "pyright", "--no-strict", "."], workspace)
    assert "reportMissingImports" not in result.stdout
    assert result.returncode == 0, result.stdout + result.stderr


def test_reflection_follows_import_and_object_aliases(workspace: Path):
    """`sig = inspect.signature; fn = lib.f; sig(fn)` observes `lib.f`."""
    (workspace / "lib.py").write_text(
        textwrap.dedent(
            """
            def observed_a(x):
                return x


            def observed_b(x):
                return x


            def observed_c(x):
                return x


            def free(x):
                return x + 1


            print(free(1), observed_a(2), observed_b(3), observed_c(4))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (workspace / "readers.py").write_text(
        textwrap.dedent(
            """
            import inspect as i
            from typing import get_type_hints as gth

            import lib

            sig = i.signature
            fn = lib.observed_b
            fn2 = fn

            print(sig(lib.observed_a))
            print(fn2.__annotations__)
            print(gth(lib.observed_c))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "lib.py"], workspace).returncode == 0
    converted = (workspace / "lib.ppy").read_text(encoding="utf-8")
    assert "def observed_a(x):" in converted
    assert "def observed_b(x):" in converted
    assert "def observed_c(x):" in converted
    assert "def free(x: int) -> int:" in converted


def test_a_shadowing_cache_keeps_its_untyped_view(workspace: Path):
    """Decoration binds at its point in the file, not to the file's imports."""
    (workspace / "shadow.py").write_text(
        textwrap.dedent(
            """
            def cache(fn):
                print(sorted(fn.__annotations__))
                return fn


            @cache
            def f(x):
                return x


            from functools import cache as real_cache

            print(f(1), real_cache)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    assert _ppy(["migrate", "shadow.py"], workspace).returncode == 0
    assert "def f(x):" in (workspace / "shadow.ppy").read_text(encoding="utf-8")
