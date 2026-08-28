"""CLI surface: convert, check, fmt, explain, inspect, cache (spec 4.3)."""

from __future__ import annotations

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
    assert "X: int = 1" in converted


def test_convert_refuses_to_clobber_without_force(workspace: Path):
    (workspace / "exists.py").write_text("X = 1\n", encoding="utf-8")
    (workspace / "exists.ppy").write_text("# handwritten\n", encoding="utf-8")
    result = _ppy(["convert", "exists.py"], workspace)
    assert result.returncode == 1
    assert "--force" in result.stderr
    assert (workspace / "exists.ppy").read_text(encoding="utf-8") == "# handwritten\n"


def test_convert_in_place_replaces_the_source(workspace: Path):
    source = workspace / "inplace.py"
    source.write_text("def f(x):\n    return x + 1\n\n\nf(1)\n", encoding="utf-8")
    result = _ppy(["convert", "inplace.py", "--in-place"], workspace)
    assert result.returncode == 0
    assert "def f(x: int) -> int:" in source.read_text(encoding="utf-8")
    assert not (workspace / "inplace.ppy").exists()


def test_convert_reports_uninferable_parameters(workspace: Path):
    (workspace / "vague.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    result = _ppy(["convert", "vague.py", "--dry-run"], workspace)
    assert "E1304" in result.stderr
    assert "Any" not in (workspace / "vague.py").read_text(encoding="utf-8")


def test_convert_flags_dynamic_features(workspace: Path):
    (workspace / "dyn.py").write_text("def f(src: str):\n    return eval(src)\n", encoding="utf-8")
    result = _ppy(["convert", "dyn.py", "--dry-run"], workspace)
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
    target.write_text("def a() -> int:\n    return 1\ndef b() -> int:\n    return 2\n", encoding="utf-8")

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
    (workspace / "args.ppy").write_text(
        "import sys\n\nprint(sys.argv[1:])\n", encoding="utf-8"
    )
    result = _ppy(["args.ppy", "--", "one", "two"], workspace)
    assert result.returncode == 0
    assert result.stdout.strip() == "['one', 'two']"


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
    assert "def total(xs: list[float])" in converted
    assert "array.array" not in converted


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
            [sys.executable, "-m", "pylint", "--enable=all", "--score=n",
             "--disable=import-error", name],
            cwd=workspace, capture_output=True, text=True, check=False,
        )
        return {
            line.rsplit("(", 1)[-1].rstrip(")")
            for line in done.stdout.splitlines()
            if line.startswith(name)
        }

    assert _findings("converted.py") <= _findings("linted.py")
