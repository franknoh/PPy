"""The `ppy` runtime package under plain CPython (spec 3.2, 6)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ppy


def test_decorators_return_the_original_callable():
    def original(x: int) -> int:
        return x * x

    decorated = ppy.pure(original)
    assert decorated is original
    assert decorated(7) == 49


def test_decorators_do_not_change_behavior():
    @ppy.pure
    @ppy.opt(3)
    @ppy.jit(cache=True, max_specializations=8)
    @ppy.parallel
    @ppy.native
    @ppy.inline
    @ppy.specialize
    @ppy.fastmath
    def kernel(x: int) -> int:
        return x + 1

    assert kernel(1) == 2
    assert kernel.__name__ == "kernel"


def test_directives_are_recorded_in_application_order():
    @ppy.pure
    @ppy.opt(2)
    def f() -> None:
        return None

    names = [d.name for d in ppy.directives_of(f)]
    assert names == ["pure", "opt"]
    assert ppy.directives_of(f)[1].options == {"level": 2}


def test_opt_rejects_an_out_of_range_level():
    with pytest.raises(ValueError):
        ppy.opt(9)


def test_decorators_preserve_coroutine_and_descriptor_behavior():
    import inspect

    @ppy.pure
    async def coroutine() -> int:
        return 1

    assert inspect.iscoroutinefunction(coroutine)

    class Holder:
        @ppy.pure
        @staticmethod
        def helper() -> int:
            return 5

    assert Holder.helper() == 5


def test_dynamic_is_both_decorator_and_context_manager():
    with ppy.dynamic:
        value = 1
    assert value == 1

    @ppy.dynamic
    def boundary() -> int:
        return 2

    assert boundary() == 2
    assert ppy.directives_of(boundary)[0].name == "dynamic"


def test_reflective_is_inert_and_recorded():
    @ppy.reflective
    def observed(x: int) -> int:
        return x * 2

    assert observed(3) == 6
    assert any(d.name == "reflective" for d in ppy.directives_of(observed))


def test_dynamic_marker_is_plain_any():
    from typing import Any

    assert ppy.Dynamic is Any


def test_class_decoration_keeps_class_identity():
    @ppy.opt(1)
    class Model:
        x: int = 1

    assert Model().x == 1
    assert ppy.directives_of(Model)[0].options == {"level": 1}


@pytest.mark.parametrize(
    ("marker", "bits", "signed"),
    [(ppy.i8, 8, True), (ppy.u16, 16, False), (ppy.i64, 64, True), (ppy.u64, 64, False)],
)
def test_integer_markers_are_annotated_aliases(marker, bits, signed):
    from typing import get_args, get_origin

    assert get_origin(marker) is not None or hasattr(marker, "__metadata__")
    base, meta = get_args(marker)
    assert base is int
    assert meta.bits == bits and meta.signed is signed


def test_float_markers_carry_precision():
    from typing import get_args

    base, meta = get_args(ppy.f32)
    assert base is float and meta.bits == 32


def test_container_markers_are_valid_annotations():
    from typing import get_args

    array = ppy.Array[float, 3]
    base, meta = get_args(array)
    assert base == tuple[float, ...]
    assert meta.length == 3

    vector = ppy.Vector[int]
    base, meta = get_args(vector)
    assert base == list[int]

    buffer = ppy.Buffer[float]
    base, meta = get_args(buffer)
    assert base is memoryview


def test_markers_are_usable_in_real_annotations():
    from typing import get_type_hints

    def f(x: ppy.i64, xs: ppy.Vector[float]) -> ppy.f32:
        return 0.0

    hints = get_type_hints(f, include_extras=True)
    assert "x" in hints and "xs" in hints and "return" in hints


def test_refinement_metadata_round_trips():
    assert ppy.Range(0, 255).low == 0
    assert ppy.Length(16).size == 16
    assert ppy.Shape("N", 768).dims == ("N", 768)
    assert ppy.NoAlias() == ppy.NoAlias()
    assert ppy.Contiguous() == ppy.Contiguous()


def test_import_hook_is_installed():
    assert ppy.is_installed()


def _run(script: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_plain_cpython_runs_a_ppy_file_with_imports(tmp_path: Path):
    (tmp_path / "helper.ppy").write_text(
        textwrap.dedent(
            """
            import ppy

            @ppy.pure
            def double(x: ppy.i64) -> ppy.i64:
                return x * 2
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.ppy").write_text("from .inner import NAME\n", encoding="utf-8")
    (tmp_path / "pkg" / "inner.ppy").write_text("NAME = 'inner'\n", encoding="utf-8")
    entry = tmp_path / "main.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import ppy
            import helper
            import pkg

            print(helper.double(21), pkg.NAME, helper.__file__.endswith('.ppy'))
            """
        ),
        encoding="utf-8",
    )

    result = _run(entry, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "42 inner True"


def test_fixed_width_contract_is_unchecked_under_plain_execution(tmp_path: Path):
    entry = tmp_path / "wide.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import ppy

            def grow(x: ppy.i64) -> ppy.i64:
                return x * x

            print(grow(10**20))
            """
        ),
        encoding="utf-8",
    )
    result = _run(entry, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(10**40)


def test_ambiguous_module_warns(tmp_path: Path):
    (tmp_path / "shadow.py").write_text("VALUE = 'py'\n", encoding="utf-8")
    (tmp_path / "shadow.ppy").write_text("VALUE = 'ppy'\n", encoding="utf-8")
    entry = tmp_path / "main.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import warnings
            import ppy

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                import shadow
            print(shadow.VALUE, any(w.category is ppy.PPyAmbiguousModuleWarning for w in caught))
            """
        ),
        encoding="utf-8",
    )
    result = _run(entry, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ppy True"


def test_importing_ppy_starts_nothing_expensive():
    """Importing the runtime must not initialize LLVM (spec 6)."""
    script = "import sys, ppy; print('llvmlite' in sys.modules, 'ppy_compiler' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "False False"


def test_a_plain_py_file_imports_a_ppy_module(tmp_path: Path):
    """The question the extension raises: ordinary CPython, ordinary `.py` entry."""
    (tmp_path / "geometry.ppy").write_text(
        textwrap.dedent(
            """
            import ppy

            @ppy.pure
            def area(width: float, height: float) -> float:
                return width * height
            """
        ),
        encoding="utf-8",
    )
    entry = tmp_path / "consumer.py"
    entry.write_text(
        textwrap.dedent(
            """
            import ppy
            import geometry

            print(geometry.area(3.0, 4.0), geometry.__file__.endswith('.ppy'))
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, entry.name], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "12.0 True"


def test_a_ppy_module_is_invisible_without_importing_ppy(tmp_path: Path):
    (tmp_path / "geometry.ppy").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text("import geometry\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "consumer.py"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "No module named 'geometry'" in result.stderr
