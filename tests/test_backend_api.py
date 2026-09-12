"""External backends: discovered from entry points without importing them, loaded when
asked for, handed the canonical IR after the shared passes and their own, and asked
to emit and build -- through `ppy emit` and `ppy build --backend` like a builtin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend import (
    BACKEND_API_VERSION,
    Backend,
    BackendConfig,
    BackendLoadError,
    available_backends,
    discover_external_backends,
    emit_format_owner,
    load_backend,
)
from ppy_compiler.driver.config import Config, load_config
from ppy_compiler.driver.ir_pipeline import canonical_ir_modules
from ppy_compiler.driver.pipeline import (
    analyze_paths,
    backend_identity,
    collect_sources,
    open_project,
)
from ppy_compiler.ir import STAGES

KERNEL = """
    from ppy import Buffer


    def scale(xs: Buffer[int], k: int, n: int) -> int:
        total: int = 0
        for i in range(n):
            xs[i] = xs[i] * k
            total = total + xs[i]
        return total


    def halve(n: int) -> int:
        return n // 2
    """

#: A backend the way a package would write one: the interface, the IR, nothing else.
DUMMY = '''
    import json
    from pathlib import Path

    from ppy_compiler.backend import (
        Backend,
        BackendValidationError,
        BuildResult,
        EmitFormat,
        ToolchainStatus,
    )
    from ppy_compiler.ir import FunctionPass, verify_or_raise
    from ppy_compiler.ir.transforms import Canonicalize

    CREATED = []


    class Tag(FunctionPass):
        """Marks every function it saw; with `break`, leaves a block without its terminator."""

        name = "dummy-tag"

        def __init__(self, breaking: bool) -> None:
            self.breaking = breaking

        def run_on_function(self, function, ctx):
            function.attributes["dummy.tagged"] = True
            if self.breaking:
                function.body.blocks[0].operations.clear()
            return True


    class DummyBackend(Backend):
        name = "dummy"

        def fingerprint(self):
            return f"dummy:{self.options.get('sdk', '0')}"

        def emit_formats(self):
            return (
                EmitFormat("dummy", ".dummy", description="a summary of the IR"),
                EmitFormat("dummy-bin", ".dbin", binary=True, description="the summary, packed"),
            )

        def register_passes(self, manager):
            manager.register_stage_pass("backend", lambda: Tag(bool(self.options.get("break"))))

        def validate(self, module, context):
            refused = str(self.options.get("refuse", ""))
            for function in module.functions.values():
                if refused and function.name.endswith(refused):
                    raise BackendValidationError(
                        self.name,
                        f"function {function.name}",
                        capability="a device kernel",
                        location=function.location,
                    )

        def _summary(self, module, context):
            verify_or_raise(module, context.registry)
            still = Canonicalize().run(module, __import__("ppy_compiler.ir", fromlist=["PassContext"]).PassContext(context.registry))
            lines = [f"module {module.name}", f"canonical: {not still}", "verified: True"]
            for name, function in sorted(module.functions.items()):
                if function.is_declaration:
                    continue
                tagged = function.attributes.get("dummy.tagged", False)
                lines.append(f"function {name} tagged={tagged} ops={sum(len(b.operations) for b in function.body.blocks)}")
            lines.append(f"dialects: {', '.join(sorted(module.dialects))}")
            lines.append(f"identity: {context.identity.get(module.name, '')}")
            lines.append(f"target: {context.target}")
            lines.append(f"flavor: {context.backend_config.get('flavor', 'plain')}")
            lines.append(f"opt-level: {context.opt_level}")
            lines.append(f"stages: {'backend' in __import__('ppy_compiler.ir', fromlist=['STAGES']).STAGES}")
            return "\\n".join(lines) + "\\n"

        def emit(self, module, format, context):
            text = self._summary(module, context)
            if format == "dummy-bin":
                return b"DUMMY\\0" + text.encode("utf-8")
            if format == "dummy":
                return text
            return super().emit(module, format, context)

        def build(self, modules, output, context):
            written = []
            for name, module in modules.items():
                path = output / f"{name}.dummy"
                path.write_text(self._summary(module, context), encoding="utf-8")
                written.append(path)
            record = output / "build.json"
            record.write_text(json.dumps({
                "modules": sorted(modules),
                "identity": dict(context.identity),
                "target": context.target,
                "opt_level": context.opt_level,
                "entry": str(context.entry) if context.entry else None,
            }, indent=1))
            written.append(record)
            context.note("dummy: nothing was compiled, everything was written")
            return BuildResult(outputs=tuple(written))

        def toolchain_status(self):
            if self.options.get("toolchain") == "missing":
                return ToolchainStatus(False, "missing dummy-sdk")
            return ToolchainStatus(True, "dummy-sdk 1.0")


    def create_backend(options):
        CREATED.append(dict(options))
        return DummyBackend(options)
    '''


def _install_backend(
    tmp_path: Path, monkeypatch, source: str, name: str = "dummy", module: str | None = None
) -> None:
    """A distribution in `tmp_path/site` registering `name` in the `ppy.backends` group."""
    module = module or f"{name}_backend"
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    info = site / f"{module}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {module}\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(f"[ppy.backends]\n{name} = {module}:create_backend\n")
    monkeypatch.syspath_prepend(str(site))
    sys.modules.pop(module, None)


@pytest.fixture(autouse=True)
def _forget_installed_backends():
    """A fake backend imported by one test is not the next test's module."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.endswith("_backend"):
            sys.modules.pop(name, None)


def _project(tmp_path: Path, config: str = "") -> Path:
    """A project in `tmp_path/proj`, beside the `site` the backends install into."""
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n" + config, encoding="utf-8")
    path = root / "kernel.ppy"
    path.write_text(textwrap.dedent(KERNEL).lstrip("\n"), encoding="utf-8")
    return path


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd.parent / "site") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


# -- the registry --------------------------------------------------------------


def test_the_builtin_backends_are_in_the_registry():
    catalog = available_backends()
    for name in ("llvm", "python", "c", "nvvm", "stablehlo", "ir"):
        assert catalog.backends[name].builtin
    llvm = load_backend("llvm")
    assert llvm.name == "llvm" and llvm.api_version == BACKEND_API_VERSION
    assert [spec.name for spec in llvm.emit_formats()] == ["llvm-ir"]
    assert emit_format_owner("c").backend.name == "c"
    assert emit_format_owner("ir").format.suffix == ".ppyir"
    assert isinstance(load_backend("python"), Backend)


def test_every_builtin_emit_kind_belongs_to_one_backend():
    from ppy_compiler.driver.emit import KINDS

    owners = {kind: emit_format_owner(kind).backend.name for kind in KINDS}
    assert set(owners) == set(KINDS)
    assert owners["llvm-ir"] == "llvm" and owners["ptx"] == "nvvm" and owners["cuda"] == "c"


def test_an_installed_backend_is_discovered_without_being_imported(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    found, problems = discover_external_backends()
    assert "dummy" in found and problems == []
    catalog = available_backends()
    assert not catalog.backends["dummy"].builtin
    assert "dummy_backend" not in sys.modules, "discovery does not import"
    backend = load_backend("dummy", {"sdk": "2"})
    assert backend.name == "dummy" and "dummy_backend" in sys.modules
    assert sys.modules["dummy_backend"].CREATED == [{"sdk": "2"}]
    assert backend.fingerprint() == "dummy:2"
    assert load_backend("dummy", BackendConfig("dummy", {"sdk": "3"})).fingerprint() == "dummy:3"


def test_an_unknown_backend_names_the_ones_there_are():
    with pytest.raises(BackendLoadError, match=r"no backend is called 'nope'.*llvm"):
        load_backend("nope")


def test_a_name_registered_twice_is_refused_not_settled_by_order(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY, name="twice", module="twice_a")
    _install_backend(tmp_path, monkeypatch, DUMMY, name="twice", module="twice_b")
    catalog = available_backends()
    assert "twice" in catalog.duplicates and len(catalog.duplicates["twice"]) == 2
    assert any("registered by more than one" in problem for problem in catalog.problems)
    with pytest.raises(BackendLoadError, match=r"more than one distribution"):
        load_backend("twice")


def test_a_backend_written_against_another_interface_version_is_refused(
    tmp_path: Path, monkeypatch
):
    _install_backend(
        tmp_path,
        monkeypatch,
        """
        from ppy_compiler.backend import Backend


        class Old(Backend):
            name = "old"
            api_version = 99


        def create_backend(options):
            return Old(options)
        """,
        name="old",
    )
    with pytest.raises(BackendLoadError, match=r"version 99.*speaks version 1"):
        load_backend("old")


def test_a_factory_that_fails_or_answers_the_wrong_thing_is_the_reason(tmp_path: Path, monkeypatch):
    _install_backend(
        tmp_path,
        monkeypatch,
        """
        def create_backend(options):
            raise RuntimeError("boom")
        """,
        name="broken",
    )
    _install_backend(
        tmp_path,
        monkeypatch,
        """
        def create_backend(options):
            return object()
        """,
        name="odd",
    )
    with pytest.raises(BackendLoadError, match=r"could not be created.*boom"):
        load_backend("broken")
    with pytest.raises(BackendLoadError, match=r"answered object, not a"):
        load_backend("odd")
    # A broken package beside a good one changes nothing for the good one.
    _install_backend(tmp_path, monkeypatch, DUMMY)
    assert load_backend("dummy").name == "dummy"


def test_a_backend_that_claims_a_builtin_name_is_reported_and_the_builtin_used(
    tmp_path: Path, monkeypatch
):
    _install_backend(tmp_path, monkeypatch, DUMMY, name="llvm", module="fake_llvm")
    catalog = available_backends()
    assert catalog.backends["llvm"].builtin
    assert any("builtin backend's name" in problem for problem in catalog.problems)
    assert load_backend("llvm").emit_formats()[0].name == "llvm-ir"


def test_a_format_registered_twice_is_a_collision(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    _install_backend(
        tmp_path, monkeypatch, DUMMY.replace('name = "dummy"', 'name = "rival"'), name="rival"
    )
    with pytest.raises(
        BackendLoadError, match=r"'dummy' is registered by backends 'dummy', 'rival'"
    ):
        emit_format_owner("dummy")
    with pytest.raises(
        BackendLoadError, match=r"no backend emits 'nothing'; the formats are .*dummy"
    ):
        emit_format_owner("nothing")


def test_the_config_table_reaches_the_backend_and_no_other(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.backends.dummy]\nsdk = "4"\ntarget = "toy"\n\n'
        "[tool.ppy.backends.other]\nflag = true\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.backend("dummy").options == {"sdk": "4", "target": "toy"}
    assert config.backend("other").get("flag") is True
    assert config.backend("absent").options == {} and config.backend("absent").name == "absent"
    assert config.backend("dummy").fingerprint() != config.backend("other").fingerprint()
    assert Config().backend("dummy").fingerprint() == BackendConfig("dummy").fingerprint()


# -- the pipeline ----------------------------------------------------------------


def test_the_backend_receives_verified_canonical_ir_after_its_own_passes(
    tmp_path: Path, monkeypatch
):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    project = open_project(path)
    bundle = analyze_paths(project, collect_sources(path), backend="llvm")
    assert bundle.ok, [str(d) for d in bundle.diagnostics.sorted()]
    backend = load_backend("dummy")
    modules = canonical_ir_modules(bundle, launches=True, backend=backend)
    module = modules["kernel"]
    from ppy_compiler.ir import PassContext, verify
    from ppy_compiler.ir.transforms import Canonicalize

    assert not verify(module, project.plugins.dialect_registry())
    assert all(
        f.attributes.get("dummy.tagged") for f in module.functions.values() if not f.is_declaration
    ), "the backend's pass ran at the backend stage"
    assert not Canonicalize().run(module, PassContext(project.plugins.dialect_registry())), (
        "the shared passes were done: canonicalization is at its fixed point"
    )
    assert STAGES[-1] == "backend" and "before-backend" in STAGES


def test_a_backend_pass_that_breaks_the_ir_is_named(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path, "[tool.ppy.backends.dummy]\nbreak = true\n")
    result = _ppy(path.parent, "emit", "dummy", path.name)
    assert result.returncode == 2
    assert "E1904" in result.stderr and "dummy-tag" in result.stderr


def test_validation_refuses_with_the_backend_the_function_and_the_place(
    tmp_path: Path, monkeypatch
):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path, '[tool.ppy.backends.dummy]\nrefuse = "scale"\n')
    result = _ppy(path.parent, "emit", "dummy", path.name)
    assert result.returncode == 2, result.stderr
    assert "E1802" in result.stderr
    assert "backend 'dummy' cannot take function" in result.stderr
    assert "kernel.ppy" in result.stderr and "needs a device kernel" in result.stderr
    build = _ppy(path.parent, "build", path.name, "--backend", "dummy", "-o", "out")
    assert build.returncode == 2 and "E1802" in build.stderr
    assert not (path.parent / "out" / "kernel.dummy").exists()


# -- ppy emit ----------------------------------------------------------------------


def test_emit_hands_the_backend_the_ir_and_prints_what_it_answers(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path, '[tool.ppy.backends.dummy]\nflavor = "mint"\ntarget = "toy"\n')
    result = _ppy(path.parent, "emit", "dummy", path.name)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("module kernel\ncanonical: True\nverified: True\n")
    assert "function kernel_scale tagged=True ops=" in result.stdout
    assert "function kernel_halve tagged=True" in result.stdout
    assert "target: toy\nflavor: mint\nopt-level: 2\nstages: True\n" in result.stdout
    identity = next(line for line in result.stdout.splitlines() if line.startswith("identity: "))
    assert len(identity.split(": ")[1]) == 64


def test_emit_writes_where_told_and_a_file_per_module_for_a_directory(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    (path.parent / "other.ppy").write_text("def twice(n: int) -> int:\n    return n * 2\n")
    named = _ppy(path.parent, "emit", "dummy", path.name, "-o", "out/kernel.txt")
    assert named.returncode == 0, named.stderr
    assert (path.parent / "out" / "kernel.txt").read_text().startswith("module kernel\n")
    refused = _ppy(path.parent, "emit", "dummy", ".")
    assert refused.returncode == 2 and "needs `-o DIR`" in refused.stderr
    whole = _ppy(path.parent, "emit", "dummy", ".", "-o", "out/dir")
    assert whole.returncode == 0, whole.stderr
    assert sorted(p.name for p in (path.parent / "out" / "dir").iterdir()) == [
        "kernel.dummy",
        "other.dummy",
    ]
    assert (path.parent / "out" / "dir" / "other.dummy").read_text().startswith("module other\n")


def test_a_binary_format_is_written_as_bytes(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    result = _ppy(path.parent, "emit", "dummy-bin", path.name, "-o", "out/kernel.dbin")
    assert result.returncode == 0, result.stderr
    data = (path.parent / "out" / "kernel.dbin").read_bytes()
    assert data.startswith(b"DUMMY\0module kernel\n")
    whole = _ppy(path.parent, "emit", "dummy-bin", ".", "-o", "out/dir")
    assert whole.returncode == 0 and (path.parent / "out" / "dir" / "kernel.dbin").exists()


def test_emit_refuses_an_unknown_format_and_the_builtin_flags(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    unknown = _ppy(path.parent, "emit", "nothing", path.name)
    assert unknown.returncode == 2
    assert "E1903" in unknown.stderr and "no backend emits 'nothing'" in unknown.stderr
    assert "dummy" in unknown.stderr and "llvm-ir" in unknown.stderr
    flagged = _ppy(path.parent, "emit", "dummy", "--format", path.name)
    assert flagged.returncode == 2 and "`--format` applies to the builtin kinds" in flagged.stderr


def test_a_missing_toolchain_is_an_error_before_any_analysis(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path, '[tool.ppy.backends.dummy]\ntoolchain = "missing"\n')
    result = _ppy(path.parent, "emit", "dummy", path.name)
    assert result.returncode == 2
    assert "E1801" in result.stderr and "missing dummy-sdk" in result.stderr
    build = _ppy(path.parent, "build", path.name, "--backend", "dummy")
    assert build.returncode == 2 and "missing dummy-sdk" in build.stderr


# -- ppy build ---------------------------------------------------------------------


def test_build_calls_the_backend_and_lists_what_it_wrote(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path, '[tool.ppy.backends.dummy]\ntarget = "toy"\n')
    result = _ppy(path.parent, "-O", "3", "build", path.name, "--backend", "dummy", "-o", "build")
    assert result.returncode == 0, result.stderr
    out = path.parent / "build"
    assert (out / "kernel.dummy").read_text().startswith("module kernel\ncanonical: True\n")
    record = json.loads((out / "build.json").read_text())
    assert record["modules"] == ["kernel"] and record["target"] == "toy"
    assert record["opt_level"] == 3 and record["entry"].endswith("kernel.ppy")
    assert len(record["identity"]["kernel"]) == 64
    assert "backend:  dummy" in result.stderr and "outputs:  2" in result.stderr
    assert "dummy: nothing was compiled" in result.stderr


def test_build_without_an_output_goes_under_the_cache(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    result = _ppy(path.parent, "build", path.name, "--backend", "dummy")
    assert result.returncode == 0, result.stderr
    assert (path.parent / ".ppy-cache" / "backends" / "dummy" / "kernel.dummy").exists()


def test_build_refuses_a_backend_that_is_not_there_or_does_not_build(tmp_path: Path, monkeypatch):
    path = _project(tmp_path)
    unknown = _ppy(path.parent, "build", path.name, "--backend", "nope")
    assert unknown.returncode == 2 and "E1903" in unknown.stderr
    assert "no backend is called 'nope'" in unknown.stderr
    source_only = _ppy(path.parent, "build", path.name, "--backend", "c")
    assert source_only.returncode == 2 and "E1802" in source_only.stderr
    assert "backend 'c' does not build" in source_only.stderr


# -- the cache ---------------------------------------------------------------------


def test_the_backend_its_fingerprint_and_its_configuration_are_in_the_identity(
    tmp_path: Path, monkeypatch
):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)

    def identity(config: str, options: dict) -> str:
        (path.parent / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n" + config)
        project = open_project(path)
        bundle = analyze_paths(project, collect_sources(path), backend="llvm")
        backend = load_backend("dummy", options)
        return backend_identity(bundle, "kernel", backend, 2).hex()

    plain = identity("", {})
    assert plain == identity("", {}), "the same inputs are the same artifact"
    assert plain != identity("", {"sdk": "9"}), "the backend's fingerprint is in the key"
    assert plain != identity('[tool.ppy.backends.dummy]\nflavor = "mint"\n', {}), (
        "the backend's configuration is in the key"
    )
    assert plain != identity('[tool.ppy.backends.dummy]\ntarget = "toy"\n', {}), (
        "the target is in the key"
    )


def test_the_llvm_key_carries_the_llvm_under_it(write, analyze, monkeypatch):
    from ppy_compiler.backend import builtin
    from ppy_compiler.driver.pipeline import module_cache_key

    path = write("k.ppy", "def f(n: int) -> int:\n    return n + 1\n")
    bundle = analyze(path, backend="llvm")
    before = module_cache_key(bundle, "k", target="llvm", opt_level=2).hex()
    monkeypatch.setattr(builtin, "distribution_version", lambda name: "0.0.0-other")
    after = module_cache_key(bundle, "k", target="llvm", opt_level=2).hex()
    assert before != after


# -- ppy doctor --------------------------------------------------------------------


def test_doctor_lists_every_backend_with_its_toolchain(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    _install_backend(
        tmp_path,
        monkeypatch,
        """
        def create_backend(options):
            raise RuntimeError("no sdk")
        """,
        name="broken",
    )
    path = _project(tmp_path, '[tool.ppy.backends.dummy]\ntoolchain = "missing"\n')
    result = _ppy(path.parent, "doctor")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "backends:" in lines
    dummy = next(line for line in lines if line.startswith("  dummy"))
    assert "unavailable, missing dummy-sdk" in dummy and "dummy_backend:create_backend" in dummy
    assert any("emits dummy, dummy-bin; fingerprint dummy:0" in line for line in lines)
    broken = next(line for line in lines if line.startswith("  broken"))
    assert "unusable" in broken and "no sdk" in broken
    llvm = next(line for line in lines if line.startswith("  llvm "))
    assert "emits llvm-ir" in "\n".join(lines) and ("available" in llvm)
