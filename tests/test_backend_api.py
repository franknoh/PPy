"""External backends: discovered from entry points without importing them, loaded when
asked for, handed the canonical IR after the shared passes and their own, and asked
to emit and build -- through `ppy emit` and `ppy build --backend` like a builtin."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    discover_format_owners,
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
        api_version = 1

        def fingerprint(self):
            return f"dummy:{self.options.get('sdk', '0')}"

        def emit_formats(self):
            return (
                EmitFormat("dummy", ".dummy", description="a summary of the IR"),
                EmitFormat("dummy-bin", ".dbin", binary=True, description="the summary, packed"),
                EmitFormat(
                    "dummy-prog",
                    ".dprog",
                    scope="program",
                    description="every module in one artifact",
                ),
                EmitFormat(
                    "dummy-prog-bin", ".dpbin", binary=True, scope="program", description="packed"
                ),
            )

        def register_passes(self, passes):
            passes.add(lambda: Tag(bool(self.options.get("break"))))

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
                if self.options.get("wrong-type"):
                    return b"bytes where the format says text"
                return text
            return super().emit(module, format, context)

        def emit_program(self, modules, format, context):
            joined = "".join(self._summary(m, context) for _n, m in sorted(modules.items()))
            head = "program of " + str(len(modules)) + ": " + ", ".join(sorted(modules))
            if format == "dummy-prog-bin":
                return b"DPROG" + (head + chr(10) + joined).encode("utf-8")
            if format == "dummy-prog":
                return head + chr(10) + joined
            return super().emit_program(modules, format, context)

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
    tmp_path: Path,
    monkeypatch,
    source: str,
    name: str = "dummy",
    module: str | None = None,
    version: str = "1.0",
    formats: tuple[str, ...] = (),
) -> None:
    """A distribution in `tmp_path/site` registering `name` in the `ppy.backends` group.

    `formats` also declares them in `ppy.backend-formats`, the optional group
    that says which backend owns which format without importing anything.
    """
    module = module or f"{name}_backend"
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    info = site / f"{module}-{version}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {module}\nVersion: {version}\n")
    entries = f"[ppy.backends]\n{name} = {module}:create_backend\n"
    if formats:
        entries += "\n[ppy.backend-formats]\n" + "".join(f"{f} = {name}\n" for f in formats)
    (info / "entry_points.txt").write_text(entries)
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
    assert any(
        "emits dummy, dummy-bin, dummy-prog, dummy-prog-bin; fingerprint dummy:0" in line
        for line in lines
    )
    broken = next(line for line in lines if line.startswith("  broken"))
    assert "unusable" in broken and "no sdk" in broken
    llvm = next(line for line in lines if line.startswith("  llvm "))
    assert "emits llvm-ir" in "\n".join(lines) and ("available" in llvm)


# -- the interface version is the backend package's own ----------------------------

#: A backend whose only interesting property is the version it declares.
VERSIONED = """
    from ppy_compiler.backend import Backend, EmitFormat


    class Versioned(Backend):
        name = "versioned"
    {declaration}

        def emit_formats(self):
            return (EmitFormat("versioned", ".v"),)


    def create_backend(options):
        return Versioned(options)
    """


def _install_versioned(tmp_path: Path, monkeypatch, declaration: str, name: str = "versioned"):
    source = VERSIONED.format(declaration=declaration).replace(
        "class Versioned", f"class {name.title()}"
    )
    source = source.replace('name = "versioned"', f'name = "{name}"').replace(
        "return Versioned(options)", f"return {name.title()}(options)"
    )
    _install_backend(tmp_path, monkeypatch, source, name=name)


def test_a_backend_declaring_the_current_version_loads(tmp_path: Path, monkeypatch):
    _install_versioned(tmp_path, monkeypatch, "    api_version = 1")
    backend = load_backend("versioned")
    assert backend.api_version == BACKEND_API_VERSION == 1


def test_a_backend_declaring_an_older_version_is_refused(tmp_path: Path, monkeypatch):
    _install_versioned(tmp_path, monkeypatch, "    api_version = 0")
    with pytest.raises(BackendLoadError) as raised:
        load_backend("versioned")
    message = str(raised.value)
    assert "versioned" in message, "the backend is named"
    assert "declares backend API version 0" in message, "what it declared is named"
    assert f"speaks version {BACKEND_API_VERSION}" in message, "what the compiler speaks is named"


def test_a_backend_that_declares_no_version_is_refused_not_assumed_current(
    tmp_path: Path, monkeypatch
):
    """The bug this guards: an old backend subclasses `Backend` and declares
    nothing, so a later compiler's `Backend.api_version` would be inherited and
    the old package would call itself current forever."""
    _install_versioned(tmp_path, monkeypatch, "")
    with pytest.raises(BackendLoadError) as raised:
        load_backend("versioned")
    message = str(raised.value)
    assert "does not declare the backend API version" in message
    assert f"api_version = {BACKEND_API_VERSION}" in message, "the fix is spelled out"
    assert "versioned" in message and "versioned_backend" in message, "backend and distribution"


def test_declaring_the_version_on_the_backends_own_base_class_counts(tmp_path: Path, monkeypatch):
    """A package with a base class of its own declares once, for all of them."""
    _install_backend(
        tmp_path,
        monkeypatch,
        """
        from ppy_compiler.backend import Backend


        class OurBackend(Backend):
            api_version = 1


        class Derived(OurBackend):
            name = "derived"


        def create_backend(options):
            return Derived(options)
        """,
        name="derived",
    )
    assert load_backend("derived").api_version == 1


def test_the_compilers_own_backends_need_no_boilerplate():
    """Builtins ship with the interface, so they track the constant rather than
    a literal; nothing about them is refused."""
    for name in ("llvm", "python", "c", "nvvm", "stablehlo", "ir"):
        assert load_backend(name).api_version == BACKEND_API_VERSION


def test_the_base_class_declares_no_version_of_its_own():
    from ppy_compiler.backend import UNDECLARED_API_VERSION

    assert Backend.api_version is UNDECLARED_API_VERSION is None


# -- a backend's passes run at the backend stage and nowhere else -------------------

#: A backend that tries to hang a pass where the shared pipeline runs.
TRESPASSER = """
    from ppy_compiler.backend import Backend, EmitFormat
    from ppy_compiler.ir import FunctionPass


    class Noop(FunctionPass):
        name = "trespass"

        def run_on_function(self, function, ctx):
            return False


    class Trespasser(Backend):
        name = "trespasser"
        api_version = 1

        def emit_formats(self):
            return (EmitFormat("trespass", ".t"),)

        def register_passes(self, passes):
            passes.register_stage_pass("{stage}", Noop)

        def emit(self, module, format, context):
            return "nothing"


    def create_backend(options):
        return Trespasser(options)
    """


@pytest.mark.parametrize(
    "stage", ["before-optimization", "after-ir-generation", "before-backend", "after-optimization"]
)
def test_a_backend_cannot_register_a_pass_before_the_backend_stage(
    tmp_path: Path, monkeypatch, stage: str
):
    _install_backend(tmp_path, monkeypatch, TRESPASSER.format(stage=stage), name="trespasser")
    path = _project(tmp_path)
    result = _ppy(path.parent, "emit", "trespass", path.name)
    assert result.returncode == 2, result.stdout
    assert "E1802" in result.stderr
    assert f"registers a pass at the {stage!r} stage" in result.stderr
    assert "backend" in result.stderr and "add(factory)" in result.stderr


def test_the_registrar_reaches_only_the_backend_stage():
    from ppy_compiler.backend import BackendError, BackendPassRegistrar
    from ppy_compiler.ir import PassContext, PassManager

    manager = PassManager(PassContext(None))
    manager.add_stage("backend")
    registrar = BackendPassRegistrar(manager, "toy")
    assert not hasattr(registrar, "add_stage"), "the manager itself is not handed over"
    sentinel = object()
    registrar.add(lambda: sentinel)
    registrar.register_stage_pass("backend", lambda: sentinel)
    assert manager.passes() == [sentinel, sentinel]
    for stage in STAGES:
        if stage == "backend":
            continue
        with pytest.raises(BackendError, match=r"registers a pass at the"):
            registrar.register_stage_pass(stage, lambda: sentinel)


def test_a_backend_stage_pass_still_runs_and_is_still_verified(tmp_path: Path, monkeypatch):
    """The valid case keeps working, and a pass that breaks the IR is still named."""
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    good = _ppy(path.parent, "emit", "dummy", path.name)
    assert good.returncode == 0 and "tagged=True" in good.stdout
    (path.parent / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\n[tool.ppy.backends.dummy]\nbreak = true\n"
    )
    broken = _ppy(path.parent, "emit", "dummy", path.name)
    assert broken.returncode == 2 and "E1904" in broken.stderr and "dummy-tag" in broken.stderr


# -- the distribution's version is part of the artifact's identity ------------------


def test_a_new_release_of_a_backend_package_is_new_artifacts(tmp_path: Path, monkeypatch):
    """Two releases can carry the same module, class, and API version and
    generate different code; an artifact from the old one must not be served."""
    path = _project(tmp_path)

    def identity(version: str) -> str:
        site = tmp_path / "site"
        if site.exists():
            shutil.rmtree(site)
        sys.modules.pop("dummy_backend", None)
        _install_backend(tmp_path, monkeypatch, DUMMY, version=version)
        project = open_project(path)
        bundle = analyze_paths(project, collect_sources(path), backend="llvm")
        backend = load_backend("dummy")
        assert backend.distribution == ("dummy_backend", version)
        return backend_identity(bundle, "kernel", backend, 2).hex()

    first = identity("1.0")
    assert first == identity("1.0"), "the same release is the same artifact"
    assert first != identity("1.1"), "a new release is a new artifact"


def test_the_builtin_backends_have_no_distribution_of_their_own():
    assert load_backend("llvm").distribution is None


# -- emit scope --------------------------------------------------------------------


def _two_module_project(tmp_path: Path, config: str = "") -> Path:
    """An entry importing a second local module, so one file target is two modules."""
    path = _project(tmp_path, config)
    (path.parent / "helper.ppy").write_text("def twice(n: int) -> int:\n    return n * 2\n")
    path.write_text(
        "import helper\n\n\ndef own(n: int) -> int:\n    return n + 1\n\n\n"
        "def use(n: int) -> int:\n    return helper.twice(n)\n",
        encoding="utf-8",
    )
    return path


def test_one_module_goes_to_standard_output_or_to_the_file_named(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    printed = _ppy(path.parent, "emit", "dummy", path.name)
    assert printed.returncode == 0 and printed.stdout.startswith("module kernel\n")
    named = _ppy(path.parent, "emit", "dummy", path.name, "-o", "one.dummy")
    assert named.returncode == 0
    assert (path.parent / "one.dummy").read_text().startswith("module kernel\n")


def test_several_modules_are_never_concatenated_into_one_file(tmp_path: Path, monkeypatch):
    """Two artifacts written end to end are not one artifact: the ambiguity is
    refused, and `-o DIR` is what writes them."""
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _two_module_project(tmp_path)
    for kind in ("dummy", "dummy-bin"):
        refused = _ppy(path.parent, "emit", kind, path.name)
        assert refused.returncode == 2, refused.stdout[:200]
        assert "E1002" in refused.stderr
        assert "2 module(s)" in refused.stderr and "one artifact per module" in refused.stderr
        assert "-o DIR" in refused.stderr
        assert "helper" in refused.stderr and "kernel" in refused.stderr
    written = _ppy(path.parent, "emit", "dummy", path.name, "-o", "out")
    assert written.returncode == 0, written.stderr
    assert sorted(p.name for p in (path.parent / "out").iterdir()) == [
        "helper.dummy",
        "kernel.dummy",
    ]
    binary = _ppy(path.parent, "emit", "dummy-bin", path.name, "-o", "bin")
    assert binary.returncode == 0
    assert (path.parent / "bin" / "helper.dbin").read_bytes().startswith(b"DUMMY\0")


def test_a_program_scoped_format_is_one_artifact_for_every_module(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _two_module_project(tmp_path)
    printed = _ppy(path.parent, "emit", "dummy-prog", path.name)
    assert printed.returncode == 0, printed.stderr
    assert printed.stdout.startswith("program of 2: helper, kernel\n")
    assert "module kernel" in printed.stdout and "module helper" in printed.stdout
    named = _ppy(path.parent, "emit", "dummy-prog", path.name, "-o", "whole.dprog")
    assert named.returncode == 0
    assert (path.parent / "whole.dprog").read_text().startswith("program of 2:")
    packed = _ppy(path.parent, "emit", "dummy-prog-bin", path.name, "-o", "whole.dpbin")
    assert packed.returncode == 0
    assert (path.parent / "whole.dpbin").read_bytes().startswith(b"DPROGprogram of 2:")


def test_a_directory_target_writes_one_file_per_module_for_either_scope(
    tmp_path: Path, monkeypatch
):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _two_module_project(tmp_path)
    per_module = _ppy(path.parent, "emit", "dummy", ".", "-o", "out")
    assert per_module.returncode == 0, per_module.stderr
    assert sorted(p.name for p in (path.parent / "out").iterdir()) == [
        "helper.dummy",
        "kernel.dummy",
    ]
    whole = _ppy(path.parent, "emit", "dummy-prog", ".", "-o", "prog")
    assert whole.returncode == 0, whole.stderr
    written = list((path.parent / "prog").iterdir())
    assert len(written) == 1 and written[0].suffix == ".dprog"


def test_a_format_scope_is_module_or_program():
    from ppy_compiler.backend import EmitFormat

    assert EmitFormat("a", ".a").scope == "module"
    with pytest.raises(ValueError, match=r"module.*program"):
        EmitFormat("a", ".a", scope="whole-world")


def test_a_backend_answering_the_wrong_type_for_a_format_is_refused(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path, "[tool.ppy.backends.dummy]\nwrong-type = true\n")
    result = _ppy(path.parent, "emit", "dummy", path.name)
    assert result.returncode == 2
    assert "E1802" in result.stderr and "answered bytes for 'dummy', a text format" in result.stderr


# -- the build road --------------------------------------------------------------


def test_a_ppyir_file_builds_through_the_backend_that_was_chosen(tmp_path: Path, monkeypatch):
    """The bug this guards: `--backend NAME` with a `.ppyir` target ran the
    LLVM road and wrote LLVM artifacts, whatever the backend said."""
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    emitted = _ppy(path.parent, "emit", "ir", path.name, "-o", "kernel.ppyir")
    assert emitted.returncode == 0, emitted.stderr
    built = _ppy(path.parent, "build", "kernel.ppyir", "--backend", "dummy", "-o", "out")
    assert built.returncode == 0, built.stderr
    assert "backend:  dummy" in built.stderr
    assert (path.parent / "out" / "kernel.dummy").read_text().startswith("module kernel\n")
    assert not (path.parent / "out" / "ppy-bindings.json").exists(), "no LLVM artifact"
    record = json.loads((path.parent / "out" / "build.json").read_text())
    assert record["modules"] == ["kernel"] and len(record["identity"]["kernel"]) == 64


def test_warm_is_the_llvm_artifact_and_is_refused_for_another_backend(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    result = _ppy(path.parent, "build", path.name, "--warm", "--backend", "dummy")
    assert result.returncode == 2
    assert "E1002" in result.stderr and "`--warm`" in result.stderr
    assert "dummy" in result.stderr and "ppy run" in result.stderr
    assert not (path.parent / ".ppy-cache" / "backends").exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--unsafe", None),
        ("--standalone", None),
        ("--host-cpu", None),
        ("--python-extension", None),
        ("--library", None),
        ("--report-opt", None),
        ("--sanitize", "bounds"),
        ("--prover", "z3"),
        ("--target", "aarch64-linux-gnu"),
        ("--pgo", "some.ppyprof"),
        ("--report-opt-json", "report.json"),
    ],
)
def test_an_llvm_only_option_is_refused_rather_than_ignored(
    tmp_path: Path, monkeypatch, flag: str, value: str | None
):
    """Every option argparse accepts either means something to the backend or
    says so; none is quietly dropped."""
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    arguments = ["build", path.name, "--backend", "dummy", flag]
    if value is not None:
        arguments.append(value)
    result = _ppy(path.parent, *arguments)
    assert result.returncode == 2, result.stderr
    assert "E1002" in result.stderr and flag in result.stderr and "dummy" in result.stderr
    assert not (path.parent / ".ppy-cache" / "backends").exists(), "nothing was built"


def test_the_target_refusal_says_where_to_name_the_target_instead(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    result = _ppy(path.parent, "build", path.name, "--backend", "dummy", "--target", "rngd")
    assert result.returncode == 2
    assert "[tool.ppy.backends.dummy] target" in result.stderr


def test_the_llvm_road_keeps_every_one_of_those_options(tmp_path: Path, monkeypatch):
    """The refusals are about the backend chosen, not about the options."""
    path = _project(tmp_path)
    result = _ppy(path.parent, "build", path.name, "--unsafe", "-o", "out")
    assert result.returncode == 0, result.stderr
    assert (path.parent / "out").is_dir()


def test_output_and_opt_level_reach_an_external_build(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY)
    path = _project(tmp_path)
    result = _ppy(path.parent, "-O", "1", "build", path.name, "--backend", "dummy", "-o", "here")
    assert result.returncode == 0, result.stderr
    record = json.loads((path.parent / "here" / "build.json").read_text())
    assert record["opt_level"] == 1


# -- finding a format without importing every backend -------------------------------


def test_a_declared_format_imports_only_its_own_backend(tmp_path: Path, monkeypatch):
    """`ppy.backend-formats` says who owns what, so one heavy or broken SDK
    package beside the one asked for is never imported."""
    _install_backend(tmp_path, monkeypatch, DUMMY, formats=("dummy", "dummy-bin"))
    _install_backend(
        tmp_path,
        monkeypatch,
        """
        raise RuntimeError("this SDK is not importable here")
        """,
        name="heavy",
    )
    declared, problems = discover_format_owners()
    assert declared["dummy"] == "dummy" and problems == []
    assert "heavy_backend" not in sys.modules and "dummy_backend" not in sys.modules
    owner = emit_format_owner("dummy")
    assert owner.backend.name == "dummy" and owner.format.suffix == ".dummy"
    assert "heavy_backend" not in sys.modules, "the other backend was never imported"


def test_a_format_declared_for_a_backend_that_does_not_emit_it_is_an_error(
    tmp_path: Path, monkeypatch
):
    _install_backend(tmp_path, monkeypatch, DUMMY, formats=("dummy-elsewhere",))
    with pytest.raises(BackendLoadError, match=r"entry point and the backend disagree"):
        emit_format_owner("dummy-elsewhere")


def test_a_format_two_distributions_declare_is_not_settled_by_order(tmp_path: Path, monkeypatch):
    _install_backend(tmp_path, monkeypatch, DUMMY, formats=("dummy",))
    _install_backend(
        tmp_path,
        monkeypatch,
        DUMMY.replace('name = "dummy"', 'name = "rival"'),
        name="rival",
        formats=("dummy",),
    )
    declared, problems = discover_format_owners()
    assert "dummy" not in declared, "neither claimant wins by luck"
    assert any("claimed by more than one distribution" in problem for problem in problems)


def test_an_undeclared_format_is_still_found_by_asking_the_backends(tmp_path: Path, monkeypatch):
    """The manifest is optional: a backend that does not declare its formats
    still works, by the slower road."""
    _install_backend(tmp_path, monkeypatch, DUMMY)
    assert emit_format_owner("dummy-bin").backend.name == "dummy"


# -- the Python backend is a backend too -------------------------------------------

#: Everything `ppy build` accepts that belongs to the LLVM road alone.
LLVM_ONLY = [
    ("--unsafe", None),
    ("--standalone", None),
    ("--host-cpu", None),
    ("--python-extension", None),
    ("--library", None),
    ("--report-opt", None),
    ("--sanitize", "bounds"),
    ("--prover", "z3"),
    ("--pgo", "some.ppyprof"),
    ("--report-opt-json", "report.json"),
    ("--target", "aarch64-linux-gnu"),
]


def _llvm_artifacts(directory: Path) -> list[str]:
    """Anything the LLVM road writes: objects, a library, a manifest, a launcher."""
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.rglob("*")
        if path.suffix in {".o", ".so", ".dylib", ".dll"} or path.name == "ppy-bindings.json"
    )


def test_the_python_backend_builds_source_and_takes_the_generic_options(tmp_path: Path):
    path = _project(tmp_path)
    plain = _ppy(path.parent, "build", path.name, "--backend", "python")
    assert plain.returncode == 0, plain.stderr
    assert "built 1 module(s)" in plain.stderr
    cache = path.parent / ".ppy-cache"
    assert cache.is_dir() and _llvm_artifacts(cache) == [], "nothing native was built"
    levelled = _ppy(path.parent, "-O", "1", "build", path.name, "--backend", "python")
    assert levelled.returncode == 0, levelled.stderr


@pytest.mark.parametrize(("flag", "value"), LLVM_ONLY)
def test_an_llvm_only_option_is_refused_for_the_python_backend(
    tmp_path: Path, flag: str, value: str | None
):
    """The bug this guards: `--backend python` sat inside the LLVM branch, so
    every one of these was accepted and quietly dropped."""
    path = _project(tmp_path)
    arguments = ["build", path.name, "--backend", "python", flag]
    if value is not None:
        arguments.append(value)
    result = _ppy(path.parent, *arguments)
    assert result.returncode != 0, result.stderr
    assert "E1002" in result.stderr
    assert flag in result.stderr and "'python'" in result.stderr
    assert "LLVM" in result.stderr, "the message says whose option it is"
    assert _llvm_artifacts(path.parent) == [], "nothing was built"


def test_warm_is_refused_for_the_python_backend_before_anything_is_built(tmp_path: Path):
    path = _project(tmp_path)
    result = _ppy(path.parent, "build", path.name, "--warm", "--backend", "python")
    assert result.returncode != 0
    assert "E1002" in result.stderr and "`--warm`" in result.stderr
    assert "'python'" in result.stderr and "ppy run" in result.stderr
    assert _llvm_artifacts(path.parent) == [] and not (path.parent / ".ppy-cache").exists()


def test_the_python_backend_refuses_a_ppyir_target(tmp_path: Path):
    """The bug this guards: `ppy build foo.ppyir --backend python` ran the LLVM
    road and wrote an object, a shared library, and a manifest."""
    path = _project(tmp_path)
    emitted = _ppy(path.parent, "emit", "ir", path.name, "-o", "kernel.ppyir")
    assert emitted.returncode == 0, emitted.stderr
    result = _ppy(path.parent, "build", "kernel.ppyir", "--backend", "python", "-o", "out")
    assert result.returncode != 0, result.stderr
    assert "E1002" in result.stderr
    assert "the Python backend builds PPY source" in result.stderr
    assert "`.ppyir`" in result.stderr, "what it cannot build is named"
    assert _llvm_artifacts(path.parent) == [], "no LLVM artifact"
    assert not (path.parent / "out").exists(), "no artifact of any other backend"


def test_the_python_backend_refuses_an_output_directory_it_would_not_write(tmp_path: Path):
    """`-o` was accepted and ignored: the directory was never even created."""
    path = _project(tmp_path)
    result = _ppy(path.parent, "build", path.name, "--backend", "python", "-o", "out")
    assert result.returncode != 0
    assert "E1002" in result.stderr and "`-o`" in result.stderr
    assert "project cache" in result.stderr, "where its modules do go"
    assert not (path.parent / "out").exists()


def test_the_llvm_road_still_answers_warm_and_a_ppyir_target(tmp_path: Path):
    """The refusals are about the backend chosen; LLVM keeps both roads."""
    path = _project(tmp_path)
    emitted = _ppy(path.parent, "emit", "ir", path.name, "-o", "kernel.ppyir")
    assert emitted.returncode == 0, emitted.stderr
    built = _ppy(path.parent, "build", "kernel.ppyir", "-o", "out")
    assert built.returncode == 0, built.stderr
    assert "ppy-bindings.json" in _llvm_artifacts(path.parent / "out")
    warmed = _ppy(path.parent, "build", path.name, "--warm")
    assert warmed.returncode == 0, warmed.stderr


def test_one_policy_decides_the_options_of_every_backend_that_is_not_llvm(tmp_path: Path):
    """Python and an installed backend are refused by the same table, so the two
    cannot drift apart."""
    from ppy_compiler.driver import commands

    assert not hasattr(commands, "_external_build_refusal"), "one helper, not two"
    options = argparse.Namespace(warm=False, unsafe=True)
    for name in ("python", "toy"):
        refusal = commands._non_llvm_build_refusal(name, options)
        assert refusal is not None and f"{name!r}" in refusal.message
    assert commands._non_llvm_build_refusal("python", argparse.Namespace(warm=False)) is None
