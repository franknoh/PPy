"""Content-addressed incremental cache (spec 27)."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import ClassVar

import pytest

from ppy_compiler.cache import CacheKey, CacheStore, environment_fingerprint
from ppy_compiler.cache.store import _in_memory


@pytest.fixture
def store(tmp_path: Path) -> CacheStore:
    store = CacheStore(tmp_path / ".ppy-cache")
    store.ensure()
    return store


def _key(source: str = "abc", **overrides) -> CacheKey:
    options = {
        "source_digest": source,
        "compiler_version": "0.1.0",
        "opt_level": 2,
        "target": "python",
    }
    options.update(overrides)
    return CacheKey.build("python", **options)


def test_keys_are_content_addressed(store: CacheStore):
    assert _key("a").hex() == _key("a").hex()
    assert _key("a").hex() != _key("b").hex()


@pytest.mark.parametrize(
    "overrides",
    [
        {"opt_level": 3},
        {"compiler_version": "0.2.0"},
        {"target": "llvm"},
        {"directives": ("f:pure",)},
        {"dependency_hashes": ("dep",)},
        {"plugin_fingerprints": ("numpy:2.0",)},
    ],
)
def test_every_key_input_changes_the_key(overrides):
    assert _key().hex() != _key(**overrides).hex()


def test_environment_fingerprint_covers_the_interpreter():
    assert environment_fingerprint() == environment_fingerprint()
    assert len(environment_fingerprint()) == 64


def test_put_and_read_round_trip(store: CacheStore):
    key = _key()
    store.put(key, "print(1)", source="foo.ppy")
    assert store.read_text(key) == "print(1)"
    assert store.get(_key("other")) is None


def test_identical_content_is_stored_once(store: CacheStore):
    first = store.put(_key("a"), "same", source="a.ppy")
    second = store.put(_key("b"), "same", source="b.ppy")
    assert first == second
    assert store.stats().entries == 2


def test_writes_are_atomic(store: CacheStore):
    key = _key()
    path = store.put(key, "payload")
    assert path.exists()
    assert not list(path.parent.glob(".tmp-*"))


def test_stats_group_by_kind(store: CacheStore):
    store.put(_key("a"), "x", kind="python")
    store.put(_key("b"), "yy", kind="llvm")
    stats = store.stats()
    assert stats.entries == 2
    assert stats.by_kind["python"][0] == 1
    assert stats.by_kind["llvm"][0] == 1
    assert stats.human_total().endswith("B")


def test_gc_keeps_reachable_artifacts(store: CacheStore):
    root = _key("root")
    dependency = _key("dep")
    store.put(dependency, "dep")
    store.put(root, "root", dependencies=[dependency.hex()])
    store.mark_root(root)

    removed, _freed = store.gc(max_age_days=None)
    assert removed == 0
    assert store.read_text(root) == "root"
    assert store.read_text(dependency) == "dep"


def test_gc_removes_unreachable_artifacts(store: CacheStore):
    orphan = _key("orphan")
    store.put(orphan, "orphan")
    removed, freed = store.gc(max_age_days=None)
    assert removed == 1 and freed > 0
    assert store.get(orphan) is None


def test_gc_removes_expired_artifacts(store: CacheStore):
    key = _key()
    store.put(key, "old")
    store.mark_root(key)
    store.connect().execute(
        "UPDATE artifacts SET accessed=? WHERE key=?", (time.time() - 86400 * 100, key.hex())
    )
    removed, _freed = store.gc(max_age_days=30)
    assert removed == 1


def test_gc_collects_orphan_objects(store: CacheStore):
    store.put(_key(), "payload")
    store.gc(max_age_days=None)
    remaining = [p for p in (store.root / "objects").rglob("*") if p.is_file()]
    assert remaining == []


def test_clean_resets_the_cache(store: CacheStore):
    store.put(_key(), "payload")
    store.clean()
    assert store.stats().entries == 0
    assert (store.root / "objects").is_dir()


def test_invalidate_by_source(store: CacheStore):
    store.put(_key("a"), "one", source="mod.ppy")
    store.put(_key("b"), "two", source="mod.ppy")
    store.put(_key("c"), "three", source="other.ppy")
    assert store.invalidate_source("mod.ppy") == 2
    assert store.stats().entries == 1


def test_two_stores_share_one_cache(tmp_path: Path):
    root = tmp_path / ".ppy-cache"
    first = CacheStore(root)
    first.ensure()
    key = _key()
    first.put(key, "shared")

    second = CacheStore(root)
    assert second.read_text(key) == "shared"


def test_function_level_abi_hash_ignores_private_bodies(write, analyze):
    from ppy_compiler.driver.pipeline import _public_abi_hash

    write(
        "dep.ppy",
        "def public(x: int) -> int:\n    return helper(x)\n\ndef helper(x: int) -> int:\n    return x + 1\n",
    )
    path = write("app.ppy", "import dep\n\ndef use() -> int:\n    return dep.public(1)\n")
    before = _public_abi_hash(analyze(path), "dep")

    (path.parent / "dep.ppy").write_text(
        "def public(x: int) -> int:\n    return helper(x)\n\ndef helper(x: int) -> int:\n    return x + 1 + 0\n",
        encoding="utf-8",
    )
    after = _public_abi_hash(analyze(path), "dep")
    assert before == after


def test_changing_a_public_signature_changes_the_abi_hash(write, analyze):
    from ppy_compiler.driver.pipeline import _public_abi_hash

    write("dep2.ppy", "def public(x: int) -> int:\n    return x\n")
    path = write("app2.ppy", "import dep2\n\ndef use() -> int:\n    return dep2.public(1)\n")
    before = _public_abi_hash(analyze(path), "dep2")

    (path.parent / "dep2.ppy").write_text(
        "def public(x: int) -> float:\n    return float(x)\n", encoding="utf-8"
    )
    after = _public_abi_hash(analyze(path), "dep2")
    assert before != after


def test_changing_a_dependency_invalidates_the_dependent(write, analyze):
    from ppy_compiler.driver.pipeline import build_python

    write("lib3.ppy", "def value() -> int:\n    return 1\n")
    path = write("main3.ppy", "import lib3\n\ndef use() -> int:\n    return lib3.value()\n")
    build_python(analyze(path))

    (path.parent / "lib3.ppy").write_text(
        "def value() -> float:\n    return 1.0\n", encoding="utf-8"
    )
    output = build_python(analyze(path))
    assert output.stats.get("cache_misses", 0) >= 1


def test_a_plugin_is_not_fingerprinted_for_a_module_that_ignores_it():
    """Fingerprinting imports the library, so it must follow actual imports."""
    from ppy_compiler.plugins.base import PluginRegistry

    calls: list[str] = []

    class _Fake:
        def __init__(self, name: str, modules: tuple[str, ...]) -> None:
            self.name = name
            self.modules = modules

        def fingerprint(self) -> str:
            calls.append(self.name)
            return f"{self.name}-v1"

        def external_types(self) -> dict[str, str]:
            return {}

    registry = PluginRegistry()
    registry.register(_Fake("heavy", ("torch",)))
    registry.register(_Fake("light", ("numpy",)))

    assert registry.fingerprints({"numpy"}) == ("light:light-v1",)
    assert calls == ["light"]

    calls.clear()
    assert len(registry.fingerprints(None)) == 2
    assert sorted(calls) == ["heavy", "light"]


def test_a_module_that_does_import_the_plugin_still_pins_its_version():
    from ppy_compiler.plugins.base import PluginRegistry

    class _Fake:
        name = "torch"
        modules = ("torch",)

        def fingerprint(self) -> str:
            return "torch=2.11"

        def external_types(self) -> dict[str, str]:
            return {}

    registry = PluginRegistry()
    registry.register(_Fake())
    assert registry.fingerprints({"torch.nn"}) == ("torch:torch=2.11",)
    assert not registry.fingerprints({"json"})


def test_the_index_is_closed_at_exit(tmp_path: Path):
    """An open sqlite handle is reported as a ResourceWarning on shutdown."""
    import subprocess
    import sys

    script = tmp_path / "use.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r})\n"
        "from ppy_compiler.cache.store import CacheStore\n"
        f"from pathlib import Path\nstore = CacheStore(Path({str(tmp_path / 'cache')!r}))\n"
        "store.connect()\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, "-W", "always", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "unclosed database" not in done.stderr


def _build(workspace: Path, *args: str):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


def test_an_object_file_is_reused_rather_than_regenerated(tmp_path: Path):
    """Code generation is deterministic, so the same key means the same bytes."""
    pytest.importorskip("llvmlite")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    (tmp_path / "hot.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef f(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "build", "hot.ppy").returncode == 0
    status = _build(tmp_path, "cache", "status").stdout
    assert "native" in status, status

    from ppy_compiler.backend.llvm import _object_key
    from ppy_compiler.driver.pipeline import analyze_paths, module_cache_key, open_project

    path = tmp_path / "hot.ppy"
    project = open_project(path)
    # The probe must speak the same cache-key dialect as `ppy build`, whose
    # default guard mode is the wrap-semantics `off`.
    project.config.llvm.safeguards = "off"
    bundle = analyze_paths(project, [path], backend="llvm")
    key = module_cache_key(bundle, "hot", target="llvm", opt_level=3)
    store = bundle.project.store
    assert store.read(_object_key(key)) is not None


def test_changing_the_source_invalidates_the_object(tmp_path: Path):
    pytest.importorskip("llvmlite")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    source = tmp_path / "hot.ppy"
    source.write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\n"
        "def f(x: int) -> int:\n    return x * 2\n\n\nprint(f(21))\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "run", "hot.ppy").stdout.strip().endswith("42")
    source.write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\n"
        "def f(x: int) -> int:\n    return x * 3\n\n\nprint(f(21))\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "run", "hot.ppy").stdout.strip().endswith("63")


def test_the_optimization_level_is_part_of_the_object_key(tmp_path: Path):
    pytest.importorskip("llvmlite")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    (tmp_path / "hot.ppy").write_text(
        "import ppy\n\n\ndef f(x: int) -> int:\n    return x * 2\n", encoding="utf-8"
    )
    from ppy_compiler.backend.llvm import _object_key
    from ppy_compiler.driver.pipeline import analyze_paths, module_cache_key, open_project

    path = tmp_path / "hot.ppy"
    bundle = analyze_paths(open_project(path), [path], backend="llvm")
    keys = {
        _object_key(module_cache_key(bundle, "hot", target="llvm", opt_level=level))
        for level in (0, 1, 2, 3)
    }
    assert len(keys) == 4


def test_is_generator_is_computed_once(tmp_path: Path):
    """`signature()` asks for it once per function per checked function."""
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "gen.ppy"
    path.write_text(
        "def plain(x: int) -> int:\n    return x\n\n\n"
        "def yielding(n: int):\n    for i in range(n):\n        yield i\n",
        encoding="utf-8",
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    functions = bundle.symbols.modules["gen"].functions
    assert not functions["plain"].is_generator
    assert functions["yielding"].is_generator
    # The answer is kept, so asking again does not walk the body a second time.
    assert functions["plain"]._generator is False
    assert functions["yielding"]._generator is True


def _project(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[tool.ppy]\nopt-level = 3\nsource-roots = ["src"]\n', encoding="utf-8"
    )
    (root / "src").mkdir(exist_ok=True)


def test_a_rebuild_with_no_change_recompiles_nothing(tmp_path: Path):
    pytest.importorskip("llvmlite")
    _project(tmp_path)
    (tmp_path / "src" / "hot.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef f(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "build", "src").returncode == 0

    from ppy_compiler.backend.llvm import _cached_lowering, _object_key
    from ppy_compiler.driver.pipeline import (
        analyze_paths,
        collect_sources,
        module_cache_key,
        open_project,
    )

    sources = list(collect_sources(tmp_path / "src", ppy_only=True))
    project = open_project(tmp_path / "src")
    # The probe must speak the same cache-key dialect as `ppy build`, whose
    # default guard mode is the wrap-semantics `off`.
    project.config.llvm.safeguards = "off"
    bundle = analyze_paths(project, sources, backend="llvm")
    key = module_cache_key(bundle, "hot", target="llvm", opt_level=3)
    assert _cached_lowering(bundle, "hot", 3) is not None, "lowering was not cached"
    assert bundle.project.store.read(_object_key(key)) is not None, "the object was not cached"


def test_only_the_changed_module_is_recompiled(tmp_path: Path):
    pytest.importorskip("llvmlite")
    _project(tmp_path)
    for name in ("a", "b"):
        (tmp_path / "src" / f"mod_{name}.ppy").write_text(
            f"import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef {name}(x: int) -> int:\n    return x * 2\n",
            encoding="utf-8",
        )
    assert _build(tmp_path, "build", "src").returncode == 0

    from ppy_compiler.backend.llvm import _object_key
    from ppy_compiler.driver.pipeline import (
        analyze_paths,
        collect_sources,
        module_cache_key,
        open_project,
    )

    def object_keys() -> dict[str, str]:
        sources = list(collect_sources(tmp_path / "src", ppy_only=True))
        project = open_project(tmp_path / "src")
        # The probe must speak the same cache-key dialect as `ppy build`,
        # whose default guard mode is the wrap-semantics `off`.
        project.config.llvm.safeguards = "off"
        bundle = analyze_paths(project, sources, backend="llvm")
        return {
            name: _object_key(module_cache_key(bundle, name, target="llvm", opt_level=3))
            for name in ("mod_a", "mod_b")
        }

    before = object_keys()
    (tmp_path / "src" / "mod_a.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef a(x: int) -> int:\n    return x * 3\n",
        encoding="utf-8",
    )
    after = object_keys()
    assert after["mod_a"] != before["mod_a"], "the changed module kept its key"
    assert after["mod_b"] == before["mod_b"], "an untouched module was invalidated"


def test_changing_a_dependency_invalidates_its_dependent(tmp_path: Path):
    """The dependent's key covers the public summaries it compiled against."""
    pytest.importorskip("llvmlite")
    _project(tmp_path)
    (tmp_path / "src" / "lib.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef base(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "app.ppy").write_text(
        "import ppy\n\nimport lib\n\n\n@ppy.pure\n@ppy.opt(3)\n"
        "def doubled(x: int) -> int:\n    return lib.base(x) + 1\n\n\nprint(doubled(10))\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "run", "src/app.ppy").stdout.strip().endswith("21")

    (tmp_path / "src" / "lib.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef base(x: int) -> int:\n    return x * 5\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "run", "src/app.ppy").stdout.strip().endswith("51")


def test_a_fully_cached_build_needs_no_llvm(tmp_path: Path):
    """Nothing is compiled, so the backend is never initialized."""
    pytest.importorskip("llvmlite")
    _project(tmp_path)
    (tmp_path / "src" / "hot.ppy").write_text(
        "import ppy\n\n\n@ppy.pure\n@ppy.opt(3)\ndef f(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )
    assert _build(tmp_path, "build", "src").returncode == 0

    import subprocess
    import sys

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        "from ppy_compiler.driver.cli import main\n"
        "raise SystemExit(main(['build', 'src']) or "
        "(1 if any('llvmlite.binding' in m for m in sys.modules) else 0))\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, str(probe)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, "a fully cached build still loaded LLVM"


def test_a_cached_lowering_round_trips():
    from ppy_compiler.backend.llvm.lowering import NativeParam, NativeSignature
    from ppy_compiler.backend.llvm.lowering_cache import decode, encode

    class _Module:
        name = "m"
        ir = "; ir"
        functions: ClassVar[dict] = {
            "m.f": type(
                "L",
                (),
                {
                    "signature": NativeSignature(
                        "m.f",
                        "ppy_m_f",
                        (NativeParam("xs", "view", "float"), NativeParam("n", "int")),
                        ("i64",),
                        releases_gil=True,
                    )
                },
            )()
        }
        rejected: ClassVar[dict] = {"m.g": "has effects"}
        fused: ClassVar[dict] = {}
        fusion_plan: ClassVar[dict] = {}
        fusion_notes: ClassVar[list] = []

    restored = decode(encode(_Module()))
    assert restored is not None
    signature = restored.signatures["m.f"]
    assert signature.symbol == "ppy_m_f"
    assert signature.releases_gil
    assert signature.parameters[0].kind == "view"
    assert signature.parameters[0].element == "float"
    assert restored.rejected == {"m.g": "has effects"}
    assert decode('{"version": 0}') is None
    assert decode("not json") is None


def test_host_targeting_is_part_of_the_cache_key(write, analyze):
    """A baseline object and a host-specific one are different artifacts."""
    from ppy_compiler.driver.pipeline import module_cache_key

    path = write("keyed.ppy", "value: int = 1\n")
    bundle = analyze(path, backend="llvm")
    bundle.project.config.llvm.safeguards = "off"

    bundle.project.config.llvm.host_cpu = False
    portable = module_cache_key(bundle, "keyed", target="llvm", opt_level=2)
    bundle.project.config.llvm.host_cpu = True
    tuned = module_cache_key(bundle, "keyed", target="llvm", opt_level=2)
    assert portable != tuned


def test_the_compiler_fingerprint_tracks_the_build(monkeypatch):
    """A dev tree keys caches on its own sources; an override wins outright."""
    from ppy_compiler.driver import pipeline

    pipeline.compiler_fingerprint.cache_clear()
    monkeypatch.setenv("PPY_COMPILER_BUILD", "release-abc")
    assert pipeline.compiler_fingerprint() == "release-abc"
    pipeline.compiler_fingerprint.cache_clear()
    monkeypatch.delenv("PPY_COMPILER_BUILD")
    fingerprint = pipeline.compiler_fingerprint()
    assert fingerprint and fingerprint != "release-abc"
    pipeline.compiler_fingerprint.cache_clear()


def test_the_cache_root_honours_the_environment(monkeypatch, tmp_path):
    """`PPY_CACHE_DIR` moves the store off a slow filesystem, per project."""
    from ppy_compiler.driver.config import Config

    monkeypatch.setenv("PPY_CACHE_DIR", str(tmp_path / "store"))
    one = Config(root=tmp_path / "one")
    two = Config(root=tmp_path / "two")
    assert one.cache_path != two.cache_path
    assert one.cache_path.is_relative_to(tmp_path / "store")
    monkeypatch.delenv("PPY_CACHE_DIR")
    assert Config(root=tmp_path / "one").cache_path == tmp_path / "one" / ".ppy-cache"


def test_a_damaged_index_is_quarantined_and_rebuilt(tmp_path: Path):
    """A cache is optional speed; damage costs a rebuild, never the answer."""
    store = CacheStore(tmp_path / "cache")
    store.put("aaaa", "first", kind="python")
    store.close()

    store.index_path.write_bytes(os.urandom(4096))
    reopened = CacheStore(tmp_path / "cache")
    assert reopened.get("aaaa") is None, "the entry is gone, which is a miss"

    reopened.put("bbbb", "second", kind="python")
    assert reopened.read_text("bbbb") == "second", "and the store works again"
    assert reopened.quarantined is not None
    assert reopened.quarantined.exists(), "the damaged index is kept, not deleted"
    assert not reopened.disabled


def test_damage_found_while_reading_is_recovered(tmp_path: Path):
    """Corruption shows up mid-query as easily as at open."""
    store = CacheStore(tmp_path / "cache")
    store.put("cccc", "value", kind="python")
    connection = store.connect()
    connection.close()  # every later statement raises ProgrammingError/DatabaseError

    # A store whose connection is unusable answers misses and repairs itself.
    store._connection = _Damaged()
    assert store.get("cccc") is None
    assert store.read_text("cccc") is None


class _Damaged:
    """A connection that fails the way a corrupt index does."""

    def execute(self, *_args, **_keywords):
        raise sqlite3.DatabaseError("database disk image is malformed")

    def close(self) -> None:
        raise sqlite3.DatabaseError("database disk image is malformed")


def test_an_object_the_index_lost_reads_as_a_miss(tmp_path: Path):
    """The index may outlive the file it points at."""
    store = CacheStore(tmp_path / "cache")
    path = store.put("dddd", "payload", kind="python")
    path.unlink()
    assert store.get("dddd") is None
    assert store.read("dddd") is None


def test_a_locked_index_is_stepped_around_not_destroyed(tmp_path: Path):
    """Busy is another process holding a good index, not a damaged one."""
    root = tmp_path / "cache"
    store = CacheStore(root)
    store.put("aaaa", "value", kind="python")
    store.close()

    holder = sqlite3.connect(store.index_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        blocked = CacheStore(root)
        # A miss is always a safe answer; destroying the file is not.
        blocked.get("aaaa")
        assert blocked.quarantined is None, "nothing was moved aside"
        assert not list(root.glob("*.corrupt-*"))
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    # The index the other process was holding is intact.
    assert CacheStore(root).read_text("aaaa") == "value"


def test_a_store_that_stepped_aside_still_answers_its_own_writes(tmp_path: Path):
    """The regression CI caught: `put` then `read_text` came back empty.

    A moment of contention used to disable the store for the rest of its
    life, and a disabled store refused even the entry it had just written --
    the object was on disk, the caller got nothing back.
    """
    root = tmp_path / "cache"
    store = CacheStore(root)
    store.disabled = True
    store._connection = _in_memory()

    store.put("bbbb", "value", kind="python")
    assert store.read_text("bbbb") == "value", "its own write, whatever the index is"


def test_contention_mid_write_costs_a_retry_and_not_the_entry(tmp_path: Path, monkeypatch):
    """Busy on the transaction is waited out, not taken as a verdict."""
    store = CacheStore(tmp_path / "cache")
    store.put("warm", "value", kind="python")

    real = store._transaction
    refusals = [0]

    def busy_once():
        """Refuse the first transaction the way a held index does."""
        if refusals[0] < 1:
            refusals[0] += 1
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = 5
            raise error
        return real()

    monkeypatch.setattr(store, "_transaction", busy_once)
    store.put("cccc", "value-c", kind="python")
    assert refusals[0] == 1, "the first attempt was refused"
    assert store.read_text("cccc") == "value-c"
    assert not store.disabled, "one busy moment is not a broken cache"


def test_a_cache_that_cannot_be_written_is_simply_no_cache(tmp_path: Path):
    """Nowhere to keep an artifact is a slow build, not a failed one."""
    root = tmp_path / "locked-out"
    root.mkdir()
    root.chmod(0o500)
    try:
        store = CacheStore(root / "cache")
        store.put("bbbb", "value", kind="python")
        assert store.get("bbbb") is None
        assert store.disabled
    finally:
        root.chmod(0o700)


def test_a_read_only_cache_still_compiles(tmp_path: Path):
    """Nowhere to write is a slow build, not a broken one."""
    root = tmp_path / "cache"
    store = CacheStore(root)
    store.put("eeee", "value", kind="python")
    store.close()
    store.index_path.write_bytes(os.urandom(2048))
    root.chmod(0o500)
    try:
        reopened = CacheStore(root)
        assert reopened.get("eeee") is None
        # An in-memory index keeps the interface working; nothing persists.
        reopened.put("ffff", "value", kind="python")
    finally:
        root.chmod(0o700)


def test_concurrent_writers_share_one_index(tmp_path: Path):
    """Several compilations against one cache is the normal case, not a race."""
    import concurrent.futures

    root = tmp_path / "cache"
    CacheStore(root).close()

    def store(index: int) -> str:
        own = CacheStore(root)
        own.put(f"{index:04x}", f"value-{index}", kind="python", dependencies=[f"dep-{index}"])
        text = own.read_text(f"{index:04x}")
        own.close()
        return text or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        written = list(pool.map(store, range(24)))
    assert written == [f"value-{i}" for i in range(24)]

    reader = CacheStore(root)
    assert all(reader.read_text(f"{i:04x}") == f"value-{i}" for i in range(24))


def test_a_failed_write_leaves_no_half_recorded_artifact(tmp_path: Path):
    """The artifact row and its dependencies land together or not at all."""
    store = CacheStore(tmp_path / "cache")
    with pytest.raises(RuntimeError), store._transaction() as connection:
        connection.execute(
            "INSERT INTO artifacts(key, kind, object, size, created, accessed, source) "
            "VALUES('abcd','python','obj',1,0,0,'')"
        )
        raise RuntimeError("interrupted")
    assert store.get("abcd") is None


def test_damage_is_judged_by_sqlites_code_not_by_its_wording():
    """A message is localized and reworded; the error code is the authority.

    "database is locked" and "database disk image is malformed" differ by a
    few words, and only one of them means the file should be thrown away.
    """
    from ppy_compiler.cache.store import _is_damage

    def raised(code: int, message: str) -> sqlite3.DatabaseError:
        error = sqlite3.DatabaseError(message)
        error.sqlite_errorcode = code
        return error

    assert _is_damage(raised(26, "file is not a database"))
    assert _is_damage(raised(11, "database disk image is malformed"))
    assert _is_damage(raised(267, "database disk image is malformed")), "an extended code"
    # Busy and locked keep their index, whatever the message happens to say.
    assert not _is_damage(raised(5, "database is locked"))
    assert not _is_damage(raised(6, "database table is locked"))
    assert not _is_damage(raised(5, "the corrupt word appears in this message"))
    # Only where a driver reports no code at all does the wording decide.
    assert _is_damage(sqlite3.DatabaseError("database disk image is malformed"))
    assert not _is_damage(sqlite3.OperationalError("database is locked"))
