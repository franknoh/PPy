"""Content-addressed incremental cache (spec 27)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ppy_compiler.cache import CacheKey, CacheStore, digest, environment_fingerprint


@pytest.fixture
def store(tmp_path: Path) -> CacheStore:
    store = CacheStore(tmp_path / ".ppy-cache")
    store.ensure()
    return store


def _key(source: str = "abc", **overrides) -> CacheKey:
    options = dict(
        source_digest=source,
        compiler_version="0.1.0",
        opt_level=2,
        target="python",
    )
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

    write("dep.ppy", "def public(x: int) -> int:\n    return helper(x)\n\ndef helper(x: int) -> int:\n    return x + 1\n")
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

    (path.parent / "dep2.ppy").write_text("def public(x: int) -> float:\n    return float(x)\n", encoding="utf-8")
    after = _public_abi_hash(analyze(path), "dep2")
    assert before != after


def test_changing_a_dependency_invalidates_the_dependent(write, analyze):
    from ppy_compiler.driver.pipeline import build_python

    write("lib3.ppy", "def value() -> int:\n    return 1\n")
    path = write("main3.ppy", "import lib3\n\ndef use() -> int:\n    return lib3.value()\n")
    build_python(analyze(path))

    (path.parent / "lib3.ppy").write_text("def value() -> float:\n    return 1.0\n", encoding="utf-8")
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
    assert registry.fingerprints({"json"}) == ()
