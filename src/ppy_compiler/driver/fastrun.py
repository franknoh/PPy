"""The warm `ppy run FILE` before anything else is imported.

A warm run used to import the command line (argparse builds the whole parser),
read `pyproject.toml`, and read and hash every source under the project root to
recompute the key of an artifact it had built before: on a program that runs
in 30 ms, that alone made `ppy run` twice as slow as `python`. This module
remembers, per program, the manifest the last full run found or built, with a
signature of what that artifact depends on, and runs it when the signature
still matches: no parser, no configuration, no hashing.

The signature is proportional to the program, not to the project:

- each source the artifact was built from (the manifest records their paths
  and content digests), by size and modification time;
- each directory an import resolves through (the manifest's search paths and
  the directories of those sources), by modification time, which moves when
  a file is added, removed, or renamed there, so a new module that would
  shadow an import is seen;
- the project's `pyproject.toml` and `ppy.toml`, the compiler's fingerprint,
  the installed distributions, and the `PPY_*` environment.

`remember` takes its stats after the build, so it first checks each source's
content against the digest the build recorded; a file edited while the build
ran is left unremembered, and the next run takes the full path.

A program whose native code Python never calls has a light plan beside its
manifest (`ppy_runtime.launch.write_light`): its generated modules compiled,
run here with nothing of the launcher imported. Anything else goes through
`ppy_runtime.launch`. Only `ppy run FILE` and `ppy run FILE -- ARGS...` take
this path; a flag of any kind means the full command line.
"""

from __future__ import annotations

import marshal
import os
import sys

__all__ = ["MARKERS", "remember", "signature", "try_warm"]

#: What marks a project root; `config.find_project_root` looks for the same,
#: and a test holds the two together.
MARKERS = ("pyproject.toml", "ppy.toml", ".git")

_SUFFIXES = (".ppy", ".py")
_INDEX_VERSION = 2
_LIGHT = "light.marshal"


def _root(file: str) -> str:
    start = os.path.dirname(file)
    directory = start
    while True:
        for marker in MARKERS:
            if os.path.exists(os.path.join(directory, marker)):
                return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return start
        directory = parent


def _index_path(file: str) -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    identity = f"{file}\0{sys.executable}\0{sys.version}"
    # `hash()` is salted per process; this name has to be the same next time.
    name = format(_stable_hash(identity), "016x")
    return os.path.join(base, "ppy", "warm-index", f"{name}.marshal")


def _stable_hash(text: str) -> int:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def _stat(path: str) -> tuple[str, int, int] | tuple[str]:
    try:
        found = os.stat(path)
    except OSError:
        return (path,)
    return (path, found.st_size, found.st_mtime_ns)


def signature(file: str, sources: tuple[str, ...], directories: tuple[str, ...]) -> tuple:  # type: ignore[type-arg]
    """What the artifact of `file` depends on, as stats (see the module doc)."""
    from ..version import compiler_fingerprint

    root = _root(file)
    installed: set[str] = set()
    for place in sys.path:
        try:
            with os.scandir(place or ".") as entries:
                installed.update(e.name for e in entries if e.name.endswith(".dist-info"))
        except OSError:
            continue
    return (
        _INDEX_VERSION,
        compiler_fingerprint(),
        tuple(_stat(path) for path in sources),
        tuple(_stat(path) for path in directories),
        tuple(_stat(os.path.join(root, name)) for name in ("pyproject.toml", "ppy.toml")),
        tuple(sorted(installed)),
        tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith("PPY_"))),
    )


def _digest(path: str) -> str | None:
    import hashlib  # pylint: disable=import-outside-toplevel

    try:
        with open(path, encoding="utf-8") as held:
            text = held.read()
    except (OSError, UnicodeDecodeError):
        return None
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def remember(file: str, manifest: str) -> None:
    """Record that `ppy run file` runs `manifest`, with the signature of what it
    was built from. Best effort: a cache that cannot be written, a manifest
    that does not say its sources, or a source that changed since it was
    built (an edit during the build) leaves nothing remembered."""
    import json  # pylint: disable=import-outside-toplevel

    try:
        with open(manifest, encoding="utf-8") as held:
            program = json.load(held).get("program") or {}
    except (OSError, ValueError):
        return
    recorded = program.get("sources")
    if not isinstance(recorded, dict) or not recorded:
        return
    for path, digest in recorded.items():
        if _digest(path) != digest:
            return
    sources = tuple(sorted(recorded))
    directories = tuple(
        sorted({os.path.dirname(p) for p in sources} | set(program.get("search_paths", ())))
    )
    light = os.path.join(os.path.dirname(manifest), _LIGHT)
    entry = (
        _INDEX_VERSION,
        sources,
        directories,
        signature(file, sources, directories),
        manifest,
        light if os.path.isfile(light) else None,
    )
    path = _index_path(file)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        draft = f"{path}.{os.getpid()}.part"
        with open(draft, "wb") as out:
            marshal.dump(entry, out)
        os.replace(draft, path)
    except (OSError, ValueError):
        pass


def try_warm(argv: list[str]) -> int | None:
    """Run the remembered artifact of `ppy run FILE [-- ARGS...]`, or None to
    take the full path."""
    if len(argv) < 2 or argv[0] != "run":
        return None
    named, rest = argv[1], argv[2:]
    if named.startswith("-") or not named.endswith(_SUFFIXES) or (rest and rest[0] != "--"):
        return None
    file = os.path.abspath(named)
    if not os.path.isfile(file):
        return None
    try:
        with open(_index_path(file), "rb") as held:
            version, sources, directories, taken, manifest, light = marshal.load(held)
    except (OSError, EOFError, ValueError, TypeError):
        return None
    if version != _INDEX_VERSION or not os.path.isfile(manifest):
        return None
    if taken != signature(file, sources, directories):
        return None
    program_args = rest[1:]
    if light is not None:
        ran = _run_light(light, program_args)
        if ran is not None:
            return ran
    from ppy_runtime.launch import main as launch

    return launch(manifest, program_args)


def _run_light(light: str, argv: list[str]) -> int | None:
    """Run a light plan as `ppy_runtime.execute` runs generated modules, or None
    when there is no usable plan."""
    try:
        with open(light, "rb") as held:
            plan = marshal.load(held)
    except (OSError, EOFError, ValueError, TypeError):
        return None
    if not isinstance(plan, dict) or plan.get("python") != sys.version:
        return None
    modules: dict = plan["modules"]  # type: ignore[type-arg]
    entry = plan["entry"]
    if entry not in modules:
        return None
    if plan["uses_ppy"]:
        # As `install_loader` does: the runtime's own finder first, then told
        # the compiler serves this process.
        try:
            import ppy  # pylint: disable=import-outside-toplevel
            from ppy import _native  # pylint: disable=import-outside-toplevel
        except ImportError:
            pass
        else:
            ppy.install()
            _native.managed()
    finder = _Finder({name: m for name, m in modules.items() if name != entry})
    sys.meta_path.insert(0, finder)
    for extra in reversed(plan["search_paths"]):
        sys.path.insert(0, extra)
    source, code = modules[entry]
    sys.argv = [source, *argv]
    module = type(sys)("__main__")
    module.__dict__.update(
        {"__file__": source, "__builtins__": __builtins__, "__package__": None, "__spec__": None}
    )
    sys.modules["__main__"] = module
    try:
        exec(code, module.__dict__)
    except SystemExit as exit_request:
        status = exit_request.code
        return status if isinstance(status, int) else (0 if status is None else 1)
    except BaseException as exc:  # noqa: BLE001 - reported to the user verbatim
        import traceback  # pylint: disable=import-outside-toplevel

        sys.stderr.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return 1
    return 0


class _Finder:
    """Serves the plan's other modules to `import`, under their `.ppy` names."""

    def __init__(self, modules: dict) -> None:  # type: ignore[type-arg]
        self.modules = modules

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[no-untyped-def]
        found = self.modules.get(fullname)
        if found is None:
            return None
        from importlib.machinery import ModuleSpec  # pylint: disable=import-outside-toplevel

        spec = ModuleSpec(fullname, _Loader(*found), origin=found[0])
        spec.has_location = True
        return spec


class _Loader:
    def __init__(self, source: str, code: object) -> None:
        self.source = source
        self.code = code

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return None

    def exec_module(self, module) -> None:  # type: ignore[no-untyped-def]
        module.__file__ = self.source
        exec(self.code, module.__dict__)
