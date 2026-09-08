"""`ppy run --profile` and `--pgo`: a program's profile written and read (spec 84).

A profiling run is the JIT with the `instrument-profile` pass in the
pipeline: every native function counts its blocks and branches into a
counter array the module carries, and the boundary records what kinds of
value each native function was called with -- an `int`, an
`ndarray[float64;4x3]`, a `DataFrame[a:int64,b:float64;1000 rows]`. When
the program ends the collector reads the arrays through the engine, joins
them with the legend each module wrote next to its counters, and writes
the `.ppyprof`; a profile already there is merged, so several runs add up.
`profile_for` hands a build the profile its configuration names, read
once, and `profile_digest` is what the cache keys carry for it.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..ir.transforms.profile import FunctionProfile, Profile, ProfileError

__all__ = [
    "Collector",
    "Profile",
    "ProfileError",
    "default_output",
    "describe",
    "load_profile",
    "profile_digest",
    "profile_for",
]

_LEGEND = re.compile(r'@"?(__ppy_prof_map_\w+)"?\s*=')


def default_output(entry: Path) -> Path:
    """`<entry>.ppyprof` in the working directory, where the next `--pgo` looks."""
    return Path.cwd() / f"{Path(entry).stem}.ppyprof"


def describe(value: Any) -> str:
    """The kind of a value at the boundary, with the shape or schema the compiler could use."""
    kind = type(value)
    if value is None:
        return "None"
    if kind in (int, float, bool, str, bytes):
        return kind.__name__
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        family = "ndarray" if kind.__module__.startswith("numpy") else kind.__name__
        try:
            spelled = "x".join(str(int(d)) for d in shape)
        except (TypeError, ValueError):
            spelled = "?"
        return f"{family}[{dtype};{spelled}]"
    if kind.__name__ == "DataFrame":
        dtypes = getattr(value, "dtypes", None)
        try:
            columns = ",".join(f"{c}:{t}" for c, t in dtypes.items())  # type: ignore[union-attr]
            rows = len(value)
        except (AttributeError, TypeError):
            return "DataFrame"
        return f"DataFrame[{columns};{rows} rows]"
    if isinstance(value, (list, tuple, dict, set)):
        return f"{kind.__name__}[{len(value)}]"
    return kind.__name__


class Collector:
    """Gathers one run's profile: boundary calls as they happen, counters at the end."""

    def __init__(self) -> None:
        self._arguments: dict[str, dict[int, dict[str, int]]] = {}
        self._calls: dict[str, int] = {}

    def wrap(self, qualname: str, function: Callable[..., Any]) -> Callable[..., Any]:
        """`function`, recording each call's argument kinds under `qualname`."""
        arguments = self._arguments.setdefault(qualname, {})
        calls = self._calls

        def recorded(*args: Any) -> Any:
            calls[qualname] = calls.get(qualname, 0) + 1
            for position, value in enumerate(args):
                kinds = arguments.setdefault(position, {})
                kind = describe(value)
                kinds[kind] = kinds.get(kind, 0) + 1
            return function(*args)

        for attribute in ("__name__", "__qualname__", "__doc__", "__module__"):
            try:
                setattr(recorded, attribute, getattr(function, attribute))
            except (AttributeError, TypeError):
                continue
        return recorded

    def harvest(self, engine, natives, compiler: str, program: str) -> Profile:  # type: ignore[no-untyped-def]
        """The run's profile: counters read through `engine`, joined with each module's legend."""
        profile = Profile(runs=1, compiler=compiler, program=program)
        for native in natives.values():
            if not native.ir:
                continue
            for legend_symbol in _LEGEND.findall(native.ir):
                counters_symbol = legend_symbol.replace(
                    "__ppy_prof_map_", "__ppy_prof_counters_", 1
                )
                legend_address = engine.global_address(legend_symbol)
                counters_address = engine.global_address(counters_symbol)
                if not legend_address or not counters_address:
                    continue
                table = json.loads(ctypes.string_at(legend_address).decode("utf-8"))
                values = list(
                    (ctypes.c_int64 * max(int(table["counters"]), 1)).from_address(counters_address)
                )
                for symbol, entry in table["functions"].items():
                    blocks = {label: values[index] for label, index in entry["blocks"].items()}
                    branches: dict[str, tuple[int, int]] = {}
                    for label, index in entry["edges"].items():
                        taken = values[index]
                        branches[label] = (taken, max(blocks.get(label, 0) - taken, 0))
                    profile.functions[str(entry["qualname"])] = FunctionProfile(
                        symbol=symbol,
                        cfg=str(entry["cfg"]),
                        calls=blocks.get(str(entry["entry"]), 0),
                        blocks=blocks,
                        branches=branches,
                        generic=str(entry.get("generic", "")),
                    )
        for qualname, positions in self._arguments.items():
            record = profile.functions.get(qualname)
            if record is None:
                record = profile.functions[qualname] = FunctionProfile(
                    calls=self._calls.get(qualname, 0)
                )
            record.arguments = {
                str(position): dict(kinds) for position, kinds in sorted(positions.items())
            }
        return profile


def load_profile(path: Path) -> Profile:
    """The profile at `path`; a `ProfileError` names what is wrong with it."""
    return Profile.load(Path(path))


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=16)
def _loaded(path: str, stamp: tuple[int, int]) -> Profile:
    del stamp
    return Profile.load(Path(path))


def profile_for(config) -> Profile | None:  # type: ignore[no-untyped-def]
    """The profile a build's configuration names, read once per file version; None without one."""
    named = getattr(config.llvm, "pgo", None)
    if not named:
        return None
    path = Path(named)
    return _loaded(str(path), _stamp(path))


@functools.lru_cache(maxsize=16)
def _digest(path: str, stamp: tuple[int, int]) -> str:
    del stamp
    return hashlib.blake2b(Path(path).read_bytes(), digest_size=16).hexdigest()


def profile_digest(named: str) -> str:
    """What a cache key carries for a profile: its content, or that it is missing."""
    path = Path(named)
    try:
        return _digest(str(path), _stamp(path))
    except OSError:
        return f"missing:{path}"
