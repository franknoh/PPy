"""`instrument-profile` and `annotate-profile`: profile-guided optimization over the IR (spec 84).

A profiling run instruments every host function: one counter per block,
and one per conditional branch for its taken edge -- a `prof.hit_if` on
the condition, so the graph itself is untouched and a profile keys on the
same blocks a plain build has. The legend from counters to functions and
labels goes on the module as `ppy.profile.map`, and the backends write it
next to the counters, so the program carries its own legend.

A profile-guided build reads the counts back at the same point of the
pipeline. Each function's graph is digested -- labels, terminators,
successors -- and a function whose digest moved is left unannotated and
named, so a stale profile is never applied to code it did not measure.
Where it matches, the function carries `ppy.profile.calls` and `hot` or
`cold`; every terminator carries `ppy.profile.count`; a `core.cond_br`
carries `ppy.weights` (taken, not taken); a loop's back edge carries
`ppy.profile.trips`. The inliner reads the hotness, the LLVM backend
writes the counts and weights as `!prof` metadata and the hotness as the
`hot` and `cold` attributes, and the report files the remarks under
`profile applied`. A profile changes what is fast, never what is computed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import region_dominators
from ..dialects import prof
from ..model import Block, Builder, IRFunction, IRModule
from ..passes import Pass, PassContext

__all__ = [
    "PROFILE_KIND",
    "PROFILE_MAP",
    "PROFILE_VERSION",
    "AnnotateProfile",
    "FunctionProfile",
    "Instrument",
    "Profile",
    "ProfileError",
    "annotate",
    "back_edges",
    "cfg_digest",
    "counters_identifier",
]

#: The module attribute the instrument pass leaves: the legend, as JSON text.
PROFILE_MAP = "ppy.profile.map"
PROFILE_KIND = "ppy.profile"
PROFILE_VERSION = 1


class ProfileError(Exception):
    """A `.ppyprof` that is missing, unreadable, or not a profile."""


def cfg_digest(function: IRFunction) -> str:
    """The shape a profile keys on: every block's label, terminator, and successors."""
    hasher = hashlib.blake2b(digest_size=8)
    for block in function.body.blocks:
        terminator = block.terminator
        line = (
            f"{block.name}|{terminator.name if terminator is not None else ''}|"
            f"{','.join(successor.name for successor in block.successors)}\n"
        )
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def counters_identifier(module_name: str) -> str:
    """The suffix of a module's counter array and legend symbols."""
    return re.sub(r"\W", "_", module_name)


def _profiled(function: IRFunction) -> bool:
    return not function.is_declaration and not function.attributes.get("gpu.kind")


# -- the profile -----------------------------------------------------------------------------------


@dataclass
class FunctionProfile:
    """What one function did: its calls, its blocks, its branches, its arguments."""

    symbol: str = ""
    cfg: str = ""
    calls: int = 0
    blocks: dict[str, int] = field(default_factory=dict)
    #: By branching block: (taken, not taken).
    branches: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: By parameter position: the kinds of value that arrived, with counts.
    arguments: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The generic this function is an instance of, when it is one.
    generic: str = ""

    def merge(self, other: FunctionProfile) -> None:
        """Add `other`'s counts, when it measured the same graph; take it whole otherwise."""
        if other.cfg != self.cfg or not self.cfg:
            arguments = self.arguments
            self.symbol, self.cfg, self.calls = other.symbol, other.cfg, other.calls
            self.blocks, self.branches = dict(other.blocks), dict(other.branches)
            self.generic = other.generic
            self.arguments = arguments
        else:
            self.calls += other.calls
            for label, count in other.blocks.items():
                self.blocks[label] = self.blocks.get(label, 0) + count
            for label, (taken, not_taken) in other.branches.items():
                before = self.branches.get(label, (0, 0))
                self.branches[label] = (before[0] + taken, before[1] + not_taken)
        for position, kinds in other.arguments.items():
            mine = self.arguments.setdefault(position, {})
            for kind, count in kinds.items():
                mine[kind] = mine.get(kind, 0) + count

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        data: dict = {"symbol": self.symbol, "cfg": self.cfg, "calls": self.calls}
        data["blocks"] = dict(self.blocks)
        data["branches"] = {label: list(pair) for label, pair in self.branches.items()}
        if self.arguments:
            data["arguments"] = {k: dict(v) for k, v in self.arguments.items()}
        if self.generic:
            data["generic"] = self.generic
        return data

    @classmethod
    def from_dict(cls, data: dict) -> FunctionProfile:  # type: ignore[type-arg]
        try:
            return cls(
                symbol=str(data.get("symbol", "")),
                cfg=str(data.get("cfg", "")),
                calls=int(data.get("calls", 0)),
                blocks={str(k): int(v) for k, v in dict(data.get("blocks", {})).items()},
                branches={
                    str(k): (int(v[0]), int(v[1]))
                    for k, v in dict(data.get("branches", {})).items()
                },
                arguments={
                    str(k): {str(kk): int(vv) for kk, vv in dict(v).items()}
                    for k, v in dict(data.get("arguments", {})).items()
                },
                generic=str(data.get("generic", "")),
            )
        except (TypeError, ValueError, IndexError, AttributeError) as error:
            raise ProfileError(f"a function record is malformed: {error}") from error


@dataclass
class Profile:
    """A `.ppyprof`: every profiled function by qualified name, over one or more runs."""

    functions: dict[str, FunctionProfile] = field(default_factory=dict)
    runs: int = 0
    program: str = ""
    compiler: str = ""
    path: str = ""

    def hot_threshold(self) -> int:
        """A function this often called is hot: a twentieth of the most-called one, at least 2."""
        hottest = max((f.calls for f in self.functions.values()), default=0)
        return max(2, hottest // 20)

    def kind(self, qualname: str) -> str | None:
        """`hot`, `warm`, or `cold` for a measured function; None for one the profile lacks."""
        record = self.functions.get(qualname)
        if record is None or not record.cfg:
            return None
        if record.calls == 0:
            return "cold"
        return "hot" if record.calls >= self.hot_threshold() else "warm"

    def merge(self, other: Profile) -> None:
        for qualname, record in other.functions.items():
            mine = self.functions.get(qualname)
            if mine is None:
                self.functions[qualname] = record
            else:
                mine.merge(record)
        self.runs += other.runs
        self.program = other.program or self.program
        self.compiler = other.compiler or self.compiler

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "kind": PROFILE_KIND,
            "version": PROFILE_VERSION,
            "compiler": self.compiler,
            "program": self.program,
            "runs": self.runs,
            "functions": {
                name: record.to_dict() for name, record in sorted(self.functions.items())
            },
        }

    @classmethod
    def from_dict(cls, data: object, path: str = "") -> Profile:
        if not isinstance(data, dict) or data.get("kind") != PROFILE_KIND:
            raise ProfileError(f"{path or 'the profile'} is not a ppy profile")
        version = data.get("version")
        if not isinstance(version, int) or version > PROFILE_VERSION:
            raise ProfileError(
                f"{path or 'the profile'} is profile version {version!r}; this compiler reads "
                f"up to {PROFILE_VERSION}"
            )
        functions = data.get("functions", {})
        if not isinstance(functions, dict):
            raise ProfileError(f"{path or 'the profile'} has no function table")
        return cls(
            functions={str(k): FunctionProfile.from_dict(v) for k, v in functions.items()},
            runs=int(data.get("runs", 0)),
            program=str(data.get("program", "")),
            compiler=str(data.get("compiler", "")),
            path=path,
        )

    @classmethod
    def load(cls, path: Path) -> Profile:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ProfileError(f"no profile at {path}; `ppy run --profile` writes one") from None
        except OSError as error:
            raise ProfileError(f"cannot read the profile {path}: {error}") from error
        try:
            data = json.loads(text)
        except ValueError as error:
            raise ProfileError(f"{path} is not a ppy profile: {error}") from error
        return cls.from_dict(data, str(path))

    def write(self, path: Path) -> Profile:
        """Write, merging into a profile already at `path`; the profile as written."""
        merged = self
        if Path(path).is_file():
            try:
                previous = Profile.load(path)
            except ProfileError:
                previous = None
            if previous is not None:
                previous.merge(self)
                merged = previous
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(merged.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        merged.path = str(path)
        return merged


# -- instrumenting ---------------------------------------------------------------------------------


class Instrument(Pass):
    """A counter in every block and on every conditional branch; the legend on the module."""

    name = "instrument-profile"

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        if PROFILE_MAP in module.attributes:
            return False
        counter = 0
        table: dict[str, dict] = {}  # type: ignore[type-arg]
        for function in module.functions.values():
            if not _profiled(function) or function.entry is None:
                continue
            digest = cfg_digest(function)
            blocks: dict[str, int] = {}
            edges: dict[str, int] = {}
            for block in function.body.blocks:
                blocks[block.name] = counter
                b = Builder().before(block.operations[0]) if block.operations else Builder(block)
                prof.hit(b, counter)
                counter += 1
            for block in function.body.blocks:
                terminator = block.terminator
                if terminator is None or terminator.name != "core.cond_br":
                    continue
                edges[block.name] = counter
                prof.hit_if(Builder().before(terminator), terminator.operands[0], counter)
                counter += 1
            entry: dict = {  # type: ignore[type-arg]
                "qualname": str(function.attributes.get("ppy.qualname", function.name)),
                "cfg": digest,
                "entry": function.entry.name,
                "blocks": blocks,
                "edges": edges,
            }
            generic = function.attributes.get("ppy.generic")
            if generic:
                entry["generic"] = str(generic)
            table[function.name] = entry
        if not table:
            return False
        module.require(prof.ProfDialect.name, prof.ProfDialect.version)
        module.attributes[PROFILE_MAP] = json.dumps(
            {"counters": counter, "functions": table}, sort_keys=True, separators=(",", ":")
        )
        ctx.remark(f"profile: {counter} counter(s) instrument {len(table)} function(s)")
        return True


# -- annotating ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Annotated:
    kind: str
    branches: int
    loops: int


def back_edges(function: IRFunction) -> list[tuple[Block, Block]]:
    """Every (latch, header) edge: a jump to a block that dominates the jumper."""
    dominators = region_dominators(function.body)
    return [
        (block, successor)
        for block in function.body.blocks
        for successor in block.successors
        if successor in dominators.get(block, frozenset())
    ]


def _edge_count(latch: Block, header: Block, record: FunctionProfile) -> int | None:
    terminator = latch.terminator
    if terminator is None:
        return None
    if terminator.name == "core.br":
        return record.blocks.get(latch.name)
    if terminator.name == "core.cond_br" and latch.name in record.branches:
        taken, not_taken = record.branches[latch.name]
        targets = [successor.block for successor in terminator.successors]
        return taken if targets.index(header) == 0 else not_taken
    return None


def annotate(function: IRFunction, record: FunctionProfile, threshold: int) -> Annotated:
    """Write `record`'s counts onto `function`; what was written."""
    calls = record.calls
    function.attributes["ppy.profile.calls"] = calls
    kind = "cold" if calls == 0 else "hot" if calls >= threshold else "warm"
    if kind != "warm":
        function.attributes[f"ppy.profile.{kind}"] = True
    branches = 0
    for block in function.body.blocks:
        terminator = block.terminator
        count = record.blocks.get(block.name)
        if terminator is None or count is None:
            continue
        terminator.attributes["ppy.profile.count"] = count
        if terminator.name == "core.cond_br" and block.name in record.branches:
            taken, not_taken = record.branches[block.name]
            terminator.attributes["ppy.weights"] = (taken, not_taken)
            branches += 1
    loops = 0
    for latch, header in back_edges(function):
        taken = _edge_count(latch, header, record)
        header_count = record.blocks.get(header.name)
        if taken is None or header_count is None or latch.terminator is None:
            continue
        entries = max(header_count - taken, 0)
        trips = taken / entries if entries else float(taken)
        latch.terminator.attributes["ppy.profile.trips"] = round(trips, 2)
        loops += 1
    return Annotated(kind, branches, loops)


class AnnotateProfile(Pass):
    """A profile's counts as attributes on the functions and terminators it measured."""

    name = "annotate-profile"

    def __init__(self, profile: Profile) -> None:
        self.profile = profile

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        changed = False
        threshold = self.profile.hot_threshold()
        for function in module.functions.values():
            if not _profiled(function):
                continue
            qualname = str(function.attributes.get("ppy.qualname", function.name))
            record = self.profile.functions.get(qualname)
            if record is None or not record.cfg:
                continue
            if record.cfg != cfg_digest(function):
                ctx.remark(
                    f"profile: stale for `{qualname}`: the function changed since the profile "
                    "was recorded; its counts were ignored"
                )
                continue
            result = annotate(function, record, threshold)
            ctx.remark(
                f"profile: `{qualname}` is {result.kind} ({record.calls} call(s)); "
                f"{result.branches} branch(es) weighted, {result.loops} loop(s) with trip counts"
            )
            changed = True
        return changed
