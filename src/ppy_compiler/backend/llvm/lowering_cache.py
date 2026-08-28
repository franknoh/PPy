"""Serializing what lowering produced, so a warm build can skip it.

Lowering a module to IR is the largest single cost of a native build, and it
depends on exactly what the module cache key already covers. What has to
survive a round trip is the IR and the ABI decisions made about each function;
everything else -- the `FunctionInfo`, the AST of a specialization candidate --
is recovered from the analysis bundle, which is cheap to rebuild.
"""

from __future__ import annotations

import json

from .fusion import FusedLoop
from .lowering import NativeParam, NativeSignature

__all__ = ["SCHEMA_VERSION", "CachedLowering", "decode", "encode"]

#: Bumped when the shape below changes, so an old entry is simply a miss.
SCHEMA_VERSION = 1


class CachedLowering:
    """What `_collect` produced for one module, minus what is recomputable."""

    __slots__ = ("fused", "ir", "notes", "plan", "rejected", "signatures")

    def __init__(
        self,
        ir: str,
        signatures: dict[str, NativeSignature],
        rejected: dict[str, str],
        fused: dict[str, FusedLoop],
        plan: dict[tuple[int, int], FusedLoop],
        notes: list[tuple[int, str]],
    ) -> None:
        self.ir = ir
        self.signatures = signatures
        self.rejected = rejected
        self.fused = fused
        self.plan = plan
        self.notes = notes


def _param(p: NativeParam) -> dict:
    return {
        "name": p.name,
        "kind": p.kind,
        "element": p.element,
        "elements": list(p.elements),
        "fields": [list(f) for f in p.fields],
        "class_name": p.class_name,
    }


def _read_param(raw: dict) -> NativeParam:
    return NativeParam(
        name=raw["name"],
        kind=raw["kind"],
        element=raw["element"],
        elements=tuple(raw["elements"]),
        fields=tuple(tuple(f) for f in raw["fields"]),
        class_name=raw["class_name"],
    )


def _signature(s: NativeSignature) -> dict:
    return {
        "qualname": s.qualname,
        "symbol": s.symbol,
        "parameters": [_param(p) for p in s.parameters],
        "returns": list(s.returns),
        "releases_gil": s.releases_gil,
    }


def _read_signature(raw: dict) -> NativeSignature:
    return NativeSignature(
        qualname=raw["qualname"],
        symbol=raw["symbol"],
        parameters=tuple(_read_param(p) for p in raw["parameters"]),
        returns=tuple(raw["returns"]),
        releases_gil=raw["releases_gil"],
    )


def _loop(loop: FusedLoop) -> dict:
    return {
        "symbol": loop.symbol,
        "arrays": list(loop.arrays),
        "scalars": list(loop.scalars),
        "reduction": loop.reduction,
        "expression": loop.expression,
        "parallel": loop.parallel,
    }


def _read_loop(raw: dict) -> FusedLoop:
    return FusedLoop(
        symbol=raw["symbol"],
        arrays=tuple(raw["arrays"]),
        scalars=tuple(raw["scalars"]),
        reduction=raw["reduction"],
        expression=raw["expression"],
        parallel=raw["parallel"],
    )


def encode(module) -> str:  # type: ignore[no-untyped-def]
    """Serialize one `NativeModule`."""
    return json.dumps(
        {
            "version": SCHEMA_VERSION,
            "ir": module.ir,
            "signatures": {q: _signature(f.signature) for q, f in module.functions.items()},
            "rejected": dict(module.rejected),
            "fused": {symbol: _loop(loop) for symbol, loop in module.fused.items()},
            "plan": [
                [list(position), _loop(loop)] for position, loop in module.fusion_plan.items()
            ],
            "notes": [list(note) for note in module.fusion_notes],
        },
        separators=(",", ":"),
    )


def decode(text: str) -> CachedLowering | None:
    """Rebuild what was cached, or None when the entry is from another shape."""
    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return None
    try:
        return CachedLowering(
            ir=raw["ir"],
            signatures={q: _read_signature(s) for q, s in raw["signatures"].items()},
            rejected=dict(raw["rejected"]),
            fused={symbol: _read_loop(loop) for symbol, loop in raw["fused"].items()},
            plan={tuple(position): _read_loop(loop) for position, loop in raw["plan"]},
            notes=[tuple(note) for note in raw["notes"]],
        )
    except (KeyError, TypeError, ValueError):
        return None
