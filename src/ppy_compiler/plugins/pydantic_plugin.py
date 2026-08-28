"""Pydantic plugin: model typing and constraint refinements (spec 23)."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, Lowering

__all__ = ["PydanticPlugin"]

PLUGIN_VERSION = 1


class PydanticPlugin:
    """Types `BaseModel` subclasses and turns field constraints into refinements."""

    name = "pydantic"
    modules = ("pydantic",)

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.schema_execution = str(self.options.get("schema-execution", "deny"))

    def fingerprint(self) -> str:
        version = "absent"
        try:
            if importlib.util.find_spec("pydantic") is not None:
                import pydantic

                version = pydantic.VERSION
        except Exception:  # noqa: BLE001
            version = "unknown"
        return f"v{PLUGIN_VERSION}:pydantic={version}:schema-exec={self.schema_execution}"

    def external_types(self) -> dict[str, str]:
        return {
            "pydantic.BaseModel": "pydantic.BaseModel",
            "pydantic.ConfigDict": "pydantic.ConfigDict",
            "pydantic.ValidationError": "pydantic.ValidationError",
        }

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        if qualname == "pydantic.BaseModel":
            return (
                T.ClassObject(
                    "pydantic.BaseModel",
                    T.Instance("pydantic.BaseModel", (), ("pydantic.BaseModel", "object")),
                ),
                Facts(),
            )
        return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        attribute = qualname.rpartition(".")[2]
        if attribute == "Field":
            # Field metadata is consumed by the annotation resolver, not at runtime.
            return CallResult(
                T.ANY, Facts(), EffectSet(), Lowering.PYTHON_FALLBACK, "field metadata"
            )
        if attribute in {"model_validate", "model_validate_json", "parse_obj"}:
            return CallResult(
                T.UNKNOWN,
                Facts(),
                EffectSet.of(Effect.ALLOC, raises=("pydantic.ValidationError",)),
                Lowering.PYTHON_FALLBACK,
                "validation stays a C-API call into the installed pydantic-core (spec 23.7)",
            )
        if attribute in {"model_dump", "model_dump_json", "dict", "json"}:
            result = T.dict_of(T.STR, T.ANY) if attribute in {"model_dump", "dict"} else T.STR
            return CallResult(
                result,
                Facts(),
                EffectSet.of(Effect.ALLOC),
                Lowering.PYTHON_FALLBACK,
                "serialization stays on the Python path",
            )
        return None

    def validator_effects(self) -> EffectSet:
        """A PPY-analyzable validator gets a checked effect summary (spec 23.5)."""
        return EffectSet.of(raises=("ValueError",))
