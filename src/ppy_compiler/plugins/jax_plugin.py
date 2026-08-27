"""JAX plugin: staged export plus PJRT execution (spec 21)."""

from __future__ import annotations

import importlib.util
from typing import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, Lowering

__all__ = ["JaxPlugin", "STAGING_MARKERS"]

PLUGIN_VERSION = 1

#: Decorators that mark a staged region PPY may export (spec 21.2).
STAGING_MARKERS = ("jax.jit", "jit", "ppy.jax")

_ARRAY = T.Instance("jax.Array", (), ("jax.Array", "object"))


class JaxPlugin:
    """Recognizes staged JAX functions; eager calls stay on the Python path."""

    name = "jax"
    modules = ("jax", "jax.numpy", "jax.lax", "jaxlib")

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.allow_build_export = bool(self.options.get("allow-build-export", False))

    def fingerprint(self) -> str:
        """JAX, jaxlib, and XLA identities all enter the cache key (spec 21.8)."""
        jax_version = jaxlib_version = "absent"
        try:
            if importlib.util.find_spec("jax") is not None:
                import jax

                jax_version = jax.__version__
                try:
                    import jaxlib

                    jaxlib_version = jaxlib.__version__
                except Exception:  # noqa: BLE001
                    jaxlib_version = "unknown"
        except Exception:  # noqa: BLE001
            jax_version = "unknown"
        return (
            f"v{PLUGIN_VERSION}:jax={jax_version}:jaxlib={jaxlib_version}"
            f":export={int(self.allow_build_export)}"
        )

    def external_types(self) -> dict[str, str]:
        return {"jax.Array": "jax.Array", "jax.numpy.ndarray": "jax.Array"}

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        if qualname.rpartition(".")[2] in {"pi", "e", "inf", "nan"}:
            return T.FLOAT, Facts()
        return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        if not qualname.startswith(("jax.numpy.", "jax.lax.", "jax.")):
            return None
        effects = EffectSet.of(Effect.ALLOC, raises=("ValueError", "TypeError"))
        # Full eager lowering of jax.numpy is explicitly out of v1 scope (21.7).
        return CallResult(
            _ARRAY,
            Facts(),
            effects,
            Lowering.PYTHON_FALLBACK,
            "eager JAX calls outside an exported region use the Python path",
        )

    def staged_region(self, decorators: Sequence[str]) -> CallResult | None:
        """A `@jax.jit` function PPY may export to StableHLO at build time."""
        if not any(marker in d for d in decorators for marker in STAGING_MARKERS):
            return None
        if not self.allow_build_export:
            return CallResult(
                _ARRAY,
                Facts(),
                EffectSet.of(Effect.EXTERNAL_UNKNOWN),
                Lowering.PYTHON_FALLBACK,
                "build-time export is disabled (`allow-build-export = false`); "
                "JAX export executes project code and needs explicit permission",
            )
        return CallResult(
            _ARRAY,
            Facts(),
            EffectSet.of(Effect.ALLOC),
            Lowering.GRAPH_REGION,
            "staged and exported to StableHLO, then executed through PJRT without per-call Python dispatch",
            ("input shape and dtype match the exported signature", "PJRT client available for the target device"),
        )
