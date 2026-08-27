"""Uvicorn/ASGI plugin: framework integration, not a native ABI (spec 22)."""

from __future__ import annotations

import ast
import importlib.util
from typing import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, Lowering

__all__ = ["UvicornPlugin", "resolve_app"]

PLUGIN_VERSION = 1


class UvicornPlugin:
    """Resolves the ASGI application statically and keeps Uvicorn as the host."""

    name = "uvicorn"
    modules = ("uvicorn",)

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}

    def fingerprint(self) -> str:
        version = "absent"
        try:
            if importlib.util.find_spec("uvicorn") is not None:
                import uvicorn

                version = uvicorn.__version__
        except Exception:  # noqa: BLE001
            version = "unknown"
        return f"v{PLUGIN_VERSION}:uvicorn={version}"

    def external_types(self) -> dict[str, str]:
        return {"uvicorn.Config": "uvicorn.Config", "uvicorn.Server": "uvicorn.Server"}

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        if qualname != "uvicorn.run":
            return None
        return CallResult(
            T.NONE,
            Facts(),
            EffectSet.of(Effect.IO, Effect.THREAD, Effect.PROCESS),
            Lowering.PYTHON_FALLBACK,
            "Uvicorn remains the ASGI host; PPY compiles the application and its hot helpers",
        )

    def reload_dirs_note(self) -> str:
        """Development reload must watch `.ppy` files (spec 22.5)."""
        return "pass `reload_includes=['*.ppy']` so watchfiles sees PPY sources"


def resolve_app(node: ast.Call) -> str | None:
    """Statically resolve the ASGI application passed to `uvicorn.run` (spec 22.2)."""
    if not node.args:
        for keyword in node.keywords:
            if keyword.arg == "app":
                return _app_name(keyword.value)
        return None
    return _app_name(node.args[0])


def _app_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Attribute):
        return ast.unparse(expr)
    return None
