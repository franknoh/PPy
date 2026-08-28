"""Uvicorn/ASGI plugin: framework integration, not a native ABI (spec 22)."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallAdjustment, CallResult, Lowering

__all__ = ["PPY_RELOAD_PATTERN", "UvicornPlugin", "resolve_app"]

#: Uvicorn's reloader watches `*.py` by default, which never sees PPY sources.
PPY_RELOAD_PATTERN = "*.ppy"

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

    def adjust_call(self, qualname: str, node: ast.Call, symbols) -> CallAdjustment | None:  # type: ignore[no-untyped-def]
        """Resolve the ASGI application statically and fix reload watching.

        Uvicorn stays the host: the only changes are to stop re-importing the
        application by string on every worker start, and to make the reloader
        watch `.ppy` sources (spec 22.2, 22.5).
        """
        if qualname != "uvicorn.run":
            return None

        keywords = {k.arg: k.value for k in node.keywords if k.arg}
        reloading = _is_true(keywords.get("reload"))
        adjustments: list[tuple[str, str]] = []
        replacement: str | None = None
        reasons: list[str] = []

        if reloading and "reload_includes" not in keywords:
            adjustments.append(("reload_includes", f"[{PPY_RELOAD_PATTERN!r}]"))
            reasons.append("reload now watches .ppy sources")

        app = resolve_app(node)
        target = node.args[0] if node.args else keywords.get("app")
        if not reloading and app is not None and isinstance(target, ast.Constant) and ":" in app:
            module_name, _, attribute = app.partition(":")
            if module_name == symbols.name and attribute.isidentifier():
                replacement = attribute
                reasons.append("the ASGI application is resolved statically")

        if replacement is None and not adjustments:
            return None
        return CallAdjustment(
            qualname=qualname,
            replace_first_argument=replacement,
            add_keywords=tuple(adjustments),
            reason="; ".join(reasons),
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


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True
