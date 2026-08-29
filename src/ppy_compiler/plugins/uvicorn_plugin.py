"""Uvicorn/ASGI plugin: framework integration, not a native ABI (spec 22).

One plugin covers the serving stack: FastAPI is an ASGI framework riding the
same `uvicorn.run`, so knowing Uvicorn and not FastAPI would split one story
in half. The FastAPI surface modeled here is the part a checked project
touches -- the application object, its route decorators, dependency markers,
and the in-process `TestClient` -- typed just enough that strict mode has a
signature for every call. Route *handlers* stay out of PPY's hands entirely:
FastAPI reads their `__annotations__` at import time, and the conversion
policy already refuses to touch functions under unvouched decorators.
"""

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

PLUGIN_VERSION = 2

#: `@app.<method>` route/registration decorators on an application or router.
_ROUTE_METHODS = frozenset(
    {
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "options",
        "head",
        "trace",
        "websocket",
        "middleware",
        "on_event",
        "exception_handler",
    }
)

#: Application methods called for their effect, not as decorators.
_APP_CALLS = frozenset({"include_router", "mount", "add_middleware", "add_api_route"})

#: `TestClient` request methods, all returning an HTTP response.
_CLIENT_METHODS = frozenset({"get", "post", "put", "delete", "patch", "options", "head", "request"})

#: Parameter markers interpreted by FastAPI, opaque values to everyone else.
_MARKER_FACTORIES = frozenset(
    {
        "Depends",
        "Query",
        "Path",
        "Body",
        "Header",
        "Cookie",
        "Form",
        "File",
        "Security",
    }
)

_APP = "fastapi.FastAPI"
_ROUTER = "fastapi.APIRouter"
_CLIENT = "fastapi.testclient.TestClient"
_RESPONSE = "httpx.Response"


class UvicornPlugin:
    """Resolves the ASGI application statically and keeps Uvicorn as the host."""

    name = "uvicorn"
    modules = ("uvicorn", "fastapi", "starlette", "httpx")

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}

    def fingerprint(self) -> str:
        versions: list[str] = []
        for library in ("uvicorn", "fastapi"):
            version = "absent"
            try:
                if importlib.util.find_spec(library) is not None:
                    import importlib as _importlib

                    version = _importlib.import_module(library).__version__
            except Exception:  # noqa: BLE001
                version = "unknown"
            versions.append(f"{library}={version}")
        return f"v{PLUGIN_VERSION}:" + ":".join(versions)

    def external_types(self) -> dict[str, str]:
        return {
            "uvicorn.Config": "uvicorn.Config",
            "uvicorn.Server": "uvicorn.Server",
            # Displays are the qualnames: an annotation resolved through this
            # table must build the same `Instance` the constructor call does.
            _APP: _APP,
            _ROUTER: _ROUTER,
            _CLIENT: _CLIENT,
            _RESPONSE: _RESPONSE,
        }

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        return None

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        del args, keywords
        if qualname == "uvicorn.run":
            return CallResult(
                T.NONE,
                Facts(),
                EffectSet.of(Effect.IO, Effect.THREAD, Effect.PROCESS),
                Lowering.PYTHON_FALLBACK,
                "Uvicorn remains the ASGI host; PPY compiles the application and its hot helpers",
            )
        stays = "FastAPI objects stay on the Python path; PPY compiles the helpers they call"
        if qualname in {_APP, _ROUTER}:
            built = _APP if qualname == _APP else _ROUTER
            return CallResult(
                T.Instance(built, (), (built, "object")),
                Facts(),
                EffectSet.of(Effect.ALLOC),
                Lowering.PYTHON_FALLBACK,
                stays,
            )
        if qualname in {_CLIENT, "starlette.testclient.TestClient"}:
            return CallResult(
                T.Instance(_CLIENT, (), (_CLIENT, "object")),
                Facts(),
                EffectSet.of(Effect.ALLOC),
                Lowering.PYTHON_FALLBACK,
                stays,
            )
        tail = qualname.rpartition(".")[2]
        if qualname.startswith("fastapi.") and tail in _MARKER_FACTORIES:
            # An opaque marker FastAPI interprets at import; a value to us.
            return CallResult(T.ANY, Facts(), EffectSet(), Lowering.PYTHON_FALLBACK, stays)
        return None

    def instance_attribute(
        self, owner: str, attr: str, facts: Facts
    ) -> tuple[T.Type, Facts] | None:
        """The FastAPI surface a checked project touches.

        Just enough signature that strict mode has one for every call: a
        route decorator hands back an opaque decorator, a client request
        hands back a response, and a response answers the usual accessors.
        The values themselves stay Python-path machinery.
        """
        del facts

        def callable_returning(result: T.Type, name: str) -> tuple[T.Type, Facts]:
            return (T.Callable_((), result, f"{owner}.{name}"), Facts())

        if owner in {_APP, _ROUTER}:
            if attr in _ROUTE_METHODS:
                return callable_returning(T.ANY, attr)
            if attr in _APP_CALLS:
                return callable_returning(T.NONE, attr)
        if owner == _CLIENT and attr in _CLIENT_METHODS:
            return callable_returning(T.Instance(_RESPONSE, (), (_RESPONSE, "object")), attr)
        if owner == _RESPONSE:
            if attr == "status_code":
                return (T.INT, Facts())
            if attr == "text":
                return (T.STR, Facts())
            if attr == "content":
                return (T.BYTES, Facts())
            if attr in {"json", "raise_for_status", "read"}:
                return callable_returning(T.ANY, attr)
            return (T.ANY, Facts())
        return None

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
