"""PPY directive objects and the inert decorators that record them."""

from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

__all__ = [
    "Directive",
    "directives_of",
    "attach",
    "pure",
    "opt",
    "jit",
    "parallel",
    "native",
    "inline",
    "noinline",
    "specialize",
    "fastmath",
    "dynamic",
    "jax",
]

_T = TypeVar("_T")

DIRECTIVE_ATTR = "__ppy_directives__"


class Directive:
    """A recorded PPY directive. Carries no runtime behavior."""

    __slots__ = ("name", "options")

    def __init__(self, name: str, options: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.options = dict(options or {})

    def __repr__(self) -> str:
        if not self.options:
            return f"Directive({self.name!r})"
        return f"Directive({self.name!r}, {self.options!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Directive):
            return NotImplemented
        return self.name == other.name and self.options == other.options

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.options.items(), key=lambda kv: kv[0]))))


def directives_of(obj: Any) -> tuple[Directive, ...]:
    return tuple(getattr(obj, DIRECTIVE_ATTR, ()))


def attach(obj: _T, directive: Directive) -> _T:
    """Record a directive on obj and return obj unchanged.

    The object is never wrapped: identity, descriptor behavior, coroutine
    behavior, and call behavior are all preserved (spec 6.1).
    """
    try:
        existing = tuple(obj.__dict__.get(DIRECTIVE_ATTR, ()))  # type: ignore[attr-defined]
    except AttributeError:
        existing = ()
    try:
        setattr(obj, DIRECTIVE_ATTR, (directive, *existing))
    except (AttributeError, TypeError):
        pass
    return obj


def _flexible(name: str) -> Any:
    """Build a decorator usable bare (@d) or called (@d(**options))."""

    def decorator(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs and callable(args[0]):
            return attach(args[0], Directive(name))
        if args:
            raise TypeError(f"@ppy.{name} takes keyword options only")

        def bind(obj: _T) -> _T:
            return attach(obj, Directive(name, kwargs))

        return bind

    decorator.__name__ = name
    decorator.__qualname__ = f"ppy.{name}"
    decorator.__doc__ = f"PPY `{name}` directive. Behaviorally inert under plain CPython."
    return decorator


pure = _flexible("pure")
jit = _flexible("jit")
parallel = _flexible("parallel")
native = _flexible("native")
inline = _flexible("inline")
noinline = _flexible("noinline")
specialize = _flexible("specialize")
fastmath = _flexible("fastmath")
jax = _flexible("jax")


def opt(level: int) -> Callable[[_T], _T]:
    """Select a per-function optimization level from 0 through 3."""
    if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 3:
        raise ValueError("@ppy.opt(level) requires an int level in 0..3")

    def bind(obj: _T) -> _T:
        return attach(obj, Directive("opt", {"level": level}))

    return bind


class _Dynamic:
    """`ppy.dynamic` used either as a decorator or as a context manager."""

    def __call__(self, obj: _T) -> _T:
        return attach(obj, Directive("dynamic"))

    def __enter__(self) -> "_Dynamic":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __repr__(self) -> str:
        return "ppy.dynamic"


dynamic = _Dynamic()
