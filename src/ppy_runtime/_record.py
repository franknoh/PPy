"""Records for the launcher's path: what `@dataclass` gives, without its import.

`dataclasses` imports `inspect`, which imports `dis`, `ast`, `tokenize` and
more: about 11 ms, a third of what a warm `ppy run` of a short program costs
over `python`. The runtime's records need little of it (fields in annotation
order, defaults, `__init__`, `__repr__`, `__eq__`, a hash when frozen, and
slots), so this writes those and nothing else. Type checkers see the real
`dataclass` and `field`, and every class keeps the meaning it had.
"""

from __future__ import annotations

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any

__all__ = ["field", "record", "replace"]

_MISSING = object()


class _Field:
    __slots__ = ("default_factory",)

    def __init__(self, default_factory: Any) -> None:
        self.default_factory = default_factory


def _field(*, default_factory: Any) -> Any:
    """A field whose default is made per instance (`field(default_factory=list)`)."""
    return _Field(default_factory)


def _names(cls: type) -> list[str]:
    names: list[str] = []
    for klass in reversed(cls.__mro__):
        # Read off the class itself: `inspect.get_annotations` is the import
        # this module exists to avoid.
        for name, spelled in klass.__dict__.get("__annotations__", {}).items():  # noqa: RUF063
            if "ClassVar" in str(spelled) or name in names:
                continue
            names.append(name)
    return names


def _record(cls: type | None = None, *, frozen: bool = False, slots: bool = False) -> Any:
    def build(cls: type) -> type:
        names = _names(cls)
        defaults: dict[str, Any] = {}
        for name in names:
            value = cls.__dict__.get(name, _MISSING)
            if value is not _MISSING:
                defaults[name] = value
        namespace: dict[str, Any] = {"_MISSING": _MISSING}
        parameters = []
        body = []
        setter = "object.__setattr__(self, {0!r}, {1})" if frozen else "self.{0} = {1}"
        for name in names:
            default = defaults.get(name, _MISSING)
            if isinstance(default, _Field):
                namespace[f"_factory_{name}"] = default.default_factory
                parameters.append(f"{name}=_MISSING")
                value = f"_factory_{name}() if {name} is _MISSING else {name}"
            elif default is not _MISSING:
                namespace[f"_default_{name}"] = default
                parameters.append(f"{name}=_default_{name}")
                value = name
            else:
                parameters.append(name)
                value = name
            body.append(setter.format(name, value))
        source = f"def __init__(self, {', '.join(parameters)}):\n    " + (
            "\n    ".join(body) or "pass"
        )
        exec(source, namespace)
        attributes = {k: v for k, v in cls.__dict__.items() if k not in names and k != "__dict__"}
        attributes.pop("__weakref__", None)
        if slots:
            attributes["__slots__"] = tuple(names)
        attributes["__init__"] = namespace["__init__"]
        attributes["__match_args__"] = tuple(names)
        attributes["_record_fields"] = tuple(names)

        def __repr__(self: Any) -> str:
            shown = ", ".join(f"{n}={getattr(self, n)!r}" for n in names)
            return f"{type(self).__qualname__}({shown})"

        def __eq__(self: Any, other: object) -> bool:
            if other.__class__ is not self.__class__:
                return NotImplemented
            return all(getattr(self, n) == getattr(other, n) for n in names)

        attributes["__repr__"] = __repr__
        attributes["__eq__"] = __eq__
        if frozen:

            def __setattr__(self: Any, name: str, value: Any) -> None:
                raise AttributeError(f"cannot assign to field {name!r}")

            def __hash__(self: Any) -> int:
                return hash(tuple(getattr(self, n) for n in names))

            attributes["__setattr__"] = __setattr__
            attributes["__delattr__"] = __setattr__
            attributes["__hash__"] = __hash__
        else:
            attributes["__hash__"] = None
        made = type(cls)(cls.__name__, cls.__bases__, attributes)
        made.__qualname__ = cls.__qualname__
        return made

    return build if cls is None else build(cls)


def _replace(instance: Any, **changes: Any) -> Any:
    """A copy of a record, or of a dataclass, with `changes`."""
    names = getattr(type(instance), "_record_fields", None)
    if names is None:
        import dataclasses  # pylint: disable=import-outside-toplevel

        return dataclasses.replace(instance, **changes)
    values = {n: getattr(instance, n) for n in names}
    values.update(changes)
    return type(instance)(**values)


if TYPE_CHECKING:
    from dataclasses import dataclass as record
    from dataclasses import field, replace
else:
    record = _record
    field = _field
    replace = _replace
