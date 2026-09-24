"""A JSON value read as a typed object: `ppy.input[Model]()` and `ppy.scan[Model]()`.

The schema is a class the program already has:

```python
@dataclass
class Order:
    id: int
    items: list[Item]
    note: str | None = None


order = ppy.input[Order]()           # one line of JSON, built into an Order
orders = ppy.input[list[Order]]()    # one line holding a JSON array of them
order = ppy.scan[Order]()            # the next JSON value, however many lines it spans
```

A dataclass or a `TypedDict` is built here, field by field: `int` takes a
JSON integer (not `true`, not `1.5`), `float` takes any JSON number, `str`,
`bool`, and `None` take their own kind, and containers, unions, `Literal`,
enums, and fixed widths such as `ppy.i32` are read as they are declared. A
field with a default may be missing; a key the class does not declare is
ignored, as pydantic ignores it by default. A pydantic model, or anything
that holds one, is validated by pydantic itself, with its own rules.

Every data error is a `ValueError` that says where it is (`$.items[2].price`)
and what was there; pydantic's `ValidationError` is one too. A JSON value
that does not parse is the `json` module's `JSONDecodeError`, also a
`ValueError`.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import re
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["depth_after", "is_schema", "reader"]


def _is_pydantic(spec: object) -> bool:
    return isinstance(spec, type) and callable(getattr(spec, "model_validate_json", None))


def _is_record(spec: object) -> bool:
    """A class a JSON object is built into."""
    if not isinstance(spec, type):
        return False
    return dataclasses.is_dataclass(spec) or typing.is_typeddict(spec) or _is_pydantic(spec)


def is_schema(spec: object) -> bool:
    """Is `spec` a record, or a list, tuple, dict, or optional of one?

    That is what makes a read JSON: a `list[int]` is still a line of fields,
    and a `list[Order]` is a JSON array.
    """
    if _is_record(spec):
        return True
    origin = typing.get_origin(spec)
    if origin in (list, tuple, dict, Sequence, Mapping, typing.Union, types.UnionType):
        return any(is_schema(argument) for argument in typing.get_args(spec))
    return False


def _uses_pydantic(spec: object) -> bool:
    if _is_pydantic(spec):
        return True
    return any(_uses_pydantic(argument) for argument in typing.get_args(spec))


def reader(spec: object) -> Callable[[str], Any]:
    """The function from JSON text to a value of `spec`, planned once."""
    if _uses_pydantic(spec):
        import pydantic  # pylint: disable=import-outside-toplevel

        return pydantic.TypeAdapter(spec).validate_json
    return lambda text: build(spec, json.loads(text), "$")


def _kind(value: object) -> str:
    """How JSON spells the kind of a decoded value, for an error message."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    return "an array" if isinstance(value, list) else "an object"


def _refuse(path: str, wanted: str, value: object) -> ValueError:
    return ValueError(f"{path}: expected {wanted}, got {_kind(value)}")


def build(spec: Any, value: Any, path: str) -> Any:
    """`value`, decoded from JSON, checked and built into `spec`."""
    if spec is Any or spec is object:
        return value
    if spec is None or spec is type(None):
        if value is not None:
            raise _refuse(path, "null", value)
        return None
    if spec is bool:
        if not isinstance(value, bool):
            raise _refuse(path, "a boolean", value)
        return value
    if spec is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise _refuse(path, "an integer", value)
        return value
    if spec is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _refuse(path, "a number", value)
        return float(value)
    if spec is str:
        if not isinstance(value, str):
            raise _refuse(path, "a string", value)
        return value
    if isinstance(spec, type) and issubclass(spec, enum.Enum):
        try:
            return spec(value)
        except ValueError:
            raise ValueError(f"{path}: {value!r} is not a {spec.__name__}") from None
    if dataclasses.is_dataclass(spec) and isinstance(spec, type):
        return _dataclass(spec, value, path)
    if typing.is_typeddict(spec):
        return _typed_dict(spec, value, path)
    origin = typing.get_origin(spec)
    arguments = typing.get_args(spec)
    if origin is typing.Annotated:
        return _annotated(arguments, value, path)
    if origin is typing.Literal:
        if value not in arguments or any(
            value == option and type(value) is not type(option) for option in arguments
        ):
            raise ValueError(f"{path}: expected one of {list(arguments)!r}, got {value!r}")
        return value
    if origin in (typing.Union, types.UnionType):
        return _union(arguments, value, path)
    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise _refuse(path, "an array", value)
        (element,) = arguments or (Any,)
        return [build(element, item, f"{path}[{i}]") for i, item in enumerate(value)]
    if origin is tuple:
        return _tuple(arguments, value, path)
    if origin in (dict, Mapping):
        if not isinstance(value, dict):
            raise _refuse(path, "an object", value)
        _, element = arguments or (str, Any)
        return {key: build(element, item, f"{path}.{key}") for key, item in value.items()}
    raise TypeError(f"{spec!r} is not something a JSON value can be read as")


def _fields(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls, include_extras=True)


def _dataclass(cls: type, value: Any, path: str) -> Any:
    if not isinstance(value, dict):
        raise _refuse(path, f"an object for {cls.__name__}", value)
    hints = _fields(cls)
    arguments: dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        if not item.init:
            continue
        if item.name not in value:
            if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING:
                raise ValueError(f"{path}: missing field `{item.name}` of {cls.__name__}")
            continue
        arguments[item.name] = build(
            hints.get(item.name, Any), value[item.name], f"{path}.{item.name}"
        )
    return cls(**arguments)


def _typed_dict(cls: Any, value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _refuse(path, f"an object for {cls.__name__}", value)
    hints = _fields(cls)
    for name in cls.__required_keys__:
        if name not in value:
            raise ValueError(f"{path}: missing key `{name}` of {cls.__name__}")
    return {
        name: build(hints[name], value[name], f"{path}.{name}") for name in hints if name in value
    }


def _annotated(arguments: tuple[Any, ...], value: Any, path: str) -> Any:
    """`ppy.i32` and the other widths: the base type, then the range."""
    base, *metadata = arguments
    built = build(base, value, path)
    for item in metadata:
        low, high = getattr(item, "low", None), getattr(item, "high", None)
        fixed = base is int and isinstance(low, int) and isinstance(high, int)
        if fixed and not low <= built <= high:
            name = f"{'i' if item.signed else 'u'}{item.bits}"
            raise ValueError(f"{path}: {built} does not fit in ppy.{name}")
    return built


def _union(arguments: tuple[Any, ...], value: Any, path: str) -> Any:
    failures: list[str] = []
    for option in arguments:
        try:
            return build(option, value, path)
        except ValueError as error:
            failures.append(str(error))
    raise ValueError(failures[0] if len(failures) == 1 else f"{path}: no option of the union fits")


def _tuple(arguments: tuple[Any, ...], value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise _refuse(path, "an array", value)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(build(arguments[0], item, f"{path}[{i}]") for i, item in enumerate(value))
    if len(value) != len(arguments):
        raise ValueError(f"{path}: expected {len(arguments)} items, got {len(value)}")
    return tuple(
        build(part, item, f"{path}[{i}]")
        for i, (part, item) in enumerate(zip(arguments, value, strict=True))
    )


#: A JSON string: it cannot hold a raw newline, so it never spans lines.
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def depth_after(line: str, depth: int) -> int:
    """How deep in brackets a JSON value is after one more of its lines."""
    text = _STRING.sub("", line)
    return depth + text.count("{") + text.count("[") - text.count("}") - text.count("]")
