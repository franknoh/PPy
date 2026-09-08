# Generics

A function may declare type parameters the way Python 3.12 spells them:

```python
def largest[T: int | float](a: T, b: T) -> T:
    return a if a > b else b
```

A call infers the type arguments from what it passes, checks each against
its bound (`E1721`), and has the declared return type with them
substituted: `largest(1, 2)` is an `int`, `largest(1.5, 2.5)` a `float`. An
unbounded parameter has no operators -- nothing says it does -- and a
bound that is a Protocol lends its methods, to operators and to attribute
calls alike:

```python
from typing import Protocol


class Named(Protocol):
    def name(self) -> str: ...


def label[T: Named](thing: T) -> str:
    return "at " + thing.name()
```

A project class is an instance of a Protocol when its members cover the
Protocol's.

## Monomorphization

Native code monomorphizes: a generic called from a native function is
lowered once per tuple of type arguments, under a name that spells them,
and the call goes straight to that instance; the generic itself keeps its
Python body for every other caller. `ppy emit ir` shows the instances,
marked `ppy.generic`. `[tool.ppy.generics]` bounds the process --
`max-specializations` per generic (`E1722`) and `max-depth` of a type
argument -- and a generic that calls itself with its own parameter wrapped
in a type is refused outright (`E1723`), because its specializations never
end.

## Static dispatch

`a + b` on a value class inside native code calls the class's own
`__add__`, lowered like any native function; native code never falls back
to Python's dynamic dispatch, and a class without a native operator is
refused with the reason.

Examples: [Generics](../howto/40_generics.md),
[Value classes](../howto/13_value_classes.md).
