# Classes the checker can see through

A class in PPY is an ordinary Python class with one requirement: its fields
are known statically. `__init__` declares them, annotations confirm them,
and from then on every attribute access is resolved on a type the checker
knows. A misspelled method is a compile-time error (`E1202`), not a
`AttributeError` at 3 a.m.

## Fields from annotations, or from `__init__`

```python
class Circle(Shape):
    radius: float

    def __init__(self, radius: float) -> None:
        super().__init__("circle")
        self.radius = radius

    def area(self) -> float:
        return 3.141592653589793 * self.radius * self.radius
```

`Shape.name` is declared in the class body and assigned in `__init__`;
`Circle.radius` the same. Inheritance and `super()` work as in Python, and
`area` is dispatched on the static type where it can be. `Box` is a
`@dataclass` of two floats, which is the one kind of class native code can
take apart: an all-scalar dataclass flattens into scalar arguments at the
boundary ([value classes](../13_value_classes/README.md)).

## Unions, narrowed

```python
@ppy.pure
def describe(value: int | str | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return "text of " + str(len(value))
    return "number " + str(value)
```

After `is None` fails, `value` is `int | str`; after `isinstance(value, str)`
fails, it is `int`. Each branch is typed with what is left, so `len(value)`
and `str(value)` both check. Class construction itself has to be
declarative — a class body that runs statements, or a base whose
`__init_subclass__` does real work, is `E1507` — because the checker judges
a class from what it can read, not from what running it would do.

## Run it

```bash
python  classes.ppy
ppy     classes.ppy
ppy run classes.ppy
```

<!-- outputs:start -->
## What it prints

**`python  classes.ppy`**

```text
circle 12.5664 12.0
none text of 5 number 7
```

**`ppy     classes.ppy`**

```text
circle 12.5664 12.0
none text of 5 number 7
```

**`ppy run classes.ppy`**

```text
circle 12.5664 12.0
none text of 5 number 7
```

<!-- outputs:end -->

## Read on

- [Narrowing](../10_narrowing/README.md) — every narrowing form the checker understands.
- [The subset](../../docs/guide/subset.md) — what makes a class declarative.

`classes.ppy` is hand-written; there is no `.py` source and no conversion step.
