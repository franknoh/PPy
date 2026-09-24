# Classes

Ordinary classes work when their fields are known statically, and a
misspelled method is a compile-time error (`E1202`). `__init__` declares the
fields, annotations confirm them, and every attribute access is resolved on
a type the checker knows.

## Run it

```bash
python  classes.ppy
ppy     classes.ppy
ppy run classes.ppy
```

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

`Shape.name` is declared in the class body and assigned in `__init__`, and
so is `Circle.radius`. Inheritance and `super()` work as in Python, and
`area` is dispatched on the static type where it can be.

`Box` is a `@dataclass` of two floats, the one kind of class native code can
take apart: an all-scalar dataclass flattens into scalar arguments at the
boundary ([value classes](../13_value_classes/README.md)).

## Narrowing a union

```python
@ppy.pure
def describe(value: int | str | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return "text of " + str(len(value))
    return "number " + str(value)
```

After `is None` fails, `value` is `int | str`. After `isinstance(value, str)`
fails, it is `int`. Each branch is typed with what is left, so `len(value)`
and `str(value)` both check.

## Declarative construction

The checker judges a class from what it can read, not from what running it
would do. These are `E1507`:

- a class body that runs statements
- a body value constructing a project descriptor whose `__set_name__` runs
  at creation
- a base whose `__init_subclass__` does real work

The safe class hoister in `ppy convert` uses the same facts.

<!-- outputs:start -->
## What it prints

**`python  classes.ppy`**, **`ppy     classes.ppy`**, **`ppy run classes.ppy`**

```text
circle 12.5664 12.0
none text of 5 number 7
```

<!-- outputs:end -->

Read on: [Narrowing](../10_narrowing/README.md) ·
[The subset](../../docs/guide/subset.md)

`classes.ppy` is hand-written; there is no `.py` source and no conversion step.
