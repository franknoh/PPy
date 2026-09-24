# Common tensors

`ppy.Tensor[dtype, shape]` describes a tensor independently of the library
that owns its storage. PPy carries its element type and dimensions through
type checking, function calls, and canonical IR.

```python
import ppy


def identity(x: ppy.Tensor[ppy.bf16, (3840,)]) -> ppy.Tensor[ppy.bf16, (3840,)]:
    return x
```

The annotation does not allocate memory, wrap an array, or import a GPU
framework.

## Element types and shapes

`ppy.bf16` names bfloat16. It stays distinct from `ppy.f16`, which names IEEE
float16, even though both occupy 16 bits.

The shape is a tuple. The comma in `(3840,)` makes it a one-dimensional
shape. Dimensions can be nonnegative integers or symbolic names, and `()`
describes a scalar tensor.

The existing `Shape` and `DType` refinements offer the same contract:

```python
from typing import Annotated
import ppy

Matrix = ppy.Tensor[ppy.f32, ("N", 4)]
Scalar = ppy.Tensor[ppy.bf16, ()]
Vector = Annotated[ppy.Tensor, ppy.DType("bfloat16"), ppy.Shape(3840)]
```

A bare `ppy.Tensor` leaves dtype and shape unspecified. Canonical lowering
requires those facts to be known from the annotation or analysis.

## Checking a value at run time

`ppy.check[TensorType](value)` validates the object's `shape` and `dtype`
metadata without importing its framework. Repeated symbolic names within one
shape must have the same size.

Ownership markers describe a relationship between caller and callee, so
`ppy.check` cannot establish `Mut` or `Owned` from one object's metadata.

## Writing through an argument

`ppy.Mut[ppy.Tensor[...]]` permits writes for the duration of a call. A
plugin declares which arguments it reads, writes, or retains.

- A borrowed argument may not be kept after the call.
- A read-only borrow cannot be passed to an operation that writes through it.

For a device plugin implementing the common tensor and borrowing APIs, a
broadcast into preallocated output can have this form:

```python
import ppy
import ppy_furiosa as fx


def broadcast(
    x: ppy.Tensor[ppy.bf16, (3840,)],
    out: ppy.Mut[ppy.Tensor[ppy.bf16, (256, 3840)]],
) -> None:
    value = fx.broadcast(x, copies=256)
    fx.store(out, value)
```

The plugin must describe `store` as a write and declare its output argument
as a mutable borrow. PPy preserves the write even though the call returns no
value.

!!! note
    An existing plugin needs to adopt these contracts before accepting this
    example. Installing PPy alone does not update a device plugin.

## Storage and backends

Without a selected device backend, a common tensor uses the shared tensor IR
type. A backend extension can choose a physical representation from the same
analyzed type and facts. Address spaces, device memory placement, and
device-specific execution settings belong to that extension.

Enabling several plugins does not let registration order choose the storage
of a common tensor. The selected backend determines which extension can
provide its representation. If multiple extensions claim it for that backend,
compilation reports the ambiguity.

## CPU paths

Common tensor annotations do not create a CPU native ABI. A CPU path that
cannot represent a tensor keeps its Python fallback. An explicit external
backend request reports a located lowering error when its representation or
operation is unsupported.

See the [plugin API](../api/plugins.md) for argument contracts and type
lowering hooks.
