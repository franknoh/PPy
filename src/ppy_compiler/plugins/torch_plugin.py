"""PyTorch plugin: direct ATen/dispatcher lowering for a curated set (spec 20)."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, Lowering

__all__ = ["ATEN_SCHEMAS", "CURATED_OPS", "TorchPlugin"]

PLUGIN_VERSION = 1

#: The v1 curated operator set (spec 20.4).
CURATED_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "pow",
        "remainder",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "reshape",
        "view",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
        "flatten",
        "contiguous",
        "clone",
        "detach",
        "to",
        "sum",
        "mean",
        "max",
        "min",
        "prod",
        "argmax",
        "argmin",
        "norm",
        "matmul",
        "mm",
        "bmm",
        "linear",
        "relu",
        "gelu",
        "sigmoid",
        "tanh",
        "softmax",
        "log_softmax",
        "cat",
        "stack",
        "zeros",
        "ones",
        "empty",
        "full",
        "arange",
        "tensor",
        "randn",
        "rand",
        "zeros_like",
        "ones_like",
    }
)

_RANDOM_OPS = frozenset({"randn", "rand", "randint", "randperm", "manual_seed"})

#: Non-tensor library functions the analyzer needs a signature for.
_UTILITIES: dict[str, tuple[str, str]] = {
    "torch.manual_seed": ("none", "Random"),
    "torch.set_num_threads": ("none", "Thread"),
    "torch.get_num_threads": ("int", ""),
    "torch.set_grad_enabled": ("none", ""),
    "torch.is_grad_enabled": ("bool", ""),
    "torch.allclose": ("bool", ""),
    "torch.equal": ("bool", ""),
    "torch.numel": ("int", ""),
    "torch.cuda.is_available": ("bool", ""),
    "torch.cuda.device_count": ("int", ""),
    "torch.cuda.synchronize": ("none", "Sync"),
    "torch.cuda.get_device_name": ("str", ""),
    "torch.cuda.get_device_capability": ("int_pair", ""),
    "torch.cuda.current_device": ("int", ""),
    "torch.cuda.empty_cache": ("none", ""),
    "torch.cuda.manual_seed": ("none", "Random"),
}

_UTILITY_TYPES: dict[str, T.Type] = {
    "none": T.NONE,
    "bool": T.BOOL,
    "int": T.INT,
    "float": T.FLOAT,
    "str": T.STR,
    "int_pair": T.Tuple_((T.INT, T.INT)),
    # `tolist` on a multi-dimensional tensor nests, so this is the 1-D shape
    # that a caller can actually feed to `array.array` or a native buffer.
    "float_list": T.list_of(T.FLOAT),  # refined by the receiver's dtype below
    "int_tuple": T.Tuple_((T.INT,), homogeneous=True),
}

#: Tensor methods and attributes with a statically known result.
_TENSOR_MEMBERS: dict[str, str] = {
    "sum": "tensor",
    "mean": "tensor",
    "max": "tensor",
    "min": "tensor",
    "prod": "tensor",
    "abs": "tensor",
    "exp": "tensor",
    "log": "tensor",
    "sqrt": "tensor",
    "relu": "tensor",
    "sigmoid": "tensor",
    "tanh": "tensor",
    "neg": "tensor",
    "clone": "tensor",
    "detach": "tensor",
    "contiguous": "tensor",
    "reshape": "tensor",
    "view": "tensor",
    "transpose": "tensor",
    "permute": "tensor",
    "squeeze": "tensor",
    "unsqueeze": "tensor",
    "flatten": "tensor",
    "to": "tensor",
    "cpu": "tensor",
    "cuda": "tensor",
    "float": "tensor",
    "double": "tensor",
    "matmul": "tensor",
    "mm": "tensor",
    "t": "tensor",
    "expand": "tensor",
    "backward": "none",
    "item": "float",
    "tolist": "float_list",
    "size": "int_tuple",
    "numel": "int",
    "dim": "int",
    "is_contiguous": "bool",
    "is_cuda": "bool_value",
    "requires_grad": "bool_value",
    "grad": "optional_tensor",
}

#: Operations that need dispatcher behavior PPY must not bypass (spec 20.3).
_DISPATCH_SENSITIVE = frozenset({"linear", "matmul", "mm", "bmm", "softmax", "log_softmax"})

_TENSOR = T.Instance("torch.Tensor", (), ("torch.Tensor", "object"))
_MODULE = T.Instance("torch.nn.Module", (), ("torch.nn.Module", "object"))

#: What every `nn.Module` -- and every project class deriving from one --
#: exposes. A method that returns the module returns `nn.Module`, not the
#: subclass; `forward` is the subclass's own and is not here.
_MODULE_MEMBERS: dict[str, T.Type] = {
    "parameters": T.Callable_((), T.instance("Iterator", _TENSOR), "torch.nn.Module.parameters"),
    "buffers": T.Callable_((), T.instance("Iterator", _TENSOR), "torch.nn.Module.buffers"),
    "named_parameters": T.Callable_(
        (), T.instance("Iterator", T.Tuple_((T.STR, _TENSOR))), "torch.nn.Module.named_parameters"
    ),
    "named_buffers": T.Callable_(
        (), T.instance("Iterator", T.Tuple_((T.STR, _TENSOR))), "torch.nn.Module.named_buffers"
    ),
    "children": T.Callable_((), T.instance("Iterator", _MODULE), "torch.nn.Module.children"),
    "modules": T.Callable_((), T.instance("Iterator", _MODULE), "torch.nn.Module.modules"),
    "named_children": T.Callable_(
        (), T.instance("Iterator", T.Tuple_((T.STR, _MODULE))), "torch.nn.Module.named_children"
    ),
    "named_modules": T.Callable_(
        (), T.instance("Iterator", T.Tuple_((T.STR, _MODULE))), "torch.nn.Module.named_modules"
    ),
    "state_dict": T.Callable_((), T.dict_of(T.STR, _TENSOR), "torch.nn.Module.state_dict"),
    "load_state_dict": T.Callable_((), T.OBJECT, "torch.nn.Module.load_state_dict"),
    "training": T.BOOL,
}
for _name in (
    "to",
    "cuda",
    "cpu",
    "eval",
    "train",
    "float",
    "half",
    "double",
    "bfloat16",
    "requires_grad_",
    "share_memory",
):
    _MODULE_MEMBERS[_name] = T.Callable_((), _MODULE, f"torch.nn.Module.{_name}")
for _name in ("zero_grad", "apply", "register_buffer", "register_parameter", "add_module"):
    _MODULE_MEMBERS[_name] = T.Callable_((), T.NONE, f"torch.nn.Module.{_name}")

#: The ATen schema each recognized call resolves to. Routing an individual
#: call to `torch.ops.aten.*` from Python is measurably slower than the
#: library's own C entry point, so this table documents the operator identity
#: for diagnostics and manifests; the path that actually pays off is compiling
#: a whole region against the C++ API (see `torch_region`).
ATEN_SCHEMAS: dict[str, str | tuple[str, str]] = {
    "add": ("add.Tensor", "add.Scalar"),
    "sub": ("sub.Tensor", "sub.Scalar"),
    "mul": ("mul.Tensor", "mul.Scalar"),
    "div": ("div.Tensor", "div.Scalar"),
    "remainder": ("remainder.Tensor", "remainder.Scalar"),
    "pow": ("pow.Tensor_Tensor", "pow.Tensor_Scalar"),
    "eq": ("eq.Tensor", "eq.Scalar"),
    "ne": ("ne.Tensor", "ne.Scalar"),
    "lt": ("lt.Tensor", "lt.Scalar"),
    "le": ("le.Tensor", "le.Scalar"),
    "gt": ("gt.Tensor", "gt.Scalar"),
    "ge": ("ge.Tensor", "ge.Scalar"),
    "neg": "neg.default",
    "abs": "abs.default",
    "exp": "exp.default",
    "log": "log.default",
    "sqrt": "sqrt.default",
    "rsqrt": "rsqrt.default",
    "sin": "sin.default",
    "cos": "cos.default",
    "tanh": "tanh.default",
    "sigmoid": "sigmoid.default",
    "relu": "relu.default",
    "gelu": "gelu.default",
    "silu": "silu.default",
    "erf": "erf.default",
    "matmul": "matmul.default",
    "mm": "mm.default",
    "bmm": "bmm.default",
    "linear": "linear.default",
    "t": "t.default",
    "contiguous": "contiguous.default",
    "clone": "clone.default",
    "detach": "detach.default",
    "sum": "sum.default",
    "mean": "mean.default",
    "prod": "prod.default",
    "argmax": "argmax.default",
    "argmin": "argmin.default",
    "sigmoid_": "sigmoid_.default",
}

#: Python operators mapped to their ATen operator names.
_OPERATORS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "remainder",
    "**": "pow",
    "@": "matmul",
    "<": "lt",
    "<=": "le",
    "==": "eq",
    "!=": "ne",
    ">": "gt",
    ">=": "ge",
    "u-": "neg",
}


class TorchPlugin:
    """Routes recognized tensor calls through PyTorch's dispatcher, never around it."""

    name = "torch"
    modules = ("torch", "torch.nn", "torch.nn.functional")

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.version_policy = str(self.options.get("version-policy", "exact-minor"))

    def fingerprint(self) -> str:
        """PyTorch version, C++ ABI, and accelerator runtime (spec 20.7)."""
        version = "absent"
        cxx11_abi = "unknown"
        cuda = "none"
        try:
            if importlib.util.find_spec("torch") is not None:
                import torch

                version = torch.__version__
                cxx11_abi = str(
                    getattr(
                        torch.compiled_with_cxx11_abi(), "real", torch.compiled_with_cxx11_abi()
                    )
                )
                cuda = str(getattr(torch.version, "cuda", None) or "none")
        except Exception:  # noqa: BLE001 - a broken install must not break analysis
            version = "unknown"
        return (
            f"v{PLUGIN_VERSION}:torch={version}:cxx11abi={cxx11_abi}"
            f":cuda={cuda}:policy={self.version_policy}"
        )

    def external_types(self) -> dict[str, str]:
        return {
            "torch.Tensor": "torch.Tensor",
            "torch.device": "torch.device",
            "torch.dtype": "torch.dtype",
            "torch.nn.Module": "torch.nn.Module",
        }

    def call_alias(self, type_name: str) -> str | None:
        """Calling an `nn.Module` instance runs its `forward`."""
        return "forward" if type_name == "torch.nn.Module" else None

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        attribute = qualname.rpartition(".")[2]
        if attribute == "Tensor":
            return T.ClassObject("torch.Tensor", _TENSOR), Facts()
        if qualname == "torch.nn.Module":
            return T.ClassObject("torch.nn.Module", _MODULE), Facts()
        if attribute in {"float32", "float16", "bfloat16", "int64", "int32", "bool"}:
            return T.Instance("torch.dtype", (), ("torch.dtype", "object")), Facts()
        if qualname == "torch.__version__":
            return T.STR, Facts()
        if qualname in _UTILITIES:
            kind, _effect = _UTILITIES[qualname]
            return T.Callable_((), _UTILITY_TYPES.get(kind, T.UNKNOWN), qualname), Facts()
        if attribute in {"cuda", "nn", "functional", "linalg", "version", "overrides"}:
            return T.Module_(qualname), Facts()
        return None

    def instance_attribute(
        self, type_name: str, attribute: str, facts: Facts | None = None
    ) -> tuple[T.Type, Facts] | None:
        """Methods and attributes of an exact `torch.Tensor` value, and of an
        `nn.Module` -- which a project class deriving from one inherits."""
        if type_name == "torch.nn.Module":
            member = _MODULE_MEMBERS.get(attribute)
            return None if member is None else (member, Facts())
        if type_name != "torch.Tensor":
            return None
        kind = _TENSOR_MEMBERS.get(attribute)
        if kind is None:
            return None
        if kind == "tensor":
            return T.Callable_((), _TENSOR, f"torch.Tensor.{attribute}"), Facts()
        if kind == "bool_value":
            return T.BOOL, Facts()
        if kind == "optional_tensor":
            return T.union(_TENSOR, T.NONE), Facts()
        if kind == "float_list":
            dtype = getattr(facts, "dtype", None) if facts is not None else None
            element = T.INT if dtype is not None and "int" in dtype else T.FLOAT
            return T.Callable_((), T.list_of(element), f"torch.Tensor.{attribute}"), Facts()
        return (
            T.Callable_((), _UTILITY_TYPES.get(kind, T.UNKNOWN), f"torch.Tensor.{attribute}"),
            Facts(),
        )

    def subscript(
        self, type_name: str, *, is_slice: bool, tupled: bool
    ) -> tuple[T.Type, Facts] | None:
        """Indexing an array: a slice stays an array, a full index is a scalar."""
        if type_name != "torch.Tensor":
            return None
        if is_slice or tupled:
            return _TENSOR, Facts()
        # A single index into an array of unknown rank may still be an array,
        # so the result is the element type only when the rank is known to be
        # one, which v1 does not track. `_TENSOR` is the safe answer.
        return _TENSOR, Facts()

    def operator(self, symbol: str) -> str | None:
        return _OPERATORS.get(symbol)

    def schema_for(self, operation: str, args: Sequence[T.Type]) -> str | None:
        """The ATen schema a curated call corresponds to, for diagnostics.

        The overload follows the static argument types, the same way the C++
        API selects between a tensor and a scalar operand.
        """
        schema = ATEN_SCHEMAS.get(operation)
        if schema is None:
            return None
        if not isinstance(schema, tuple):
            return f"aten::{schema}"
        tensor_form, scalar_form = schema
        if len(args) != 2:
            return None
        second = T.strip_literal(args[1])
        is_tensor = isinstance(second, T.Instance) and second.name == "torch.Tensor"
        return f"aten::{tensor_form if is_tensor else scalar_form}"

    def _operand(self, t: T.Type) -> str:
        base = T.strip_literal(t)
        if isinstance(base, T.Instance):
            if base.name == "torch.Tensor":
                return "tensor"
            if base.name in {"int", "float", "bool"}:
                return "scalar"
        return "other"

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        if qualname in _UTILITIES:
            kind, effect = _UTILITIES[qualname]
            effects = EffectSet.of(*((Effect(effect),) if effect else ()))
            return CallResult(
                _UTILITY_TYPES.get(kind, T.UNKNOWN),
                Facts(),
                effects,
                Lowering.PYTHON_FALLBACK,
                "a library utility call, not a tensor operation",
            )

        operation = qualname.rpartition(".")[2]
        if operation not in CURATED_OPS:
            return None

        effects = EffectSet.of(Effect.ALLOC, raises=("RuntimeError", "TypeError"))
        if operation in _RANDOM_OPS:
            effects = effects.add(Effect.RANDOM)

        lowering, reason, guards = self._lowering(operation, args, keywords)
        schema = self.schema_for(operation, [t for t, _facts in args])
        if schema is not None:
            reason = f"{reason} (`{schema}`)"
        result_type: T.Type = _TENSOR
        if operation in {"argmax", "argmin"} and not args:
            result_type = T.INT
        return CallResult(
            result_type, Facts(exact_class="torch.Tensor"), effects, lowering, reason, guards
        )

    def _lowering(
        self,
        operation: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> tuple[Lowering, str, tuple[str, ...]]:
        for arg_type, _facts in args:
            base = T.strip_literal(arg_type)
            if isinstance(base, (T.AnyType, T.UnknownType)):
                return Lowering.PYTHON_FALLBACK, "an operand has no static type", ()
            if isinstance(base, T.Instance) and base.name not in {
                "torch.Tensor",
                "int",
                "float",
                "bool",
            }:
                return (
                    Lowering.PYTHON_FALLBACK,
                    f"operand `{base}` is outside the curated domain",
                    (),
                )

        guards = (
            "exact torch.Tensor (no tensor subclass)",
            "no __torch_function__ override in effect",
            "supported device and layout",
            "operator schema matches the installed build",
        )
        if operation in _DISPATCH_SENSITIVE:
            return (
                Lowering.DIRECT_NATIVE_CALL,
                (
                    f"`{operation}` has an ATen C++ counterpart that keeps autograd and "
                    "device keys; it is used when the whole function compiles into a region"
                ),
                guards,
            )
        return (
            Lowering.DIRECT_NATIVE_CALL,
            (
                f"`{operation}` has an ATen C++ counterpart; it is used when the whole "
                "function compiles into a region"
            ),
            guards,
        )
