"""PyTorch plugin: direct ATen/dispatcher lowering for a curated set (spec 20)."""

from __future__ import annotations

import importlib.util
from typing import Sequence

from ..analysis import types as T
from ..analysis.effects import Effect, EffectSet
from ..analysis.refinements import Facts
from .base import CallResult, Lowering

__all__ = ["TorchPlugin", "CURATED_OPS"]

PLUGIN_VERSION = 1

#: The v1 curated operator set (spec 20.4).
CURATED_OPS = frozenset({
    "add", "sub", "mul", "div", "neg", "abs", "pow", "remainder",
    "eq", "ne", "lt", "le", "gt", "ge",
    "reshape", "view", "transpose", "permute", "squeeze", "unsqueeze", "flatten",
    "contiguous", "clone", "detach", "to",
    "sum", "mean", "max", "min", "prod", "argmax", "argmin", "norm",
    "matmul", "mm", "bmm", "linear",
    "relu", "gelu", "sigmoid", "tanh", "softmax", "log_softmax",
    "cat", "stack",
    "zeros", "ones", "empty", "full", "arange", "tensor", "randn", "rand", "zeros_like", "ones_like",
})

_RANDOM_OPS = frozenset({"randn", "rand", "randint", "randperm"})

#: Operations that need dispatcher behavior PPY must not bypass (spec 20.3).
_DISPATCH_SENSITIVE = frozenset({"linear", "matmul", "mm", "bmm", "softmax", "log_softmax"})

_TENSOR = T.Instance("torch.Tensor", (), ("torch.Tensor", "object"))

#: Python operators mapped to their ATen operator names.
_OPERATORS = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "remainder",
    "**": "pow", "@": "matmul",
    "<": "lt", "<=": "le", "==": "eq", "!=": "ne", ">": "gt", ">=": "ge",
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
                cxx11_abi = str(getattr(torch.compiled_with_cxx11_abi(), "real", torch.compiled_with_cxx11_abi()))
                cuda = str(getattr(torch.version, "cuda", None) or "none")
        except Exception:  # noqa: BLE001 - a broken install must not break analysis
            version = "unknown"
        return f"v{PLUGIN_VERSION}:torch={version}:cxx11abi={cxx11_abi}:cuda={cuda}:policy={self.version_policy}"

    def external_types(self) -> dict[str, str]:
        return {
            "torch.Tensor": "torch.Tensor",
            "torch.device": "torch.device",
            "torch.dtype": "torch.dtype",
            "torch.nn.Module": "torch.nn.Module",
        }

    def attribute_type(self, qualname: str) -> tuple[T.Type, Facts] | None:
        attribute = qualname.rpartition(".")[2]
        if attribute == "Tensor":
            return T.ClassObject("torch.Tensor", _TENSOR), Facts()
        if attribute in {"float32", "float16", "bfloat16", "int64", "int32", "bool"}:
            return T.Instance("torch.dtype", (), ("torch.dtype", "object")), Facts()
        return None

    def operator(self, symbol: str) -> str | None:
        return _OPERATORS.get(symbol)

    def call(
        self,
        qualname: str,
        args: Sequence[tuple[T.Type, Facts]],
        keywords: dict[str, tuple[T.Type, Facts]],
    ) -> CallResult | None:
        operation = qualname.rpartition(".")[2]
        if operation not in CURATED_OPS:
            return None

        effects = EffectSet.of(Effect.ALLOC, raises=("RuntimeError", "TypeError"))
        if operation in _RANDOM_OPS:
            effects = effects.add(Effect.RANDOM)

        lowering, reason, guards = self._lowering(operation, args, keywords)
        result_type: T.Type = _TENSOR
        if operation in {"argmax", "argmin"} and not args:
            result_type = T.INT
        return CallResult(result_type, Facts(exact_class="torch.Tensor"), effects, lowering, reason, guards)

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
                "torch.Tensor", "int", "float", "bool"
            }:
                return Lowering.PYTHON_FALLBACK, f"operand `{base}` is outside the curated domain", ()

        guards = (
            "exact torch.Tensor (no tensor subclass)",
            "no __torch_function__ override in effect",
            "supported device and layout",
            "operator schema matches the installed build",
        )
        if operation in _DISPATCH_SENSITIVE:
            return (
                Lowering.DIRECT_NATIVE_CALL,
                f"`{operation}` lowered through the ATen dispatcher, preserving autograd and device keys",
                guards,
            )
        return (
            Lowering.DIRECT_NATIVE_CALL,
            f"`{operation}` lowered to an unboxed dispatcher call for its known schema",
            guards,
        )
