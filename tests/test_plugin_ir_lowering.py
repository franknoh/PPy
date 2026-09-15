"""Public plugin contracts reach canonical IR without a CPU ABI."""

from __future__ import annotations

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect, EffectSet
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.backend import Backend, BackendValidationError
from ppy_compiler.driver.ir_pipeline import canonical_ir_modules, optimize_shared_ir
from ppy_compiler.driver.pipeline import analyze_paths, open_project
from ppy_compiler.ir import Dialect, DialectType, OpSpec, PtrType, verify
from ppy_compiler.lowering import lower_module_to_ir
from ppy_compiler.plugins import CallResult, DialectOperationSpec, Plugin, PluginRegistry

POINTER = T.Instance("sample.Pointer", (), ("sample.Pointer", "object"))
TENSOR = T.Instance("sample.Tensor", (), ("sample.Tensor", "object"))
FACTS = Facts(dtype="float32", shape=(4, 8))
IR_TENSOR = DialectType("sample", "tensor", ("float32", 4, 8))


class SampleDialect(Dialect):
    name = "sample"
    version = 3

    def verify_type(self, t):
        return None if t.name == "tensor" else "unknown sample type"

    def register_operations(self, registry):
        registry.add_op(OpSpec("sample.make", pure=True, operands=0, results=1))
        registry.add_op(OpSpec("sample.stop", terminator=True, operands=0, results=0, successors=0))
        registry.add_op(OpSpec("sample.scalar", operands=1, results=1))
        registry.add_op(OpSpec("sample.scale", pure=True, operands=2, results=1))
        # A call's stronger write effect must win over the general pure flag.
        registry.add_op(OpSpec("sample.store", pure=True, operands=1, results=0))


class SamplePlugin(Plugin):
    name = "sample"
    modules = ("sample",)

    def external_types(self):
        return {"sample.Tensor": "sample.Tensor", "sample.Pointer": "sample.Pointer"}

    def lower_type(self, type_, facts):
        if type_ == POINTER:
            return PtrType(IR_TENSOR)
        if type_ == TENSOR:
            return DialectType(
                "sample", "tensor", (facts.dtype or "float32", *(facts.shape or (4, 8)))
            )
        return None

    def register_dialects(self, registry):
        registry.register(SampleDialect())

    def call(self, qualname, args, keywords):
        if qualname == "sample.stop":
            return CallResult(T.NONE, lowering=DialectOperationSpec("sample", "stop"))
        if qualname == "sample.scalar":
            return CallResult(
                T.INT,
                effects=EffectSet.of(Effect.WRITE_MEMORY),
                lowering=DialectOperationSpec("sample", "scalar"),
            )
        if qualname == "sample.axis":
            return CallResult(
                T.INT,
                Facts(constant=1, has_constant=True),
                effects=EffectSet.of(Effect.IO),
                lowering=DialectOperationSpec("sample", "axis"),
            )
        if qualname == "sample.make":
            return CallResult(TENSOR, FACTS, lowering=DialectOperationSpec("sample", "make"))
        if qualname == "sample.scale":
            return CallResult(
                TENSOR,
                FACTS,
                lowering=DialectOperationSpec(
                    "sample",
                    "scale",
                    (("mode", "linear"),),
                    keyword_operands=("factor",),
                    keyword_attributes=("axis",),
                ),
                guards=("shape matches",),
            )
        if qualname == "sample.store":
            return CallResult(
                T.NONE,
                effects=EffectSet.of(Effect.WRITE_MEMORY),
                lowering=DialectOperationSpec("sample", "store"),
            )
        return None


class SampleBackend(Backend):
    name = "sample"
    api_version = 1


def bundle_for(write, source):
    path = write("kernel.ppy", source)
    project = open_project(path, config_overrides={"strict": False})
    project.plugins.register(SamplePlugin())
    return analyze_paths(project, [path], backend="llvm")


def lower(bundle, **kwargs):
    analysis = bundle.analysis.modules["kernel"]
    functions = {
        info.qualname: (info, analysis.functions[info.qualname], info.node)
        for info in bundle.symbols.modules["kernel"].functions.values()
    }
    return lower_module_to_ir(analysis, functions, plugins=bundle.project.plugins, **kwargs)


def test_type_hook_preserves_facts_and_declines_unknown_types():
    registry = PluginRegistry()
    registry.register(Plugin())
    registry.register(SamplePlugin())
    assert registry.lower_type(TENSOR, FACTS) == IR_TENSOR
    assert registry.lower_type(TENSOR, Facts(dtype="int16", shape=(3,))) == DialectType(
        "sample", "tensor", ("int16", 3)
    )
    assert registry.lower_type(T.STR, Facts()) is None


def test_actual_custom_types_across_arguments_results_locals_and_calls(write):
    bundle = bundle_for(
        write,
        """
        from sample import Tensor
        import sample
        def helper(x: Tensor) -> Tensor:
            value = sample.scale(x, factor=2, axis=1)
            value = sample.scale(value, factor=3, axis=1)
            return value
        def entry(x: Tensor) -> Tensor:
            return helper(x)
    """,
    )
    lowered = lower(bundle)
    assert not lowered.rejected
    assert len(lowered.functions) == 2
    for function in lowered.module.functions.values():
        assert function.params[0][1] == IR_TENSOR
        assert function.results == (IR_TENSOR,)
    assert not verify(lowered.module, bundle.project.plugins.dialect_registry())
    ops = [op for function in lowered.module.functions.values() for op in function.operations()]
    scale = next(op for op in ops if op.name == "sample.scale")
    assert scale.operands[0].type == IR_TENSOR
    assert scale.attributes["axis"] == 1
    assert scale.attributes["mode"] == "linear"
    assert scale.attributes["guards"] == ("shape matches",)
    assert scale.location.line == 4
    assert any(op.name == "core.call" and op.result.type == IR_TENSOR for op in ops)


def test_void_write_survives_shared_optimization(write):
    bundle = bundle_for(
        write,
        """
        import sample
        def entry() -> None:
            value = sample.make()
            sample.store(value)
            return
    """,
    )
    lowered = lower(bundle)
    assert not lowered.rejected
    optimize_shared_ir(lowered.module, 3, bundle.project.plugins)
    ops = list(lowered.module.functions["kernel_entry"].operations())
    store = next(op for op in ops if op.name == "sample.store")
    assert not store.results
    assert store.attributes["effects"] == ("write_memory",)
    assert store.operands[0].type == IR_TENSOR


@pytest.mark.parametrize(
    "keyword,reason", [("axis=n", "compile-time constant"), ("other=n", "no declared role")]
)
def test_keywords_require_explicit_roles_and_constant_attributes(write, keyword, reason):
    bundle = bundle_for(
        write,
        f"""
        import sample
        from sample import Tensor
        def entry(x: Tensor, n: int) -> Tensor:
            return sample.scale(x, factor=n, {keyword})
    """,
    )
    lowered = lower(bundle)
    assert reason in lowered.rejected["kernel.entry"]


def test_custom_type_cpu_boundary_falls_back_without_crashing(write):
    bundle = bundle_for(
        write,
        """
        from sample import Tensor
        def entry(x: Tensor) -> Tensor:
            return x
    """,
    )
    lowered = lower(bundle, cpu_compatible=True)
    assert not lowered.functions
    assert "no native ABI" in lowered.rejected["kernel.entry"]


def test_explicit_backend_reports_plain_function_failure_and_normal_path_falls_back(write):
    bundle = bundle_for(
        write,
        """
        def good(n: int) -> int:
            return n + 1
        def bad(text: str) -> str:
            return text
    """,
    )
    modules = canonical_ir_modules(bundle)
    assert "kernel" in modules
    with pytest.raises(
        BackendValidationError, match=r"kernel.bad.*no canonical IR representation"
    ) as caught:
        canonical_ir_modules(bundle, backend=SampleBackend())
    assert caught.value.location.line == 3
    assert "kernel.ppy" in str(caught.value)


def test_analysis_keeps_original_contract_and_result_facts(write):
    bundle = bundle_for(
        write,
        """
        import sample
        def entry():
            return sample.make()
    """,
    )
    analysis = bundle.analysis.modules["kernel"]
    note = next(note for note in analysis.lowerings.values() if note.qualname == "sample.make")
    assert note.spec == DialectOperationSpec("sample", "make")
    assert note.result_type == TENSOR
    assert note.facts == FACTS


def test_annotated_facts_survive_cross_module_custom_calls(write):
    write(
        "helper.ppy",
        """
        from typing import Annotated
        import ppy
        from sample import Tensor
        def identity(x: Annotated[Tensor, ppy.DType("int16"), ppy.Shape(3)]) -> Annotated[Tensor, ppy.DType("int16"), ppy.Shape(3)]:
            return x
    """,
    )
    bundle = bundle_for(
        write,
        """
        from typing import Annotated
        import ppy
        import helper
        from sample import Tensor
        def entry(x: Annotated[Tensor, ppy.DType("int16"), ppy.Shape(3)]) -> Annotated[Tensor, ppy.DType("int16"), ppy.Shape(3)]:
            return helper.identity(x)
    """,
    )
    modules = canonical_ir_modules(bundle, backend=SampleBackend())
    assert set(modules) == {"helper", "kernel"}
    expected = DialectType("sample", "tensor", ("int16", 3))
    for module in modules.values():
        assert module.dialects["sample"] == 3
        assert not verify(module, bundle.project.plugins.dialect_registry())
        for function in module.functions.values():
            assert function.params[0][1] == expected
            assert function.results == (expected,)
    call = next(
        op
        for op in modules["kernel"].functions["kernel_entry"].operations()
        if op.name == "core.call"
    )
    assert call.result.type == expected


def test_lower_function_utility_accepts_custom_signature(write):
    from ppy_compiler.lowering import lower_function

    bundle = bundle_for(
        write,
        """
        from sample import Tensor
        def identity(x: Tensor) -> Tensor:
            return x
    """,
    )
    info = bundle.symbols.modules["kernel"].functions["identity"]
    module = lower_function(
        bundle.analysis.modules["kernel"],
        info,
        info.node,
        plugins=bundle.project.plugins,
        symbol="renamed",
    )
    function = module.functions["kernel_identity"]
    assert function.params[0][1] == IR_TENSOR
    assert function.results == (IR_TENSOR,)
    assert function.attributes["ppy.symbol"] == "renamed"


def test_void_result_is_not_a_value(write):
    bundle = bundle_for(
        write,
        """
        import sample
        def bad() -> int:
            value = sample.make()
            result = sample.store(value)
            return 1
    """,
    )
    assert "void dialect operation" in lower(bundle).rejected["kernel.bad"]


def test_custom_pointer_parameters_results_and_rebinding_keep_nested_type(write):
    bundle = bundle_for(
        write,
        """
        from sample import Pointer
        def identity(x: Pointer) -> Pointer:
            value = x
            value = x
            return value
        def entry(x: Pointer) -> Pointer:
            return identity(x)
    """,
    )
    lowered = lower(bundle)
    assert not lowered.rejected
    assert lowered.module.dialects["sample"] == 3
    assert not verify(lowered.module, bundle.project.plugins.dialect_registry())
    for function in lowered.module.functions.values():
        assert function.params[0][1] == PtrType(IR_TENSOR)
        assert function.results == (PtrType(IR_TENSOR),)


def test_constant_attribute_does_not_erase_effectful_expression(write):
    bundle = bundle_for(
        write,
        """
        import sample
        from sample import Tensor
        def entry(x: Tensor) -> Tensor:
            return sample.scale(x, factor=2, axis=sample.axis())
    """,
    )
    assert "compile-time constant" in lower(bundle).rejected["kernel.entry"]


def test_cpu_rejects_surviving_custom_operation_and_callers_only(write):
    from ppy_compiler.backend.llvm.ir_pipeline import lower_module_via_ir

    bundle = bundle_for(
        write,
        """
        import sample
        def custom(n: int) -> int:
            return sample.scalar(n)
        def caller(n: int) -> int:
            return custom(n)
        def ordinary(n: int) -> int:
            return n + 1
    """,
    )
    analysis = bundle.analysis.modules["kernel"]
    functions = {
        info.qualname: (info, analysis.functions[info.qualname], info.node)
        for info in bundle.symbols.modules["kernel"].functions.values()
    }
    result = lower_module_via_ir(analysis, functions, plugins=bundle.project.plugins, opt_level=0)
    assert set(result.functions) == {"kernel.ordinary"}
    assert "sample.scalar" in result.rejected["kernel.custom"]
    assert "no LLVM lowering" in result.rejected["kernel.caller"]
    assert "ppy_kernel_ordinary" in result.ir


def test_cpu_rejects_custom_ir_result_even_with_scalar_source_return(write):
    class ResultPlugin(Plugin):
        name = "custom_result"

        def lower_type(self, type_, facts):
            return IR_TENSOR if type_ == T.INT else None

    bundle = bundle_for(
        write,
        """
        import sample
        def entry() -> int:
            return sample.scalar(1)
    """,
    )
    bundle.project.plugins.register(ResultPlugin())
    canonical = lower(bundle)
    assert not canonical.rejected
    assert canonical.module.functions["kernel_entry"].results == (IR_TENSOR,)
    assert canonical.functions["kernel.entry"].signature.native is None
    cpu = lower(bundle, cpu_compatible=True)
    assert not cpu.functions
    assert "no CPU native ABI" in cpu.rejected["kernel.entry"]


def test_custom_zero_successor_terminator_closes_source_block(write):
    bundle = bundle_for(
        write,
        """
        import sample
        def entry() -> None:
            sample.stop()
            sample.scalar(7)
    """,
    )
    lowered = lower(bundle)
    assert not lowered.rejected
    function = lowered.module.functions["kernel_entry"]
    assert [operation.name for operation in function.operations()] == ["sample.stop"]
    assert not verify(lowered.module, bundle.project.plugins.dialect_registry())
