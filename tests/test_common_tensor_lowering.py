"""Common tensors retain shape, dtype and backend-independent value semantics."""

from __future__ import annotations

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect, EffectSet
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.backend import Backend
from ppy_compiler.driver.ir_pipeline import canonical_ir_modules
from ppy_compiler.driver.pipeline import analyze_paths, open_project
from ppy_compiler.ir import BF16, F16, Dialect, DialectType, OpSpec
from ppy_compiler.plugins import CallArgument, CallResult, DialectOperationSpec, Plugin


def test_bfloat16_is_distinct_scalar_and_tensor_type_through_codec():
    from ppy_compiler.ir import Builder, IRModule, decode, encode, parse_type, verify
    from ppy_compiler.ir.dialects import core, tensor
    from ppy_compiler.ir.types import is_arithmetic, is_scalar

    assert BF16 != F16
    assert str(BF16) == "bf16"
    assert parse_type("bf16") == BF16
    assert is_scalar(BF16) and is_arithmetic(BF16)
    declared = tensor.tensor_type(BF16, (256, 3840))
    assert parse_type(str(declared)) == declared
    module = IRModule("bf16_values")
    module.require("tensor", 1)
    function = module.add_function("identity", [("x", declared)], [declared])
    builder = Builder(function.add_entry_block())
    core.ret(builder, function.entry.arguments[0])
    assert not verify(module)
    text = encode(module)
    assert "tensor.tensor<bf16, 256, 3840>" in text
    restored = decode(text)
    assert restored.functions["identity"].results == (declared,)
    assert encode(restored) == text
    assert not verify(restored)


def test_cpu_emitter_rejects_bfloat16_instead_of_substituting_float16():
    from ppy_compiler.backend.llvm.from_ir import EmitError, emit_module
    from ppy_compiler.ir import Builder, IRModule
    from ppy_compiler.ir.dialects import core

    module = IRModule("bf16_cpu")
    function = module.add_function("identity", [("x", BF16)], [BF16])
    builder = Builder(function.add_entry_block())
    core.ret(builder, function.entry.arguments[0])
    with pytest.raises(EmitError, match="bf16 has no LLVM representation"):
        emit_module(module)


def test_generic_tensor_memory_lowering_refuses_bfloat16_without_backend_support():
    from ppy_compiler.ir import BufferType, Builder, IRModule, PassContext
    from ppy_compiler.ir.dialects import core, tensor
    from ppy_compiler.ir.transforms import LoweringError, LowerTensor

    module = IRModule("bf16_memory")
    module.require("tensor", 1)
    function = module.add_function("copy", [("x", BufferType(BF16)), ("out", BufferType(BF16))], [])
    builder = Builder(function.add_entry_block())
    value = tensor.load(builder, function.entry.arguments[0], tensor.tensor_type(BF16, (4,)))
    tensor.store(builder, value, function.entry.arguments[1])
    core.ret(builder)
    with pytest.raises(LoweringError, match=r"bfloat16.*backend"):
        LowerTensor().run(module, PassContext())


COMMON_TENSOR = T.Instance("ppy.Tensor", (), ("ppy.Tensor", "object"))
BROADCAST_SOURCE = """
    import ppy
    import ppy_furiosa as fx

    def broadcast(x: ppy.Tensor[ppy.bf16, (3840,)], out: ppy.Mut[ppy.Tensor[ppy.bf16, (256, 3840)]]) -> None:
        value = fx.broadcast(x, copies=256)
        fx.store(out, value)
"""


class FixtureDialect(Dialect):
    name = "fixture"
    version = 2

    def register_operations(self, registry):
        registry.add_op(
            OpSpec(
                "fixture.broadcast",
                pure=True,
                operands=1,
                results=1,
                required_attributes=("copies",),
            )
        )
        registry.add_op(OpSpec("fixture.store", operands=2, results=0))

    def verify_type(self, type_):
        return (
            None
            if type_.name == "tensor" and type_.args[0] in {"alpha", "beta"}
            else "invalid fixture tensor"
        )


class FixturePlugin(Plugin):
    name = "fixture"
    modules = ("ppy_furiosa",)

    def register_dialects(self, registry):
        registry.register(FixtureDialect())

    def call(self, qualname, args, keywords):
        if qualname == "ppy_furiosa.broadcast":
            copies = keywords["copies"][1].constant
            source = args[0][1]
            return CallResult(
                COMMON_TENSOR,
                Facts(dtype=source.dtype, shape=(copies, *source.shape), ownership="owned"),
                lowering=DialectOperationSpec(
                    "fixture", "broadcast", keyword_attributes=("copies",)
                ),
                arguments=(CallArgument(0, "borrowed"),),
            )
        if qualname == "ppy_furiosa.store":
            return CallResult(
                T.NONE,
                effects=EffectSet.of(Effect.WRITE_MEMORY),
                lowering=DialectOperationSpec("fixture", "store"),
                arguments=(CallArgument(0, "mut"), CallArgument(1, "borrowed")),
            )
        return None


class PhysicalPlugin(Plugin):
    def __init__(self, backend_name):
        super().__init__()
        self.name = backend_name

    def lower_type_for_backend(self, type_, facts, backend):
        if isinstance(type_, T.Instance) and type_.name == "ppy.Tensor" and backend == self.name:
            return DialectType("fixture", "tensor", (backend, BF16, *facts.shape))
        return None

    def lower_type(self, type_, facts):
        # Old unscoped hooks must not capture common Tensor values.
        if isinstance(type_, T.Instance) and type_.name == "ppy.Tensor":
            return DialectType("fixture", "tensor", (self.name, BF16, 999))
        return None


class FixtureBackend(Backend):
    api_version = 1

    def __init__(self, name):
        super().__init__()
        self.name = name

    def emit(self, module, format, context):  # pylint: disable=redefined-builtin
        from ppy_compiler.ir import encode

        assert format == self.name + "-source"
        return encode(module, context.registry)


def fixture_bundle(write, source=BROADCAST_SOURCE, physical=("alpha", "beta")):
    path = write("common.ppy", source)
    project = open_project(path)
    project.plugins.register(FixturePlugin())
    for name in physical:
        project.plugins.register(PhysicalPlugin(name))
    return analyze_paths(project, [path], backend="llvm")


def test_user_broadcast_and_store_remain_common_tensors_without_backend(write):
    from ppy_compiler.ir import decode, encode, verify
    from ppy_compiler.ir.dialects import tensor

    bundle = fixture_bundle(write)
    assert not bundle.diagnostics.has_errors(), [str(d) for d in bundle.diagnostics.sorted()]
    modules = canonical_ir_modules(bundle)
    module = modules["common"]
    function = module.functions["common_broadcast"]
    expected = (tensor.tensor_type(BF16, (3840,)), tensor.tensor_type(BF16, (256, 3840)))
    assert tuple(type_ for _name, type_ in function.params) == expected
    assert "ownership" not in function.param_attributes[0]
    assert function.param_attributes[1]["ownership"] == "mut"
    ops = list(function.operations())
    broadcast = next(op for op in ops if op.name == "fixture.broadcast")
    store = next(op for op in ops if op.name == "fixture.store")
    assert broadcast.result.type == expected[1]
    assert broadcast.attributes["copies"] == 256
    assert tuple(value.type for value in store.operands) == (expected[1], expected[1])
    assert store.attributes["effects"] == ("write_memory",)
    assert not store.results
    registry = bundle.project.plugins.dialect_registry()
    assert not verify(module, registry)
    assert encode(decode(encode(module, registry), registry), registry) == encode(module, registry)


def test_two_backends_select_physical_types_independently_of_plugin_order(write):
    for order in (("alpha", "beta"), ("beta", "alpha")):
        bundle = fixture_bundle(write, physical=order)
        for backend in ("alpha", "beta"):
            module = canonical_ir_modules(bundle, backend=FixtureBackend(backend))["common"]
            function = module.functions["common_broadcast"]
            assert function.params[0][1] == DialectType("fixture", "tensor", (backend, BF16, 3840))
            assert function.params[1][1] == DialectType(
                "fixture", "tensor", (backend, BF16, 256, 3840)
            )
            assert function.param_attributes[1]["ownership"] == "mut"
            broadcast = next(op for op in function.operations() if op.name == "fixture.broadcast")
            assert broadcast.result.type == function.params[1][1]


def test_common_tensor_cpu_path_falls_back(write):
    from ppy_compiler.backend.llvm.ir_pipeline import lower_module_via_ir

    bundle = fixture_bundle(write)
    analysis = bundle.analysis.modules["common"]
    info = bundle.symbols.modules["common"].functions["broadcast"]
    result = lower_module_via_ir(
        analysis,
        {info.qualname: (info, analysis.functions[info.qualname], info.node)},
        plugins=bundle.project.plugins,
    )
    assert not result.functions
    assert info.qualname in result.rejected
    assert not result.ir


def test_bfloat16_tensor_arithmetic_verifies_and_stays_tensor_when_requested():
    from ppy_compiler.driver.ir_pipeline import optimize_shared_ir
    from ppy_compiler.ir import Builder, IRModule, verify
    from ppy_compiler.ir.dialects import core, tensor

    declared = tensor.tensor_type(BF16, (4,))
    module = IRModule("bf16_arithmetic")
    module.require("tensor", 1)
    function = module.add_function("divide", [("a", declared), ("b", declared)], [declared])
    builder = Builder(function.add_entry_block())
    value = tensor.elementwise(builder, "div", *function.entry.arguments)
    core.ret(builder, value)
    assert not verify(module)
    optimize_shared_ir(module, 2, lower_tensors=False)
    assert function.results == (declared,)
    assert not verify(module)


def test_emit_driver_threads_the_selected_backend_to_type_lowering(write, monkeypatch):
    import argparse
    import io

    import ppy_compiler.backend as backends
    import ppy_compiler.driver.emit as emit_driver
    from ppy_compiler.backend import EmitFormat
    from ppy_compiler.backend.registry import FormatOwner
    from ppy_compiler.driver.reporting import Reporter

    bundle = fixture_bundle(write)
    target = bundle.project.root / "common.ppy"
    monkeypatch.setattr(emit_driver, "open_project", lambda _target: bundle.project)

    def owner(kind, _config):
        name = kind.removesuffix("-source")
        return FormatOwner(FixtureBackend(name), EmitFormat(kind, ".txt", requires_toolchain=False))

    monkeypatch.setattr(backends, "emit_format_owner", owner)
    for name in ("alpha", "beta"):
        output = bundle.project.root / (name + ".txt")
        options = argparse.Namespace(target=target, output=output, kind=name + "-source")
        stream = io.StringIO()
        assert emit_driver.run_emit(options, Reporter(color=False, stream=stream)) == 0, (
            stream.getvalue()
        )
        emitted = output.read_text()
        assert f"fixture.tensor<{name}, bf16, 3840>" in emitted
        assert f"fixture.tensor<{name}, bf16, 256, 3840>" in emitted
        assert "fixture.store" in emitted


def test_common_tensor_shapes_survive_cross_module_arguments_and_results(write):
    from ppy_compiler.ir.dialects import tensor

    write(
        "helper.ppy",
        """
        import ppy
        def identity(x: ppy.Tensor[ppy.bf16, (3,)]) -> ppy.Tensor[ppy.bf16, (3,)]:
            return x
    """,
    )
    bundle = fixture_bundle(
        write,
        """
        import ppy
        import helper
        def entry(x: ppy.Tensor[ppy.bf16, (3,)]) -> ppy.Tensor[ppy.bf16, (3,)]:
            return helper.identity(x)
    """,
    )
    for backend in (None, FixtureBackend("beta")):
        modules = canonical_ir_modules(bundle, backend=backend)
        expected = (
            tensor.tensor_type(BF16, (3,))
            if backend is None
            else DialectType("fixture", "tensor", ("beta", BF16, 3))
        )
        for module in modules.values():
            for function in module.functions.values():
                assert function.params[0][1] == expected
                assert function.results == (expected,)
                assert "ownership" not in function.param_attributes[0]


def test_backend_type_conflict_is_located_at_the_source_function(write):
    from ppy_compiler.backend import BackendValidationError

    bundle = fixture_bundle(write)
    bundle.project.plugins.register(PhysicalPlugin("alpha"))
    with pytest.raises(BackendValidationError, match=r"both lower.*ppy.Tensor.*alpha") as caught:
        canonical_ir_modules(bundle, backend=FixtureBackend("alpha"))
    assert caught.value.location.file.endswith("common.ppy")
    assert caught.value.location.line == 4


def test_unknown_common_tensor_metadata_is_rejected_without_inventing_shape(write):
    from ppy_compiler.backend import BackendValidationError

    bundle = fixture_bundle(
        write,
        """
        import ppy
        def identity(x: ppy.Tensor) -> ppy.Tensor:
            return x
    """,
        physical=(),
    )
    with pytest.raises(BackendValidationError, match="known dtype and shape"):
        canonical_ir_modules(bundle, backend=FixtureBackend("alpha"))


def test_rank_zero_and_symbolic_common_tensor_shapes_are_preserved(write):
    from ppy_compiler.ir import encode

    bundle = fixture_bundle(
        write,
        """
        import ppy
        def scalar(x: ppy.Tensor[ppy.bf16, ()]) -> ppy.Tensor[ppy.bf16, ()]:
            return x
        def symbolic(x: ppy.Tensor[ppy.bf16, ("N", 3)]) -> ppy.Tensor[ppy.bf16, ("N", 3)]:
            return x
    """,
        physical=(),
    )
    module = canonical_ir_modules(bundle)["common"]
    text = encode(module, bundle.project.plugins.dialect_registry())
    assert "tensor.tensor<bf16>" in text
    assert "tensor.tensor<bf16, N, 3>" in text


def test_bfloat16_broadcast_cannot_change_element_format():
    from ppy_compiler.ir import Builder, IRModule, verify
    from ppy_compiler.ir.dialects import core, tensor

    source = tensor.tensor_type(BF16, (4,))
    result = tensor.tensor_type(F16, (2, 4))
    module = IRModule("invalid_broadcast")
    module.require("tensor", 1)
    function = module.add_function("bad", [("x", source)], [result])
    builder = Builder(function.add_entry_block())
    value = builder.create("tensor.broadcast", function.entry.arguments, (result,)).result
    core.ret(builder, value)
    assert any("element" in str(problem) for problem in verify(module))


def test_bfloat16_out_of_range_integer_cast_is_not_folded_as_wrapping_integer():
    from ppy_compiler.driver.ir_pipeline import optimize_shared_ir
    from ppy_compiler.ir import F64, I8, Builder, IRModule, verify
    from ppy_compiler.ir.dialects import core

    for source in (BF16, F64):
        module = IRModule("float_cast")
        function = module.add_function("cast", [], [I8])
        builder = Builder(function.add_entry_block())
        core.ret(builder, core.cast(builder, core.const(builder, 256.0, source), I8))
        assert not verify(module)
        optimize_shared_ir(module, 2, lower_tensors=False)
        assert any(op.name == "core.cast" for op in function.operations())
        assert not any(
            op.name == "core.const" and op.result.type == I8 for op in function.operations()
        )


def test_bfloat16_literals_are_not_folded_using_unrounded_host_values():
    from ppy_compiler.driver.ir_pipeline import optimize_shared_ir
    from ppy_compiler.ir import BOOL, F64, Builder, IRModule, verify
    from ppy_compiler.ir.dialects import core

    module = IRModule("bf16_literals")
    compare = module.add_function("compare", [], [BOOL])
    builder = Builder(compare.add_entry_block())
    left = core.const(builder, 257.0, BF16)
    right = core.const(builder, 256.0, BF16)
    core.ret(builder, core.cmp(builder, "eq", left, right))
    widen = module.add_function("widen", [], [F64])
    builder = Builder(widen.add_entry_block())
    core.ret(builder, core.cast(builder, core.const(builder, 257.0, BF16), F64))
    optimize_shared_ir(module, 2, lower_tensors=False)
    assert any(op.name == "core.cmp" for op in compare.operations())
    assert any(op.name == "core.cast" for op in widen.operations())
    assert not verify(module)


@pytest.mark.parametrize("common_tensor", [False, True])
def test_legacy_tensor_lowering_depends_on_common_tensor_module_opt_in(write, common_tensor):
    from ppy_compiler.driver.ir_pipeline import prepare_for_backend
    from ppy_compiler.driver.pipeline import backend_context
    from ppy_compiler.ir import F64, BufferType, Builder, Pass
    from ppy_compiler.ir.dialects import core, tensor

    class InsertTensor(Pass):
        name = "insert-legacy-tensor"

        def run(self, module, ctx):
            module.require("tensor", 1)
            function = module.add_function("legacy_fill", [("out", BufferType(F64))], [])
            builder = Builder(function.add_entry_block())
            value = tensor.fill(
                builder, core.const(builder, 2.0, F64), tensor.tensor_type(F64, (4,))
            )
            tensor.store(builder, value, function.entry.arguments[0])
            core.ret(builder)
            return True

    class LegacyPlugin(Plugin):
        name = "legacy_tensor"

        def register_passes(self, manager):
            manager.register_stage_pass("after-ir-generation", InsertTensor)

    class LegacyBackend(FixtureBackend):
        def validate(self, module, context):
            has_tensor_ops = any(
                op.dialect == "tensor"
                for function in module.functions.values()
                for op in function.operations()
            )
            assert has_tensor_ops == common_tensor

    source = (
        "import ppy\ndef identity(x: ppy.Tensor[ppy.bf16, (4,)]) -> ppy.Tensor[ppy.bf16, (4,)]:\n    return x\n"
        if common_tensor
        else "def identity(x: int) -> int:\n    return x\n"
    )
    bundle = fixture_bundle(write, source, physical=())
    bundle.project.plugins.register(LegacyPlugin())
    backend = LegacyBackend("legacy")
    module = canonical_ir_modules(bundle, backend=backend)["common"]
    assert bool(module.attributes.get("ppy.common_tensor")) == common_tensor
    prepare_for_backend(module, backend, backend_context(bundle, backend))


@pytest.mark.parametrize("cross_module", [False, True])
def test_concrete_tensor_generalizes_to_symbolic_helper_shape(write, cross_module):
    from ppy_compiler.ir import encode, verify

    helper = """
        import ppy
        def takes(x: ppy.Tensor[ppy.bf16, ("N", 4)]) -> None:
            pass
    """
    source = """
        import ppy
        def caller(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> None:
            takes(x)
        def generalize(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> ppy.Tensor[ppy.bf16, ("N", 4)]:
            return x
    """
    if cross_module:
        write("helper.ppy", helper)
        source = source.replace("import ppy", "import ppy\n        from helper import takes")
    else:
        source = helper + source
    bundle = fixture_bundle(write, source, physical=())
    for backend in (FixtureBackend("logical"), None):
        modules = canonical_ir_modules(bundle, backend=backend)
        text = "\n".join(encode(module) for module in modules.values())
        assert "common_caller" in text
        assert "common_generalize" in text
        if cross_module:  # The empty local helper is removed by shared optimization.
            assert "tensor.shape_cast" in text
        assert "tensor.tensor<bf16, 8, 4>" in text
        assert "tensor.tensor<bf16, N, 4>" in text
        assert all(not verify(module) for module in modules.values())


@pytest.mark.parametrize(
    "source_shape,target_shape,source_dtype,target_dtype,accepted",
    [
        ((8, 4), ("N", 4), BF16, BF16, True),
        ((8, 8), ("N", "N"), BF16, BF16, True),
        ((8, 4), ("N", "N"), BF16, BF16, False),
        ((8, 4), ("N", 5), BF16, BF16, False),
        (("N", 4), (8, 4), BF16, BF16, False),
        ((8, 4), ("N",), BF16, BF16, False),
        ((8, 4), ("N", 4), BF16, F16, False),
    ],
)
def test_tensor_shape_cast_preserves_constraints(
    source_shape, target_shape, source_dtype, target_dtype, accepted
):
    from ppy_compiler.ir import Builder, IRModule, verify
    from ppy_compiler.ir.dialects import core, tensor

    module = IRModule("shape_cast")
    module.require("tensor", 1)
    source = tensor.tensor_type(source_dtype, source_shape)
    target = tensor.tensor_type(target_dtype, target_shape)
    function = module.add_function("generalize", [("x", source)], [target])
    b = Builder(function.add_entry_block())
    value = b.create("tensor.shape_cast", (function.entry.arguments[0],), (target,)).result
    core.ret(b, value)
    assert (not verify(module)) == accepted


def test_scalar_void_helper_call_is_allowed_only_as_statement(write):
    from ppy_compiler.backend import BackendValidationError

    helper = """
        def takes(x: int) -> None:
            pass
        def caller(x: int) -> None:
            takes(x)
    """
    bundle = fixture_bundle(write, helper, physical=())
    for backend in (None, FixtureBackend("logical")):
        modules = canonical_ir_modules(bundle, backend=backend)
        assert "common_caller" in modules["common"].functions
    bundle = fixture_bundle(
        write, helper.replace("            takes(x)", "            return takes(x)"), physical=()
    )
    with pytest.raises(BackendValidationError, match=r"must return a value|returns nothing"):
        canonical_ir_modules(bundle, backend=FixtureBackend("logical"))


@pytest.mark.parametrize("target_shape", ['("N", 5)', '("N", "N")', "(8, 5)"])
def test_driver_rejects_incompatible_tensor_shape_generalization(write, target_shape):
    from ppy_compiler.backend import BackendValidationError

    source = f"""
        import ppy
        def takes(x: ppy.Tensor[ppy.bf16, {target_shape}]) -> None:
            pass
        def caller(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> None:
            takes(x)
    """
    bundle = fixture_bundle(write, source, physical=())
    assert "common_caller" not in canonical_ir_modules(bundle)["common"].functions
    with pytest.raises(BackendValidationError, match=r"expected.*tensor.tensor"):
        canonical_ir_modules(bundle, backend=FixtureBackend("logical"))


def test_legacy_tensor_lowering_rejects_shape_abstraction_explicitly():
    from ppy_compiler.ir import Builder, IRModule, PassContext
    from ppy_compiler.ir.dialects import core, tensor
    from ppy_compiler.ir.transforms import LoweringError, LowerTensor

    module = IRModule("shape_cast_memory")
    module.require("tensor", 1)
    function = module.add_function("generalize", [], [])
    b = Builder(function.add_entry_block())
    value = tensor.fill(b, core.const(b, 1.0, F16), tensor.tensor_type(F16, (8, 4)))
    b.create("tensor.shape_cast", (value,), (tensor.tensor_type(F16, ("N", 4)),))
    core.ret(b)
    with pytest.raises(LoweringError, match="shape_cast requires a backend"):
        LowerTensor().run(module, PassContext())
