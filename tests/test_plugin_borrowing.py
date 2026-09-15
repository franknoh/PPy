"""Plugin call ownership contracts and backend-aware type lowering."""

from __future__ import annotations

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.driver.pipeline import analyze_paths, open_project
from ppy_compiler.ir import I32, I64
from ppy_compiler.plugins import (
    ArgumentOwnership,
    CallArgument,
    CallResult,
    Lowering,
    Plugin,
    PluginError,
    PluginRegistry,
    RejectSpec,
)

TENSOR = T.Instance("borrowlib.Tensor", (), ("borrowlib.Tensor", "object"))
COMMON = T.Instance("ppy.Tensor", (), ("ppy.Tensor", "object"))


class BorrowPlugin(Plugin):
    name = "borrowlib"
    modules = ("borrowlib",)

    def external_types(self):
        return {"borrowlib.Tensor": "borrowlib.Tensor"}

    def call(self, qualname, args, keywords):
        if qualname == "borrowlib.read":
            return CallResult(
                T.NONE,
                lowering=Lowering.DIALECT_OPERATION,
                arguments=(CallArgument(0, ArgumentOwnership.BORROWED),),
            )
        if qualname == "borrowlib.write":
            return CallResult(
                T.NONE,
                lowering=Lowering.DIALECT_OPERATION,
                reason="write requires writable storage",
                arguments=(CallArgument(0, ArgumentOwnership.MUT),),
            )
        if qualname == "borrowlib.retain":
            return CallResult(
                T.NONE,
                arguments=(
                    CallArgument("value", ArgumentOwnership.OWNED, "retained beyond the call"),
                ),
            )
        if qualname == "borrowlib.reject_result":
            return CallResult(T.NONE, lowering=Lowering.REJECT, reason="device mode is disabled")
        if qualname == "borrowlib.reject_spec":
            return CallResult(T.NONE, lowering=RejectSpec("layout is unsupported"))
        return None


def _bundle(write, source: str):
    path = write("borrowed.ppy", source)
    project = open_project(path, config_overrides={"strict": False})
    project.plugins.register(BorrowPlugin())
    return analyze_paths(project, [path], backend="llvm")


def _errors(bundle):
    return [d for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"]


def test_explicit_borrow_does_not_escape_and_mut_tracks_the_aliased_parameter(write):
    bundle = _bundle(
        write,
        """
        from typing import Annotated
        import ppy
        import borrowlib
        from borrowlib import Tensor

        Value = Annotated[Tensor, ppy.DType("float32"), ppy.Shape(4)]

        def inspect(x: ppy.Borrowed[Value]) -> None:
            alias = x
            borrowlib.read(alias)

        def update(x: ppy.Mut[Value]) -> None:
            # A broad rebinding keeps x's dtype, shape, and ownership facts.
            alias: Tensor = x
            borrowlib.write(alias)
        """,
    )
    assert not _errors(bundle)
    analysis = bundle.analysis.modules["borrowed"].functions
    assert "x" not in analysis["borrowed.inspect"].escaping
    assert "x" not in analysis["borrowed.update"].escaping
    assert analysis["borrowed.update"].mutated_params == {"x"}


def test_mut_and_owned_contracts_reject_read_only_arguments_with_the_contract_reason(write):
    bundle = _bundle(
        write,
        """
        import ppy
        import borrowlib
        from borrowlib import Tensor

        def bad_write(x: ppy.Borrowed[Tensor]) -> None:
            borrowlib.write(x)

        def bad_retain(x: ppy.Borrowed[Tensor]) -> None:
            borrowlib.retain(value=x)
        """,
    )
    messages = [d.message for d in _errors(bundle)]
    assert any("write requires writable storage" in message for message in messages)
    assert any("retained beyond the call" in message for message in messages)


def test_unknown_plugin_arguments_keep_the_conservative_retained_escape(write):
    bundle = _bundle(
        write,
        """
        import ppy
        import borrowlib
        from borrowlib import Tensor

        def unknown_contract(x: ppy.Owned[Tensor]) -> None:
            borrowlib.unknown(x)
        """,
    )
    analysis = bundle.analysis.modules["borrowed"].functions["borrowed.unknown_contract"]
    assert "x" in analysis.escaping


def test_starred_plugin_arguments_stay_conservative_when_positions_are_unknown(write):
    bundle = _bundle(
        write,
        """
        import ppy
        import borrowlib
        from borrowlib import Tensor

        def expanded(values: ppy.Borrowed[list[Tensor]]) -> None:
            borrowlib.read(*values)
        """,
    )
    errors = _errors(bundle)
    assert [error.code for error in errors] == ["E1612"]
    analysis = bundle.analysis.modules["borrowed"].functions["borrowed.expanded"]
    assert "values" in analysis.escaping


def test_reject_diagnostics_include_call_result_and_reject_spec_reasons(write):
    bundle = _bundle(
        write,
        """
        import borrowlib

        def rejected() -> None:
            borrowlib.reject_result()
            borrowlib.reject_spec()
        """,
    )
    errors = _errors(bundle)
    assert [error.code for error in errors] == ["E1802", "E1802"]
    assert errors[0].span is not None and errors[0].span.line == 4
    assert "device mode is disabled" in errors[0].message
    assert "layout is unsupported" in errors[1].message


def test_tensor_facts_are_checked_at_rebinding_return_and_local_call_boundaries(write):
    bundle = _bundle(
        write,
        """
        from typing import Annotated
        import ppy
        from borrowlib import Tensor

        F32x4 = Annotated[Tensor, ppy.DType("float32"), ppy.Shape(4)]
        F16x8 = Annotated[Tensor, ppy.DType("float16"), ppy.Shape(8)]

        def takes(value: F32x4) -> None:
            return

        def bad_assign(value: F16x8) -> None:
            local: F32x4 = value

        def bad_return(value: F16x8) -> F32x4:
            return value

        def bad_call(value: F16x8) -> None:
            takes(value)
        """,
    )
    messages = [d.message for d in _errors(bundle)]
    assert sum("dtype" in message or "shape" in message for message in messages) == 3


def test_inferred_local_and_generic_returns_keep_tensor_facts(write):
    bundle = _bundle(
        write,
        """
        import ppy

        def identity(x: ppy.Tensor[ppy.bf16, (8,)]):
            return x

        def generic_identity[T](x: ppy.Tensor[ppy.bf16, (8,)]):
            return x

        def bad_local(x: ppy.Tensor[ppy.bf16, (8,)]) -> ppy.Tensor[ppy.bf16, (4,)]:
            return identity(x)

        def bad_generic(x: ppy.Tensor[ppy.bf16, (8,)]) -> ppy.Tensor[ppy.bf16, (4,)]:
            return generic_identity(x)
        """,
    )
    errors = _errors(bundle)
    assert [error.code for error in errors] == ["E1303", "E1303"]
    assert all("shape" in error.message for error in errors)


def test_symbolic_shapes_unify_and_repeated_symbols_remain_consistent(write):
    bundle = _bundle(
        write,
        """
        import ppy

        def symbolic(x: ppy.Tensor[ppy.bf16, ("N", 4)]) -> None:
            return

        def repeated(x: ppy.Tensor[ppy.bf16, ("N", "N")]) -> None:
            return

        def concrete(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> None:
            return

        def caller(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> None:
            symbolic(x)
            alias: ppy.Tensor[ppy.bf16, ("N", 4)] = x
            concrete(alias)

        def bad_concrete(x: ppy.Tensor[ppy.bf16, (8, 5)]) -> None:
            symbolic(x)

        def bad_repeated(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> None:
            repeated(x)

        def fixed(x: ppy.Tensor[ppy.bf16, (8, 4)]) -> None:
            return

        def bad_incoming_symbols(x: ppy.Tensor[ppy.bf16, ("N", "N")]) -> None:
            fixed(x)
        """,
    )
    errors = _errors(bundle)
    assert [error.code for error in errors] == ["E1301", "E1301", "E1301"]
    assert all("shape" in error.message for error in errors)


def test_subsequent_assignment_honors_the_original_tensor_declaration(write):
    bundle = _bundle(
        write,
        """
        import ppy

        def replace(
            x: ppy.Tensor[ppy.bf16, (4,)],
            y: ppy.Tensor[ppy.bf16, (8,)],
        ) -> None:
            local: ppy.Tensor[ppy.bf16, (4,)] = x
            local = y
        """,
    )
    errors = _errors(bundle)
    assert [error.code for error in errors] == ["E1301"]
    assert "shape" in errors[0].message


def test_subsequent_assignment_keeps_contextual_fresh_display_compatibility(write):
    bundle = _bundle(
        write,
        """
        class Animal:
            pass

        class Dog(Animal):
            pass

        def replace(x: Dog) -> None:
            items: list[Animal] = [x]
            items = [x]
        """,
    )
    assert not _errors(bundle)


def test_known_mut_helper_is_call_scoped_and_propagates_plugin_writes(write):
    bundle = _bundle(
        write,
        """
        import ppy
        import borrowlib

        def helper(x: ppy.Mut[ppy.Tensor[ppy.bf16, (4,)]]) -> None:
            borrowlib.write(x)

        def caller(x: ppy.Mut[ppy.Tensor[ppy.bf16, (4,)]]) -> None:
            helper(x)
        """,
    )
    assert not _errors(bundle)
    functions = bundle.analysis.modules["borrowed"].functions
    helper = functions["borrowed.helper"]
    caller = functions["borrowed.caller"]
    assert helper.escaping == set() and caller.escaping == set()
    assert helper.mutated_params == {"x"}
    assert caller.delegated_writes == {"x"}
    assert Effect.WRITE_MEMORY in helper.effects
    assert Effect.WRITE_MEMORY in caller.effects


def test_helper_write_propagation_excludes_its_borrowed_inputs(write):
    bundle = _bundle(
        write,
        """
        import ppy
        import borrowlib

        def helper(
            src: ppy.Borrowed[ppy.Tensor],
            dst: ppy.Mut[ppy.Tensor],
        ) -> None:
            borrowlib.read(src)
            borrowlib.write(dst)

        def caller(
            src: ppy.Borrowed[ppy.Tensor],
            dst: ppy.Mut[ppy.Tensor],
        ) -> None:
            helper(src=src, dst=dst)
        """,
    )
    assert not _errors(bundle)
    functions = bundle.analysis.modules["borrowed"].functions
    assert functions["borrowed.helper"].mutated_params == {"dst"}
    assert functions["borrowed.caller"].delegated_writes == {"dst"}


def test_expanded_container_arguments_are_evaluated_and_escape_borrowed_roots(write):
    bundle = _bundle(
        write,
        """
        import ppy
        import borrowlib

        def keyword(x: ppy.Borrowed[ppy.Tensor[ppy.bf16, (4,)]]) -> None:
            borrowlib.read(**{"value": x})

        def positional(x: ppy.Borrowed[ppy.Tensor[ppy.bf16, (4,)]]) -> None:
            borrowlib.read(*[x])
        """,
    )
    errors = _errors(bundle)
    assert [error.code for error in errors] == ["E1612", "E1612"]
    functions = bundle.analysis.modules["borrowed"].functions
    assert functions["borrowed.keyword"].escaping == {"x"}
    assert functions["borrowed.positional"].escaping == {"x"}


def test_backend_type_claims_are_selected_explicitly_and_conflicts_are_rejected():
    class Legacy(Plugin):
        name = "legacy"

        def lower_type(self, type_, facts):
            return I64 if type_ in {TENSOR, COMMON} else None

    class Device(Plugin):
        name = "device"

        def lower_type_for_backend(self, type_, facts, backend):
            return I32 if type_ == COMMON and backend == "device" else None

    registry = PluginRegistry()
    registry.register(Legacy())
    registry.register(Device())
    assert registry.lower_type(TENSOR, Facts()) == I64
    assert registry.lower_type(COMMON, Facts()) is None
    assert registry.lower_type(COMMON, Facts(), backend="device") == I32

    class Rival(Device):
        name = "rival"

    registry.register(Rival())
    try:
        registry.lower_type(COMMON, Facts(), backend="device")
    except PluginError as error:
        assert "device" in str(error) and "rival" in str(error)
    else:
        raise AssertionError("competing backend type claims were accepted")


def test_backend_type_provider_fingerprint_applies_without_a_claimed_import():
    class Provider(Plugin):
        name = "device"

        def __init__(self, version: str):
            super().__init__()
            self.version = version

        def fingerprint(self):
            return self.version

        def lower_type_for_backend(self, type_, facts, backend):
            return I32 if type_ == COMMON and backend == "device" else None

    first = PluginRegistry()
    first.register(Provider("v1"))
    second = PluginRegistry()
    second.register(Provider("v2"))
    assert first.fingerprints({"ppy"}) == ("device:v1",)
    assert second.fingerprints({"ppy"}) == ("device:v2",)
