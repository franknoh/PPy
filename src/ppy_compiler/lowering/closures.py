"""Nested functions, lambdas, and functions as values, lowered natively.

A function value is a handle to a closure: a one-record sequence whose first
word is the address of its native entry and whose other words are cells, the
variables it shares with the function that made it. A cell is a one-word
sequence of its own. The function that owns a shared variable reads and
writes it through the cell, and so does every closure made over it, which is
CPython's late binding and `nonlocal` both: a closure sees the variable as it
is when the closure runs.

A closure's entry takes the closure as its first parameter and then the
arguments a call through the value passes, each as its type says (a scalar,
or a handle). A call through a value loads the entry's address from the
closure's first word and calls it indirectly with the usual status and result
slots, so a closure can raise, fall back, or return a collection like any
native function.

A top-level function used as a value is wrapped in an entry that calls it,
with no cells. The collector sees a closure's cells, and a cell's handle when
it holds one, so a closure that reaches itself is freed with its cycle.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from types import MappingProxyType

from ..analysis import types as T
from ..analysis.closures import (
    Scope,
    free_names,
    is_plain_callable,
    shared_with_closures,
)
from ..backend.llvm.lowering import Unsupported, _scalar_name
from ..ir import I64, PtrType, Successor, Value
from ..ir.dialects import core
from .collections import HANDLE, Held, Kind, Shape, shape_of

__all__ = ["ClosureLowering"]


def _plain(typed: T.Callable_) -> T.Callable_:
    return T.Callable_(
        tuple(T.Param(f"arg{i}", p.type) for i, p in enumerate(typed.params)), typed.ret
    )


class ClosureLowering:  # pylint: disable=attribute-defined-outside-init
    """The closure half of lowering one function; mixed into `_FunctionLowering`."""

    #: A closure's entry: the names its cells stand for, in the closure's
    #: word order, and their types.
    captures: Mapping[str, T.Type] = MappingProxyType({})
    _closure_env: Value | None = None

    # -- cells --------------------------------------------------------------------

    def _closure_setup(self) -> None:
        """No shared variables yet: `_setup_closure` finds them once the
        parameters are bound."""
        #: Shared variables by name: the cell's handle, the variable's type,
        #: and the pointer its reads and writes go through.
        self._cells: dict[str, Value] = {}
        self._cell_types: dict[str, T.Type] = {}
        #: The cells this function made, let go when it returns.
        self._owned_cells: list[Value] = []
        self._cell_slots: dict[str, Value] = {}

    def _setup_closure(self, node: ast.FunctionDef) -> None:
        """Point every shared variable at its cell: the ones this closure was
        made with, and the ones it shares with closures it makes."""
        if self._closure_env is not None and self.captures:
            record = core.cast(self.b, self._field_address(self._closure_env, 0), PtrType(I64))  # type: ignore[attr-defined]
            for index, (name, typed) in enumerate(self.captures.items()):
                address = core.ptr_offset(self.b, record, self._word(index + 1))  # type: ignore[attr-defined]
                cell = core.load(self.b, core.cast(self.b, address, PtrType(HANDLE)))
                self._place(name, cell, typed)
        shared = shared_with_closures(node) - set(self.captures)
        for name in sorted(shared):
            typed = self._local_type(name)
            if typed is None:
                continue
            cell = self._new_cell(typed)
            self._owned_cells.append(cell)
            self._move_into(name, cell, typed)

    def _local_type(self, name: str) -> T.Type | None:
        analysis = self.frontend.analysis.functions.get(self.info.qualname)  # type: ignore[attr-defined]
        if analysis is not None and name in analysis.locals:
            return analysis.locals[name]
        for parameter in self.info.params:  # type: ignore[attr-defined]
            if parameter.name == name:
                return parameter.type
        return None

    def _cell_shape(self, typed: T.Type) -> Shape:
        shape = shape_of(typed, self._records())  # type: ignore[attr-defined]
        if shape is None or not (shape.reference or shape.kind in {"int", "float", "bool"}):
            raise Unsupported(f"a closure shares a `{typed}` variable, which has no native cell")
        return shape

    def _new_cell(self, typed: T.Type) -> Value:
        shape = self._cell_shape(typed)
        self._use_collections()  # type: ignore[attr-defined]
        return self._rt(  # type: ignore[attr-defined]
            "ppy_seq_new",
            (
                self._word(1),  # type: ignore[attr-defined]
                self._word(1),  # type: ignore[attr-defined]
                self._word(shape.floats),  # type: ignore[attr-defined]
                self._word(shape.handles | (shape.leaves << 32)),  # type: ignore[attr-defined]
            ),
            HANDLE,
        )

    def _place(self, name: str, cell: Value, typed: T.Type) -> None:
        """Read and write `name` through `cell` from here on."""
        shape = self._cell_shape(typed)
        address = self._field_address(cell, 0)  # type: ignore[attr-defined]
        self._cells[name] = cell
        self._cell_types[name] = typed
        if shape.reference:
            kind = self._reference_of_type(typed)  # type: ignore[attr-defined]
            assert kind is not None
            pointer = core.cast(self.b, address, PtrType(HANDLE))
            self.collections[name] = Held(kind, pointer)  # type: ignore[attr-defined]
            self.slots.pop(name, None)  # type: ignore[attr-defined]
        else:
            scalar = _scalar_name(typed)
            assert scalar is not None
            from .ast_to_ir import _scalar_type  # pylint: disable=import-outside-toplevel

            pointer = core.cast(self.b, address, PtrType(_scalar_type(scalar)))
            self.slots[name] = pointer  # type: ignore[attr-defined]
        self._cell_slots[name] = pointer

    def _check_cells(self) -> None:
        """Every shared variable was read and written through its cell: a
        lowering that gave one a slot of its own (a `for` loop's counter)
        would have split it from the closures that share it."""
        for name, pointer in self._cell_slots.items():
            held = self.collections.get(name)  # type: ignore[attr-defined]
            if self.slots.get(name) is not pointer and (held is None or held.slot is not pointer):  # type: ignore[attr-defined]
                raise Unsupported(
                    f"`{name}` is shared with a closure and used where native code "
                    "keeps a variable of its own"
                )

    def _move_into(self, name: str, cell: Value, typed: T.Type) -> None:
        """A variable becomes `cell`'s: a parameter's value moves in, and
        anything else starts empty, as a new cell is."""
        if name in self.buffers:  # type: ignore[attr-defined]
            # A lent buffer has no handle to move in; `written_params` holds a
            # parameter a closure shares by handle, so this is a lowering bug.
            raise Unsupported(f"`{name}` is a buffer a closure shares")
        held = self.collections.get(name)  # type: ignore[attr-defined]
        slot = self.slots.get(name)  # type: ignore[attr-defined]
        self._place(name, cell, typed)
        if held is not None:
            # The reference the parameter took is the cell's now.
            core.store(self.b, core.load(self.b, held.slot), self.collections[name].slot)  # type: ignore[attr-defined]
        elif slot is not None:
            target = self.slots[name]  # type: ignore[attr-defined]
            value = core.load(self.b, slot)
            core.store(self.b, self._coerce_type(value, target.type.pointee), target)  # type: ignore[attr-defined]
        elif self._cell_shape(typed).reference:
            none = self._rt("ppy_coll_none", (), HANDLE)  # type: ignore[attr-defined]
            core.store(self.b, none, self.collections[name].slot)  # type: ignore[attr-defined]

    def _declare_nonlocal(self, node: ast.Nonlocal) -> None:
        """`nonlocal total`: the name is already the enclosing function's cell."""
        for name in node.names:
            if not self._is_cell(name):
                raise Unsupported(f"`nonlocal {name}` names no variable a closure shares natively")

    def _is_cell(self, name: str) -> bool:
        return name in self._cells

    def _release_collections(self) -> None:
        cells = self._cells
        kept = {
            name: self.collections.pop(name) for name in list(self.collections) if name in cells
        }  # type: ignore[attr-defined]
        super()._release_collections()  # type: ignore[misc]
        self.collections.update(kept)  # type: ignore[attr-defined]
        for cell in self._owned_cells:
            self._release(cell)  # type: ignore[attr-defined]

    # -- making closures -------------------------------------------------------------

    def _captured(self, node: Scope) -> dict[str, T.Type]:
        return {
            name: self._cell_types[name] for name in sorted(free_names(node)) if name in self._cells
        }

    def _closure(self, entry: str, captured: dict[str, T.Type]) -> Value:
        """A new closure over `entry` and the cells of `captured`: an owned handle."""
        self._use_collections()  # type: ignore[attr-defined]
        count = len(captured)
        mask = sum(1 << (index + 1) for index in range(count))
        made = self._rt(  # type: ignore[attr-defined]
            "ppy_seq_new",
            (self._word(1), self._word(1 + count), self._word(0), self._word(mask)),  # type: ignore[attr-defined]
            HANDLE,
        )
        record = core.cast(self.b, self._field_address(made, 0), PtrType(I64))  # type: ignore[attr-defined]
        core.store(self.b, core.function_address(self.b, entry), record)
        for index, name in enumerate(captured):
            cell = self._cells[name]
            self._retain(cell)  # type: ignore[attr-defined]
            address = core.ptr_offset(self.b, record, self._word(index + 1))  # type: ignore[attr-defined]
            core.store(self.b, cell, core.cast(self.b, address, PtrType(HANDLE)))
        return made

    def _define_closure(self, node: ast.FunctionDef) -> None:
        """`def inner(...)` inside a function: `inner` is bound to a new closure."""
        args = node.args
        if (
            node.decorator_list
            or args.vararg
            or args.kwarg
            or args.kwonlyargs
            or args.defaults
            or args.posonlyargs
        ):
            raise Unsupported(
                f"a nested `{node.name}` with decorators, defaults, or special "
                "parameters has no native lowering"
            )
        qualname = f"{self.info.qualname}.<locals>.{node.name}"  # type: ignore[attr-defined]
        info = self.frontend.analysis.symbols.nested.get(qualname)  # type: ignore[attr-defined]
        if info is None or info.node is not node:
            raise Unsupported(f"`{node.name}` is a nested function the checker did not see")
        typed = T.Callable_(tuple(T.Param(p.name, p.type) for p in info.params), info.ret)
        if not is_plain_callable(typed):
            raise Unsupported(f"`{node.name}` is not a function a closure can hold")
        captured = self._captured(node)
        function, _signature = self.frontend.closure_code(info, node, captured)  # type: ignore[attr-defined]
        shape = shape_of(_plain(typed), self._records())  # type: ignore[attr-defined]
        assert shape is not None
        self._bind(node.name, shape, self._closure(function.name, captured), owned=True)  # type: ignore[attr-defined]

    def _function_value(self, node: ast.expr) -> Value | None:
        """A lambda, or a top-level function named as a value: a new closure,
        owned. None for anything else."""
        if isinstance(node, ast.Lambda):
            typed = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
            if not isinstance(typed, T.Callable_) or not is_plain_callable(typed):
                raise Unsupported("a lambda whose type native code does not know")
            args = node.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.posonlyargs:
                raise Unsupported("a lambda with defaults or special parameters")
            info, wrapper = self.frontend.lambda_code(node, typed, self.info)  # type: ignore[attr-defined]
            captured = self._captured(wrapper)
            function, _signature = self.frontend.closure_code(info, wrapper, captured)  # type: ignore[attr-defined]
            return self._closure(function.name, captured)
        qualname = self._named_function(node)
        if qualname is None:
            return None
        function, _signature = self.frontend.adapter_code(qualname)  # type: ignore[attr-defined]
        return self._closure(function.name, {})

    def _named_function(self, node: ast.expr) -> str | None:
        """The module function a name used as a value stands for, if it is one."""
        if not isinstance(node, ast.Name) or node.id in self.collections:  # type: ignore[attr-defined]
            return None
        if node.id in self.slots or self._is_cell(node.id):  # type: ignore[attr-defined]
            return None
        typed = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if not isinstance(typed, T.Callable_) or not typed.qualname:
            return None
        info = self.frontend.analysis.symbols.functions.get(node.id)  # type: ignore[attr-defined]
        if info is None or info.qualname != typed.qualname or info.enclosing or info.owner:
            return None
        return info.qualname

    def _function_shape(self, node: ast.expr) -> Shape | None:
        """The function value an expression denotes, as native code holds it."""
        if isinstance(node, ast.Name) and node.id in self.collections:  # type: ignore[attr-defined]
            held = self.collections[node.id].kind  # type: ignore[attr-defined]
            return held if isinstance(held, Shape) and held.kind == "function" else None
        if isinstance(node, ast.Name) and self._named_function(node) is None:
            return None
        typed = T.strip_literal(self._type_of(node))  # type: ignore[attr-defined]
        if not isinstance(typed, T.Callable_):
            return None
        if isinstance(node, ast.Attribute):
            # A field holding a function, not a method bound to its object.
            owner = self._object_of(node.value)  # type: ignore[attr-defined]
            if owner is None or node.attr not in self._layout(owner).fields:  # type: ignore[attr-defined]
                return None
        found = shape_of(typed, self._records())  # type: ignore[attr-defined]
        return found if found is not None and found.kind == "function" else None

    # -- hooks into the collection half ----------------------------------------------------

    def _reference_of(self, node: ast.expr) -> Kind | Shape | None:
        found = super()._reference_of(node)  # type: ignore[misc]
        if found is not None:
            return found
        return self._function_shape(node)

    def _handle(self, node: ast.expr) -> tuple[Value, bool]:
        if isinstance(node, (ast.Lambda, ast.Name)):
            made = self._function_value(node)
            if made is not None:
                return made, True
        return super()._handle(node)  # type: ignore[misc]

    # -- calling through a value -------------------------------------------------------

    def _calls_value(self, node: ast.Call) -> bool:
        """Whether `node` calls a function value rather than a function by name."""
        func = node.func
        if isinstance(func, ast.Name):
            held = self.collections.get(func.id)  # type: ignore[attr-defined]
            return (
                held is not None and isinstance(held.kind, Shape) and held.kind.kind == "function"
            )
        if isinstance(func, ast.Lambda):
            return True
        if isinstance(func, ast.Call) and self._derivative_spec(func) is not None:  # type: ignore[attr-defined]
            # `ppy.grad(f)(x)` is a derivative, lowered where it is called.
            return False
        if isinstance(func, (ast.Call, ast.Subscript, ast.Attribute)):
            return self._function_shape(func) is not None
        return False

    def _value_call(self, node: ast.Call, *, discard_result: bool = False) -> Value:
        """`f(x)` where `f` is a function value: an indirect call of its entry."""
        if node.keywords:
            raise Unsupported("a function value takes positional arguments natively")
        shape = self._reference_of(node.func)
        if not isinstance(shape, Shape) or shape.kind != "function":
            raise Unsupported(f"`{ast.unparse(node.func)}` is not a function value")
        typed = shape.class_args[0]
        assert isinstance(typed, T.Callable_)
        signature = self.frontend.callable_signature(typed)  # type: ignore[attr-defined]
        closure, owned = self._handle(node.func)
        waiting = len(self._temporaries)  # type: ignore[attr-defined]
        arguments = self._call_arguments(signature, node.args, ast.unparse(node.func))  # type: ignore[attr-defined]
        temporaries = self._temporaries[waiting:]  # type: ignore[attr-defined]
        del self._temporaries[waiting:]  # type: ignore[attr-defined]
        record = core.cast(self.b, self._field_address(closure, 0), PtrType(I64))  # type: ignore[attr-defined]
        code = core.load(self.b, record)
        values = self._call_indirect(code, (closure, *arguments), signature.results)
        for handle in temporaries:
            self._release(handle)  # type: ignore[attr-defined]
        if owned:
            self._release(closure)  # type: ignore[attr-defined]
        if not values:
            if not discard_result:
                raise Unsupported(f"`{ast.unparse(node.func)}` returns nothing a caller can use")
            return core.const(self.b, 0, I64)
        result = values[0]
        if discard_result and result.type == HANDLE:
            self._release(result)  # type: ignore[attr-defined]
        return result

    def _call_indirect(
        self, code: Value, arguments: tuple[Value, ...], results: tuple
    ) -> tuple[Value, ...]:
        """The indirect twin of `_call_native`."""
        if not self._exceptions_on():  # type: ignore[attr-defined]
            return tuple(core.call_indirect(self.b, code, arguments, results).results)
        made = core.call_indirect(self.b, code, arguments, results, capture_status=True)
        *values, status = made.results
        answered = self._block("call.ok")  # type: ignore[attr-defined]
        failed = self._block("call.failed")  # type: ignore[attr-defined]
        answers = core.cmp(self.b, "eq", status, self._word(0))  # type: ignore[attr-defined]
        core.cond_br(self.b, answers, Successor(answered), Successor(failed))
        self.b.at_end(failed)
        raised = core.cmp(self.b, "eq", status, self._word(-1))  # type: ignore[attr-defined]
        core.guard(self.b, raised, "contract", "a call fell back")
        self._go_raise()  # type: ignore[attr-defined]
        self.b.at_end(answered)
        return tuple(values)

    # -- map and filter ----------------------------------------------------------------

    def _map_filter(self, node: ast.expr) -> tuple[str, ast.expr, ast.expr] | None:
        """`map(f, xs)` or `filter(f, xs)` with the builtin: which, `f`, and `xs`."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"map", "filter"}
            and len(node.args) == 2
            and not node.keywords
        ):
            return None
        name = node.func.id
        if name in self.collections or name in self.slots or self._is_cell(name):  # type: ignore[attr-defined]
            return None
        if not isinstance(T.strip_literal(self._type_of(node)), T.Instance):  # type: ignore[attr-defined]
            return None
        return name, node.args[0], node.args[1]

    def _is_generator(self, node: ast.expr) -> bool:
        return self._map_filter(node) is not None or super()._is_generator(node)  # type: ignore[misc]

    def _inline(self, node: ast.expr, *args: object, **kwargs: object) -> None:
        """`map` and `filter` are generator expressions over their iterable:
        `map(f, xs)` is `(f(x) for x in xs)` with `f` taken once, first."""
        found = self._map_filter(node)
        if found is None:
            super()._inline(node, *args, **kwargs)  # type: ignore[misc]
            return
        which, function, iterable = found
        self._walks_made = getattr(self, "_walks_made", 0) + 1
        item = ast.Name(id=f".each{self._walks_made}", ctx=ast.Load())
        if isinstance(function, ast.Lambda):
            if len(function.args.args) != 1 or function.args.vararg or function.args.defaults:
                raise Unsupported(f"`{which}` calls a function of one argument natively")
            item = ast.Name(id=function.args.args[0].arg, ctx=ast.Load())
            applied = function.body
        elif which == "filter" and isinstance(function, ast.Constant) and function.value is None:
            applied = item
        else:
            shape = self._reference_of(function)
            if not isinstance(shape, Shape) or shape.kind != "function":
                raise Unsupported(f"`{ast.unparse(function)}` is not a function native code holds")
            held = f".fn{self._walks_made}"
            handle, owned = self._handle(function)
            self._bind(held, shape, handle, owned)  # type: ignore[attr-defined]
            applied = ast.Call(ast.Name(id=held, ctx=ast.Load()), [item], [])
            typed = shape.class_args[0]
            assert isinstance(typed, T.Callable_)
            self.frontend.analysis.node_types[id(applied)] = typed.ret  # type: ignore[attr-defined]
        target = ast.Name(id=item.id, ctx=ast.Store())
        if which == "map":
            made = ast.GeneratorExp(applied, [ast.comprehension(target, iterable, [], 0)])
        else:
            element = ast.Name(id=item.id, ctx=ast.Load())
            made = ast.GeneratorExp(element, [ast.comprehension(target, iterable, [applied], 0)])
        ast.copy_location(made, node)
        ast.fix_missing_locations(made)
        types = self.frontend.analysis.node_types  # type: ignore[attr-defined]
        types[id(made)] = self._type_of(node)  # type: ignore[attr-defined]
        # Types are kept by node identity: the nodes live as long as the types.
        self.frontend.synthetic.append(made)  # type: ignore[attr-defined]
        super()._inline(made, *args, **kwargs)  # type: ignore[misc]
