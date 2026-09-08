"""The NVVM backend: device code as LLVM IR for NVPTX, and PTX from it (spec 75).

A module's kernels and device functions are emitted the way the LLVM
backend emits host code -- the same operation lowerings, the same phis --
under the NVPTX triple: a kernel is a `ptx_kernel` returning nothing, a
device function returns its value directly (nothing on a device falls
back), a thread's position is an `llvm.nvvm.read.ptx.sreg.*` register,
a barrier is `llvm.nvvm.barrier0`, shared memory is an `addrspace(3)`
array of the module, a shuffle is `llvm.nvvm.shfl.sync.*` (a 64-bit value
as two halves), and the math library is libdevice's `__nv_*`. `ptx_from_ir`
links libdevice, keeps what the kernels reach, and asks LLVM's NVPTX
backend for PTX; a machine without that backend, or without the CUDA
toolkit's libdevice, is told so and keeps its CPU compiler whole.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...ir import BoolType, FloatType, IRModule, Operation, PtrType
from ...ir.dialects.gpu import kind_of
from ...target import host_target
from ..llvm.from_ir import EmitError, _FunctionEmitter, _ModuleEmitter

__all__ = [
    "DEFAULT_ARCH",
    "NvvmError",
    "available",
    "device_functions",
    "emit_module",
    "libdevice_path",
    "ptx_from_ir",
]

TRIPLE = "nvptx64-nvidia-cuda"
DATA_LAYOUT = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
#: The architecture PTX is written for unless `PPY_CUDA_ARCH` says otherwise; a
#: driver compiles PTX for an older architecture forward, so a modest one
#: runs everywhere the runtime does.
DEFAULT_ARCH = "sm_70"
_REGISTERS = {"thread_id": "tid", "block_id": "ctaid", "block_dim": "ntid", "grid_dim": "nctaid"}
_SHUFFLES = {"idx": ("idx", 0x1F), "up": ("up", 0), "down": ("down", 0x1F), "xor": ("bfly", 0x1F)}
#: libdevice's name for each math operation; the f32 form adds `f`.
_LIBDEVICE = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
    "exp2": "exp2",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "sqrt": "sqrt",
    "pow": "pow",
    "abs": "fabs",
    "fabs": "fabs",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "round": "round",
    "fma": "fma",
    "minimum": "fmin",
    "maximum": "fmax",
    "copysign": "copysign",
    "atan2": "atan2",
    "hypot": "hypot",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "expm1": "expm1",
    "log1p": "log1p",
    "erf": "erf",
    "erfc": "erfc",
}


class NvvmError(Exception):
    """Device code the NVVM backend cannot write, or a toolchain it does not have."""


def available() -> bool:
    """Whether this LLVM has the NVPTX backend."""
    try:
        from llvmlite import binding as llvm

        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        llvm.Target.from_triple(TRIPLE)
    except (ImportError, RuntimeError):
        return False
    return True


def libdevice_path() -> Path | None:
    """The CUDA toolkit's libdevice bitcode, by `CUDA_HOME`, `CUDA_PATH`, or the usual place."""
    spelled = os.environ.get("PPY_LIBDEVICE")
    if spelled:
        return Path(spelled) if Path(spelled).is_file() else None
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"]
    for root in roots:
        if root:
            candidate = Path(root) / "nvvm" / "libdevice" / "libdevice.10.bc"
            if candidate.is_file():
                return candidate
    return None


def device_functions(module: IRModule) -> list[str]:
    """The names of `module`'s kernels and device functions."""
    return [
        f.name for f in module.functions.values() if not f.is_declaration and kind_of(f) != "host"
    ]


def emit_module(module: IRModule) -> str:
    """The NVVM IR text of `module`'s device code; nothing when it has none."""
    from llvmlite import ir

    if not device_functions(module):
        return ""
    try:
        return str(_DeviceModuleEmitter(ir, module).run())
    except EmitError as error:
        raise NvvmError(str(error)) from error


def ptx_from_ir(text: str, arch: str | None = None) -> str:
    """PTX for `arch` from NVVM IR text, libdevice linked in and pruned to what is used."""
    from llvmlite import binding as llvm

    if not available():
        raise NvvmError("this LLVM has no NVPTX backend; PTX needs one")
    arch = arch or os.environ.get("PPY_CUDA_ARCH") or DEFAULT_ARCH
    module = llvm.parse_assembly(text)
    module.verify()
    if any(f.name.startswith("__nv_") and f.is_declaration for f in module.functions):
        found = libdevice_path()
        if found is None:
            raise NvvmError(
                "the math library on a device is libdevice, which was not found; "
                "set CUDA_HOME to the CUDA toolkit or PPY_LIBDEVICE to libdevice.10.bc"
            )
        library = llvm.parse_bitcode(found.read_bytes())
        library.triple = module.triple
        library.data_layout = module.data_layout
        for function in library.functions:
            if not function.is_declaration:
                function.linkage = "internal"
        module.link_in(library)
    target = llvm.Target.from_triple(TRIPLE)
    machine = target.create_target_machine(cpu=arch, features="+ptx70", opt=2)
    builder = llvm.PassBuilder(machine, llvm.PipelineTuningOptions())
    passes = builder.getModulePassManager()
    passes.add_global_dead_code_eliminate_pass()
    passes.run(module, builder)
    return machine.emit_assembly(module)


class _DeviceModuleEmitter(_ModuleEmitter):
    """The LLVM module emitter under the NVPTX triple, for device code only."""

    def __init__(self, ir, module: IRModule) -> None:  # type: ignore[no-untyped-def]
        super().__init__(ir, module, host_target())
        self.llvm.triple = TRIPLE
        self.llvm.data_layout = DATA_LAYOUT
        self.shared = 0

    def function_type(self, function):  # type: ignore[no-untyped-def]
        ir = self.ir
        atoms = [atom for _name, t in function.params for atom in self.boundary_atoms(t)]
        if kind_of(function) == "kernel":
            return ir.FunctionType(ir.VoidType(), atoms)
        results = [atom for t in function.results for atom in self.boundary_atoms(t)]
        if len(results) > 1:
            raise EmitError(f"@{function.name}: a device function returns one scalar")
        return ir.FunctionType(results[0] if results else ir.VoidType(), atoms)

    def run(self):  # type: ignore[no-untyped-def]
        ir = self.ir
        device = [
            f
            for f in self.module.functions.values()
            if not f.is_declaration and kind_of(f) != "host"
        ]
        for function in device:
            symbol = str(function.attributes.get("ppy.symbol", function.name))
            declared = ir.Function(self.llvm, self.function_type(function), name=symbol)
            if kind_of(function) == "kernel":
                declared.calling_convention = "ptx_kernel"
            else:
                declared.linkage = "internal"
            self.functions[function.name] = declared
        for function in device:
            _DeviceFunctionEmitter(self, function).run()
        return self.llvm


class _DeviceFunctionEmitter(_FunctionEmitter):
    """A kernel or a device function: no fallback, values returned directly."""

    def run(self) -> None:
        ir = self.ir
        body = self.function.body
        for block in body.blocks:
            self.blocks[id(block)] = self.llvm.append_basic_block(block.name)
        self.entry = self.blocks[id(body.blocks[0])]
        self.builder = ir.IRBuilder(self.entry)
        self._bind_parameters()
        for block in body.blocks[1:]:
            self._make_phis(block)
        for block in body.blocks:
            self.builder.position_at_end(self.blocks[id(block)])
            for op in block.operations:
                self._op(op)
        for phis in self.phis.values():
            for phi in phis:
                for value, block in phi.incoming:
                    phi.node.add_incoming(value, block)

    def _op(self, op: Operation) -> None:
        if op.dialect == "gpu":
            self._gpu(op)
        elif op.dialect == "math":
            self._libdevice(op)
        elif op.name == "core.guard":
            raise EmitError(f"@{self.function.name}: a guard has no device form")
        else:
            super()._op(op)

    def _ret(self, op: Operation) -> None:
        if not op.operands:
            self.builder.ret_void()
            return
        (value,) = op.operands
        (atom,) = self._to_atoms(self.function.results[0], self.value(value))
        self.builder.ret(atom)

    def _call(self, op: Operation) -> None:
        callee_name = op.attributes["callee"].name  # type: ignore[union-attr]
        callee = self.owner.functions.get(callee_name)
        target = self.owner.module.functions.get(callee_name)
        if callee is None or target is None:
            raise EmitError(f"call to @{callee_name}, which was not emitted")
        arguments: list = []
        for operand, (_name, t) in zip(op.operands, target.params, strict=True):
            arguments.extend(self._to_atoms(t, self.value(operand)))
        result = self.builder.call(callee, arguments)
        if op.results:
            self.set(op.results[0], self._from_atoms(target.results[0], [result], 0)[0])

    # -- the gpu dialect ----------------------------------------------------------

    def _gpu(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        i32 = ir.IntType(32)
        name = op.local_name
        if name in _REGISTERS:
            register = f"llvm.nvvm.read.ptx.sreg.{_REGISTERS[name]}.{op.attributes['dim']}"
            read = b.call(self.intrinsic(register, i32, []), [])
            self.set(op.result, b.zext(read, ir.IntType(64)))
        elif name == "barrier":
            b.call(self.intrinsic("llvm.nvvm.barrier0", ir.VoidType(), []), [])
        elif name == "subgroup_barrier":
            b.call(
                self.intrinsic("llvm.nvvm.bar.warp.sync", ir.VoidType(), [i32]),
                [ir.Constant(i32, -1)],
            )
        elif name == "shared_alloc":
            pointer = op.result.type
            assert isinstance(pointer, PtrType)
            element = self.owner.llvm_type(pointer.pointee)
            count = int(op.attributes["count"])  # type: ignore[call-overload]
            self.owner.shared += 1
            array = ir.GlobalVariable(
                self.owner.llvm,
                ir.ArrayType(element, count),
                name=f"{self.llvm.name}_shared{self.owner.shared}",
                addrspace=3,
            )
            array.linkage = "internal"
            array.initializer = ir.Constant(ir.ArrayType(element, count), ir.Undefined)
            array.align = 8
            self.set(op.result, b.addrspacecast(array, element.as_pointer()))
        elif name == "private_alloc":
            pointer = op.result.type
            assert isinstance(pointer, PtrType)
            element = self.owner.llvm_type(pointer.pointee)
            count = int(op.attributes["count"])  # type: ignore[call-overload]
            self.set(op.result, self.entry_alloca(ir.ArrayType(element, count), "private"))
        elif name == "subgroup_shuffle":
            self._shuffle(op)
        else:
            raise EmitError(f"{op.name} is the host's; a kernel does not launch")

    def _shuffle(self, op: Operation) -> None:
        ir = self.ir
        b = self.builder
        i32, i64 = ir.IntType(32), ir.IntType(64)
        value, lane = (self.value(v) for v in op.operands)
        t = op.result.type
        mode, clamp = _SHUFFLES[str(op.attributes["mode"])]
        lane32 = b.trunc(lane, i32)
        mask, c = ir.Constant(i32, -1), ir.Constant(i32, clamp)

        def shuffle(word, suffix: str):  # type: ignore[no-untyped-def]
            kind = ir.FloatType() if suffix == "f32" else i32
            function = self.intrinsic(
                f"llvm.nvvm.shfl.sync.{mode}.{suffix}", kind, [i32, kind, i32, i32]
            )
            return b.call(function, [mask, word, lane32, c])

        if isinstance(t, FloatType) and t.width == 32:
            self.set(op.result, shuffle(value, "f32"))
            return
        if isinstance(t, BoolType):
            self.set(op.result, b.trunc(shuffle(b.zext(value, i32), "i32"), ir.IntType(1)))
            return
        width = t.width  # type: ignore[union-attr]
        bits = b.bitcast(value, i64) if isinstance(t, FloatType) else value
        if width <= 32:
            moved = shuffle(b.sext(bits, i32) if width < 32 else bits, "i32")
            self.set(op.result, b.trunc(moved, self.owner.llvm_type(t)) if width < 32 else moved)
            return
        low = shuffle(b.trunc(bits, i32), "i32")
        high = shuffle(b.trunc(b.lshr(bits, ir.Constant(i64, 32)), i32), "i32")
        joined = b.or_(b.shl(b.zext(high, i64), ir.Constant(i64, 32)), b.zext(low, i64))
        self.set(
            op.result,
            b.bitcast(joined, self.owner.llvm_type(t)) if isinstance(t, FloatType) else joined,
        )

    # -- math: libdevice ---------------------------------------------------------

    def _libdevice(self, op: Operation) -> None:
        t = op.results[0].type
        if not isinstance(t, FloatType) or t.width not in {32, 64}:
            raise EmitError(f"{op.name} over {t} has no libdevice form")
        found = _LIBDEVICE.get(op.local_name)
        if found is None:
            raise EmitError(f"{op.name} has no libdevice form")
        llvm_type = self.owner.llvm_type(t)
        arguments = [self.value(v) for v in op.operands]
        symbol = f"__nv_{found}{'f' if t.width == 32 else ''}"
        function = self.extern(symbol, llvm_type, [llvm_type] * len(arguments))
        self.set(op.results[0], self.builder.call(function, arguments))
