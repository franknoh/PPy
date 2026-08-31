"""LLVM ORC/MCJIT compilation host (spec 16.9)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

__all__ = ["JitEngine", "LlvmUnavailable", "available", "llvm_status"]


class LlvmUnavailable(RuntimeError):
    """The LLVM backend cannot run in this environment."""


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("llvmlite.binding") is not None


def llvm_status() -> tuple[str, str]:
    try:
        import llvmlite
        from llvmlite import binding
    except Exception as exc:  # noqa: BLE001
        return "unavailable", str(exc)
    try:
        version = ".".join(str(p) for p in binding.llvm_version_info)
    except Exception:  # noqa: BLE001
        version = "unknown"
    return "available", f"llvmlite {llvmlite.__version__}, LLVM {version}"


_INITIALIZED = False


def _initialize() -> None:
    # LLVM must be initialized exactly once per process; a module flag is the
    # honest way to say that.
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return
    try:
        from llvmlite import binding
    except Exception as exc:
        raise LlvmUnavailable(f"llvmlite is not installed: {exc}") from exc
    # Newer llvmlite initializes LLVM automatically and rejects the old calls.
    for step in ("initialize", "initialize_native_target", "initialize_native_asmprinter"):
        function = getattr(binding, step, None)
        if function is None:
            continue
        with contextlib.suppress(RuntimeError):
            function()
    _INITIALIZED = True


def host_target() -> tuple[str, str]:
    """The host CPU's name and feature string, or empty when unknown."""
    from llvmlite import binding

    _initialize()
    try:
        return binding.get_host_cpu_name(), binding.get_host_cpu_features().flatten()
    except (RuntimeError, AttributeError):
        return "", ""


def _host_machine(opt: int):  # type: ignore[no-untyped-def]
    """A target machine for the CPU actually running this process.

    JIT-compiled code never leaves this machine, so it may use every feature
    the host has (AVX2/FMA and friends); measured 14% on a matmul kernel.
    An LLVM that does not recognize the host falls back to the baseline.
    """
    from llvmlite import binding

    target = binding.Target.from_default_triple()
    try:
        return target.create_target_machine(
            cpu=binding.get_host_cpu_name(),
            features=binding.get_host_cpu_features().flatten(),
            opt=opt,
        )
    except RuntimeError:
        return target.create_target_machine(opt=opt)


@dataclass(slots=True)
class JitEngine:
    """Owns one execution engine and the modules added to it."""

    opt_level: int = 2
    engine: object | None = None
    target_machine: object | None = None
    _baseline: object | None = None
    _modules: list[object] = field(default_factory=list)

    def open(self) -> JitEngine:
        from llvmlite import binding

        _initialize()
        self.target_machine = _host_machine(min(self.opt_level, 3))
        backing = binding.parse_assembly("")
        self.engine = binding.create_mcjit_compiler(backing, self.target_machine)
        return self

    def add(self, ir: str) -> object:
        from llvmlite import binding

        if self.engine is None:
            self.open()
        module = binding.parse_assembly(ir)
        module.verify()
        self._optimize(module)
        self.engine.add_module(module)  # type: ignore[union-attr]
        self._modules.append(module)
        return module

    def finalize(self) -> None:
        if self.engine is not None:
            self.engine.finalize_object()  # type: ignore[union-attr]
            self.engine.run_static_constructors()  # type: ignore[union-attr]

    def address(self, symbol: str) -> int:
        if self.engine is None:
            raise LlvmUnavailable("the execution engine is not open")
        return self.engine.get_function_address(symbol)  # type: ignore[union-attr]

    def optimized_ir(self, ir: str) -> str:
        from llvmlite import binding

        _initialize()
        module = binding.parse_assembly(ir)
        module.verify()
        self._optimize(module)
        return str(module)

    def _optimize(self, module: object, machine: object | None = None) -> None:
        """Run the LLVM pipeline, tuned for `machine` when one is given.

        Optimization level never implies unsafe arithmetic or fast math: those
        require an explicit directive (spec 3.4, 12.5), so no fast-math flags
        are set here at any level.
        """
        from llvmlite import binding

        level = min(self.opt_level, 3)
        if hasattr(binding, "create_pass_builder"):
            options = binding.create_pipeline_tuning_options(speed_level=level)
            options.loop_vectorization = level >= 2
            options.slp_vectorization = level >= 3
            options.loop_unrolling = level >= 3
            machine = machine or self.target_machine or self.baseline_machine()
            builder = binding.create_pass_builder(machine, options)
            builder.getModulePassManager().run(module, builder)  # type: ignore[arg-type]
            return
        legacy = binding.create_pass_manager_builder()
        legacy.opt_level = level
        legacy.loop_vectorize = level >= 2
        legacy.slp_vectorize = level >= 3
        manager = binding.create_module_pass_manager()
        legacy.populate(manager)
        manager.run(module)  # type: ignore[arg-type]

    def object_machine(self, host_cpu: bool = False):  # type: ignore[no-untyped-def]
        """The machine object code is emitted through.

        `host_cpu` is the opt-in `--host-cpu` build: the artifact gets this
        machine's instruction set and stops being portable, which is the
        caller's stated intent. Without it, emitted code is baseline.
        """
        if not host_cpu:
            return self.baseline_machine()
        if self.target_machine is None:
            _initialize()
            self.target_machine = _host_machine(min(self.opt_level, 3))
        return self.target_machine

    def baseline_machine(self):  # type: ignore[no-untyped-def]
        """A target machine for the portable baseline rather than this host.

        Object code emitted through this goes into artifacts that may run on
        another machine, exactly like a C compiler's output without
        `-march=native`. Only in-process JIT code targets the exact host.
        """
        from llvmlite import binding

        if self._baseline is None:
            _initialize()
            self._baseline = binding.Target.from_default_triple().create_target_machine(
                opt=min(self.opt_level, 3)
            )
        return self._baseline

    def close(self) -> None:
        self._modules.clear()
        self.engine = None
        self.target_machine = None
        self._baseline = None
