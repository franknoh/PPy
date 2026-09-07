"""C for the simd, cpu, atomic, and concurrency dialects, and for core
operations over vectors.

A vector is a struct of lanes, and every operation on it is a `static
inline` helper made once per (operation, type) and called at the site, so
the unit stays plain C11 that any compiler vectorizes as it can. Atomics
are the `__atomic` builtins GCC and Clang share, threads are pthreads, and
the synchronization objects are spun on exactly as the LLVM backend spins
on them.
"""

from __future__ import annotations

from ...ir import BoolType, FloatType, IntType, IRType, Operation, PtrType, VectorType

__all__ = ["PRELUDE", "emit_atomic", "emit_concurrency", "emit_cpu", "emit_simd", "vector_core"]

#: Macros the hints and the threads need, emitted once when a unit uses them.
PRELUDE = {
    "pause": """#if defined(__x86_64__) || defined(__i386__)
#define PPY_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define PPY_PAUSE() __asm__ __volatile__("yield")
#else
#define PPY_PAUSE() ((void)0)
#endif
""",
    "prefetch": """#if defined(__GNUC__) || defined(__clang__)
#define PPY_PREFETCH(p, rw, locality) __builtin_prefetch((p), (rw), (locality))
#else
#define PPY_PREFETCH(p, rw, locality) ((void)(p))
#endif
""",
    "atomic": """#if !defined(__GNUC__) && !defined(__clang__)
#error "the atomic operations in this unit need the __atomic builtins of GCC or Clang"
#endif
""",
}

_ORDER = {
    "relaxed": "__ATOMIC_RELAXED",
    "acquire": "__ATOMIC_ACQUIRE",
    "release": "__ATOMIC_RELEASE",
    "acq_rel": "__ATOMIC_ACQ_REL",
    "seq_cst": "__ATOMIC_SEQ_CST",
}


class EmitError(Exception):
    """IR the C backend cannot lower; the verifier should have caught it."""


# -- vectors ---------------------------------------------------------------------


def vector_name(owner, t: VectorType) -> str:  # type: ignore[no-untyped-def]
    """The typedef of `vector<T, N>`, made on first use."""
    element = t.element
    if isinstance(element, BoolType):
        tag = "b"
    elif isinstance(element, FloatType):
        tag = f"f{element.width}"
    elif isinstance(element, IntType):
        tag = f"{'i' if element.signed else 'u'}{element.width}"
    else:
        tag = "i64"
    name = f"ppy_v{t.count}{tag}"
    if name not in owner.unit.aggregates:
        owner.unit.aggregates[name] = (
            f"typedef struct {name} {{\n    {owner.c_type(element)} lanes[{t.count}];\n}} {name};\n"
        )
    return name


def _helper(owner, name: str, source: str) -> str:  # type: ignore[no-untyped-def]
    owner.unit.helpers.setdefault(name, source)
    return name


def _lane_loop(count: int, body: str) -> str:
    return f"    for (int i = 0; i < {count}; i++) {{\n        {body}\n    }}\n"


def _binary_helper(
    owner, t: VectorType, name: str, expression: str, result: VectorType | None = None
) -> str:  # type: ignore[no-untyped-def]
    """`static inline R name(V a, V b)` computing `expression` per lane."""
    vector = vector_name(owner, t)
    out = vector_name(owner, result or t)
    key = f"{vector}_{name}"
    source = (
        f"static inline {out} {key}({vector} a, {vector} b) {{\n    {out} r;\n"
        + _lane_loop(t.count, f"r.lanes[i] = {expression};")
        + "    return r;\n}\n"
    )
    return _helper(owner, key, source)


def _lane_arith(owner, element: IRType, symbol: str, a: str, b: str) -> str:  # type: ignore[no-untyped-def]
    """One lane's `a symbol b`, wrapping through the unsigned type for integers."""
    if isinstance(element, IntType | type(None)) and isinstance(element, IntType):
        unsigned = f"uint{element.width}_t"
        spelled = owner.c_type(element)
        return f"(({spelled})((({unsigned})({a})) {symbol} (({unsigned})({b}))))"
    if isinstance(element, FloatType):
        return f"({a} {symbol} {b})"
    return f"((int64_t)(((uint64_t)({a})) {symbol} ((uint64_t)({b}))))"


def vector_core(fe, op: Operation, name: str) -> None:  # type: ignore[no-untyped-def]
    """A core operation whose operands or result are vectors."""
    owner = fe.owner
    if name in {"add", "sub", "mul", "div"}:
        t = op.result.type
        assert isinstance(t, VectorType)
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[name]
        if name == "div" and not isinstance(t.element, FloatType):
            raise EmitError(f"vector div over {t.element} has no C lowering")
        overflow = op.attributes.get("overflow", "python")
        if not isinstance(t.element, FloatType) and overflow not in {"wrap", "proven"}:
            raise EmitError(f"vector {name} carries `wrap` or `proven`, not `{overflow}`")
        helper = _binary_helper(
            owner, t, name, _lane_arith(owner, t.element, symbol, "a.lanes[i]", "b.lanes[i]")
        )
        a, b = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{helper}({a}, {b})")
        return
    if name == "neg":
        t = op.result.type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        expression = (
            "-a.lanes[i]"
            if isinstance(t.element, FloatType)
            else _lane_arith(owner, t.element, "-", "0", "a.lanes[i]")
        )
        helper = _helper(
            owner,
            f"{vector}_neg",
            f"static inline {vector} {vector}_neg({vector} a) {{\n    {vector} r;\n"
            + _lane_loop(t.count, f"r.lanes[i] = {expression};")
            + "    return r;\n}\n",
        )
        fe.define(op.result, f"{helper}({fe.value(op.operands[0])})")
        return
    if name in {"and", "or", "xor"}:
        t = op.result.type
        assert isinstance(t, VectorType)
        symbol = {"and": "&", "or": "|", "xor": "^"}[name]
        helper = _binary_helper(owner, t, name, f"a.lanes[i] {symbol} b.lanes[i]")
        a, b = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{helper}({a}, {b})")
        return
    if name == "cmp":
        t = op.operands[0].type
        assert isinstance(t, VectorType)
        predicate = str(op.attributes["predicate"])
        symbol = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[predicate]
        if predicate == "ne" and isinstance(t.element, FloatType):
            expression = "(a.lanes[i] < b.lanes[i] || a.lanes[i] > b.lanes[i])"
        else:
            expression = f"a.lanes[i] {symbol} b.lanes[i]"
        result = VectorType(BoolType(), t.count)
        helper = _binary_helper(owner, t, f"cmp_{predicate}", expression, result)
        a, b = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{helper}({a}, {b})")
        return
    if name == "select":
        t = op.result.type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        mask = vector_name(owner, VectorType(BoolType(), t.count))
        helper = _helper(
            owner,
            f"{vector}_select",
            f"static inline {vector} {vector}_select({mask} m, {vector} a, {vector} b) {{\n"
            f"    {vector} r;\n"
            + _lane_loop(t.count, "r.lanes[i] = m.lanes[i] ? a.lanes[i] : b.lanes[i];")
            + "    return r;\n}\n",
        )
        c, a, b = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{helper}({c}, {a}, {b})")
        return
    if name == "cast":
        source = op.operands[0].type
        target = op.result.type
        assert isinstance(source, VectorType) and isinstance(target, VectorType)
        from_name = vector_name(owner, source)
        to_name = vector_name(owner, target)
        if isinstance(target.element, BoolType):
            zero = "0.0" if isinstance(source.element, FloatType) else "0"
            expression = f"a.lanes[i] != {zero}"
        else:
            expression = owner.cast("a.lanes[i]", target.element)
        helper = _helper(
            owner,
            f"{to_name}_from_{from_name}",
            f"static inline {to_name} {to_name}_from_{from_name}({from_name} a) {{\n"
            f"    {to_name} r;\n"
            + _lane_loop(target.count, f"r.lanes[i] = {expression};")
            + "    return r;\n}\n",
        )
        fe.define(op.result, f"{helper}({fe.value(op.operands[0])})")
        return
    raise EmitError(f"{op.name} over vectors has no C lowering")


def emit_simd(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    owner = fe.owner
    name = op.local_name
    if name == "splat":
        t = op.result.type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        helper = _helper(
            owner,
            f"{vector}_splat",
            f"static inline {vector} {vector}_splat({owner.c_type(t.element)} x) {{\n"
            f"    {vector} r;\n" + _lane_loop(t.count, "r.lanes[i] = x;") + "    return r;\n}\n",
        )
        fe.define(op.result, f"{helper}({fe.value(op.operands[0])})")
        return
    if name == "load":
        t = op.result.type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        helper = _helper(
            owner,
            f"{vector}_load",
            f"static inline {vector} {vector}_load(const {owner.c_type(t.element)} *p) {{\n"
            f"    {vector} r;\n" + _lane_loop(t.count, "r.lanes[i] = p[i];") + "    return r;\n}\n",
        )
        fe.define(op.result, f"{helper}({fe.value(op.operands[0])})")
        return
    if name == "store":
        vector_value, pointer = op.operands
        t = vector_value.type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        helper = _helper(
            owner,
            f"{vector}_store",
            f"static inline void {vector}_store({vector} v, {owner.c_type(t.element)} *p) {{\n"
            + _lane_loop(t.count, "p[i] = v.lanes[i];")
            + "}\n",
        )
        fe.body.append(f"    {helper}({fe.value(vector_value)}, {fe.value(pointer)});")
        return
    if name == "extract":
        vector_value, index = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{vector_value}.lanes[{index}]")
        return
    if name == "insert":
        t = op.result.type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        helper = _helper(
            owner,
            f"{vector}_insert",
            f"static inline {vector} {vector}_insert("
            f"{vector} v, {owner.c_type(t.element)} x, int64_t i) {{\n"
            "    v.lanes[i] = x;\n    return v;\n}\n",
        )
        vector_value, value, index = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{helper}({vector_value}, {value}, {index})")
        return
    if name == "shuffle":
        source = op.operands[0].type
        result = op.result.type
        assert isinstance(source, VectorType) and isinstance(result, VectorType)
        lanes = op.attributes["mask"]
        assert isinstance(lanes, tuple)
        from_name = vector_name(owner, source)
        to_name = vector_name(owner, result)
        key = f"{from_name}_shuffle_{'_'.join(str(int(m)) for m in lanes)}"
        picks = "".join(
            f"    r.lanes[{i}] = {'a' if m < source.count else 'b'}.lanes[{m % source.count}];\n"
            for i, m in enumerate(int(m) for m in lanes)
        )
        helper = _helper(
            owner,
            key,
            f"static inline {to_name} {key}({from_name} a, {from_name} b) {{\n    {to_name} r;\n"
            + picks
            + "    return r;\n}\n",
        )
        a, b = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"{helper}({a}, {b})")
        return
    if name.startswith("reduce_"):
        kind = name.removeprefix("reduce_")
        t = op.operands[0].type
        assert isinstance(t, VectorType)
        vector = vector_name(owner, t)
        element = owner.c_type(t.element)
        if kind == "add":
            step = _lane_arith(owner, t.element, "+", "acc", "v.lanes[i]")
            body = f"acc = {step};"
        else:
            symbol = "<" if kind == "min" else ">"
            body = f"acc = v.lanes[i] {symbol} acc ? v.lanes[i] : acc;"
        helper = _helper(
            owner,
            f"{vector}_reduce_{kind}",
            f"static inline {element} {vector}_reduce_{kind}({vector} v) {{\n"
            f"    {element} acc = v.lanes[0];\n"
            f"    for (int i = 1; i < {t.count}; i++) {{\n        {body}\n    }}\n"
            "    return acc;\n}\n",
        )
        fe.define(op.result, f"{helper}({fe.value(op.operands[0])})")
        return
    raise EmitError(f"{op.name} has no C lowering")


# -- cpu ------------------------------------------------------------------------------


def emit_cpu(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    if op.local_name == "prefetch":
        fe.owner.unit.prelude.setdefault("prefetch", PRELUDE["prefetch"])
        rw = 1 if op.attributes.get("rw", "read") == "write" else 0
        locality = int(op.attributes.get("locality", 3))  # type: ignore[call-overload]
        fe.body.append(f"    PPY_PREFETCH({fe.value(op.operands[0])}, {rw}, {locality});")
        return
    if op.local_name == "pause":
        fe.owner.unit.prelude.setdefault("pause", PRELUDE["pause"])
        fe.body.append("    PPY_PAUSE();")
        return
    raise EmitError(f"{op.name} has no C lowering")


# -- atomic ----------------------------------------------------------------------------


def emit_atomic(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    fe.owner.unit.prelude.setdefault("atomic", PRELUDE["atomic"])
    name = op.local_name
    order = _ORDER[str(op.attributes.get("order", "seq_cst"))]
    if name == "load":
        fe.define(op.result, f"__atomic_load_n({fe.value(op.operands[0])}, {order})")
        return
    if name == "store":
        value, pointer = (fe.value(v) for v in op.operands)
        fe.body.append(f"    __atomic_store_n({pointer}, {value}, {order});")
        return
    if name == "exchange":
        pointer, value = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"__atomic_exchange_n({pointer}, {value}, {order})")
        return
    if name.startswith("fetch_"):
        pointer, value = (fe.value(v) for v in op.operands)
        fe.define(op.result, f"__atomic_{name}({pointer}, {value}, {order})")
        return
    if name == "compare_exchange":
        pointer, expected, desired = (fe.value(v) for v in op.operands)
        success = _ORDER[str(op.attributes["success"])]
        failure = _ORDER[str(op.attributes["failure"])]
        found = fe.define(op.results[0], expected)
        fe.define(
            op.results[1],
            f"__atomic_compare_exchange_n({pointer}, &{found}, {desired}, 0, {success}, {failure})",
        )
        return
    if name == "fence":
        fe.body.append(f"    __atomic_thread_fence({order});")
        return
    raise EmitError(f"{op.name} has no C lowering")


# -- concurrency --------------------------------------------------------------------------

_MUTEX_LOCK = """static inline void ppy_mutex_lock(int64_t *slot) {
    for (;;) {
        int64_t expected = 0;
        if (__atomic_compare_exchange_n(
                slot, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return;
        }
        PPY_PAUSE();
    }
}
"""
_CONDITION_WAIT = """static inline void ppy_condition_wait(int64_t *condition, int64_t *mutex) {
    int64_t generation = __atomic_load_n(condition, __ATOMIC_ACQUIRE);
    __atomic_store_n(mutex, 0, __ATOMIC_RELEASE);
    while (__atomic_load_n(condition, __ATOMIC_ACQUIRE) == generation) {
        PPY_PAUSE();
    }
    ppy_mutex_lock(mutex);
}
"""
_BARRIER = """static inline void ppy_barrier(int64_t *slots, int64_t parties) {
    int64_t generation = __atomic_load_n(slots + 1, __ATOMIC_ACQUIRE);
    if (__atomic_fetch_add(slots, 1, __ATOMIC_ACQ_REL) + 1 == parties) {
        __atomic_store_n(slots, 0, __ATOMIC_RELEASE);
        __atomic_fetch_add(slots + 1, 1, __ATOMIC_ACQ_REL);
        return;
    }
    while (__atomic_load_n(slots + 1, __ATOMIC_ACQUIRE) == generation) {
        PPY_PAUSE();
    }
}
"""


def emit_concurrency(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    owner = fe.owner
    if owner.target.os == "windows":
        raise EmitError(f"{op.name} has no lowering for {owner.target.triple} yet")
    owner.unit.prelude.setdefault("pause", PRELUDE["pause"])
    owner.unit.prelude.setdefault("atomic", PRELUDE["atomic"])
    owner.unit.headers.update({"pthread.h", "stdlib.h"})
    owner.unit.pthread = True
    name = op.local_name
    if name == "spawn":
        _spawn(fe, op)
        return
    if name == "join":
        handle = fe.value(op.operands[0])
        slot = fe.fresh("joined")
        fe.declarations.append(f"    void *{slot};")
        fe.fail_unless(f"pthread_join((pthread_t){handle}, &{slot}) == 0", "join.ok")
        fe.define(op.result, owner.cast(f"(intptr_t){slot}", "int64_t"))
        return
    if name == "mutex_lock":
        _helper(owner, "ppy_mutex_lock", _MUTEX_LOCK)
        fe.body.append(f"    ppy_mutex_lock({fe.value(op.operands[0])});")
        return
    if name == "mutex_unlock":
        fe.body.append(f"    __atomic_store_n({fe.value(op.operands[0])}, 0, __ATOMIC_RELEASE);")
        return
    if name == "condition_wait":
        _helper(owner, "ppy_mutex_lock", _MUTEX_LOCK)
        _helper(owner, "ppy_condition_wait", _CONDITION_WAIT)
        condition, mutex = (fe.value(v) for v in op.operands)
        fe.body.append(f"    ppy_condition_wait({condition}, {mutex});")
        return
    if name == "condition_notify":
        fe.body.append(f"    __atomic_fetch_add({fe.value(op.operands[0])}, 1, __ATOMIC_ACQ_REL);")
        return
    if name == "barrier":
        _helper(owner, "ppy_barrier", _BARRIER)
        slots, parties = (fe.value(v) for v in op.operands)
        fe.body.append(f"    ppy_barrier({slots}, {parties});")
        return
    if name == "thread_id":
        fe.define(op.result, "(int64_t)pthread_self()")
        return
    raise EmitError(f"{op.name} has no C lowering")


def _spawn(fe, op: Operation) -> None:  # type: ignore[no-untyped-def]
    """A heap context of the arguments, a trampoline that unpacks it and
    calls the function, and `pthread_create` on the two."""
    owner = fe.owner
    callee_name = op.attributes["callee"].name  # type: ignore[union-attr]
    target = owner.module.functions.get(callee_name)
    if target is None:
        raise EmitError(f"spawn of @{callee_name}, which was not emitted")
    symbol = owner.symbol_of(target)
    context = f"ppy_ctx_{symbol}"
    atoms = [atom for _name, t in target.params for atom in owner.atoms(t)]
    if context not in owner.unit.aggregates:
        fields = "".join(f"    {owner.declare(atom, f'a{i}')};\n" for i, atom in enumerate(atoms))
        owner.unit.aggregates[context] = f"typedef struct {context} {{\n{fields}}} {context};\n"
    trampoline = f"ppy_thread_{symbol}"
    if trampoline not in owner.unit.trampolines:
        arguments = ", ".join([*(f"c->a{i}" for i in range(len(atoms))), "&out"])
        owner.unit.trampolines[trampoline] = (
            f"static void *{trampoline}(void *raw) {{\n"
            f"    {context} *c = ({context} *)raw;\n"
            "    int64_t out = 0;\n"
            f"    int32_t status = {symbol}({arguments});\n"
            "    free(raw);\n"
            "    return (void *)(intptr_t)status;\n"
            "}\n"
        )
    raw = fe.fresh("ctx")
    fe.declarations.append(f"    {context} *{raw};")
    fe.body.append(f"    {raw} = ({context} *)malloc(sizeof({context}));")
    fe.fail_unless(f"{raw} != NULL", "spawn.alloc")
    position = 0
    for operand in op.operands:
        for atom in fe.flatten(operand):
            fe.body.append(f"    {raw}->a{position} = {atom};")
            position += 1
    handle = fe.fresh("thread")
    fe.declarations.append(f"    pthread_t {handle};")
    fe.fail_unless(f"pthread_create(&{handle}, NULL, {trampoline}, {raw}) == 0", "spawn.ok")
    fe.define(op.result, owner.cast(handle, "int64_t"))


def is_vector(t: IRType) -> bool:
    return isinstance(t, VectorType)


def touches_vector(op: Operation) -> bool:
    return any(isinstance(v.type, VectorType) for v in (*op.operands, *op.results))


def _unused(t: PtrType) -> PtrType:
    return t
