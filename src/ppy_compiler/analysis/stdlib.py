"""Signatures and effects for commonly used standard-library callables.

These are the stub summaries the analyzer needs to type a call without
inventing `Any` (spec 8.2, 25.1). A module absent from this table stays
effect-unknown, which is an error in strict mode rather than a silent `Any`.
"""

from __future__ import annotations

from . import types as T
from .effects import Effect, EffectSet
from .refinements import Facts, IntRange

__all__ = [
    "INSTANCE_ATTRS",
    "EXTERNAL_TYPES",
    "instance_attribute",
    "lookup",
    "call",
    "MODULE_ATTRIBUTES",
    "ARRAY_TYPECODES",
]

#: `array` type codes and the element type each denotes.
ARRAY_TYPECODES: dict[str, T.Type] = {
    "b": T.INT, "B": T.INT, "h": T.INT, "H": T.INT, "i": T.INT, "I": T.INT,
    "l": T.INT, "L": T.INT, "q": T.INT, "Q": T.INT,
    "f": T.FLOAT, "d": T.FLOAT,
}

_IO = EffectSet.of(Effect.IO)
_TIME = EffectSet.of(Effect.TIME)
_RANDOM = EffectSet.of(Effect.RANDOM)
_ALLOC = EffectSet.of(Effect.ALLOC)


def _fn(qualname: str, ret: T.Type, effects: EffectSet = EffectSet()) -> tuple[T.Type, EffectSet]:
    return T.Callable_((), ret, qualname), effects


_THREAD = T.Instance("threading.Thread", (), ("threading.Thread", "object"))
_LOCK = T.Instance("threading.Lock", (), ("threading.Lock", "object"))
_EVENT = T.Instance("threading.Event", (), ("threading.Event", "object"))
_THREAD_EFFECTS = EffectSet.of(Effect.THREAD, Effect.SYNC)

#: Attributes of a standard-library instance the analyzer knows.
INSTANCE_ATTRS: dict[str, dict[str, tuple[T.Type, EffectSet]]] = {
    "threading.Thread": {
        "start": (T.Callable_((), T.NONE, "threading.Thread.start"), _THREAD_EFFECTS),
        "join": (T.Callable_((), T.NONE, "threading.Thread.join"), _THREAD_EFFECTS),
        "is_alive": (T.Callable_((), T.BOOL, "threading.Thread.is_alive"), _THREAD_EFFECTS),
        "name": (T.STR, EffectSet()),
        "daemon": (T.BOOL, EffectSet()),
        "ident": (T.union(T.INT, T.NONE), EffectSet()),
    },
    "threading.Lock": {
        "acquire": (T.Callable_((), T.BOOL, "threading.Lock.acquire"), _THREAD_EFFECTS),
        "release": (T.Callable_((), T.NONE, "threading.Lock.release"), _THREAD_EFFECTS),
        "locked": (T.Callable_((), T.BOOL, "threading.Lock.locked"), EffectSet()),
        "__enter__": (T.Callable_((), T.BOOL, "threading.Lock.__enter__"), _THREAD_EFFECTS),
        "__exit__": (T.Callable_((), T.BOOL, "threading.Lock.__exit__"), _THREAD_EFFECTS),
    },
    "threading.Event": {
        "set": (T.Callable_((), T.NONE, "threading.Event.set"), _THREAD_EFFECTS),
        "clear": (T.Callable_((), T.NONE, "threading.Event.clear"), _THREAD_EFFECTS),
        "is_set": (T.Callable_((), T.BOOL, "threading.Event.is_set"), EffectSet()),
        "wait": (T.Callable_((), T.BOOL, "threading.Event.wait"), _THREAD_EFFECTS),
    },
}


#: Standard-library classes usable as annotations, with the name to display.
EXTERNAL_TYPES: dict[str, str] = {
    "threading.Thread": "threading.Thread",
    "threading.Lock": "threading.Lock",
    "threading.RLock": "threading.Lock",
    "threading.Event": "threading.Event",
}


def instance_attribute(name: str, attribute: str) -> tuple[T.Type, EffectSet] | None:
    """An attribute of a standard-library instance, if the analyzer knows it."""
    return INSTANCE_ATTRS.get(name, {}).get(attribute)


#: Callables the analyzer knows the result type and effects of.
_FUNCTIONS: dict[str, tuple[T.Type, EffectSet]] = {
    "threading.Thread": (T.Callable_((), _THREAD, "threading.Thread"), _THREAD_EFFECTS),
    "threading.Lock": (T.Callable_((), _LOCK, "threading.Lock"), _THREAD_EFFECTS),
    "threading.RLock": (T.Callable_((), _LOCK, "threading.RLock"), _THREAD_EFFECTS),
    "threading.Event": (T.Callable_((), _EVENT, "threading.Event"), _THREAD_EFFECTS),
    "threading.current_thread": (
        T.Callable_((), _THREAD, "threading.current_thread"), _THREAD_EFFECTS
    ),
    "threading.active_count": (
        T.Callable_((), T.INT, "threading.active_count"), _THREAD_EFFECTS
    ),
    "threading.get_ident": (T.Callable_((), T.INT, "threading.get_ident"), _THREAD_EFFECTS),

    # The runtime's own import-hook API. Installing the finder mutates
    # `sys.meta_path`, which is global state.
    "ppy.install": _fn("ppy.install", T.NONE, EffectSet.of(Effect.WRITE_GLOBAL)),
    "ppy.uninstall": _fn("ppy.uninstall", T.NONE, EffectSet.of(Effect.WRITE_GLOBAL)),
    "ppy.is_installed": _fn("ppy.is_installed", T.BOOL, EffectSet.of(Effect.READ_GLOBAL)),
    "ppy.add_import_root": _fn(
        "ppy.add_import_root", T.NONE, EffectSet.of(Effect.WRITE_GLOBAL)
    ),
    "ppy.import_roots": _fn(
        "ppy.import_roots",
        T.Tuple_((T.STR,), homogeneous=True),
        EffectSet.of(Effect.READ_GLOBAL),
    ),

    "time.time": _fn("time.time", T.FLOAT, _TIME),
    "time.perf_counter": _fn("time.perf_counter", T.FLOAT, _TIME),
    "time.perf_counter_ns": _fn("time.perf_counter_ns", T.INT, _TIME),
    "time.monotonic": _fn("time.monotonic", T.FLOAT, _TIME),
    "time.monotonic_ns": _fn("time.monotonic_ns", T.INT, _TIME),
    "time.process_time": _fn("time.process_time", T.FLOAT, _TIME),
    "time.time_ns": _fn("time.time_ns", T.INT, _TIME),
    "time.sleep": _fn("time.sleep", T.NONE, _TIME | EffectSet.of(Effect.SYNC)),

    "random.random": _fn("random.random", T.FLOAT, _RANDOM),
    "random.randint": _fn("random.randint", T.INT, _RANDOM),
    "random.randrange": _fn("random.randrange", T.INT, _RANDOM),
    "random.uniform": _fn("random.uniform", T.FLOAT, _RANDOM),
    "random.gauss": _fn("random.gauss", T.FLOAT, _RANDOM),
    "random.seed": _fn("random.seed", T.NONE, _RANDOM),
    "random.shuffle": _fn("random.shuffle", T.NONE, _RANDOM | EffectSet.of(Effect.WRITE_OBJECT)),

    "os.getenv": _fn("os.getenv", T.union(T.STR, T.NONE), _IO),
    "os.getcwd": _fn("os.getcwd", T.STR, _IO),
    "os.cpu_count": _fn("os.cpu_count", T.union(T.INT, T.NONE), _IO),
    "os.path.join": _fn("os.path.join", T.STR, _ALLOC),
    "os.path.exists": _fn("os.path.exists", T.BOOL, _IO),
    "os.path.basename": _fn("os.path.basename", T.STR, _ALLOC),
    "os.path.dirname": _fn("os.path.dirname", T.STR, _ALLOC),
    "os.path.abspath": _fn("os.path.abspath", T.STR, _IO),
    "os.path.splitext": _fn("os.path.splitext", T.Tuple_((T.STR, T.STR)), _ALLOC),

    "sys.exit": _fn("sys.exit", T.NEVER, EffectSet.of(raises=("SystemExit",))),
    "sys.getsizeof": _fn("sys.getsizeof", T.INT),
    "sys.getrecursionlimit": _fn("sys.getrecursionlimit", T.INT),

    "json.dumps": _fn("json.dumps", T.STR, _ALLOC | EffectSet.of(raises=("TypeError",))),
    "json.loads": _fn("json.loads", T.ANY, _ALLOC | EffectSet.of(raises=("ValueError",))),

    "statistics.mean": _fn("statistics.mean", T.FLOAT, EffectSet.of(raises=("ValueError",))),
    "statistics.median": _fn("statistics.median", T.FLOAT, EffectSet.of(raises=("ValueError",))),
    "statistics.stdev": _fn("statistics.stdev", T.FLOAT, EffectSet.of(raises=("ValueError",))),

    "itertools.count": _fn("itertools.count", T.instance("Iterator", T.INT), _ALLOC),

    # The element type follows the type code, which `call` resolves.
    "array.array": _fn("array.array", T.instance("array", T.UNKNOWN), _ALLOC),
    "functools.reduce": _fn(
        "functools.reduce", T.ANY, _ALLOC | EffectSet.of(Effect.PYTHON_CALLBACK)
    ),
}

#: Module attributes with a known type.
MODULE_ATTRIBUTES: dict[str, tuple[T.Type, Facts]] = {
    "sys.argv": (T.list_of(T.STR), Facts()),
    "sys.executable": (T.STR, Facts()),
    "sys.platform": (T.STR, Facts()),
    "sys.maxsize": (T.INT, Facts(int_range=IntRange(0, None))),
    "sys.version": (T.STR, Facts()),
    "os.sep": (T.STR, Facts(length=1)),
    "os.linesep": (T.STR, Facts()),
    "os.name": (T.STR, Facts()),
}


def lookup(qualname: str) -> tuple[T.Type, EffectSet] | None:
    return _FUNCTIONS.get(qualname)


def call(qualname: str, args: list[tuple[T.Type, Facts]]) -> tuple[T.Type, EffectSet] | None:
    """Signatures whose result the arguments decide."""
    if qualname == "array.array" and args:
        code = args[0][1]
        element = (
            ARRAY_TYPECODES.get(code.constant)
            if code.has_constant and isinstance(code.constant, str)
            else None
        )
        if element is None:
            return None
        return (
            T.instance("array", element),
            _ALLOC | EffectSet.of(raises=("ValueError", "TypeError")),
        )
    return None
