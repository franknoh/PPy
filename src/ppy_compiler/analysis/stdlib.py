"""Signatures and effects for commonly used standard-library callables.

These are the stub summaries the analyzer needs to type a call without
inventing `Any` (spec 8.2, 25.1). A module absent from this table stays
effect-unknown, which is an error in strict mode rather than a silent `Any`.
"""

from __future__ import annotations

import ast
import contextlib
import re as _re

from . import types as T
from .effects import Effect, EffectSet
from .refinements import Facts, IntRange

__all__ = [
    "ARRAY_TYPECODES",
    "EXTERNAL_TYPES",
    "INSTANCE_ATTRS",
    "MODULE_ATTRIBUTES",
    "call",
    "instance_attribute",
    "lookup",
]

#: `array` type codes and the element type each denotes.
#: A byte array stores bytes, and says so: `Buffer[ppy.i8]` accepts it and
#: the ABI passes one byte per element. Reading one still hands out an `int`.
_I8 = T.Instance("i8", (), ("i8", "int", "object"))
_U8 = T.Instance("u8", (), ("u8", "int", "object"))

ARRAY_TYPECODES: dict[str, T.Type] = {
    "b": _I8,
    "B": _U8,
    "h": T.INT,
    "H": T.INT,
    "i": T.INT,
    "I": T.INT,
    "l": T.INT,
    "L": T.INT,
    "q": T.INT,
    "Q": T.INT,
    "f": T.FLOAT,
    "d": T.FLOAT,
}

_IO = EffectSet.of(Effect.IO)
_NETWORK = EffectSet.of(Effect.NETWORK, Effect.IO, raises=("OSError",))
_TIME = EffectSet.of(Effect.TIME)
_RANDOM = EffectSet.of(Effect.RANDOM)
_ALLOC = EffectSet.of(Effect.ALLOC)


_NO_EFFECTS = EffectSet()


def _fn(qualname: str, ret: T.Type, effects: EffectSet = _NO_EFFECTS) -> tuple[T.Type, EffectSet]:
    return T.Callable_((), ret, qualname), effects


_TEXT_STREAM = T.Instance("io.TextIOWrapper", (), ("io.TextIOWrapper", "object"))

_PATH = T.Instance("pathlib.Path", (), ("pathlib.Path", "pathlib.PurePath", "object"))
_AST = T.Instance("ast.AST", (), ("ast.AST", "object"))
_AST_MODULE = T.Instance("ast.Module", (), ("ast.Module", "ast.mod", "ast.AST", "object"))


def _path_methods() -> dict[str, tuple[T.Type, EffectSet]]:
    """What a `Path` does: the filesystem calls carry the IO effect, the
    spellings of other paths carry allocation, the parts are strings."""
    table: dict[str, tuple[T.Type, EffectSet]] = {}
    for name in ("exists", "is_file", "is_dir", "is_absolute", "is_symlink"):
        table[name] = (T.Callable_((), T.BOOL, f"pathlib.Path.{name}"), _IO)
    for name in ("read_text",):
        table[name] = (T.Callable_((), T.STR, f"pathlib.Path.{name}"), _IO | _ALLOC)
    table["read_bytes"] = (T.Callable_((), T.BYTES, "pathlib.Path.read_bytes"), _IO | _ALLOC)
    for name in ("write_text", "write_bytes"):
        table[name] = (T.Callable_((), T.INT, f"pathlib.Path.{name}"), _IO)
    for name in ("mkdir", "unlink", "rmdir", "touch", "rename", "replace", "chmod"):
        table[name] = (T.Callable_((), T.NONE, f"pathlib.Path.{name}"), _IO)
    for name in ("resolve", "absolute", "expanduser", "readlink"):
        table[name] = (T.Callable_((), _PATH, f"pathlib.Path.{name}"), _IO | _ALLOC)
    for name in ("with_suffix", "with_name", "with_stem", "joinpath", "relative_to"):
        table[name] = (T.Callable_((), _PATH, f"pathlib.Path.{name}"), _ALLOC)
    for name in ("glob", "rglob", "iterdir"):
        table[name] = (
            T.Callable_((), T.instance("Iterator", _PATH), f"pathlib.Path.{name}"),
            _IO | _ALLOC,
        )
    table["open"] = (T.Callable_((), _TEXT_STREAM, "pathlib.Path.open"), _IO | _ALLOC)
    table["samefile"] = (T.Callable_((), T.BOOL, "pathlib.Path.samefile"), _IO)
    table["stat"] = (T.Callable_((), T.ANY, "pathlib.Path.stat"), _IO)
    for name in ("name", "stem", "suffix", "anchor", "drive", "root"):
        table[name] = (T.STR, _NO_EFFECTS)
    table["suffixes"] = (T.list_of(T.STR), _NO_EFFECTS)
    table["parts"] = (T.Tuple_((T.STR,), homogeneous=True), _NO_EFFECTS)
    table["parent"] = (_PATH, _NO_EFFECTS)
    table["parents"] = (T.instance("Sequence", _PATH), _NO_EFFECTS)
    return table


def _ast_functions() -> dict[str, tuple[T.Type, EffectSet]]:
    """The `ast` module's functions: parsing allocates and may raise, the
    walkers hand out nodes, `unparse` and `dump` hand out text."""
    table: dict[str, tuple[T.Type, EffectSet]] = {}
    table["ast.parse"] = _fn(
        "ast.parse", _AST_MODULE, _ALLOC | EffectSet.of(raises=("SyntaxError",))
    )
    for name in ("walk", "iter_child_nodes"):
        table[f"ast.{name}"] = _fn(f"ast.{name}", T.instance("Iterator", _AST), _ALLOC)
    for name in ("unparse", "dump"):
        table[f"ast.{name}"] = _fn(f"ast.{name}", T.STR, _ALLOC)
    table["ast.get_source_segment"] = _fn("ast.get_source_segment", T.union(T.STR, T.NONE), _ALLOC)
    table["ast.get_docstring"] = _fn("ast.get_docstring", T.union(T.STR, T.NONE), _ALLOC)
    for name in ("copy_location", "fix_missing_locations", "increment_lineno"):
        table[f"ast.{name}"] = _fn(f"ast.{name}", _AST)
    table["ast.literal_eval"] = _fn("ast.literal_eval", T.ANY, EffectSet.of(raises=("ValueError",)))
    table["ast.iter_fields"] = _fn("ast.iter_fields", T.instance("Iterator", T.ANY), _ALLOC)
    return table


_DOMAIN = EffectSet.of(raises=("ValueError",))
_OVERFLOW = EffectSet.of(raises=("OverflowError",))
_DOMAIN_OR_OVERFLOW = EffectSet.of(raises=("ValueError", "OverflowError"))


def _math() -> dict[str, tuple[T.Type, EffectSet]]:
    """The `math` module, which every numeric kernel imports and none of which
    was modeled: `math.tanh` in a reward function was an unknown signature,
    and everything computed from it followed. Pure functions all; the
    effects are the exceptions the C implementation raises.
    """
    floats_total = [
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cbrt",
        "copysign",
        "cos",
        "cosh",
        "degrees",
        "dist",
        "erf",
        "erfc",
        "exp2",
        "expm1",
        "fabs",
        "fmod",
        "fsum",
        "hypot",
        "log1p",
        "nextafter",
        "prod",
        "radians",
        "remainder",
        "sin",
        "sinh",
        "sqrt",
        "sumprod",
        "tan",
        "tanh",
        "ulp",
    ]
    floats_domain = [
        "acos",
        "acosh",
        "asin",
        "atanh",
        "gamma",
        "lgamma",
        "log",
        "log10",
        "log2",
        "sqrt",
        "fmod",
        "remainder",
    ]
    floats_overflow = ["exp", "pow", "ldexp", "cosh", "sinh"]
    ints = ["ceil", "floor", "trunc", "isqrt", "gcd", "lcm"]
    ints_domain = ["factorial", "comb", "perm", "isqrt"]
    bools = ["isclose", "isfinite", "isinf", "isnan"]
    table: dict[str, tuple[T.Type, EffectSet]] = {}
    for name in floats_total:
        table[f"math.{name}"] = _fn(f"math.{name}", T.FLOAT)
    for name in floats_domain:
        table[f"math.{name}"] = _fn(f"math.{name}", T.FLOAT, _DOMAIN)
    for name in floats_overflow:
        table[f"math.{name}"] = _fn(f"math.{name}", T.FLOAT, _OVERFLOW)
    table["math.fma"] = _fn("math.fma", T.FLOAT, _DOMAIN_OR_OVERFLOW)
    for name in ints:
        table[f"math.{name}"] = _fn(f"math.{name}", T.INT)
    for name in ints_domain:
        table[f"math.{name}"] = _fn(f"math.{name}", T.INT, _DOMAIN)
    for name in bools:
        table[f"math.{name}"] = _fn(f"math.{name}", T.BOOL)
    table["math.frexp"] = _fn("math.frexp", T.Tuple_((T.FLOAT, T.INT)))
    table["math.modf"] = _fn("math.modf", T.Tuple_((T.FLOAT, T.FLOAT)))
    return table


_THREAD = T.Instance("threading.Thread", (), ("threading.Thread", "object"))
_LOCK = T.Instance("threading.Lock", (), ("threading.Lock", "object"))
_EVENT = T.Instance("threading.Event", (), ("threading.Event", "object"))
_THREAD_EFFECTS = EffectSet.of(Effect.THREAD, Effect.SYNC)

#: Attributes of a standard-library instance the analyzer knows.
INSTANCE_ATTRS: dict[str, dict[str, tuple[T.Type, EffectSet]]] = {
    "pathlib.Path": _path_methods(),
    "pathlib.PurePath": _path_methods(),
    # The standard streams. Reading is the effect it looks like, so a function
    # that reads input is never mistaken for a pure one.
    "io.TextIOWrapper": {
        "read": (T.Callable_((), T.STR, "io.TextIOWrapper.read"), _IO | _ALLOC),
        "readline": (T.Callable_((), T.STR, "io.TextIOWrapper.readline"), _IO | _ALLOC),
        "readlines": (
            T.Callable_((), T.list_of(T.STR), "io.TextIOWrapper.readlines"),
            _IO | _ALLOC,
        ),
        "write": (T.Callable_((), T.INT, "io.TextIOWrapper.write"), _IO),
        "writelines": (T.Callable_((), T.NONE, "io.TextIOWrapper.writelines"), _IO),
        "flush": (T.Callable_((), T.NONE, "io.TextIOWrapper.flush"), _IO),
        "close": (T.Callable_((), T.NONE, "io.TextIOWrapper.close"), _IO),
        "isatty": (T.Callable_((), T.BOOL, "io.TextIOWrapper.isatty"), _IO),
        "fileno": (T.Callable_((), T.INT, "io.TextIOWrapper.fileno"), _IO),
        "readable": (T.Callable_((), T.BOOL, "io.TextIOWrapper.readable"), EffectSet()),
        "writable": (T.Callable_((), T.BOOL, "io.TextIOWrapper.writable"), EffectSet()),
        "closed": (T.BOOL, EffectSet()),
        "encoding": (T.STR, EffectSet()),
    },
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

#: The real class hierarchy of each opaque type, so that an `ast.Call` is an
#: `ast.AST` where one is expected. Read from the classes themselves.
EXTERNAL_MRO: dict[str, tuple[str, ...]] = {}


def _opaque(module: str, names: tuple[str, ...]) -> None:
    """Classes an annotation may name without the analyzer modeling them.

    A parameter typed `Path` or `ast.expr` is a value the program passes
    around and hands to the library; the analyzer needs to accept the name,
    not to know what a `Path` can do. An attribute or call on one is still
    unknown, and says so. `pathlib.Path` was "not a type the project can
    analyze" in every file that took a path.
    """
    library = __import__(module)

    def spelled(base: type) -> str:
        # `libcst.Call` is defined in `libcst._nodes.expression`; the
        # annotation, and the table, say `libcst.BaseExpression`.
        if base.__module__ == "builtins":
            return base.__name__
        if getattr(library, base.__name__, None) is base:
            return f"{module}.{base.__name__}"
        return f"{base.__module__}.{base.__name__}"

    for name in names:
        qualname = f"{module}.{name}"
        EXTERNAL_TYPES[qualname] = qualname
        cls = getattr(library, name, None)
        if isinstance(cls, type):
            EXTERNAL_MRO[qualname] = tuple(spelled(base) for base in cls.__mro__)


_opaque("pathlib", ("Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath"))
# The converter's own dependency, but the analyzer runs without it.
with contextlib.suppress(ImportError):
    _opaque(
        "libcst",
        (
            "CSTNode",
            "Module",
            "BaseStatement",
            "BaseSmallStatement",
            "SimpleStatementLine",
            "BaseExpression",
            "Name",
            "Attribute",
            "Call",
            "FunctionDef",
            "ClassDef",
            "CSTTransformer",
            "CSTVisitor",
        ),
    )
_opaque("re", ("Pattern", "Match"))
_opaque("datetime", ("datetime", "date", "time", "timedelta", "timezone"))
_opaque("collections", ("deque", "OrderedDict", "Counter", "defaultdict", "ChainMap"))
_opaque("decimal", ("Decimal",))
_opaque("fractions", ("Fraction",))
_opaque("uuid", ("UUID",))
_opaque("io", ("TextIOWrapper", "BytesIO", "StringIO", "BufferedReader", "BufferedWriter"))
_opaque("types", ("ModuleType", "FunctionType", "SimpleNamespace"))
_opaque("enum", ("Enum", "IntEnum", "Flag", "IntFlag"))
_opaque("argparse", ("Namespace", "ArgumentParser"))
_opaque("subprocess", ("CompletedProcess", "Popen"))
_opaque("logging", ("Logger",))
_opaque(
    "sqlite3",
    ("Connection", "Cursor", "Row", "Error", "DatabaseError", "OperationalError", "IntegrityError"),
)
# Every node class of the `ast` module, from the module itself: a compiler
# written in Python names them in most signatures.
_opaque(
    "ast",
    tuple(
        name
        for name in dir(ast)
        if isinstance(getattr(ast, name), type) and issubclass(getattr(ast, name), ast.AST)
    ),
)


def instance_attribute(name: str, attribute: str) -> tuple[T.Type, EffectSet] | None:
    """An attribute of a standard-library instance, if the analyzer knows it."""
    return INSTANCE_ATTRS.get(name, {}).get(attribute)


#: Callables the analyzer knows the result type and effects of.
_FUNCTIONS: dict[str, tuple[T.Type, EffectSet]] = {
    "pathlib.Path": _fn("pathlib.Path", _PATH, _ALLOC),
    "pathlib.PurePath": _fn("pathlib.PurePath", _PATH, _ALLOC),
    **_ast_functions(),
    "threading.Thread": (T.Callable_((), _THREAD, "threading.Thread"), _THREAD_EFFECTS),
    "threading.Lock": (T.Callable_((), _LOCK, "threading.Lock"), _THREAD_EFFECTS),
    "threading.RLock": (T.Callable_((), _LOCK, "threading.RLock"), _THREAD_EFFECTS),
    "threading.Event": (T.Callable_((), _EVENT, "threading.Event"), _THREAD_EFFECTS),
    "threading.current_thread": (
        T.Callable_((), _THREAD, "threading.current_thread"),
        _THREAD_EFFECTS,
    ),
    "threading.active_count": (T.Callable_((), T.INT, "threading.active_count"), _THREAD_EFFECTS),
    "threading.get_ident": (T.Callable_((), T.INT, "threading.get_ident"), _THREAD_EFFECTS),
    # The runtime's own import-hook API. Installing the finder mutates
    # `sys.meta_path`, which is global state.
    # Reading straight into a buffer: an IO effect like any other read, and
    # the write lands in the caller's memory (spec 28).
    "ppy.read_ints": _fn(
        "ppy.read_ints", T.INT, _IO | EffectSet.of(Effect.WRITE_OBJECT, raises=("TypeError",))
    ),
    "ppy.read_token": _fn(
        "ppy.read_token", T.INT, _IO | EffectSet.of(Effect.WRITE_OBJECT, raises=("TypeError",))
    ),
    "ppy.reader_available": _fn("ppy.reader_available", T.BOOL, EffectSet()),
    "ppy.install": _fn("ppy.install", T.NONE, EffectSet.of(Effect.WRITE_GLOBAL)),
    "ppy.uninstall": _fn("ppy.uninstall", T.NONE, EffectSet.of(Effect.WRITE_GLOBAL)),
    "ppy.is_installed": _fn("ppy.is_installed", T.BOOL, EffectSet.of(Effect.READ_GLOBAL)),
    "ppy.add_import_root": _fn("ppy.add_import_root", T.NONE, EffectSet.of(Effect.WRITE_GLOBAL)),
    "ppy.import_roots": _fn(
        "ppy.import_roots",
        T.Tuple_((T.STR,), homogeneous=True),
        EffectSet.of(Effect.READ_GLOBAL),
    ),
    # Whether `.ppy` imports may be served natively, and which have been:
    # process-wide state of the hook, like the finder itself.
    "ppy.ffi.library": _fn("ppy.ffi.library", T.instance("ppy.ffi.Library"), EffectSet()),
    "ppy.ffi.bind": _fn("ppy.ffi.bind", T.ANY, EffectSet()),
    "ppy.ffi.LengthOf": _fn("ppy.ffi.LengthOf", T.ANY, EffectSet()),
    "ppy.native.extern": _fn("ppy.native.extern", T.ANY, EffectSet()),
    "ppy.native.export": _fn("ppy.native.export", T.ANY, EffectSet()),
    "ppy.cpu.target": _fn("ppy.cpu.target", T.ANY, EffectSet()),
    "ppy.xla.jit": _fn("ppy.xla.jit", T.ANY, EffectSet()),
    "ppy.xla.compile": _fn("ppy.xla.compile", T.ANY, EffectSet()),
    # Asking the PJRT bridge which devices exist reads process-wide state.
    "ppy.xla.devices": _fn("ppy.xla.devices", T.list_of(T.STR), EffectSet.of(Effect.READ_GLOBAL)),
    "ppy.xla.default_device": _fn(
        "ppy.xla.default_device", T.union(T.STR, T.NONE), EffectSet.of(Effect.READ_GLOBAL)
    ),
    "ppy.xla.device_put": _fn("ppy.xla.device_put", T.ANY, EffectSet.of(Effect.READ_GLOBAL)),
    "ppy.cuda.kernel": _fn("ppy.cuda.kernel", T.ANY, EffectSet()),
    "ppy.cuda.device": _fn("ppy.cuda.device", T.ANY, EffectSet()),
    "ppy.hip.kernel": _fn("ppy.hip.kernel", T.ANY, EffectSet()),
    "ppy.hip.device": _fn("ppy.hip.device", T.ANY, EffectSet()),
    "ppy.parallel.range": _fn(
        "ppy.parallel.range", T.Instance("range", (), ("range", "object")), EffectSet()
    ),
    "ppy.native_import": _fn("ppy.native_import", T.BOOL, EffectSet.of(Effect.WRITE_GLOBAL)),
    "ppy.native_imports": _fn(
        "ppy.native_imports",
        T.dict_of(T.STR, T.Tuple_((T.STR,), homogeneous=True)),
        EffectSet.of(Effect.READ_GLOBAL),
    ),
    "socket.socket": _fn("socket.socket", T.instance("socket.socket"), _NETWORK | _ALLOC),
    "socket.create_connection": _fn(
        "socket.create_connection", T.instance("socket.socket"), _NETWORK | _ALLOC
    ),
    "socket.gethostbyname": _fn("socket.gethostbyname", T.STR, _NETWORK),
    "urllib.request.urlopen": _fn("urllib.request.urlopen", T.ANY, _NETWORK | _ALLOC),
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
    **_math(),
}

#: Module attributes with a known type.
MODULE_ATTRIBUTES: dict[str, tuple[T.Type, Facts]] = {
    "ppy.native": (T.Module_("ppy.native"), Facts()),
    "ppy.ffi": (T.Module_("ppy.ffi"), Facts()),
    "ppy.simd": (T.Module_("ppy.simd"), Facts()),
    "ppy.cpu": (T.Module_("ppy.cpu"), Facts()),
    "ppy.atomic": (T.Module_("ppy.atomic"), Facts()),
    "ppy.concurrent": (T.Module_("ppy.concurrent"), Facts()),
    "ppy.xla": (T.Module_("ppy.xla"), Facts()),
    "ppy.cuda": (T.Module_("ppy.cuda"), Facts()),
    "ppy.hip": (T.Module_("ppy.hip"), Facts()),
    "ppy.aio": (T.Module_("ppy.aio"), Facts()),
    "ppy.ffi.nullable": (T.ANY, Facts()),
    "math.pi": (T.FLOAT, Facts()),
    "math.e": (T.FLOAT, Facts()),
    "math.tau": (T.FLOAT, Facts()),
    "math.inf": (T.FLOAT, Facts()),
    "math.nan": (T.FLOAT, Facts()),
    "sys.argv": (T.list_of(T.STR), Facts()),
    "sys.path": (T.list_of(T.STR), Facts()),
    "sys.modules": (T.dict_of(T.STR, T.OBJECT), Facts()),
    "sys.version_info": (T.Tuple_((T.INT, T.INT, T.INT, T.STR, T.INT)), Facts()),
    "sys.stdin": (_TEXT_STREAM, Facts()),
    "sys.stdout": (_TEXT_STREAM, Facts()),
    "sys.stderr": (_TEXT_STREAM, Facts()),
    "sys.executable": (T.STR, Facts()),
    "sys.platform": (T.STR, Facts()),
    "sys.maxsize": (T.INT, Facts(int_range=IntRange(0, None))),
    "sys.version": (T.STR, Facts()),
    "os.sep": (T.STR, Facts(length=1)),
    "os.linesep": (T.STR, Facts()),
    "os.name": (T.STR, Facts()),
}


# -- re: a pattern over bytes, matched natively --------------------------------------
#: `re.compile(rb"...")` at module level compiles to a native matcher; the
#: checker types the pattern and its matches from these.
_RE_PATTERN = T.Instance("re.Pattern", (), ("re.Pattern", "object"))
_RE_MATCH = T.Instance("re.Match", (), ("re.Match", "object"))
_RE_OPTIONAL_MATCH = T.union(_RE_MATCH, T.NONE)
_RE_TEXT = T.union(T.BYTES, T.STR)
_RE_ERROR = EffectSet.of(raises=("re.error",))
_RE_SEARCH = _ALLOC | EffectSet.of(raises=("TypeError",))

_FUNCTIONS.update(
    {
        "re.compile": (T.Callable_((), _RE_PATTERN, "re.compile"), _RE_ERROR | _ALLOC),
        "re.search": (T.Callable_((), _RE_OPTIONAL_MATCH, "re.search"), _RE_ERROR | _RE_SEARCH),
        "re.match": (T.Callable_((), _RE_OPTIONAL_MATCH, "re.match"), _RE_ERROR | _RE_SEARCH),
        "re.fullmatch": (
            T.Callable_((), _RE_OPTIONAL_MATCH, "re.fullmatch"),
            _RE_ERROR | _RE_SEARCH,
        ),
        "re.escape": (T.Callable_((), _RE_TEXT, "re.escape"), _ALLOC),
    }
)
INSTANCE_ATTRS["re.Pattern"] = {
    "search": (T.Callable_((), _RE_OPTIONAL_MATCH, "re.Pattern.search"), _RE_SEARCH),
    "match": (T.Callable_((), _RE_OPTIONAL_MATCH, "re.Pattern.match"), _RE_SEARCH),
    "fullmatch": (T.Callable_((), _RE_OPTIONAL_MATCH, "re.Pattern.fullmatch"), _RE_SEARCH),
    "pattern": (_RE_TEXT, EffectSet()),
    "flags": (T.INT, EffectSet()),
    "groups": (T.INT, EffectSet()),
}
INSTANCE_ATTRS["re.Match"] = {
    "start": (T.Callable_((), T.INT, "re.Match.start"), EffectSet.of(raises=("IndexError",))),
    "end": (T.Callable_((), T.INT, "re.Match.end"), EffectSet.of(raises=("IndexError",))),
    "span": (
        T.Callable_((), T.Tuple_((T.INT, T.INT)), "re.Match.span"),
        EffectSet.of(raises=("IndexError",)),
    ),
    "group": (
        T.Callable_((), T.union(_RE_TEXT, T.NONE), "re.Match.group"),
        _ALLOC | EffectSet.of(raises=("IndexError",)),
    ),
    "pos": (T.INT, EffectSet()),
    "endpos": (T.INT, EffectSet()),
    "lastindex": (T.union(T.INT, T.NONE), EffectSet()),
    "re": (_RE_PATTERN, EffectSet()),
    "string": (_RE_TEXT, EffectSet()),
}
_opaque("re", ("Pattern", "Match"))
MODULE_ATTRIBUTES.update(
    {
        f"re.{name}": (T.INT, Facts(constant=int(flag), has_constant=True))
        for name, flag in (
            ("IGNORECASE", _re.IGNORECASE),
            ("I", _re.IGNORECASE),
            ("MULTILINE", _re.MULTILINE),
            ("M", _re.MULTILINE),
            ("DOTALL", _re.DOTALL),
            ("S", _re.DOTALL),
            ("ASCII", _re.ASCII),
            ("A", _re.ASCII),
            ("VERBOSE", _re.VERBOSE),
            ("X", _re.VERBOSE),
        )
    }
)


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
