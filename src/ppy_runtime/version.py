"""The runtime's version: the same string the compiler and `ppy` report.

`ppy_compiler.version.COMPILER_VERSION` is the literal the packaging build
reads, and this is the runtime's copy of it -- the runtime cannot import
the compiler, and a built artifact may run where the compiler is not
installed. A test holds the two together.
"""

VERSION = "0.3.0"
