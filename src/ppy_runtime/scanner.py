"""The input scanner, in C, defined once for everyone who reads standard input.

`ppy.scan` and `ppy.read_*` under CPython load these functions from a small
shared library compiled on first use; a standalone binary links the same
text in as its runtime shims. One grammar, one implementation, two homes,
so a token means the same thing whichever way a program was built.

The grammar:

* Whitespace is ASCII: space, tab, newline, carriage return, form feed,
  vertical tab -- what `bytes.split()` splits on.
* A token is a maximal run of non-whitespace bytes. Reading one stops
  *before* the whitespace that ends it, so the newline after a token is
  still there for a line read that follows.
* A line is every byte up to a newline, which is consumed and not part of
  the line; the last line of the input needs no newline.
* An integer token is `[+-]?[0-9]+` and fits in a signed 64-bit word.
  Anything else where an integer is expected is an error, never skipped.

The functions keep no state between calls beyond the byte buffer itself:
a token or a line that does not fit the room it was given is cut there,
`*more` says so, and the caller continues with `continuing = 1`, which
means "do not skip whitespace first". The Python fallback in `ppy._io`
implements exactly this contract.
"""

from __future__ import annotations

__all__ = ["FUNCTIONS", "INTERNAL", "STATEFUL", "library_source"]

#: The buffered byte source. `peek` looks without consuming. This is the one
#: function that owns state, which is why a header-only unit refuses it.
NEXT = """{
    static char room[1 << 16];
    static long filled = 0;
    static long position = 0;
    if (position == filled) {
        filled = (long)fread(room, 1, sizeof room, stdin);
        if (filled <= 0) {
            filled = 0;
            position = 0;
            return -1;
        }
        position = 0;
    }
    if (peek) {
        return (unsigned char)room[position];
    }
    return (unsigned char)room[position++];
}"""

_IS_SPACE = "(c == ' ' || c == '\\n' || c == '\\r' || c == '\\t' || c == '\\f' || c == '\\v')"


def _token(slot: str) -> str:
    return f"""{{
    int c = ppy_rt_next(1);
    if (!continuing) {{
        while (c != -1 && {_IS_SPACE}) {{
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }}
    }}
    int64_t count = 0;
    while (c != -1 && !{_IS_SPACE}) {{
        if (count == capacity) {{
            *more = 1;
            return count;
        }}
        data[count++] = ({slot})c;
        ppy_rt_next(0);
        c = ppy_rt_next(1);
    }}
    *more = 0;
    return count;
}}"""


READ_TOKEN = _token("int8_t")
READ_TOKEN_WIDE = _token("int64_t")

READ_LINE = """{
    int c = ppy_rt_next(1);
    if (c == -1 && !continuing) {
        *more = 0;
        return -1;
    }
    int64_t count = 0;
    while (c != -1 && c != '\\n') {
        if (count == capacity) {
            *more = 1;
            return count;
        }
        data[count++] = (int8_t)c;
        ppy_rt_next(0);
        c = ppy_rt_next(1);
    }
    if (c == '\\n') {
        ppy_rt_next(0);
    }
    *more = 0;
    return count;
}"""

#: `[+-]?[0-9]+` within a signed 64-bit word, or the offending token copied
#: into `bad` (cut to fit with its terminating NUL, the rest consumed) and
#: `~count` back.
READ_INTS = """{
    int64_t count = 0;
    while (count < capacity) {
        int c = ppy_rt_next(1);
        while (c != -1 && IS_SPACE) {
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        if (c == -1) {
            break;
        }
        int64_t seen = 0;
        int negative = 0;
        int digits = 0;
        int failed = 0;
        uint64_t magnitude = 0;
        if (c == '-' || c == '+') {
            negative = c == '-';
            if (seen < bad_capacity) {
                bad[seen] = (int8_t)c;
            }
            seen++;
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        while (c != -1 && !IS_SPACE) {
            if (seen < bad_capacity) {
                bad[seen] = (int8_t)c;
            }
            seen++;
            if (c < '0' || c > '9') {
                failed = 1;
            } else if (!failed) {
                uint64_t digit = (uint64_t)(c - '0');
                if (magnitude > 922337203685477580ULL
                    || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
                    failed = 1;
                } else {
                    magnitude = magnitude * 10 + digit;
                    digits++;
                }
            }
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        if (failed || digits == 0) {
            bad[seen < bad_capacity - 1 ? seen : bad_capacity - 1] = 0;
            return ~count;
        }
        data[count++] = negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
    }
    return count;
}""".replace("IS_SPACE", _IS_SPACE)

#: The fields of a line already read, each an integer as Python's `int()`
#: reads an ASCII one -- a sign, digits, single underscores between them --
#: into the `room` slots at `data`; the fields past the room are checked and
#: counted but not stored. Returns how many fields the line held, -2 for a
#: field that is not an integer (copied into `bad`, cut to fit with its NUL),
#: -3 for one outside 64 bits. The caller holds the line, so a field this
#: cannot read is handed to `int()` itself, which reads every form it does.
PARSE_INTS = """{
    int64_t count = 0;
    int64_t i = 0;
    while (i < length) {
        int c = (unsigned char)text[i];
        if (IS_SPACE) {
            i++;
            continue;
        }
        int64_t seen = 0;
        int negative = 0;
        int digits = 0;
        int failed = 0;
        int last_underscore = 1;
        uint64_t magnitude = 0;
        if (c == '-' || c == '+') {
            negative = c == '-';
            bad[seen++] = (int8_t)c;
            i++;
            c = i < length ? (unsigned char)text[i] : -1;
        }
        while (i < length && !IS_SPACE) {
            if (seen < bad_capacity - 1) {
                bad[seen] = (int8_t)c;
            }
            seen++;
            if (c == '_') {
                if (last_underscore) {
                    failed = 2;
                }
                last_underscore = 1;
            } else if (c < '0' || c > '9') {
                failed = 2;
            } else {
                uint64_t digit = (uint64_t)(c - '0');
                if (magnitude > 922337203685477580ULL
                    || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
                    failed = failed ? failed : 3;
                } else if (!failed) {
                    magnitude = magnitude * 10 + digit;
                }
                digits++;
                last_underscore = 0;
            }
            i++;
            c = i < length ? (unsigned char)text[i] : -1;
        }
        if (digits == 0 || last_underscore) {
            failed = 2;
        }
        if (failed) {
            bad[seen < bad_capacity - 1 ? seen : bad_capacity - 1] = 0;
            return -(int64_t)failed;
        }
        if (count < room) {
            data[count] = negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
        }
        count++;
    }
    return count;
}""".replace("IS_SPACE", _IS_SPACE)

#: What a standalone binary does where Python would raise: say so and stop.
_STANDALONE_FAIL = """{
    fputs(message, stderr);
    fputc('\\n', stderr);
    exit(1);
}"""

#: `ppy.input[int]()` with no interpreter: one line, read as Python's `int()`
#: reads it -- surrounding whitespace, a sign, digits with single underscores
#: between them -- or an error that ends the program.
INPUT_INT = """{
    static int8_t line[1 << 16];
    int8_t more = 0;
    int64_t length = ppy_rt_read_line(line, (int64_t)sizeof line, 0, &more);
    if (length < 0) {
        ppy_rt_fail("ppy: EOFError: the input ended where a line was expected");
    }
    if (more) {
        ppy_rt_fail("ppy: ValueError: a line read as an integer is longer than the room for it");
    }
    int64_t start = 0;
    int64_t end = length;
    while (start < end && IS_SPACE_AT(start)) {
        start++;
    }
    while (end > start && IS_SPACE_AT(end - 1)) {
        end--;
    }
    int negative = 0;
    if (start < end && (line[start] == '-' || line[start] == '+')) {
        negative = line[start] == '-';
        start++;
    }
    int digits = 0;
    int last_underscore = 1;
    uint64_t magnitude = 0;
    for (int64_t i = start; i < end; i++) {
        int c = line[i];
        if (c == '_') {
            if (last_underscore) {
                ppy_rt_fail("ppy: ValueError: invalid literal for int()");
            }
            last_underscore = 1;
            continue;
        }
        if (c < '0' || c > '9') {
            ppy_rt_fail("ppy: ValueError: invalid literal for int()");
        }
        uint64_t digit = (uint64_t)(c - '0');
        if (magnitude > 922337203685477580ULL
            || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
            ppy_rt_fail("ppy: OverflowError: the integer does not fit in 64 bits");
        }
        magnitude = magnitude * 10 + digit;
        digits++;
        last_underscore = 0;
    }
    if (digits == 0 || last_underscore) {
        ppy_rt_fail("ppy: ValueError: invalid literal for int()");
    }
    return negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
}""".replace("IS_SPACE_AT(start)", _IS_SPACE.replace("c ==", "line[start] ==")).replace(
    "IS_SPACE_AT(end - 1)", _IS_SPACE.replace("c ==", "line[end - 1] ==")
)

#: `ppy.scan[Buffer[int]](n)` with no interpreter: exactly `count` integers
#: into `data`, or an error that ends the program -- where a token is not an
#: integer, or where the input ends first, since a value that was never read
#: is not a zero.
FILL_INTS = """{
    int8_t bad[64];
    int64_t got = ppy_rt_read_ints(data, count, bad, (int64_t)sizeof bad);
    if (got < 0) {
        ppy_rt_fail("ppy: ValueError: expected an integer token");
    }
    if (got < count) {
        ppy_rt_fail("ppy: EOFError: the input ended before every integer was read");
    }
    return got;
}"""

#: `ppy.scan[int]()` with no interpreter.
SCAN_INT = """{
    int64_t slot = 0;
    int8_t bad[64];
    int64_t got = ppy_rt_read_ints(&slot, 1, bad, (int64_t)sizeof bad);
    if (got < 0) {
        ppy_rt_fail("ppy: ValueError: expected an integer token");
    }
    if (got == 0) {
        ppy_rt_fail("ppy: EOFError: the input ended where an integer was expected");
    }
    return slot;
}"""

#: The text of one integer field as Python's `int()` reads it: a sign, digits
#: with single underscores between them. 0 is a number in `*out`, 1 is not an
#: integer, 2 is one past 64 bits.
INT_TEXT = """{
    int64_t start = 0;
    int negative = 0;
    if (start < length && (text[start] == '-' || text[start] == '+')) {
        negative = text[start] == '-';
        start++;
    }
    int digits = 0;
    int last_underscore = 1;
    uint64_t magnitude = 0;
    for (int64_t i = start; i < length; i++) {
        int c = text[i];
        if (c == '_') {
            if (last_underscore) {
                return 1;
            }
            last_underscore = 1;
            continue;
        }
        if (c < '0' || c > '9') {
            return 1;
        }
        uint64_t digit = (uint64_t)(c - '0');
        if (magnitude > 922337203685477580ULL
            || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
            return 2;
        }
        magnitude = magnitude * 10 + digit;
        digits++;
        last_underscore = 0;
    }
    if (digits == 0 || last_underscore) {
        return 1;
    }
    *out = negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
    return 0;
}"""

#: The text of one float field as Python's `float()` reads it: a decimal with
#: an optional exponent, single underscores between digits, or `inf`,
#: `infinity`, `nan` in any case, each with an optional sign. Hexadecimal,
#: which `strtod` would take, is not Python's. 1 for a float in `*out`.
FLOAT_TEXT = """{
    char clean[512];
    int64_t used = 0;
    if (length <= 0 || length >= (int64_t)sizeof clean) {
        return 0;
    }
    int64_t at = 0;
    int negative = 0;
    if (text[at] == '-' || text[at] == '+') {
        negative = text[at] == '-';
        at++;
    }
    int64_t rest = length - at;
    char lower[9];
    if (rest == 3 || rest == 8) {
        for (int64_t i = 0; i < rest; i++) {
            int c = text[at + i];
            lower[i] = (char)(c >= 'A' && c <= 'Z' ? c + 32 : c);
        }
        lower[rest] = 0;
        if (strcmp(lower, "inf") == 0 || strcmp(lower, "infinity") == 0) {
            *out = negative ? -HUGE_VAL : HUGE_VAL;
            return 1;
        }
        if (strcmp(lower, "nan") == 0) {
            *out = negative ? -NAN : NAN;
            return 1;
        }
    }
    int digits = 0;
    int exponent_digits = 0;
    int seen_point = 0;
    int seen_exponent = 0;
    if (negative) {
        clean[used++] = '-';
    }
    for (int64_t i = at; i < length; i++) {
        int c = text[i];
        if (c >= '0' && c <= '9') {
            if (seen_exponent) {
                exponent_digits++;
            } else {
                digits++;
            }
            clean[used++] = (char)c;
        } else if (c == '_') {
            int before = i > at && text[i - 1] >= '0' && text[i - 1] <= '9';
            int after = i + 1 < length && text[i + 1] >= '0' && text[i + 1] <= '9';
            if (!before || !after) {
                return 0;
            }
        } else if (c == '.' && !seen_point && !seen_exponent) {
            seen_point = 1;
            clean[used++] = '.';
        } else if ((c == 'e' || c == 'E') && !seen_exponent && digits > 0) {
            seen_exponent = 1;
            clean[used++] = 'e';
            if (i + 1 < length && (text[i + 1] == '-' || text[i + 1] == '+')) {
                clean[used++] = (char)text[++i];
            }
        } else {
            return 0;
        }
    }
    if (digits == 0 || (seen_exponent && exponent_digits == 0)) {
        return 0;
    }
    clean[used] = 0;
    char *end = NULL;
    *out = strtod(clean, &end);
    return end == clean + used;
}"""

#: Where the fields of the line being read live: the line, and a cursor and
#: its length. The one owner of that state, so a header refuses it.
LINE_HOLD = """{
    static int8_t room[1 << 16];
    static int64_t bounds[2];
    *where = bounds;
    return room;
}"""

#: `ppy.input[tuple[...]]()` and `ppy.input[Buffer[T]]()`: read one line and
#: count its fields. `expected` is how many a tuple needs, or -1 for a buffer,
#: which takes what the line has.
LINE_OPEN = """{
    int64_t *bounds = NULL;
    int8_t *line = ppy_rt_line_hold(&bounds);
    int8_t more = 0;
    int64_t length = ppy_rt_read_line(line, (int64_t)(1 << 16), 0, &more);
    if (length < 0) {
        ppy_rt_fail("ppy: EOFError: the input ended where a line was expected");
    }
    if (more) {
        ppy_rt_fail("ppy: ValueError: a line of fields is longer than the room for it");
    }
    if (length > 0 && line[length - 1] == '\\r') {
        length--;
    }
    bounds[0] = 0;
    bounds[1] = length;
    int64_t count = 0;
    int64_t i = 0;
    while (i < length) {
        int c = line[i];
        if (IS_SPACE) {
            i++;
            continue;
        }
        count++;
        while (i < length) {
            c = line[i];
            if (IS_SPACE) {
                break;
            }
            i++;
        }
    }
    if (expected >= 0 && count != expected) {
        char message[128];
        snprintf(message, sizeof message,
                 "ppy: ValueError: expected %lld field(s) on the line, got %lld",
                 (long long)expected, (long long)count);
        ppy_rt_fail(message);
    }
    return count;
}""".replace("IS_SPACE", _IS_SPACE)

#: The next field of the open line: its start, and its length in `*length`.
LINE_FIELD = """{
    int64_t *bounds = NULL;
    int8_t *line = ppy_rt_line_hold(&bounds);
    int64_t i = bounds[0];
    int c = 0;
    while (i < bounds[1]) {
        c = line[i];
        if (!IS_SPACE) {
            break;
        }
        i++;
    }
    int64_t start = i;
    while (i < bounds[1]) {
        c = line[i];
        if (IS_SPACE) {
            break;
        }
        i++;
    }
    bounds[0] = i;
    *length = i - start;
    return line + start;
}""".replace("IS_SPACE", _IS_SPACE)

LINE_INT = """{
    int64_t length = 0;
    const int8_t *text = ppy_rt_line_field(&length);
    int64_t value = 0;
    int status = ppy_rt_int_text(text, length, &value);
    if (status == 2) {
        ppy_rt_fail("ppy: OverflowError: the integer does not fit in 64 bits");
    }
    if (status != 0) {
        ppy_rt_fail("ppy: ValueError: invalid literal for int()");
    }
    return value;
}"""

LINE_FLOAT = """{
    int64_t length = 0;
    const int8_t *text = ppy_rt_line_field(&length);
    double value = 0.0;
    if (!ppy_rt_float_text(text, length, &value)) {
        ppy_rt_fail("ppy: ValueError: could not convert string to float");
    }
    return value;
}"""

#: The fields of the open line, into a buffer made to hold them.
LINE_INTS = """{
    for (int64_t i = 0; i < count; i++) {
        data[i] = ppy_rt_line_int();
    }
    return count;
}"""

LINE_FLOATS = """{
    for (int64_t i = 0; i < count; i++) {
        data[i] = ppy_rt_line_float();
    }
    return count;
}"""

#: `ppy.input[float]()` with no interpreter: the whole line, as `float(input())`.
INPUT_FLOAT = """{
    int64_t count = ppy_rt_line_open(-1);
    if (count != 1) {
        ppy_rt_fail("ppy: ValueError: could not convert string to float");
    }
    return ppy_rt_line_float();
}"""

#: `ppy.scan[float]()` with no interpreter: the next token as a float.
SCAN_FLOAT = """{
    int8_t token[512];
    int8_t more = 0;
    int64_t length = ppy_rt_read_token(token, (int64_t)sizeof token, 0, &more);
    if (length == 0 && !more) {
        ppy_rt_fail("ppy: EOFError: the input ended where a float was expected");
    }
    double value = 0.0;
    if (more || !ppy_rt_float_text(token, length, &value)) {
        ppy_rt_fail("ppy: ValueError: could not convert string to float");
    }
    return value;
}"""

#: `ppy.scan[Buffer[float]](n)` with no interpreter: exactly `count` floats.
FILL_FLOATS = """{
    for (int64_t i = 0; i < count; i++) {
        data[i] = ppy_rt_scan_float();
    }
    return count;
}"""

#: A value read as `ppy.i32` and its kind: the declared width holds it, or
#: the program stops as Python's read raises.
CHECK_WIDTH = """{
    if (value < low || value > high) {
        ppy_rt_fail("ppy: OverflowError: the integer does not fit its declared width");
    }
    return value;
}"""

#: A float the way Python's `repr` writes it: the shortest digits that read
#: back as the same double, fixed notation for exponents from -4 to 15 and
#: scientific otherwise, and a `.0` on an integral value.
PRINT_F64 = """{
    if (value != value) {
        fputs("nan", stdout);
        return;
    }
    if (value > DBL_MAX || value < -DBL_MAX) {
        fputs(value < 0 ? "-inf" : "inf", stdout);
        return;
    }
    if (value == 0.0) {
        /* Only the sign tells -0.0 from 0.0, and division shows it. */
        fputs(1.0 / value < 0 ? "-0.0" : "0.0", stdout);
        return;
    }
    char text[40];
    for (int precision = 1; precision <= 17; precision++) {
        snprintf(text, sizeof text, "%.*e", precision - 1, value);
        if (strtod(text, NULL) == value) {
            break;
        }
    }
    char digits[24];
    int count = 0;
    char *at = text;
    if (*at == '-') {
        fputc('-', stdout);
        at++;
    }
    while (*at && *at != 'e') {
        if (*at != '.') {
            digits[count++] = *at;
        }
        at++;
    }
    while (count > 1 && digits[count - 1] == '0') {
        count--;
    }
    int exponent = atoi(at + 1);
    if (exponent >= -4 && exponent < 16) {
        if (exponent < 0) {
            fputs("0.", stdout);
            for (int i = 0; i < -exponent - 1; i++) {
                fputc('0', stdout);
            }
            fwrite(digits, 1, (size_t)count, stdout);
            return;
        }
        for (int i = 0; i <= exponent; i++) {
            fputc(i < count ? digits[i] : '0', stdout);
        }
        fputc('.', stdout);
        if (count > exponent + 1) {
            fwrite(digits + exponent + 1, 1, (size_t)(count - exponent - 1), stdout);
        } else {
            fputc('0', stdout);
        }
        return;
    }
    fputc(digits[0], stdout);
    if (count > 1) {
        fputc('.', stdout);
        fwrite(digits + 1, 1, (size_t)(count - 1), stdout);
    }
    printf("e%c%02d", exponent < 0 ? '-' : '+', exponent < 0 ? -exponent : exponent);
}"""


#: name -> (result, parameters, body, headers, needs). The order is the
#: order a C file defines them in.
FUNCTIONS: dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]] = {
    "ppy_rt_next": ("int", ("int peek",), NEXT, ("stdio.h",), ()),
    "ppy_rt_read_token": (
        "int64_t",
        ("int8_t *data", "int64_t capacity", "int8_t continuing", "int8_t *more"),
        READ_TOKEN,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_read_token_wide": (
        "int64_t",
        ("int64_t *data", "int64_t capacity", "int8_t continuing", "int8_t *more"),
        READ_TOKEN_WIDE,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_read_line": (
        "int64_t",
        ("int8_t *data", "int64_t capacity", "int8_t continuing", "int8_t *more"),
        READ_LINE,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_read_ints": (
        "int64_t",
        ("int64_t *data", "int64_t capacity", "int8_t *bad", "int64_t bad_capacity"),
        READ_INTS,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_parse_ints": (
        "int64_t",
        (
            "const char *text",
            "int64_t length",
            "int64_t *data",
            "int64_t room",
            "int8_t *bad",
            "int64_t bad_capacity",
        ),
        PARSE_INTS,
        ("stdint.h",),
        (),
    ),
    "ppy_rt_fail": (
        "void",
        ("const char *message",),
        _STANDALONE_FAIL,
        ("stdio.h", "stdlib.h"),
        (),
    ),
    "ppy_rt_input_int": (
        "int64_t",
        (),
        INPUT_INT,
        ("stdint.h",),
        ("ppy_rt_read_line", "ppy_rt_fail"),
    ),
    "ppy_rt_fill_ints": (
        "int64_t",
        ("int64_t *data", "int64_t count"),
        FILL_INTS,
        ("stdint.h",),
        ("ppy_rt_read_ints", "ppy_rt_fail"),
    ),
    "ppy_rt_scan_int": (
        "int64_t",
        (),
        SCAN_INT,
        ("stdint.h",),
        ("ppy_rt_read_ints", "ppy_rt_fail"),
    ),
    "ppy_rt_int_text": (
        "int",
        ("const int8_t *text", "int64_t length", "int64_t *out"),
        INT_TEXT,
        ("stdint.h",),
        (),
    ),
    "ppy_rt_float_text": (
        "int",
        ("const int8_t *text", "int64_t length", "double *out"),
        FLOAT_TEXT,
        ("math.h", "stdint.h", "stdlib.h", "string.h"),
        (),
    ),
    "ppy_rt_line_hold": ("int8_t *", ("int64_t **where",), LINE_HOLD, ("stdint.h",), ()),
    "ppy_rt_line_open": (
        "int64_t",
        ("int64_t expected",),
        LINE_OPEN,
        ("stdint.h", "stdio.h"),
        ("ppy_rt_line_hold", "ppy_rt_read_line", "ppy_rt_fail"),
    ),
    "ppy_rt_line_field": (
        "const int8_t *",
        ("int64_t *length",),
        LINE_FIELD,
        ("stdint.h",),
        ("ppy_rt_line_hold",),
    ),
    "ppy_rt_line_int": (
        "int64_t",
        (),
        LINE_INT,
        ("stdint.h",),
        ("ppy_rt_line_field", "ppy_rt_int_text", "ppy_rt_fail"),
    ),
    "ppy_rt_line_float": (
        "double",
        (),
        LINE_FLOAT,
        ("stdint.h",),
        ("ppy_rt_line_field", "ppy_rt_float_text", "ppy_rt_fail"),
    ),
    "ppy_rt_line_ints": (
        "int64_t",
        ("int64_t *data", "int64_t count"),
        LINE_INTS,
        ("stdint.h",),
        ("ppy_rt_line_int",),
    ),
    "ppy_rt_line_floats": (
        "int64_t",
        ("double *data", "int64_t count"),
        LINE_FLOATS,
        ("stdint.h",),
        ("ppy_rt_line_float",),
    ),
    "ppy_rt_input_float": (
        "double",
        (),
        INPUT_FLOAT,
        ("stdint.h",),
        ("ppy_rt_line_open", "ppy_rt_line_float", "ppy_rt_fail"),
    ),
    "ppy_rt_scan_float": (
        "double",
        (),
        SCAN_FLOAT,
        ("stdint.h",),
        ("ppy_rt_read_token", "ppy_rt_float_text", "ppy_rt_fail"),
    ),
    "ppy_rt_fill_floats": (
        "int64_t",
        ("double *data", "int64_t count"),
        FILL_FLOATS,
        ("stdint.h",),
        ("ppy_rt_scan_float",),
    ),
    "ppy_rt_check_width": (
        "int64_t",
        ("int64_t value", "int64_t low", "int64_t high"),
        CHECK_WIDTH,
        ("stdint.h",),
        ("ppy_rt_fail",),
    ),
    "ppy_rt_print_f64": (
        "void",
        ("double value",),
        PRINT_F64,
        ("float.h", "stdio.h", "stdlib.h"),
        (),
    ),
}

#: The functions that own process state.
STATEFUL = frozenset({"ppy_rt_next", "ppy_rt_input_int", "ppy_rt_line_hold"})

#: The functions only the scanner itself calls: private to whichever unit carries them.
INTERNAL = frozenset({"ppy_rt_next"})

#: What the runtime's shared library carries: the scanner without the
#: standalone-only failure path, which has Python's exceptions instead.
LIBRARY = (
    "ppy_rt_next",
    "ppy_rt_read_token",
    "ppy_rt_read_token_wide",
    "ppy_rt_read_line",
    "ppy_rt_read_ints",
    "ppy_rt_parse_ints",
)


def library_source() -> str:
    """The reader `ppy._io` compiles once and loads with ctypes."""
    lines = ["#include <stdint.h>", "#include <stdio.h>", ""]
    for name in LIBRARY:
        result, parameters, body, _headers, _needs = FUNCTIONS[name]
        storage = "static " if name in INTERNAL else ""
        spelled = result if result.endswith("*") else f"{result} "
        lines.append(f"{storage}{spelled}{name}({', '.join(parameters) or 'void'}) {body}")
        lines.append("")
    return "\n".join(lines)
