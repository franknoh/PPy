/* Strings: Python's `str` as a collection of family 4.

   A string is a handle `ppy_coll_make` made, so it is counted, released, and
   freed the way a collection is. Its header words:

     [0] length in bytes           [1] capacity (words)   [2] the bytes
     [3] length in code points     [4] hash, 0 until asked
     [5] 1 when every byte is ASCII
     [6] 1 while the bytes sit in the header's own block, just after it
     [16] to [20] the handle's place on its thread's heap (collections.c)

   A one-character ASCII string is not allocated: `ppy_str_char` hands out
   one of 128 static strings whose count of references never reaches zero,
   as CPython keeps its own. A builder keeps its count of code points and
   its ASCII flag as it grows, so finishing one does not read it again.

   The bytes are UTF-8 with a NUL after the last. A string is immutable once
   `ppy_str_seal` has counted its code points; a builder is a string being
   written, sealed when it is done. Every function here that makes a string
   hands the caller its one reference.

   Indices are code points, as Python's are. An ASCII string finds one at
   once; any other walks its bytes. UTF-8 orders strings the way code points
   do, so comparing bytes is comparing strings. What Python would raise for
   is the caller's to check first, or a result here says so: a NULL string
   or a status the caller guards on. */

uint8_t *ppy_str_raw(int8_t *handle) {
    return (uint8_t *)(intptr_t)((int64_t *)handle)[2];
}

int8_t *ppy_str_data(int8_t *handle) {
    return (int8_t *)(intptr_t)((int64_t *)handle)[2];
}

int64_t ppy_str_bytes(int8_t *handle) {
    return ((int64_t *)handle)[0];
}

int64_t ppy_str_len(int8_t *handle) {
    return ((int64_t *)handle)[3];
}

int64_t ppy_str_ascii(int8_t *handle) {
    return ((int64_t *)handle)[5];
}

/* Room for `bytes` bytes and the NUL after them. */
void ppy_str_reserve(int8_t *handle, int64_t bytes) {
    int64_t *header = (int64_t *)handle;
    int64_t words = bytes / 8 + 1;
    if (words <= header[1]) {
        return;
    }
    int64_t room = header[1] * 2 > words ? header[1] * 2 : words;
    void *grown = NULL;
    if (header[6]) {
        /* Out of the header's block and into one of their own. */
        grown = malloc((size_t)(room * 8));
        if (grown != NULL) {
            memcpy(grown, (void *)(intptr_t)header[2], (size_t)(header[1] * 8));
        }
        header[6] = 0;
    } else {
        grown = realloc((void *)(intptr_t)header[2], (size_t)(room * 8));
    }
    if (grown == NULL) {
        ppy_coll_fail();
    }
    header[1] = room;
    header[2] = (int64_t)(intptr_t)grown;
}

/* A string of `bytes` zero bytes, to be written and sealed. */
int8_t *ppy_str_make(int64_t bytes) {
    int8_t *handle = ppy_coll_make(4, 0, 1, 0, 0, 1, bytes / 8 + 1);
    ((int64_t *)handle)[0] = bytes;
    return handle;
}

/* The one-character string of ASCII byte `c`: static, never freed. */
int8_t *ppy_str_char(int64_t c) {
    /* Laid out as a string made by `ppy_coll_make` is, bytes after the
       twenty-one header words, and on no heap: the collector reads a
       string it meets, and never frees one it did not make. */
    static int64_t table[128][22];
    int64_t *header = table[c & 127];
    if (header[11] == 0) {
        header[1] = 1;
        header[2] = (int64_t)(intptr_t)(header + 21);
        header[3] = 1;
        header[5] = 1;
        header[8] = 1;
        header[12] = 4;
        header[0] = 1;
        ((uint8_t *)(header + 21))[0] = (uint8_t)c;
        header[11] = INT64_MAX / 2;
    }
    return (int8_t *)header;
}

/* Count the code points, mark ASCII, end with a NUL: the string is done. */
void ppy_str_seal(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    uint8_t *data = ppy_str_raw(handle);
    int64_t count = 0;
    int64_t ascii = 1;
    for (int64_t i = 0; i < header[0]; i++) {
        if ((data[i] & 0xC0) != 0x80) {
            count++;
        }
        if (data[i] >= 0x80) {
            ascii = 0;
        }
    }
    data[header[0]] = 0;
    header[3] = count;
    header[4] = 0;
    header[5] = ascii;
}

int8_t *ppy_str_new(const int8_t *data, int64_t bytes) {
    if (bytes == 1 && (uint8_t)data[0] < 0x80) {
        return ppy_str_char(data[0]);
    }
    int8_t *handle = ppy_str_make(bytes);
    if (bytes > 0) {
        memcpy(ppy_str_raw(handle), data, (size_t)bytes);
    }
    ppy_str_seal(handle);
    return handle;
}

int8_t *ppy_str_empty(void) {
    return ppy_str_new((const int8_t *)"", 0);
}

/* -- builders ----------------------------------------------------------- */

/* An empty string to write into: its count of code points kept as it grows. */
int8_t *ppy_str_builder(int64_t hint) {
    int8_t *handle = ppy_str_make(0);
    int64_t *header = (int64_t *)handle;
    header[3] = 0;
    header[5] = 1;
    ppy_str_reserve(handle, hint > 0 ? hint : 0);
    return handle;
}

void ppy_str_add_bytes(int8_t *builder, const int8_t *data, int64_t bytes) {
    int64_t *header = (int64_t *)builder;
    if (bytes <= 0) {
        return;
    }
    ppy_str_reserve(builder, header[0] + bytes);
    uint8_t *into = ppy_str_raw(builder) + header[0];
    memmove(into, data, (size_t)bytes);
    for (int64_t i = 0; i < bytes; i++) {
        if (into[i] >= 0x80) {
            header[5] = 0;
        }
        if ((into[i] & 0xC0) != 0x80) {
            header[3]++;
        }
    }
    header[0] += bytes;
}

void ppy_str_add(int8_t *builder, int8_t *part) {
    int64_t bytes = ppy_str_bytes(part);
    int64_t *header = (int64_t *)builder;
    ppy_str_reserve(builder, header[0] + bytes);
    memmove(ppy_str_raw(builder) + header[0], ppy_str_raw(part), (size_t)bytes);
    header[0] += bytes;
    header[3] += ppy_str_len(part);
    header[5] &= ppy_str_ascii(part);
}

/* The builder done: its NUL written, and the one-character strings the
   static ones. */
int8_t *ppy_str_finish(int8_t *builder) {
    int64_t *header = (int64_t *)builder;
    uint8_t *data = ppy_str_raw(builder);
    if (header[0] == 1 && data[0] < 0x80) {
        int8_t *one = ppy_str_char(data[0]);
        ppy_coll_release(builder);
        return one;
    }
    data[header[0]] = 0;
    header[4] = 0;
    return builder;
}

/* `s += part` where the caller's reference to `s` is the only one: the
   bytes go on the end of `s` itself, as CPython appends in place. Takes the
   caller's reference to `s` and hands back one to the result. */
int8_t *ppy_str_extend(int8_t *handle, int8_t *part) {
    int64_t *header = (int64_t *)handle;
    if (header[11] != 1) {
        int8_t *made = ppy_str_builder(header[0] + ppy_str_bytes(part));
        ppy_str_add(made, handle);
        ppy_str_add(made, part);
        ppy_coll_release(handle);
        return ppy_str_finish(made);
    }
    ppy_str_add(handle, part);
    ppy_str_raw(handle)[header[0]] = 0;
    header[4] = 0;
    return handle;
}

/* -- UTF-8 -------------------------------------------------------------- */

/* The bytes one code point takes, from its first byte. */
int64_t ppy_str_step(int8_t *handle, int64_t at) {
    int64_t *header = (int64_t *)handle;
    uint8_t *data = ppy_str_raw(handle);
    int64_t next = at + 1;
    while (next < header[0] && (data[next] & 0xC0) == 0x80) {
        next++;
    }
    return next - at;
}

int64_t ppy_str_decode(const uint8_t *data, int64_t at, int64_t *width) {
    uint8_t lead = data[at];
    if (lead < 0x80) {
        *width = 1;
        return lead;
    }
    if (lead < 0xE0) {
        *width = 2;
        return ((int64_t)(lead & 0x1F) << 6) | (data[at + 1] & 0x3F);
    }
    if (lead < 0xF0) {
        *width = 3;
        return ((int64_t)(lead & 0x0F) << 12) | ((int64_t)(data[at + 1] & 0x3F) << 6)
               | (data[at + 2] & 0x3F);
    }
    *width = 4;
    return ((int64_t)(lead & 0x07) << 18) | ((int64_t)(data[at + 1] & 0x3F) << 12)
           | ((int64_t)(data[at + 2] & 0x3F) << 6) | (data[at + 3] & 0x3F);
}

int64_t ppy_str_encode(int64_t code, uint8_t *out) {
    if (code < 0x80) {
        out[0] = (uint8_t)code;
        return 1;
    }
    if (code < 0x800) {
        out[0] = (uint8_t)(0xC0 | (code >> 6));
        out[1] = (uint8_t)(0x80 | (code & 0x3F));
        return 2;
    }
    if (code < 0x10000) {
        out[0] = (uint8_t)(0xE0 | (code >> 12));
        out[1] = (uint8_t)(0x80 | ((code >> 6) & 0x3F));
        out[2] = (uint8_t)(0x80 | (code & 0x3F));
        return 3;
    }
    out[0] = (uint8_t)(0xF0 | (code >> 18));
    out[1] = (uint8_t)(0x80 | ((code >> 12) & 0x3F));
    out[2] = (uint8_t)(0x80 | ((code >> 6) & 0x3F));
    out[3] = (uint8_t)(0x80 | (code & 0x3F));
    return 4;
}

void ppy_str_add_repeat(int8_t *builder, int64_t code, int64_t count) {
    uint8_t one[4];
    int64_t width = ppy_str_encode(code, one);
    for (int64_t i = 0; i < count; i++) {
        ppy_str_add_bytes(builder, (const int8_t *)one, width);
    }
}

/* Where code point `index` starts, for 0 <= index <= length. */
int64_t ppy_str_offset(int8_t *handle, int64_t index) {
    int64_t *header = (int64_t *)handle;
    if (header[5] || index <= 0) {
        return index > 0 ? index : 0;
    }
    if (index >= header[3]) {
        return header[0];
    }
    uint8_t *data = ppy_str_raw(handle);
    int64_t at = 0;
    while (index > 0 && at < header[0]) {
        at++;
        while (at < header[0] && (data[at] & 0xC0) == 0x80) {
            at++;
        }
        index--;
    }
    return at;
}

/* The code point index of byte `at`, which starts one. */
int64_t ppy_str_index_of(int8_t *handle, int64_t at) {
    if (ppy_str_ascii(handle)) {
        return at;
    }
    uint8_t *data = ppy_str_raw(handle);
    int64_t count = 0;
    for (int64_t i = 0; i < at; i++) {
        if ((data[i] & 0xC0) != 0x80) {
            count++;
        }
    }
    return count;
}

/* A new string of bytes `from` to `to`, which bound code points. */
int8_t *ppy_str_span(int8_t *handle, int64_t from, int64_t to) {
    int64_t bytes = to > from ? to - from : 0;
    const uint8_t *data = ppy_str_raw(handle) + from;
    if (bytes == 1 && data[0] < 0x80) {
        return ppy_str_char(data[0]);
    }
    if (!ppy_str_ascii(handle)) {
        return ppy_str_new((const int8_t *)data, bytes);
    }
    /* A piece of ASCII is ASCII, and its length is its count. */
    int8_t *made = ppy_str_make(bytes);
    int64_t *header = (int64_t *)made;
    memcpy(ppy_str_raw(made), data, (size_t)bytes);
    ppy_str_raw(made)[bytes] = 0;
    header[3] = bytes;
    header[5] = 1;
    return made;
}

/* `s[index]`, for 0 <= index < len(s). */
int8_t *ppy_str_at(int8_t *handle, int64_t index) {
    int64_t at = ppy_str_offset(handle, index);
    return ppy_str_span(handle, at, at + ppy_str_step(handle, at));
}

/* The code point at `index`, for 0 <= index < len(s). */
int64_t ppy_str_code_at(int8_t *handle, int64_t index) {
    int64_t width = 0;
    return ppy_str_decode(ppy_str_raw(handle), ppy_str_offset(handle, index), &width);
}

/* `ord(s)` of a one-character string. */
int64_t ppy_str_ord(int8_t *handle) {
    int64_t width = 0;
    return ppy_str_decode(ppy_str_raw(handle), 0, &width);
}

/* `chr(code)`, for a code point UTF-8 can carry. */
int8_t *ppy_str_chr(int64_t code) {
    uint8_t one[4];
    int64_t width = ppy_str_encode(code, one);
    return ppy_str_new((const int8_t *)one, width);
}

/* -- comparing ---------------------------------------------------------- */

int64_t ppy_str_order(int8_t *a, int8_t *b) {
    if (a == b) {
        return 0;
    }
    int64_t la = ppy_str_bytes(a);
    int64_t lb = ppy_str_bytes(b);
    int64_t n = la < lb ? la : lb;
    int c = n > 0 ? memcmp(ppy_str_raw(a), ppy_str_raw(b), (size_t)n) : 0;
    if (c != 0) {
        return c < 0 ? -1 : 1;
    }
    return la < lb ? -1 : la > lb ? 1 : 0;
}

int64_t ppy_str_hash(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    if (header[4] != 0) {
        return header[4];
    }
    uint8_t *data = ppy_str_raw(handle);
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int64_t i = 0; i < header[0]; i++) {
        h ^= data[i];
        h *= 0x100000001B3ULL;
    }
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    if (h == 0) {
        h = 1;
    }
    header[4] = (int64_t)h;
    return header[4];
}

int64_t ppy_str_equal(int8_t *a, int8_t *b) {
    if (a == b) {
        return 1;
    }
    int64_t *ha = (int64_t *)a;
    int64_t *hb = (int64_t *)b;
    if (ha[0] != hb[0]) {
        return 0;
    }
    if (ha[4] != 0 && hb[4] != 0 && ha[4] != hb[4]) {
        return 0;
    }
    return memcmp(ppy_str_raw(a), ppy_str_raw(b), (size_t)ha[0]) == 0;
}

int64_t ppy_str_equal_bytes(int8_t *a, const int8_t *data, int64_t bytes) {
    if (ppy_str_bytes(a) != bytes) {
        return 0;
    }
    return bytes == 0 || memcmp(ppy_str_raw(a), data, (size_t)bytes) == 0;
}

/* -- making from others --------------------------------------------------- */

int8_t *ppy_str_concat(int8_t *a, int8_t *b) {
    int8_t *made = ppy_str_builder(ppy_str_bytes(a) + ppy_str_bytes(b));
    ppy_str_add(made, a);
    ppy_str_add(made, b);
    return ppy_str_finish(made);
}

int8_t *ppy_str_repeat(int8_t *handle, int64_t count) {
    int64_t bytes = ppy_str_bytes(handle);
    if (count <= 0 || bytes == 0) {
        return ppy_str_empty();
    }
    int8_t *made = ppy_str_builder(bytes * count);
    for (int64_t i = 0; i < count; i++) {
        ppy_str_add(made, handle);
    }
    return ppy_str_finish(made);
}

/* Python's slice of a sequence of `length`: `given` bit 0 says a start was
   written, bit 1 a stop. Writes the first index, the step, and returns how
   many items the slice takes. */
int64_t ppy_str_slice_bounds(int64_t length, int64_t *start, int64_t *stop, int64_t step,
                             int64_t given) {
    int64_t low = *start;
    int64_t high = *stop;
    if (!(given & 1)) {
        low = step < 0 ? INT64_MAX : 0;
    }
    if (!(given & 2)) {
        high = step < 0 ? INT64_MIN : INT64_MAX;
    }
    if (low < 0) {
        low += length;
        if (low < 0) {
            low = step < 0 ? -1 : 0;
        }
    } else if (low >= length) {
        low = step < 0 ? length - 1 : length;
    }
    if (high < 0) {
        high += length;
        if (high < 0) {
            high = step < 0 ? -1 : 0;
        }
    } else if (high >= length) {
        high = step < 0 ? length - 1 : length;
    }
    *start = low;
    *stop = high;
    if (step < 0) {
        return high < low ? (low - high - 1) / (-step) + 1 : 0;
    }
    return low < high ? (high - low - 1) / step + 1 : 0;
}

/* `s[start:stop:step]`, a step other than 0. */
int8_t *ppy_str_slice(int8_t *handle, int64_t start, int64_t stop, int64_t step, int64_t given) {
    int64_t length = ppy_str_len(handle);
    int64_t count = ppy_str_slice_bounds(length, &start, &stop, step, given);
    if (count <= 0) {
        return ppy_str_empty();
    }
    if (step == 1) {
        return ppy_str_span(handle, ppy_str_offset(handle, start),
                            ppy_str_offset(handle, start + count));
    }
    uint8_t *data = ppy_str_raw(handle);
    if (ppy_str_ascii(handle)) {
        int8_t *made = ppy_str_make(count);
        uint8_t *out = ppy_str_raw(made);
        for (int64_t i = 0; i < count; i++) {
            out[i] = data[start + i * step];
        }
        ppy_str_seal(made);
        return made;
    }
    int64_t *offsets = (int64_t *)malloc((size_t)(length + 1) * sizeof(int64_t));
    if (offsets == NULL) {
        ppy_coll_fail();
    }
    int64_t at = 0;
    for (int64_t i = 0; i <= length; i++) {
        offsets[i] = at;
        if (i < length) {
            at += ppy_str_step(handle, at);
        }
    }
    int8_t *made = ppy_str_builder(count);
    for (int64_t i = 0; i < count; i++) {
        int64_t index = start + i * step;
        ppy_str_add_bytes(made, (const int8_t *)(data + offsets[index]),
                          offsets[index + 1] - offsets[index]);
    }
    free(offsets);
    return ppy_str_finish(made);
}

/* -- searching ------------------------------------------------------------ */

/* Python's ADJUST_INDICES: `given` bit 0 a start, bit 1 an end. */
void ppy_str_adjust(int64_t length, int64_t *start, int64_t *end, int64_t given) {
    int64_t low = (given & 1) ? *start : 0;
    int64_t high = (given & 2) ? *end : length;
    if (high > length) {
        high = length;
    } else if (high < 0) {
        high += length;
        if (high < 0) {
            high = 0;
        }
    }
    if (low < 0) {
        low += length;
        if (low < 0) {
            low = 0;
        }
    }
    *start = low;
    *end = high;
}

/* The first byte offset in [from, to) where `needle` starts, or -1; from
   the right when `right`. */
int64_t ppy_str_search(const uint8_t *data, int64_t from, int64_t to, const uint8_t *needle,
                       int64_t bytes, int64_t right) {
    if (bytes == 0) {
        return right ? to : from;
    }
    if (to - from < bytes) {
        return -1;
    }
    if (right) {
        for (int64_t at = to - bytes; at >= from; at--) {
            if (data[at] == needle[0] && memcmp(data + at, needle, (size_t)bytes) == 0) {
                return at;
            }
        }
        return -1;
    }
    int64_t last = to - bytes;
    int64_t at = from;
    while (at <= last) {
        const uint8_t *hit = (const uint8_t *)memchr(data + at, needle[0], (size_t)(last - at + 1));
        if (hit == NULL) {
            return -1;
        }
        at = (int64_t)(hit - data);
        if (memcmp(data + at, needle, (size_t)bytes) == 0) {
            return at;
        }
        at++;
    }
    return -1;
}

/* `s.find(sub, start, end)` and `rfind`: a code point index, or -1. */
int64_t ppy_str_find(int8_t *handle, int8_t *sub, int64_t start, int64_t end, int64_t given,
                     int64_t right) {
    int64_t length = ppy_str_len(handle);
    ppy_str_adjust(length, &start, &end, given);
    if (end - start < ppy_str_len(sub)) {
        return -1;
    }
    int64_t from = ppy_str_offset(handle, start);
    int64_t to = ppy_str_offset(handle, end);
    int64_t at = ppy_str_search(ppy_str_raw(handle), from, to, ppy_str_raw(sub),
                                ppy_str_bytes(sub), right);
    return at < 0 ? -1 : ppy_str_index_of(handle, at);
}

int64_t ppy_str_contains(int8_t *handle, int8_t *sub) {
    return ppy_str_search(ppy_str_raw(handle), 0, ppy_str_bytes(handle), ppy_str_raw(sub),
                          ppy_str_bytes(sub), 0) >= 0;
}

/* `s.count(sub, start, end)`: occurrences that do not overlap. */
int64_t ppy_str_count(int8_t *handle, int8_t *sub, int64_t start, int64_t end, int64_t given) {
    int64_t length = ppy_str_len(handle);
    ppy_str_adjust(length, &start, &end, given);
    if (end - start < ppy_str_len(sub)) {
        return 0;
    }
    if (ppy_str_bytes(sub) == 0) {
        return end - start + 1;
    }
    int64_t from = ppy_str_offset(handle, start);
    int64_t to = ppy_str_offset(handle, end);
    int64_t count = 0;
    int64_t bytes = ppy_str_bytes(sub);
    while (1) {
        int64_t at = ppy_str_search(ppy_str_raw(handle), from, to, ppy_str_raw(sub), bytes, 0);
        if (at < 0) {
            return count;
        }
        count++;
        from = at + bytes;
    }
}

/* `s.startswith(prefix, start, end)`, or `endswith` when `tail`. */
int64_t ppy_str_affix(int8_t *handle, int8_t *affix, int64_t start, int64_t end, int64_t given,
                      int64_t tail) {
    int64_t length = ppy_str_len(handle);
    ppy_str_adjust(length, &start, &end, given);
    int64_t width = ppy_str_len(affix);
    if (end - width < start) {
        return 0;
    }
    if (width == 0) {
        return 1;
    }
    int64_t from = ppy_str_offset(handle, start);
    int64_t to = ppy_str_offset(handle, end);
    int64_t bytes = ppy_str_bytes(affix);
    if (to - from < bytes) {
        return 0;
    }
    int64_t at = tail ? to - bytes : from;
    return memcmp(ppy_str_raw(handle) + at, ppy_str_raw(affix), (size_t)bytes) == 0;
}

/* `s.replace(old, new, count)`, every occurrence for a negative count. */
int8_t *ppy_str_replace(int8_t *handle, int8_t *old, int8_t *fresh, int64_t count) {
    uint8_t *data = ppy_str_raw(handle);
    int64_t bytes = ppy_str_bytes(handle);
    int64_t width = ppy_str_bytes(old);
    int8_t *made = ppy_str_builder(bytes);
    if (count < 0) {
        count = INT64_MAX;
    }
    if (width == 0) {
        int64_t at = 0;
        int64_t done = 0;
        while (done < count) {
            ppy_str_add(made, fresh);
            done++;
            if (at >= bytes) {
                break;
            }
            int64_t step = ppy_str_step(handle, at);
            ppy_str_add_bytes(made, (const int8_t *)(data + at), step);
            at += step;
        }
        ppy_str_add_bytes(made, (const int8_t *)(data + at), bytes - at);
        return ppy_str_finish(made);
    }
    int64_t at = 0;
    int64_t done = 0;
    while (done < count) {
        int64_t hit = ppy_str_search(data, at, bytes, ppy_str_raw(old), width, 0);
        if (hit < 0) {
            break;
        }
        ppy_str_add_bytes(made, (const int8_t *)(data + at), hit - at);
        ppy_str_add(made, fresh);
        at = hit + width;
        done++;
    }
    ppy_str_add_bytes(made, (const int8_t *)(data + at), bytes - at);
    return ppy_str_finish(made);
}

/* -- characters ----------------------------------------------------------- */

/* `str.isspace` of one code point: Python's whitespace, every one of it. */
int64_t ppy_str_space(int64_t code) {
    if (code == ' ' || (code >= 0x09 && code <= 0x0D) || (code >= 0x1C && code <= 0x1F)) {
        return 1;
    }
    if (code < 0x80) {
        return 0;
    }
    return code == 0x85 || code == 0xA0 || code == 0x1680 || (code >= 0x2000 && code <= 0x200A)
           || code == 0x2028 || code == 0x2029 || code == 0x202F || code == 0x205F
           || code == 0x3000;
}

/* Whether `code` is one of the code points of `chars`. */
int64_t ppy_str_member(int8_t *chars, int64_t code) {
    uint8_t *data = ppy_str_raw(chars);
    int64_t bytes = ppy_str_bytes(chars);
    int64_t at = 0;
    while (at < bytes) {
        int64_t width = 0;
        if (ppy_str_decode(data, at, &width) == code) {
            return 1;
        }
        at += width;
    }
    return 0;
}

/* `s.strip(chars)`: mode 0 both ends, 1 the left, 2 the right; whitespace
   where `chars` is NULL. */
int8_t *ppy_str_strip(int8_t *handle, int8_t *chars, int64_t mode) {
    uint8_t *data = ppy_str_raw(handle);
    int64_t bytes = ppy_str_bytes(handle);
    int64_t from = 0;
    int64_t to = bytes;
    if (mode != 2) {
        while (from < to) {
            int64_t width = 0;
            int64_t code = ppy_str_decode(data, from, &width);
            if (chars == NULL ? !ppy_str_space(code) : !ppy_str_member(chars, code)) {
                break;
            }
            from += width;
        }
    }
    if (mode != 1) {
        while (to > from) {
            int64_t start = to - 1;
            while (start > from && (data[start] & 0xC0) == 0x80) {
                start--;
            }
            int64_t width = 0;
            int64_t code = ppy_str_decode(data, start, &width);
            if (chars == NULL ? !ppy_str_space(code) : !ppy_str_member(chars, code)) {
                break;
            }
            to = start;
        }
    }
    return ppy_str_span(handle, from, to);
}

int8_t *ppy_str_remove_affix(int8_t *handle, int8_t *affix, int64_t tail) {
    int64_t bytes = ppy_str_bytes(handle);
    int64_t width = ppy_str_bytes(affix);
    if (width == 0 || width > bytes) {
        return ppy_str_span(handle, 0, bytes);
    }
    uint8_t *data = ppy_str_raw(handle);
    if (tail) {
        if (memcmp(data + bytes - width, ppy_str_raw(affix), (size_t)width) == 0) {
            return ppy_str_span(handle, 0, bytes - width);
        }
    } else if (memcmp(data, ppy_str_raw(affix), (size_t)width) == 0) {
        return ppy_str_span(handle, width, bytes);
    }
    return ppy_str_span(handle, 0, bytes);
}

/* ASCII case: 0 lower, 1 upper, 2 capitalize, 3 title, 4 swapcase. NULL for
   a string that is not ASCII, whose case Python maps through its tables. */
int8_t *ppy_str_case(int8_t *handle, int64_t mode) {
    if (!ppy_str_ascii(handle)) {
        return NULL;
    }
    int64_t bytes = ppy_str_bytes(handle);
    uint8_t *data = ppy_str_raw(handle);
    int8_t *made = ppy_str_make(bytes);
    uint8_t *out = ppy_str_raw(made);
    int word = 0;
    for (int64_t i = 0; i < bytes; i++) {
        uint8_t c = data[i];
        int lower = c >= 'a' && c <= 'z';
        int upper = c >= 'A' && c <= 'Z';
        int up = mode == 1 || (mode == 2 && i == 0) || (mode == 3 && !word)
                 || (mode == 4 && lower);
        if (mode == 4 && upper) {
            up = 0;
        }
        if (up && lower) {
            c = (uint8_t)(c - 32);
        } else if (!up && upper && mode != 1) {
            c = (uint8_t)(c + 32);
        }
        out[i] = c;
        word = lower || upper;
    }
    ppy_str_seal(made);
    return made;
}

/* `isdigit` and the rest over ASCII: 0 digit, 1 alpha, 2 alnum, 3 upper,
   4 lower, 5 space (any string), 6 decimal and numeric. 1 true, 0 false,
   -1 for a string whose answer needs Python's tables. */
int64_t ppy_str_is(int8_t *handle, int64_t test) {
    int64_t bytes = ppy_str_bytes(handle);
    uint8_t *data = ppy_str_raw(handle);
    if (bytes == 0) {
        return 0;
    }
    if (test == 5) {
        int64_t at = 0;
        while (at < bytes) {
            int64_t width = 0;
            if (!ppy_str_space(ppy_str_decode(data, at, &width))) {
                return 0;
            }
            at += width;
        }
        return 1;
    }
    if (!ppy_str_ascii(handle)) {
        return -1;
    }
    int cased = 0;
    for (int64_t i = 0; i < bytes; i++) {
        uint8_t c = data[i];
        int digit = c >= '0' && c <= '9';
        int lower = c >= 'a' && c <= 'z';
        int upper = c >= 'A' && c <= 'Z';
        if ((test == 0 || test == 6) && !digit) {
            return 0;
        }
        if (test == 1 && !(lower || upper)) {
            return 0;
        }
        if (test == 2 && !(lower || upper || digit)) {
            return 0;
        }
        if (test == 3 && lower) {
            return 0;
        }
        if (test == 4 && upper) {
            return 0;
        }
        if (lower || upper) {
            cased = 1;
        }
    }
    if (test == 3 || test == 4) {
        return cased;
    }
    return 1;
}

/* `s.zfill(width)`: zeros after a sign, to `width` code points. */
int8_t *ppy_str_zfill(int8_t *handle, int64_t width) {
    int64_t length = ppy_str_len(handle);
    int64_t bytes = ppy_str_bytes(handle);
    if (width <= length) {
        return ppy_str_span(handle, 0, bytes);
    }
    uint8_t *data = ppy_str_raw(handle);
    int8_t *made = ppy_str_builder(bytes + width - length);
    int64_t at = 0;
    if (bytes > 0 && (data[0] == '+' || data[0] == '-')) {
        ppy_str_add_bytes(made, (const int8_t *)data, 1);
        at = 1;
    }
    ppy_str_add_repeat(made, '0', width - length);
    ppy_str_add_bytes(made, (const int8_t *)(data + at), bytes - at);
    return ppy_str_finish(made);
}

/* `ljust` (0), `rjust` (1), `center` (2) to `width` with `fill`. */
int8_t *ppy_str_pad(int8_t *handle, int64_t width, int64_t fill, int64_t mode) {
    int64_t length = ppy_str_len(handle);
    int64_t bytes = ppy_str_bytes(handle);
    if (width <= length) {
        return ppy_str_span(handle, 0, bytes);
    }
    int64_t spare = width - length;
    int64_t left = mode == 0 ? 0 : mode == 1 ? spare : spare / 2 + (spare & width & 1);
    int8_t *made = ppy_str_builder(bytes + spare * 4);
    ppy_str_add_repeat(made, fill, left);
    ppy_str_add(made, handle);
    ppy_str_add_repeat(made, fill, spare - left);
    return ppy_str_finish(made);
}

/* -- lists of strings ------------------------------------------------------ */

/* A new, empty list of strings: a sequence of one handle word each. */
int8_t *ppy_str_list(void) {
    return ppy_seq_new(0, 1, 0, 1);
}

/* `piece` onto the end of a list, which takes its reference. */
void ppy_str_list_take(int8_t *list, int8_t *piece) {
    int8_t *slot = ppy_seq_push_back(list);
    memcpy(slot, &piece, (size_t)8);
}

/* `s.split(sep, maxsplit)`, on whitespace where `sep` is NULL. */
int8_t *ppy_str_split(int8_t *handle, int8_t *sep, int64_t limit) {
    int8_t *list = ppy_str_list();
    uint8_t *data = ppy_str_raw(handle);
    int64_t bytes = ppy_str_bytes(handle);
    if (limit < 0) {
        limit = INT64_MAX;
    }
    if (sep != NULL) {
        int64_t width = ppy_str_bytes(sep);
        int64_t at = 0;
        int64_t done = 0;
        while (done < limit) {
            int64_t hit = ppy_str_search(data, at, bytes, ppy_str_raw(sep), width, 0);
            if (hit < 0) {
                break;
            }
            ppy_str_list_take(list, ppy_str_span(handle, at, hit));
            at = hit + width;
            done++;
        }
        ppy_str_list_take(list, ppy_str_span(handle, at, bytes));
        return list;
    }
    int64_t at = 0;
    int64_t done = 0;
    while (at < bytes) {
        int64_t width = 0;
        while (at < bytes && ppy_str_space(ppy_str_decode(data, at, &width))) {
            at += width;
        }
        if (at >= bytes) {
            break;
        }
        if (done >= limit) {
            ppy_str_list_take(list, ppy_str_span(handle, at, bytes));
            return list;
        }
        int64_t from = at;
        while (at < bytes && !ppy_str_space(ppy_str_decode(data, at, &width))) {
            at += width;
        }
        ppy_str_list_take(list, ppy_str_span(handle, from, at));
        done++;
    }
    return list;
}

/* The start of the code point that ends before byte `at`. */
int64_t ppy_str_back(const uint8_t *data, int64_t at) {
    at--;
    while (at > 0 && (data[at] & 0xC0) == 0x80) {
        at--;
    }
    return at;
}

/* `s.rsplit(sep, maxsplit)`: splits from the right, the list in order. */
int8_t *ppy_str_rsplit(int8_t *handle, int8_t *sep, int64_t limit) {
    if (limit < 0) {
        return ppy_str_split(handle, sep, limit);
    }
    int8_t *reversed = ppy_str_list();
    uint8_t *data = ppy_str_raw(handle);
    int64_t bytes = ppy_str_bytes(handle);
    if (sep != NULL) {
        int64_t width = ppy_str_bytes(sep);
        int64_t end = bytes;
        int64_t done = 0;
        while (done < limit) {
            int64_t hit = ppy_str_search(data, 0, end, ppy_str_raw(sep), width, 1);
            if (hit < 0) {
                break;
            }
            ppy_str_list_take(reversed, ppy_str_span(handle, hit + width, end));
            end = hit;
            done++;
        }
        ppy_str_list_take(reversed, ppy_str_span(handle, 0, end));
    } else {
        int64_t at = bytes;
        int64_t done = 0;
        int64_t width = 0;
        while (at > 0) {
            while (at > 0 && ppy_str_space(ppy_str_decode(data, ppy_str_back(data, at), &width))) {
                at = ppy_str_back(data, at);
            }
            if (at <= 0) {
                break;
            }
            if (done >= limit) {
                ppy_str_list_take(reversed, ppy_str_span(handle, 0, at));
                break;
            }
            int64_t end = at;
            while (at > 0 && !ppy_str_space(ppy_str_decode(data, ppy_str_back(data, at), &width))) {
                at = ppy_str_back(data, at);
            }
            ppy_str_list_take(reversed, ppy_str_span(handle, at, end));
            done++;
        }
    }
    ppy_seq_reverse(reversed);
    return reversed;
}

/* `s.splitlines(keepends)`, at every boundary Python's lines end at. */
int8_t *ppy_str_splitlines(int8_t *handle, int64_t keep) {
    int8_t *list = ppy_str_list();
    uint8_t *data = ppy_str_raw(handle);
    int64_t bytes = ppy_str_bytes(handle);
    int64_t from = 0;
    int64_t at = 0;
    while (at < bytes) {
        int64_t width = 0;
        int64_t code = ppy_str_decode(data, at, &width);
        int boundary = code == '\n' || code == '\r' || code == 0x0B || code == 0x0C
                       || code == 0x1C || code == 0x1D || code == 0x1E || code == 0x85
                       || code == 0x2028 || code == 0x2029;
        if (!boundary) {
            at += width;
            continue;
        }
        int64_t end = at + width;
        if (code == '\r' && end < bytes && data[end] == '\n') {
            end++;
        }
        ppy_str_list_take(list, ppy_str_span(handle, from, keep ? end : at));
        from = end;
        at = end;
    }
    if (from < bytes) {
        ppy_str_list_take(list, ppy_str_span(handle, from, bytes));
    }
    return list;
}

/* `sep.join(list)`. */
int8_t *ppy_str_join(int8_t *sep, int8_t *list) {
    int64_t count = ppy_coll_len(list);
    int64_t total = 0;
    for (int64_t i = 0; i < count; i++) {
        int8_t *piece = NULL;
        memcpy(&piece, ppy_seq_at(list, i), (size_t)8);
        total += ppy_str_bytes(piece) + ppy_str_bytes(sep);
    }
    int8_t *made = ppy_str_builder(total);
    for (int64_t i = 0; i < count; i++) {
        int8_t *piece = NULL;
        memcpy(&piece, ppy_seq_at(list, i), (size_t)8);
        if (i > 0) {
            ppy_str_add(made, sep);
        }
        ppy_str_add(made, piece);
    }
    return ppy_str_finish(made);
}

/* `s.partition(sep)` (or `rpartition` when `right`) into `out[0..2]`. */
void ppy_str_partition(int8_t *handle, int8_t *sep, int64_t right, int8_t **out) {
    int64_t bytes = ppy_str_bytes(handle);
    int64_t width = ppy_str_bytes(sep);
    int64_t hit = ppy_str_search(ppy_str_raw(handle), 0, bytes, ppy_str_raw(sep), width, right);
    if (hit < 0) {
        out[0] = right ? ppy_str_empty() : ppy_str_span(handle, 0, bytes);
        out[1] = ppy_str_empty();
        out[2] = right ? ppy_str_span(handle, 0, bytes) : ppy_str_empty();
        return;
    }
    out[0] = ppy_str_span(handle, 0, hit);
    out[1] = ppy_str_span(handle, hit, hit + width);
    out[2] = ppy_str_span(handle, hit + width, bytes);
}

/* Whether a list of strings holds one equal to `wanted`. */
int64_t ppy_str_list_has(int8_t *list, int8_t *wanted) {
    int64_t count = ppy_coll_len(list);
    for (int64_t i = 0; i < count; i++) {
        int8_t *piece = NULL;
        memcpy(&piece, ppy_seq_at(list, i), (size_t)8);
        if (ppy_str_equal(piece, wanted)) {
            return 1;
        }
    }
    return 0;
}

/* Take the string at `index` out of the list, closing the gap; the caller owns it. */
int8_t *ppy_str_list_remove(int8_t *list, int64_t index) {
    int64_t count = ppy_coll_len(list);
    int8_t *taken = NULL;
    memcpy(&taken, ppy_seq_at(list, index), (size_t)8);
    for (int64_t i = index; i + 1 < count; i++) {
        memcpy(ppy_seq_at(list, i), ppy_seq_at(list, i + 1), (size_t)8);
    }
    ppy_seq_pop_back(list);
    return taken;
}

/* `list.insert(index, piece)`, the index clamped as `list` clamps it; the
   list takes the reference. */
void ppy_str_list_insert(int8_t *list, int64_t index, int8_t *piece) {
    int64_t count = ppy_coll_len(list);
    if (index < 0) {
        index += count;
        if (index < 0) {
            index = 0;
        }
    }
    if (index > count) {
        index = count;
    }
    ppy_seq_push_back(list);
    for (int64_t i = count; i > index; i--) {
        memcpy(ppy_seq_at(list, i), ppy_seq_at(list, i - 1), (size_t)8);
    }
    memcpy(ppy_seq_at(list, index), &piece, (size_t)8);
}

int64_t ppy_str_list_index(int8_t *list, int8_t *wanted) {
    int64_t count = ppy_coll_len(list);
    for (int64_t i = 0; i < count; i++) {
        int8_t *piece = NULL;
        memcpy(&piece, ppy_seq_at(list, i), (size_t)8);
        if (ppy_str_equal(piece, wanted)) {
            return i;
        }
    }
    return -1;
}

int64_t ppy_str_list_count(int8_t *list, int8_t *wanted) {
    int64_t count = ppy_coll_len(list);
    int64_t found = 0;
    for (int64_t i = 0; i < count; i++) {
        int8_t *piece = NULL;
        memcpy(&piece, ppy_seq_at(list, i), (size_t)8);
        found += ppy_str_equal(piece, wanted) ? 1 : 0;
    }
    return found;
}

/* Whether the bytes are UTF-8 Python could decode: no stray continuation,
   no overlong form, no surrogate, nothing past U+10FFFF. */
int64_t ppy_str_valid(const uint8_t *data, int64_t bytes) {
    int64_t at = 0;
    while (at < bytes) {
        uint8_t lead = data[at];
        int64_t width = lead < 0x80                    ? 1
                        : lead >= 0xC2 && lead < 0xE0 ? 2
                        : lead >= 0xE0 && lead < 0xF0 ? 3
                        : lead >= 0xF0 && lead < 0xF5 ? 4
                                                      : 0;
        if (width == 0 || at + width > bytes) {
            return 0;
        }
        for (int64_t i = 1; i < width; i++) {
            if ((data[at + i] & 0xC0) != 0x80) {
                return 0;
            }
        }
        if (width == 3) {
            int64_t code = ((int64_t)(lead & 0x0F) << 12) | ((int64_t)(data[at + 1] & 0x3F) << 6);
            if (code < 0x800 || (code >= 0xD800 && code <= 0xDFFF)) {
                return 0;
            }
        }
        if (width == 4) {
            int64_t code = ((int64_t)(lead & 0x07) << 18) | ((int64_t)(data[at + 1] & 0x3F) << 12);
            if (code < 0x10000 || code > 0x10FFFF) {
                return 0;
            }
        }
        at += width;
    }
    return 1;
}

/* A copy of the bytes for a caller that frees them with `free`: the
   Python boundary, which reads a returned string once. */
int8_t *ppy_str_export(int8_t *handle) {
    int64_t bytes = ppy_str_bytes(handle);
    int8_t *copy = (int8_t *)malloc((size_t)bytes + 1);
    if (copy == NULL) {
        ppy_coll_fail();
    }
    memcpy(copy, ppy_str_raw(handle), (size_t)bytes + 1);
    return copy;
}

/* -- numbers as text ------------------------------------------------------ */

int64_t ppy_str_int_text(int64_t value, char *out) {
    char digits[24];
    int64_t count = 0;
    uint64_t magnitude = value < 0 ? (uint64_t)0 - (uint64_t)value : (uint64_t)value;
    do {
        digits[count++] = (char)('0' + magnitude % 10);
        magnitude /= 10;
    } while (magnitude > 0);
    int64_t used = 0;
    if (value < 0) {
        out[used++] = '-';
    }
    while (count > 0) {
        out[used++] = digits[--count];
    }
    return used;
}

/* A float as Python's `repr` writes it: the shortest digits that read back
   as the same double, fixed notation for exponents from -4 to 15 and
   scientific otherwise, and a `.0` on an integral value. */
int64_t ppy_str_float_text(double value, char *out) {
    if (value != value) {
        memcpy(out, "nan", 3);
        return 3;
    }
    if (value > DBL_MAX || value < -DBL_MAX) {
        if (value < 0) {
            memcpy(out, "-inf", 4);
            return 4;
        }
        memcpy(out, "inf", 3);
        return 3;
    }
    if (value == 0.0) {
        /* Only the sign tells -0.0 from 0.0, and division shows it. */
        if (1.0 / value < 0) {
            memcpy(out, "-0.0", 4);
            return 4;
        }
        memcpy(out, "0.0", 3);
        return 3;
    }
    char text[40];
    for (int precision = 1; precision <= 17; precision++) {
        snprintf(text, sizeof text, "%.*e", precision - 1, value);
        if (strtod(text, NULL) == value) {
            break;
        }
    }
    char digits[24];
    int64_t count = 0;
    int64_t used = 0;
    char *at = text;
    if (*at == '-') {
        out[used++] = '-';
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
    int64_t exponent = atoi(at + 1);
    if (exponent >= -4 && exponent < 16) {
        if (exponent < 0) {
            out[used++] = '0';
            out[used++] = '.';
            for (int64_t i = 0; i < -exponent - 1; i++) {
                out[used++] = '0';
            }
            memcpy(out + used, digits, (size_t)count);
            return used + count;
        }
        for (int64_t i = 0; i <= exponent; i++) {
            out[used++] = i < count ? digits[i] : '0';
        }
        out[used++] = '.';
        if (count > exponent + 1) {
            memcpy(out + used, digits + exponent + 1, (size_t)(count - exponent - 1));
            used += count - exponent - 1;
        } else {
            out[used++] = '0';
        }
        return used;
    }
    out[used++] = digits[0];
    if (count > 1) {
        out[used++] = '.';
        memcpy(out + used, digits + 1, (size_t)(count - 1));
        used += count - 1;
    }
    used += snprintf(out + used, 8, "e%c%02d", exponent < 0 ? '-' : '+',
                     (int)(exponent < 0 ? -exponent : exponent));
    return used;
}

int8_t *ppy_str_from_int(int64_t value) {
    char text[24];
    int64_t used = ppy_str_int_text(value, text);
    return ppy_str_new((const int8_t *)text, used);
}

int8_t *ppy_str_from_float(double value) {
    char text[40];
    int64_t used = ppy_str_float_text(value, text);
    return ppy_str_new((const int8_t *)text, used);
}

int8_t *ppy_str_from_bool(int64_t value) {
    return value ? ppy_str_new((const int8_t *)"True", 4) : ppy_str_new((const int8_t *)"False", 5);
}

void ppy_str_add_int(int8_t *builder, int64_t value) {
    char text[24];
    ppy_str_add_bytes(builder, (const int8_t *)text, ppy_str_int_text(value, text));
}

void ppy_str_add_float(int8_t *builder, double value) {
    char text[40];
    ppy_str_add_bytes(builder, (const int8_t *)text, ppy_str_float_text(value, text));
}

void ppy_str_add_bool(int8_t *builder, int64_t value) {
    if (value) {
        ppy_str_add_bytes(builder, (const int8_t *)"True", 4);
    } else {
        ppy_str_add_bytes(builder, (const int8_t *)"False", 5);
    }
}

/* `repr(s)` of an ASCII string, as Python quotes it; 0 for one that is not
   ASCII, whose printable characters Python decides from its tables. */
int64_t ppy_str_add_repr(int8_t *builder, int8_t *handle) {
    if (!ppy_str_ascii(handle)) {
        return 0;
    }
    uint8_t *data = ppy_str_raw(handle);
    int64_t bytes = ppy_str_bytes(handle);
    int single = 0;
    int dual = 0;
    for (int64_t i = 0; i < bytes; i++) {
        single |= data[i] == '\'';
        dual |= data[i] == '"';
    }
    char quote = single && !dual ? '"' : '\'';
    ppy_str_add_bytes(builder, (const int8_t *)&quote, 1);
    for (int64_t i = 0; i < bytes; i++) {
        uint8_t c = data[i];
        char text[8];
        int64_t used = 0;
        if (c == (uint8_t)quote || c == '\\') {
            text[used++] = '\\';
            text[used++] = (char)c;
        } else if (c == '\t') {
            memcpy(text, "\\t", 2);
            used = 2;
        } else if (c == '\n') {
            memcpy(text, "\\n", 2);
            used = 2;
        } else if (c == '\r') {
            memcpy(text, "\\r", 2);
            used = 2;
        } else if (c < 0x20 || c == 0x7F) {
            used = snprintf(text, sizeof text, "\\x%02x", c);
        } else {
            text[used++] = (char)c;
        }
        ppy_str_add_bytes(builder, (const int8_t *)text, used);
    }
    ppy_str_add_bytes(builder, (const int8_t *)&quote, 1);
    return 1;
}

/* `repr(list_of_strings)` onto the builder, as `[` each repr `, ` `]`; 0 when
   one of them is not ASCII. */
int64_t ppy_str_add_list_repr(int8_t *builder, int8_t *list) {
    int64_t count = ppy_coll_len(list);
    ppy_str_add_bytes(builder, (const int8_t *)"[", 1);
    for (int64_t i = 0; i < count; i++) {
        int8_t *piece = NULL;
        memcpy(&piece, ppy_seq_at(list, i), (size_t)8);
        if (i > 0) {
            ppy_str_add_bytes(builder, (const int8_t *)", ", 2);
        }
        if (!ppy_str_add_repr(builder, piece)) {
            return 0;
        }
    }
    ppy_str_add_bytes(builder, (const int8_t *)"]", 1);
    return 1;
}

/* `int(s, base)`: 0 with the value in `*out`, 1 where Python raises
   ValueError, 2 where the value does not fit 64 bits, 3 for text Python
   reads with its tables (digits outside ASCII). */
int64_t ppy_str_to_int(int8_t *handle, int64_t base, int64_t *out) {
    uint8_t *data = ppy_str_raw(handle);
    int64_t from = 0;
    int64_t to = ppy_str_bytes(handle);
    int64_t width = 0;
    while (from < to && ppy_str_space(ppy_str_decode(data, from, &width))) {
        from += width;
    }
    while (to > from && ppy_str_space(ppy_str_decode(data, ppy_str_back(data, to), &width))) {
        to = ppy_str_back(data, to);
    }
    for (int64_t i = from; i < to; i++) {
        if (data[i] >= 0x80) {
            return 3;
        }
    }
    int negative = 0;
    if (from < to && (data[from] == '+' || data[from] == '-')) {
        negative = data[from] == '-';
        from++;
    }
    int prefixed = 0;
    if (to - from >= 2 && data[from] == '0') {
        uint8_t mark = (uint8_t)(data[from + 1] | 0x20);
        int64_t named = mark == 'x' ? 16 : mark == 'o' ? 8 : mark == 'b' ? 2 : 0;
        if (named != 0 && (base == 0 || base == named)) {
            base = named;
            from += 2;
            prefixed = 1;
        }
    }
    int leading_zero_only = 0;
    if (base == 0) {
        base = 10;
        leading_zero_only = from < to && data[from] == '0';
    }
    int digits = 0;
    int last_underscore = prefixed ? 0 : 1;
    int nonzero = 0;
    uint64_t magnitude = 0;
    uint64_t limit = negative ? (uint64_t)INT64_MAX + 1 : (uint64_t)INT64_MAX;
    int overflow = 0;
    for (int64_t i = from; i < to; i++) {
        uint8_t c = data[i];
        if (c == '_') {
            if (last_underscore) {
                return 1;
            }
            last_underscore = 1;
            continue;
        }
        int64_t digit = c >= '0' && c <= '9'   ? c - '0'
                        : c >= 'a' && c <= 'z' ? c - 'a' + 10
                        : c >= 'A' && c <= 'Z' ? c - 'A' + 10
                                               : 99;
        if (digit >= base) {
            return 1;
        }
        if (digit != 0) {
            nonzero = 1;
        }
        if (!overflow) {
            if (magnitude > (limit - (uint64_t)digit) / (uint64_t)base) {
                overflow = 1;
            } else {
                magnitude = magnitude * (uint64_t)base + (uint64_t)digit;
            }
        }
        digits++;
        last_underscore = 0;
    }
    if (digits == 0 || last_underscore) {
        return 1;
    }
    if (leading_zero_only && nonzero) {
        return 1;
    }
    if (overflow) {
        return 2;
    }
    *out = negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
    return 0;
}

/* `float(s)`: 1 with the value in `*out`, 0 where Python raises or reads
   with its tables. */
int64_t ppy_str_to_float(int8_t *handle, double *out) {
    uint8_t *data = ppy_str_raw(handle);
    int64_t from = 0;
    int64_t to = ppy_str_bytes(handle);
    int64_t width = 0;
    while (from < to && ppy_str_space(ppy_str_decode(data, from, &width))) {
        from += width;
    }
    while (to > from && ppy_str_space(ppy_str_decode(data, ppy_str_back(data, to), &width))) {
        to = ppy_str_back(data, to);
    }
    int64_t length = to - from;
    const uint8_t *text = data + from;
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
    if (rest == 3 || rest == 8) {
        char lower[9];
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
}

/* -- format specs ------------------------------------------------------- */

/* A format spec, `[[fill]align][sign][z][#][0][width][grouping][.precision][type]`,
   read into `spec`: [0] fill [1] align [2] sign [3] z [4] alternate [5] width
   [6] grouping [7] precision (-1 none) [8] type (0 none) [9] zero flag.
   0 for a spec this runtime does not write, whose call stays in Python. */
int64_t ppy_str_spec(const int8_t *text, int64_t length, int64_t *spec) {
    const uint8_t *s = (const uint8_t *)text;
    int64_t at = 0;
    spec[0] = ' ';
    spec[1] = 0;
    spec[2] = '-';
    spec[3] = 0;
    spec[4] = 0;
    spec[5] = 0;
    spec[6] = 0;
    spec[7] = -1;
    spec[8] = 0;
    spec[9] = 0;
    for (int64_t i = 0; i < length; i++) {
        if (s[i] >= 0x80) {
            return 0;
        }
    }
    if (length >= 2 && (s[1] == '<' || s[1] == '>' || s[1] == '^' || s[1] == '=')) {
        spec[0] = s[0];
        spec[1] = s[1];
        at = 2;
    } else if (length >= 1 && (s[0] == '<' || s[0] == '>' || s[0] == '^' || s[0] == '=')) {
        spec[1] = s[0];
        at = 1;
    }
    if (at < length && (s[at] == '+' || s[at] == '-' || s[at] == ' ')) {
        spec[2] = s[at++];
    }
    if (at < length && s[at] == 'z') {
        spec[3] = 1;
        at++;
    }
    if (at < length && s[at] == '#') {
        spec[4] = 1;
        at++;
    }
    if (at < length && s[at] == '0') {
        spec[9] = 1;
        if (spec[1] == 0) {
            spec[0] = '0';
            spec[1] = '=';
        }
        at++;
    }
    while (at < length && s[at] >= '0' && s[at] <= '9') {
        if (spec[5] > 100000) {
            return 0;
        }
        spec[5] = spec[5] * 10 + (s[at++] - '0');
    }
    if (at < length && (s[at] == ',' || s[at] == '_')) {
        spec[6] = s[at++];
    }
    if (at < length && s[at] == '.') {
        at++;
        if (at >= length || s[at] < '0' || s[at] > '9') {
            return 0;
        }
        spec[7] = 0;
        while (at < length && s[at] >= '0' && s[at] <= '9') {
            if (spec[7] > 100000) {
                return 0;
            }
            spec[7] = spec[7] * 10 + (s[at++] - '0');
        }
    }
    if (at < length) {
        spec[8] = s[at++];
    }
    return at == length;
}

/* The digits `digits` (with no sign) grouped every `every` by `mark`, and
   zero-filled to `least` characters when `least` is not 0, as CPython's
   grouping fills them; into the builder. */
void ppy_str_add_grouped(int8_t *builder, const char *digits, int64_t count, int64_t mark,
                         int64_t every, int64_t least) {
    if (count == 0 && least <= 0) {
        return;
    }
    char *out = (char *)malloc((size_t)(count * 2 + least * 2 + 8));
    if (out == NULL) {
        ppy_coll_fail();
    }
    int64_t used = 0;
    int64_t remaining = count;
    int64_t min_width = least;
    while (1) {
        int64_t take = remaining > min_width ? remaining : min_width;
        take = take > 1 ? take : 1;
        take = take < every ? take : every;
        if (mark == 0) {
            take = remaining > min_width ? remaining : min_width;
            take = take > 1 ? take : 1;
        }
        int64_t zeros = take - remaining > 0 ? take - remaining : 0;
        int64_t chars = take - zeros;
        for (int64_t i = 0; i < chars; i++) {
            out[used++] = digits[remaining - 1 - i];
        }
        for (int64_t i = 0; i < zeros; i++) {
            out[used++] = '0';
        }
        min_width -= take;
        remaining -= chars;
        if (remaining <= 0 && min_width <= 0) {
            break;
        }
        if (mark == 0) {
            break;
        }
        out[used++] = (char)mark;
        min_width -= 1;
    }
    for (int64_t i = 0, j = used - 1; i < j; i++, j--) {
        char swap = out[i];
        out[i] = out[j];
        out[j] = swap;
    }
    ppy_str_add_bytes(builder, (const int8_t *)out, used);
    free(out);
}

/* A number laid out by `spec`: its sign, a prefix, its digits (grouped),
   the rest (a fraction, an exponent), padded to the width. */
void ppy_str_add_number(int8_t *builder, const int64_t *spec, int negative, const char *prefix,
                        const char *digits, int64_t count, const char *rest, int64_t every) {
    char sign[2] = {0, 0};
    int64_t signs = 0;
    if (negative) {
        sign[0] = '-';
        signs = 1;
    } else if (spec[2] == '+' || spec[2] == ' ') {
        sign[0] = (char)spec[2];
        signs = 1;
    }
    int64_t prefixes = (int64_t)strlen(prefix);
    int64_t rests = (int64_t)strlen(rest);
    int64_t align = spec[1] != 0 ? spec[1] : '>';
    int64_t grouped = count;
    if (spec[6] != 0 && count > 0) {
        grouped = count + (count - 1) / every;
    }
    int64_t body = signs + prefixes + grouped + rests;
    int64_t spare = spec[5] > body ? spec[5] - body : 0;
    if (align == '=' && spec[0] == '0' && spare > 0) {
        ppy_str_add_bytes(builder, (const int8_t *)sign, signs);
        ppy_str_add_bytes(builder, (const int8_t *)prefix, prefixes);
        int64_t least = spec[5] - signs - prefixes - rests;
        ppy_str_add_grouped(builder, digits, count, spec[6], every, least);
        ppy_str_add_bytes(builder, (const int8_t *)rest, rests);
        return;
    }
    int64_t left = align == '<' ? 0 : align == '^' ? spare / 2 : spare;
    if (align == '=') {
        ppy_str_add_bytes(builder, (const int8_t *)sign, signs);
        ppy_str_add_bytes(builder, (const int8_t *)prefix, prefixes);
        ppy_str_add_repeat(builder, spec[0], spare);
        ppy_str_add_grouped(builder, digits, count, spec[6], every, 0);
        ppy_str_add_bytes(builder, (const int8_t *)rest, rests);
        return;
    }
    ppy_str_add_repeat(builder, spec[0], left);
    ppy_str_add_bytes(builder, (const int8_t *)sign, signs);
    ppy_str_add_bytes(builder, (const int8_t *)prefix, prefixes);
    ppy_str_add_grouped(builder, digits, count, spec[6], every, 0);
    ppy_str_add_bytes(builder, (const int8_t *)rest, rests);
    ppy_str_add_repeat(builder, spec[0], spare - left);
}

/* `format(value, spec)` of a float into the builder; 0 where it stays Python's. */
int64_t ppy_str_format_float(int8_t *builder, double value, const int8_t *text, int64_t length) {
    int64_t spec[10];
    if (!ppy_str_spec(text, length, spec)) {
        return 0;
    }
    int64_t type = spec[8];
    if (type != 0 && type != 'e' && type != 'E' && type != 'f' && type != 'F' && type != 'g'
        && type != 'G' && type != '%') {
        return 0;
    }
    if (spec[4]) {
        return 0;
    }
    int negative = value < 0 || (value == 0.0 && 1.0 / value < 0);
    double magnitude = negative ? -value : value;
    if (value != value) {
        negative = 0;
    }
    char body[1024];
    int64_t precision = spec[7];
    if (value != value || magnitude > DBL_MAX) {
        const char *word = value != value ? "nan" : "inf";
        if (type == 'E' || type == 'F' || type == 'G') {
            word = value != value ? "NAN" : "INF";
        }
        snprintf(body, sizeof body, "%s%s", word, type == '%' ? "%" : "");
    } else if (type == 0 && precision < 0) {
        int64_t used = ppy_str_float_text(magnitude, body);
        body[used] = 0;
    } else {
        if (precision > 100) {
            return 0;
        }
        int64_t p = precision < 0 ? 6 : precision;
        double shown = type == '%' ? magnitude * 100.0 : magnitude;
        char letter = type == '%' ? 'f' : (char)type;
        if (type == 0) {
            /* Like `g`, but scientific from an exponent of p - 1, and a
               fixed result keeps a digit after the point. */
            if (p == 0) {
                p = 1;
            }
            char probe[64];
            snprintf(probe, sizeof probe, "%.*e", (int)(p - 1), shown);
            int64_t exponent = atoi(strchr(probe, 'e') + 1);
            if (exponent >= -4 && exponent < p - 1) {
                snprintf(body, sizeof body, "%.*f", (int)(p - 1 - exponent), shown);
                char *point = strchr(body, '.');
                if (point != NULL) {
                    char *last = body + strlen(body) - 1;
                    while (last > point + 1 && *last == '0') {
                        *last-- = 0;
                    }
                } else {
                    strcat(body, ".0");
                }
            } else {
                char *mark = strchr(probe, 'e');
                char *last = mark - 1;
                while (*last == '0') {
                    last--;
                }
                if (*last == '.') {
                    last--;
                }
                int64_t kept = (int64_t)(last - probe + 1);
                memcpy(body, probe, (size_t)kept);
                strcpy(body + kept, mark);
            }
        } else {
            char format[8];
            snprintf(format, sizeof format, "%%.%d%c", (int)p, letter);
            snprintf(body, sizeof body, format, shown);
        }
        if (type == '%') {
            strcat(body, "%");
        }
    }
    int finite = value == value && magnitude <= DBL_MAX;
    if (spec[3] && negative && finite) {
        int zero = 1;
        for (char *c = body; *c; c++) {
            if (*c >= '1' && *c <= '9') {
                zero = 0;
            }
        }
        if (zero) {
            negative = 0;
        }
    }
    /* The digits before the point are what grouping groups. */
    int64_t count = 0;
    while (body[count] >= '0' && body[count] <= '9') {
        count++;
    }
    char digits[1024];
    memcpy(digits, body, (size_t)count);
    if (!finite) {
        int64_t plain[10];
        memcpy(plain, spec, sizeof plain);
        plain[6] = 0;
        ppy_str_add_number(builder, plain, negative, "", digits, count, body + count, 3);
        return 1;
    }
    ppy_str_add_number(builder, spec, negative, "", digits, count, body + count, 3);
    return 1;
}

/* `format(value, spec)` of an int into the builder; 0 where it stays Python's. */
int64_t ppy_str_format_int(int8_t *builder, int64_t value, const int8_t *text, int64_t length) {
    int64_t spec[10];
    if (!ppy_str_spec(text, length, spec)) {
        return 0;
    }
    int64_t type = spec[8];
    if (type == 'e' || type == 'E' || type == 'f' || type == 'F' || type == 'g' || type == 'G'
        || type == '%') {
        return ppy_str_format_float(builder, (double)value, text, length);
    }
    if (spec[7] >= 0 || spec[3]) {
        return 0;
    }
    int64_t base = type == 'b' ? 2 : type == 'o' ? 8 : type == 'x' || type == 'X' ? 16 : 10;
    if (type != 0 && type != 'd' && base == 10) {
        return 0;
    }
    if (spec[6] == ',' && base != 10) {
        return 0;
    }
    char digits[72];
    int64_t count = 0;
    uint64_t magnitude = value < 0 ? (uint64_t)0 - (uint64_t)value : (uint64_t)value;
    const char *alphabet = type == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
    do {
        digits[count++] = alphabet[magnitude % (uint64_t)base];
        magnitude /= (uint64_t)base;
    } while (magnitude > 0);
    for (int64_t i = 0, j = count - 1; i < j; i++, j--) {
        char swap = digits[i];
        digits[i] = digits[j];
        digits[j] = swap;
    }
    const char *prefix = "";
    if (spec[4]) {
        prefix = base == 2 ? "0b" : base == 8 ? "0o" : type == 'X' ? "0X" : base == 16 ? "0x" : "";
    }
    ppy_str_add_number(builder, spec, value < 0, prefix, digits, count, "", base == 10 ? 3 : 4);
    return 1;
}

/* `format(value, spec)` of a string into the builder; 0 where it stays Python's. */
int64_t ppy_str_format_str(int8_t *builder, int8_t *handle, const int8_t *text, int64_t length) {
    int64_t spec[10];
    if (!ppy_str_spec(text, length, spec)) {
        return 0;
    }
    if ((spec[8] != 0 && spec[8] != 's') || spec[2] != '-' || spec[4] || spec[6] || spec[3]
        || spec[1] == '=' || spec[9]) {
        return 0;
    }
    int64_t shown = ppy_str_len(handle);
    int64_t bytes = ppy_str_bytes(handle);
    if (spec[7] >= 0 && spec[7] < shown) {
        shown = spec[7];
        bytes = ppy_str_offset(handle, shown);
    }
    int64_t spare = spec[5] > shown ? spec[5] - shown : 0;
    int64_t align = spec[1] != 0 ? spec[1] : '<';
    int64_t left = align == '<' ? 0 : align == '^' ? spare / 2 : spare;
    ppy_str_add_repeat(builder, spec[0], left);
    ppy_str_add_bytes(builder, (const int8_t *)ppy_str_raw(handle), bytes);
    ppy_str_add_repeat(builder, spec[0], spare - left);
    return 1;
}

/* -- keys ----------------------------------------------------------------- */

/* A key's words hashed, where `text` marks the words that are strings. */
int64_t ppy_str_key_hash(const int64_t *key, int64_t keys, int64_t text) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int64_t w = 0; w < keys; w++) {
        uint64_t word = (uint64_t)key[w];
        if ((text >> w) & 1) {
            word = (uint64_t)ppy_str_hash((int8_t *)(intptr_t)key[w]);
        }
        uint64_t z = word + h;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        h = z ^ (z >> 31);
    }
    return (int64_t)h;
}

/* Two keys in order, -1, 0, or 1: strings by their text. */
int64_t ppy_str_key_order(const int64_t *a, const int64_t *b, int64_t keys, int64_t text) {
    for (int64_t w = 0; w < keys; w++) {
        if ((text >> w) & 1) {
            int64_t order = ppy_str_order((int8_t *)(intptr_t)a[w], (int8_t *)(intptr_t)b[w]);
            if (order != 0) {
                return order;
            }
            continue;
        }
        if (a[w] != b[w]) {
            return a[w] < b[w] ? -1 : 1;
        }
    }
    return 0;
}
