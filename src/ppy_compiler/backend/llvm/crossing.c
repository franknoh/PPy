/* Containers across the Python boundary, in the generated wrapper.

   A native function that takes a `list`, a `dict`, or a `set` works on a
   handle into the collections runtime (`ppy_runtime/collections.c`). This
   is the crossing a generated wrapper does for it: each argument copied
   into native memory, the result copied out, and, when the function writes
   through a parameter, every container that came in copied back into the
   caller's object, which stays the same object. It does what
   `ppy_runtime/collection_boundary.py` does, without a Python frame.

   Identity is kept both ways. An object met twice among the arguments, or
   inside them, is one handle with two references, so `f(xs, xs)` and the
   shared rows of `[[0] * n] * m` stay shared; a handle that came from an
   object comes back as that object. An element is written back only where
   it changed, so an element the call left alone keeps its identity.

   Nothing here runs Python code: the checks are exact type tests, and the
   only objects compared are ints, floats, and strings. A value of the
   wrong type refuses the crossing, and the Python body runs instead.

   The runtime's functions are reached through `ppy_rt`, which the binder
   fills (`ppy_runtime`) with the addresses of the runtime the native code
   uses: the handles must come from the same heap. */

#include <string.h>

#define PX_INT 1
#define PX_FLOAT 2
#define PX_BOOL 3
#define PX_STR 4
#define PX_TUPLE 5
#define PX_LIST 6
#define PX_DICT 7
#define PX_SET 8

/* One type at the boundary: a scalar, a string, a tuple of scalars (`part`
   spells each word: 'i', 'f', or 'b'), or a container of them. */
typedef struct ppy_xs {
    int kind;
    int parts;
    const char *part;
    const struct ppy_xs *key;
    const struct ppy_xs *value;
} ppy_xs;

typedef struct {
    int8_t *(*seq_new)(int64_t, int64_t, int64_t, int64_t);
    int8_t *(*map_new)(int64_t, int64_t, int64_t, int64_t);
    void (*push_many)(int8_t *, const int8_t *, int64_t);
    void (*put_many)(int8_t *, const int8_t *, const int8_t *, int64_t);
    void (*copy_out)(int8_t *, int8_t *, int8_t *);
    int64_t (*len)(int8_t *);
    void (*retain)(int8_t *);
    void (*release)(int8_t *);
    void (*text_keys)(int8_t *, int64_t);
    void (*str_new_many)(const int8_t *, const int64_t *, int64_t, int64_t *);
} ppy_rt_fns;

static ppy_rt_fns ppy_rt;
static int ppy_rt_ready = 0;

/* `ppy_runtime(addresses)`: the runtime's functions, in `ppy_rt`'s order. */
static PyObject *ppy_runtime(PyObject *self, PyObject *args) {
    unsigned long long a[10];
    (void)self;
    if (!PyArg_ParseTuple(args, "KKKKKKKKKK", &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6],
                          &a[7], &a[8], &a[9])) {
        return NULL;
    }
    for (int i = 0; i < 10; i++) {
        if (a[i] == 0) {
            ppy_rt_ready = 0;
            Py_RETURN_FALSE;
        }
    }
    *(void **)(&ppy_rt.seq_new) = (void *)(uintptr_t)a[0];
    *(void **)(&ppy_rt.map_new) = (void *)(uintptr_t)a[1];
    *(void **)(&ppy_rt.push_many) = (void *)(uintptr_t)a[2];
    *(void **)(&ppy_rt.put_many) = (void *)(uintptr_t)a[3];
    *(void **)(&ppy_rt.copy_out) = (void *)(uintptr_t)a[4];
    *(void **)(&ppy_rt.len) = (void *)(uintptr_t)a[5];
    *(void **)(&ppy_rt.retain) = (void *)(uintptr_t)a[6];
    *(void **)(&ppy_rt.release) = (void *)(uintptr_t)a[7];
    *(void **)(&ppy_rt.text_keys) = (void *)(uintptr_t)a[8];
    *(void **)(&ppy_rt.str_new_many) = (void *)(uintptr_t)a[9];
    ppy_rt_ready = 1;
    Py_RETURN_TRUE;
}

/* -- one call's crossing ------------------------------------------------- */

/* A container of the call: the Python object and its handle. `incoming` is
   one that came in with the arguments, which a call that writes copies
   back; `synced` is one copied out already. The object is held. */
typedef struct {
    PyObject *obj;
    int8_t *handle;
    const ppy_xs *spec;
    int incoming;
    int synced;
} px_entry;

#define PX_INLINE 16

typedef struct {
    px_entry *entries;
    Py_ssize_t count, room;
    /* Open addressing over entry indices plus one, by object and by handle. */
    Py_ssize_t *by_obj, *by_handle;
    Py_ssize_t slots;
    /* The references the call holds, let go of at the end. */
    int8_t **owned;
    Py_ssize_t owned_count, owned_room;
    px_entry entries0[PX_INLINE];
    Py_ssize_t by_obj0[PX_INLINE * 2], by_handle0[PX_INLINE * 2];
    int8_t *owned0[PX_INLINE * 2];
} ppy_cross;

static void px_begin(ppy_cross *x) {
    x->entries = x->entries0;
    x->count = 0;
    x->room = PX_INLINE;
    x->by_obj = x->by_obj0;
    x->by_handle = x->by_handle0;
    x->slots = PX_INLINE * 2;
    memset(x->by_obj0, 0, sizeof(x->by_obj0));
    memset(x->by_handle0, 0, sizeof(x->by_handle0));
    x->owned = x->owned0;
    x->owned_count = 0;
    x->owned_room = PX_INLINE * 2;
}

/* Every reference the call held let go of, and the objects it held. */
static void px_end(ppy_cross *x) {
    for (Py_ssize_t i = 0; i < x->owned_count; i++) {
        ppy_rt.release(x->owned[i]);
    }
    x->owned_count = 0;
    for (Py_ssize_t i = 0; i < x->count; i++) {
        Py_DECREF(x->entries[i].obj);
    }
    x->count = 0;
    if (x->owned != x->owned0) {
        PyMem_Free(x->owned);
        x->owned = x->owned0;
    }
    if (x->entries != x->entries0) {
        PyMem_Free(x->entries);
        x->entries = x->entries0;
    }
    if (x->by_obj != x->by_obj0) {
        PyMem_Free(x->by_obj);
        PyMem_Free(x->by_handle);
        x->by_obj = x->by_obj0;
        x->by_handle = x->by_handle0;
    }
}

static int px_own(ppy_cross *x, int8_t *handle) {
    if (x->owned_count == x->owned_room) {
        Py_ssize_t room = x->owned_room * 2;
        int8_t **grown = (int8_t **)PyMem_Malloc((size_t)room * sizeof(int8_t *));
        if (grown == NULL) {
            return -1;
        }
        memcpy(grown, x->owned, (size_t)x->owned_count * sizeof(int8_t *));
        if (x->owned != x->owned0) {
            PyMem_Free(x->owned);
        }
        x->owned = grown;
        x->owned_room = room;
    }
    x->owned[x->owned_count++] = handle;
    return 0;
}

static size_t px_hash(const void *pointer, Py_ssize_t slots) {
    return (size_t)((((uintptr_t)pointer) >> 4) * 0x9E3779B97F4A7C15ULL) & (size_t)(slots - 1);
}

static void px_index(ppy_cross *x, Py_ssize_t at) {
    size_t i = px_hash(x->entries[at].obj, x->slots);
    while (x->by_obj[i] != 0) {
        i = (i + 1) & (size_t)(x->slots - 1);
    }
    x->by_obj[i] = at + 1;
    i = px_hash(x->entries[at].handle, x->slots);
    while (x->by_handle[i] != 0) {
        i = (i + 1) & (size_t)(x->slots - 1);
    }
    x->by_handle[i] = at + 1;
}

static px_entry *px_by_obj(ppy_cross *x, PyObject *obj) {
    size_t i = px_hash(obj, x->slots);
    while (x->by_obj[i] != 0) {
        px_entry *e = &x->entries[x->by_obj[i] - 1];
        if (e->obj == obj) {
            return e;
        }
        i = (i + 1) & (size_t)(x->slots - 1);
    }
    return NULL;
}

static px_entry *px_by_handle(ppy_cross *x, int8_t *handle) {
    size_t i = px_hash(handle, x->slots);
    while (x->by_handle[i] != 0) {
        px_entry *e = &x->entries[x->by_handle[i] - 1];
        if (e->handle == handle) {
            return e;
        }
        i = (i + 1) & (size_t)(x->slots - 1);
    }
    return NULL;
}

/* A container met for the first time: its index, or -1 out of memory. */
static Py_ssize_t px_add(ppy_cross *x, PyObject *obj, int8_t *handle, const ppy_xs *spec,
                         int incoming) {
    if (x->count == x->room) {
        Py_ssize_t room = x->room * 2;
        px_entry *grown = (px_entry *)PyMem_Malloc((size_t)room * sizeof(px_entry));
        if (grown == NULL) {
            return -1;
        }
        memcpy(grown, x->entries, (size_t)x->count * sizeof(px_entry));
        if (x->entries != x->entries0) {
            PyMem_Free(x->entries);
        }
        x->entries = grown;
        x->room = room;
    }
    if ((x->count + 1) * 2 > x->slots) {
        Py_ssize_t slots = x->slots * 2;
        Py_ssize_t *by_obj = (Py_ssize_t *)PyMem_Calloc((size_t)slots, sizeof(Py_ssize_t));
        Py_ssize_t *by_handle = (Py_ssize_t *)PyMem_Calloc((size_t)slots, sizeof(Py_ssize_t));
        if (by_obj == NULL || by_handle == NULL) {
            PyMem_Free(by_obj);
            PyMem_Free(by_handle);
            return -1;
        }
        if (x->by_obj != x->by_obj0) {
            PyMem_Free(x->by_obj);
            PyMem_Free(x->by_handle);
        }
        x->by_obj = by_obj;
        x->by_handle = by_handle;
        x->slots = slots;
        for (Py_ssize_t at = 0; at < x->count; at++) {
            px_index(x, at);
        }
    }
    Py_ssize_t at = x->count++;
    px_entry *e = &x->entries[at];
    Py_INCREF(obj);
    e->obj = obj;
    e->handle = handle;
    e->spec = spec;
    e->incoming = incoming;
    e->synced = 0;
    px_index(x, at);
    return at;
}

/* -- the layout of an element ------------------------------------------- */

static int64_t px_words(const ppy_xs *s) {
    return s->kind == PX_TUPLE ? s->parts : 1;
}

static int64_t px_floats(const ppy_xs *s) {
    if (s->kind == PX_TUPLE) {
        int64_t mask = 0;
        for (int i = 0; i < s->parts; i++) {
            if (s->part[i] == 'f') {
                mask |= (int64_t)1 << i;
            }
        }
        return mask;
    }
    return s->kind == PX_FLOAT ? 1 : 0;
}

static int px_reference(const ppy_xs *s) {
    return s->kind == PX_STR || s->kind >= PX_LIST;
}

/* The handle mask a collection of these takes: a string is a leaf too. */
static int64_t px_handles(const ppy_xs *s) {
    if (!px_reference(s)) {
        return 0;
    }
    return s->kind == PX_STR ? (int64_t)1 | ((int64_t)1 << 32) : 1;
}

/* -- in ------------------------------------------------------------------ */

/* 0 done, -1 refused (the Python body runs), -2 a Python error is set. */
static int px_in(ppy_cross *x, PyObject *o, const ppy_xs *s, int8_t **out);

static inline int px_scalar(PyObject *o, char kind, int64_t *word) {
    if (kind == 'f') {
        if (!PyFloat_CheckExact(o)) {
            return -1;
        }
        double value = PyFloat_AS_DOUBLE(o);
        memcpy(word, &value, 8);
        return 0;
    }
    if (kind == 'b') {
        if (!PyBool_Check(o)) {
            return -1;
        }
        *word = o == Py_True;
        return 0;
    }
    if (!PyLong_CheckExact(o)) {
        return -1;
    }
#if PY_VERSION_HEX >= 0x030C0000
    if (PyUnstable_Long_IsCompact((PyLongObject *)o)) {
        *word = (int64_t)PyUnstable_Long_CompactValue((PyLongObject *)o);
        return 0;
    }
#endif
    int overflow = 0;
    long long value = PyLong_AsLongLongAndOverflow(o, &overflow);
    if (overflow != 0 || (value == -1 && PyErr_Occurred())) {
        PyErr_Clear();
        return -1;
    }
    *word = (int64_t)value;
    return 0;
}

/* One element as its words; a handle made for it carries one reference. */
static int px_word(ppy_cross *x, PyObject *o, const ppy_xs *s, int64_t *words) {
    switch (s->kind) {
    case PX_INT:
        return px_scalar(o, 'i', words);
    case PX_FLOAT:
        return px_scalar(o, 'f', words);
    case PX_BOOL:
        return px_scalar(o, 'b', words);
    case PX_TUPLE:
        if (!PyTuple_CheckExact(o) || PyTuple_GET_SIZE(o) != s->parts) {
            return -1;
        }
        for (int i = 0; i < s->parts; i++) {
            if (px_scalar(PyTuple_GET_ITEM(o, i), s->part[i], words + i) != 0) {
                return -1;
            }
        }
        return 0;
    case PX_STR: {
        if (!PyUnicode_CheckExact(o)) {
            return -1;
        }
        Py_ssize_t size = 0;
        const char *data = PyUnicode_AsUTF8AndSize(o, &size);
        if (data == NULL) {
            /* A lone surrogate has no UTF-8. */
            PyErr_Clear();
            return -1;
        }
        int64_t length = (int64_t)size;
        ppy_rt.str_new_many((const int8_t *)data, &length, 1, words);
        return 0;
    }
    default: {
        int8_t *handle = NULL;
        int done = px_in(x, o, s, &handle);
        if (done == 0) {
            words[0] = (int64_t)(intptr_t)handle;
        }
        return done;
    }
    }
}

/* Let go of the handles placed in `count` elements' words so far. */
static void px_drop(const ppy_xs *s, const int64_t *words, Py_ssize_t count) {
    if (s == NULL || !px_reference(s)) {
        return;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        ppy_rt.release((int8_t *)(intptr_t)words[i]);
    }
}

static int px_fill(ppy_cross *x, int8_t *handle, PyObject *o, const ppy_xs *s) {
    int64_t small[64];
    const ppy_xs *v = s->value;
    const ppy_xs *k = s->key;
    int64_t w = v != NULL ? px_words(v) : 0;
    int64_t kw = k != NULL ? px_words(k) : 0;
    if (s->kind == PX_LIST) {
        /* The sequence was made as long as the list, its words zero: each
           element is written in place, and a handle written is the
           sequence's, let go of with it if the rest is refused. */
        Py_ssize_t n = PyList_GET_SIZE(o);
        int64_t *records = (int64_t *)(intptr_t)((int64_t *)handle)[2];
        PyObject **items = ((PyListObject *)o)->ob_item;
        if (v->kind == PX_INT || v->kind == PX_FLOAT) {
            /* Numbers, the common case, checked and copied in one loop. */
            char kind = v->kind == PX_INT ? 'i' : 'f';
            for (Py_ssize_t i = 0; i < n; i++) {
                if (px_scalar(items[i], kind, records + i) != 0) {
                    return -1;
                }
            }
            return 0;
        }
        for (Py_ssize_t i = 0; i < n; i++) {
            int done = px_word(x, items[i], v, records + i * w);
            if (done != 0) {
                return done;
            }
        }
        return 0;
    }
    Py_ssize_t n = s->kind == PX_DICT ? PyDict_GET_SIZE(o) : PySet_GET_SIZE(o);
    if (n == 0) {
        return 0;
    }
    size_t need = (size_t)n * (size_t)(w + kw);
    int64_t *buffer = need <= 64 ? small : (int64_t *)PyMem_Malloc(need * 8);
    if (buffer == NULL) {
        PyErr_NoMemory();
        return -2;
    }
    int64_t *keys = buffer;
    int64_t *values = buffer + (size_t)n * (size_t)kw;
    int done = 0;
    Py_ssize_t made = 0;
    {
        PyObject *key = NULL, *item = NULL;
        Py_ssize_t keys_made = 0;
        if (s->kind == PX_DICT) {
            Py_ssize_t position = 0;
            while (PyDict_Next(o, &position, &key, &item)) {
                done = px_word(x, key, k, keys + keys_made * kw);
                if (done != 0) {
                    break;
                }
                keys_made++;
                done = px_word(x, item, v, values + made * w);
                if (done != 0) {
                    break;
                }
                made++;
            }
        } else {
            PyObject *walk = PyObject_GetIter(o);
            if (walk == NULL) {
                done = -2;
            }
            while (done == 0 && (key = PyIter_Next(walk)) != NULL) {
                done = px_word(x, key, k, keys + keys_made * kw);
                Py_DECREF(key);
                if (done == 0) {
                    keys_made++;
                }
            }
            if (done == 0 && PyErr_Occurred()) {
                done = -2;
            }
            Py_XDECREF(walk);
        }
        if (done == 0 && keys_made != n) {
            done = -1;
        }
        if (done != 0) {
            px_drop(k, keys, keys_made);
            px_drop(v, values, made);
        } else {
            ppy_rt.put_many(handle, (const int8_t *)keys, (const int8_t *)values, (int64_t)n);
            /* A map holds its own reference to each string key it keeps. */
            px_drop(k, keys, n);
        }
    }
    if (buffer != small) {
        PyMem_Free(buffer);
    }
    return done;
}

/* Whether two types are one: an object met again as another type has
   another layout, and is refused. */
static int px_same_spec(const ppy_xs *a, const ppy_xs *b) {
    if (a == b) {
        return 1;
    }
    if (a == NULL || b == NULL || a->kind != b->kind || a->parts != b->parts) {
        return 0;
    }
    if (a->parts > 0 && memcmp(a->part, b->part, (size_t)a->parts) != 0) {
        return 0;
    }
    return px_same_spec(a->key, b->key) && px_same_spec(a->value, b->value);
}

static int px_in(ppy_cross *x, PyObject *o, const ppy_xs *s, int8_t **out) {
    px_entry *seen = px_by_obj(x, o);
    if (seen != NULL) {
        if (!px_same_spec(seen->spec, s)) {
            return -1;
        }
        ppy_rt.retain(seen->handle);
        *out = seen->handle;
        return 0;
    }
    if (s->kind == PX_LIST ? !PyList_CheckExact(o)
        : s->kind == PX_DICT ? !PyDict_CheckExact(o)
                             : !Py_IS_TYPE(o, &PySet_Type)) {
        return -1;
    }
    const ppy_xs *v = s->value;
    int64_t words = v != NULL ? px_words(v) : 0;
    int64_t floats = v != NULL ? px_floats(v) : 0;
    int64_t handles = v != NULL ? px_handles(v) : 0;
    int8_t *handle = s->kind == PX_LIST
                         ? ppy_rt.seq_new((int64_t)PyList_GET_SIZE(o), words, floats, handles)
                         : ppy_rt.map_new(px_words(s->key), words, floats, handles);
    if (s->kind != PX_LIST && s->key->kind == PX_STR) {
        ppy_rt.text_keys(handle, 1);
    }
    /* The call holds its own reference to every handle it made, so none is
       freed, and its address given to something new, while the call runs. */
    if (px_add(x, o, handle, s, 1) < 0 || px_own(x, handle) < 0) {
        ppy_rt.release(handle);
        PyErr_NoMemory();
        return -2;
    }
    ppy_rt.retain(handle);
    int done = px_fill(x, handle, o, s);
    if (done != 0) {
        ppy_rt.release(handle);
        return done;
    }
    *out = handle;
    return 0;
}

/* A top-level argument: a handle the call owns. */
static int px_argument(ppy_cross *x, PyObject *o, const ppy_xs *s, int8_t **out) {
    int done = px_in(x, o, s, out);
    if (done == 0 && px_own(x, *out) < 0) {
        ppy_rt.release(*out);
        PyErr_NoMemory();
        return -2;
    }
    return done;
}

/* -- out ----------------------------------------------------------------- */

static PyObject *px_out(ppy_cross *x, int8_t *handle, const ppy_xs *s, int rewrite);

static PyObject *px_scalar_value(int64_t word, char kind) {
    if (kind == 'f') {
        double value;
        memcpy(&value, &word, 8);
        return PyFloat_FromDouble(value);
    }
    if (kind == 'b') {
        return PyBool_FromLong(word != 0);
    }
    return PyLong_FromLongLong((long long)word);
}

/* Whether `old` already is the value of these words: then it stays, and so
   does its identity. Only numbers, strings, and tuples of numbers are asked. */
static int px_holds(PyObject *old, const int64_t *words, const ppy_xs *s) {
    switch (s->kind) {
    case PX_INT:
    case PX_FLOAT:
    case PX_BOOL: {
        int64_t mine = 0;
        char kind = s->kind == PX_INT ? 'i' : s->kind == PX_FLOAT ? 'f' : 'b';
        return px_scalar(old, kind, &mine) == 0 && mine == words[0];
    }
    case PX_TUPLE: {
        if (!PyTuple_CheckExact(old) || PyTuple_GET_SIZE(old) != s->parts) {
            return 0;
        }
        for (int i = 0; i < s->parts; i++) {
            int64_t mine = 0;
            if (px_scalar(PyTuple_GET_ITEM(old, i), s->part[i], &mine) != 0 ||
                mine != words[i]) {
                return 0;
            }
        }
        return 1;
    }
    case PX_STR: {
        if (!PyUnicode_CheckExact(old)) {
            return 0;
        }
        const int64_t *header = (const int64_t *)(intptr_t)words[0];
        Py_ssize_t size = 0;
        const char *data = PyUnicode_AsUTF8AndSize(old, &size);
        if (data == NULL) {
            PyErr_Clear();
            return 0;
        }
        return (int64_t)size == header[0] &&
               memcmp(data, (const void *)(intptr_t)header[2], (size_t)size) == 0;
    }
    default:
        return 0;
    }
}

/* One element's Python value, a new reference. */
static PyObject *px_value(ppy_cross *x, const int64_t *words, const ppy_xs *s, int rewrite) {
    switch (s->kind) {
    case PX_INT:
        return PyLong_FromLongLong((long long)words[0]);
    case PX_FLOAT:
        return px_scalar_value(words[0], 'f');
    case PX_BOOL:
        return PyBool_FromLong(words[0] != 0);
    case PX_TUPLE: {
        PyObject *made = PyTuple_New(s->parts);
        if (made == NULL) {
            return NULL;
        }
        for (int i = 0; i < s->parts; i++) {
            PyObject *part = px_scalar_value(words[i], s->part[i]);
            if (part == NULL) {
                Py_DECREF(made);
                return NULL;
            }
            PyTuple_SET_ITEM(made, i, part);
        }
        return made;
    }
    case PX_STR: {
        const int64_t *header = (const int64_t *)(intptr_t)words[0];
        return PyUnicode_DecodeUTF8((const char *)(intptr_t)header[2], (Py_ssize_t)header[0],
                                    NULL);
    }
    default:
        return px_out(x, (int8_t *)(intptr_t)words[0], s, rewrite);
    }
}

/* `old` where it already holds the value, else the value made: a new reference. */
static PyObject *px_kept(ppy_cross *x, PyObject *old, const int64_t *words, const ppy_xs *s,
                         int rewrite) {
    if (old != NULL && px_holds(old, words, s)) {
        Py_INCREF(old);
        return old;
    }
    return px_value(x, words, s, rewrite);
}

static int px_rewrite_list(ppy_cross *x, PyObject *made, const int64_t *values, Py_ssize_t n,
                           const ppy_xs *v, int64_t w, int rewrite) {
    if (PyList_GET_SIZE(made) == n && (v->kind == PX_INT || v->kind == PX_FLOAT)) {
        char kind = v->kind == PX_INT ? 'i' : 'f';
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *old = PyList_GET_ITEM(made, i);
            int64_t mine = 0;
            if (px_scalar(old, kind, &mine) == 0 && mine == values[i]) {
                continue;
            }
            PyObject *item = px_scalar_value(values[i], kind);
            if (item == NULL) {
                return -1;
            }
            PyList_SetItem(made, i, item);
        }
        return 0;
    }
    if (PyList_GET_SIZE(made) == n) {
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *old = PyList_GET_ITEM(made, i);
            PyObject *item = px_kept(x, old, values + i * w, v, rewrite);
            if (item == NULL) {
                return -1;
            }
            if (item == old) {
                Py_DECREF(item);
                continue;
            }
            /* The list may have been resized by nothing here: no Python ran. */
            PyList_SetItem(made, i, item);
        }
        return 0;
    }
    PyObject *items = PyList_New(n);
    if (items == NULL) {
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = px_value(x, values + i * w, v, rewrite);
        if (item == NULL) {
            Py_DECREF(items);
            return -1;
        }
        PyList_SET_ITEM(items, i, item);
    }
    int done = PyList_SetSlice(made, 0, PY_SSIZE_T_MAX, items);
    Py_DECREF(items);
    return done;
}

static int px_rewrite_dict(ppy_cross *x, PyObject *made, const int64_t *keys,
                           const int64_t *values, Py_ssize_t n, const ppy_xs *k,
                           const ppy_xs *v, int64_t kw, int64_t w, int rewrite) {
    /* The same keys in the same order: each value set in place, so a walk
       over the dict the caller is in the middle of goes on as it would. */
    int same = PyDict_GET_SIZE(made) == n;
    PyObject *old_keys = NULL;
    if (same && n > 0) {
        old_keys = PyList_New(n);
        if (old_keys == NULL) {
            return -1;
        }
        Py_ssize_t position = 0, i = 0;
        PyObject *key = NULL, *item = NULL;
        while (same && PyDict_Next(made, &position, &key, &item)) {
            if (!px_holds(key, keys + i * kw, k)) {
                same = 0;
                break;
            }
            Py_INCREF(key);
            PyList_SET_ITEM(old_keys, i, key);
            i++;
        }
    }
    if (!same) {
        Py_XDECREF(old_keys);
        old_keys = NULL;
        PyDict_Clear(made);
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *key = old_keys != NULL ? PyList_GET_ITEM(old_keys, i) : NULL;
        if (key != NULL) {
            Py_INCREF(key);
        } else {
            key = px_value(x, keys + i * kw, k, rewrite);
        }
        if (key == NULL) {
            Py_XDECREF(old_keys);
            return -1;
        }
        PyObject *old = same ? PyDict_GetItemWithError(made, key) : NULL;
        PyObject *item = px_kept(x, old, values + i * w, v, rewrite);
        if (item == NULL) {
            Py_DECREF(key);
            Py_XDECREF(old_keys);
            return -1;
        }
        int done = item == old ? 0 : PyDict_SetItem(made, key, item);
        Py_DECREF(key);
        Py_DECREF(item);
        if (done < 0) {
            Py_XDECREF(old_keys);
            return -1;
        }
    }
    Py_XDECREF(old_keys);
    return 0;
}

static int px_rewrite_set(ppy_cross *x, PyObject *made, const int64_t *keys, Py_ssize_t n,
                          const ppy_xs *k, int64_t kw, int rewrite, int fresh) {
    PyObject *items = PyList_New(n);
    if (items == NULL) {
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *key = px_value(x, keys + i * kw, k, rewrite);
        if (key == NULL) {
            Py_DECREF(items);
            return -1;
        }
        PyList_SET_ITEM(items, i, key);
    }
    if (!fresh && PySet_GET_SIZE(made) == n) {
        /* The same members: the set is left as it is, its order with it. */
        int all = 1;
        for (Py_ssize_t i = 0; all && i < n; i++) {
            int found = PySet_Contains(made, PyList_GET_ITEM(items, i));
            if (found < 0) {
                Py_DECREF(items);
                return -1;
            }
            all = found;
        }
        if (all) {
            Py_DECREF(items);
            return 0;
        }
    }
    if (!fresh) {
        PySet_Clear(made);
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        if (PySet_Add(made, PyList_GET_ITEM(items, i)) < 0) {
            Py_DECREF(items);
            return -1;
        }
    }
    Py_DECREF(items);
    return 0;
}

/* The Python object of a handle, a new reference: the one it came from, its
   contents set again when `rewrite`, or a new container. */
static PyObject *px_out(ppy_cross *x, int8_t *handle, const ppy_xs *s, int rewrite) {
    px_entry *known = px_by_handle(x, handle);
    if (known != NULL && (!rewrite || known->synced)) {
        Py_INCREF(known->obj);
        return known->obj;
    }
    PyObject *made;
    int fresh = known == NULL;
    if (known != NULL) {
        made = known->obj;
        known->synced = 1;
    } else {
        made = s->kind == PX_LIST ? PyList_New(0) : s->kind == PX_DICT ? PyDict_New() : PySet_New(NULL);
        if (made == NULL) {
            return NULL;
        }
        Py_ssize_t at = px_add(x, made, handle, s, 0);
        if (at < 0) {
            Py_DECREF(made);
            return PyErr_NoMemory();
        }
        /* The entry holds it now; this reference is the one handed out. */
        x->entries[at].synced = 1;
    }
    if (!fresh) {
        Py_INCREF(made);
    }
    const ppy_xs *v = s->value;
    const ppy_xs *k = s->key;
    int64_t w = v != NULL ? px_words(v) : 0;
    int64_t kw = k != NULL ? px_words(k) : 0;
    Py_ssize_t n = (Py_ssize_t)ppy_rt.len(handle);
    int64_t small[64];
    size_t need = (size_t)n * (size_t)(w + kw);
    int64_t *buffer = need <= 64 ? small : (int64_t *)PyMem_Malloc(need * 8 + 8);
    if (buffer == NULL) {
        Py_DECREF(made);
        return PyErr_NoMemory();
    }
    int64_t *keys = buffer;
    int64_t *values = buffer + (size_t)n * (size_t)kw;
    ppy_rt.copy_out(handle, (int8_t *)keys, (int8_t *)values);
    int done;
    if (s->kind == PX_LIST) {
        done = px_rewrite_list(x, made, values, n, v, w, rewrite);
    } else if (s->kind == PX_DICT) {
        done = px_rewrite_dict(x, made, keys, values, n, k, v, kw, w, rewrite);
    } else {
        done = px_rewrite_set(x, made, keys, n, k, kw, rewrite, fresh);
    }
    if (buffer != small) {
        PyMem_Free(buffer);
    }
    if (done < 0) {
        Py_DECREF(made);
        return NULL;
    }
    return made;
}

/* After a call that wrote through a parameter: every container that came in
   copied back, the ones the call no longer reaches too. */
static int px_sync(ppy_cross *x) {
    for (Py_ssize_t i = 0; i < x->count; i++) {
        if (!x->entries[i].incoming || x->entries[i].synced) {
            continue;
        }
        PyObject *done = px_out(x, x->entries[i].handle, x->entries[i].spec, 1);
        if (done == NULL) {
            return -1;
        }
        Py_DECREF(done);
    }
    return 0;
}

/* The handle a call returned, owned: its Python object. */
static PyObject *px_result(ppy_cross *x, int8_t *handle, const ppy_xs *s) {
    if (px_own(x, handle) < 0) {
        ppy_rt.release(handle);
        return PyErr_NoMemory();
    }
    return px_out(x, handle, s, 0);
}
