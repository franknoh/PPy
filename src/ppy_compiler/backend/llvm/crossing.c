/* Containers and objects across the Python boundary, in the generated wrapper.

   A native function that takes a `list`, a `dict`, a `set`, or an instance
   of a project class works on a handle into the collections runtime
   (`ppy_runtime/collections.c`). This is the crossing a generated wrapper
   does for it: each argument copied into native memory, the result copied
   out, and, when the function writes through a parameter, every container
   and object that came in copied back into the caller's object, which stays
   the same object. It does what `ppy_runtime/collection_boundary.py` does,
   without a Python frame.

   Identity is kept both ways. An object met twice among the arguments, or
   inside them, is one handle with two references, so `f(xs, xs)` and the
   shared rows of `[[0] * n] * m` stay shared; a handle that came from an
   object comes back as that object. An element or a field is written back
   only where it changed, so one the call left alone keeps its identity.

   A container's checks are exact type tests, and the only values compared
   are numbers and strings. An object's fields are read with `getattr`, as
   Python reads them, so a property may run; a container is read again
   after one did, and the crossing is refused where it changed. A value of
   the wrong type refuses the crossing, and the Python body runs instead.

   The runtime's functions are reached through `ppy_rt`, which the binder
   fills (`ppy_runtime`) with the addresses of the runtime the native code
   uses: the handles must come from the same heap. A signature's classes
   are a static table (`px_classes`) whose Python classes are found at the
   first call, through a resolver the binder hands over: a module may define
   a class after the function that takes it. */

#include <string.h>

#define PX_INT 1
#define PX_FLOAT 2
#define PX_BOOL 3
#define PX_STR 4
#define PX_TUPLE 5
#define PX_LIST 6
#define PX_DICT 7
#define PX_SET 8
#define PX_OBJECT 9
#define PX_RECORD 10

/* One type at the boundary: a scalar, a string, a tuple of scalars (`part`
   spells each word: 'i', 'f', or 'b'), a container, an object of a class
   (`cls`, in the signature's table; `nullable` where it may be None), or a
   value class's fields in place (`cls`, its words spelled in `part`). */
typedef struct ppy_xs {
    int kind;
    int parts;
    const char *part;
    const struct ppy_xs *key;
    const struct ppy_xs *value;
    int cls;
    int nullable;
} ppy_xs;

/* A class's field: its name, its first word in the record, and its type. */
typedef struct {
    const char *name;
    int64_t offset;
    const ppy_xs *spec;
} px_field;

/* A class whose instances cross, as native code lays them out
   (`ppy_runtime.abi.CrossingClass`). */
typedef struct {
    const char *qualname;
    int record;
    int64_t tag, words, floats, handles;
    int nfields;
    const px_field *fields;
    int nbases;
    const int *bases;
    /* Found at the first call: the Python class, and the fields' names. */
    PyTypeObject *type;
    PyObject **names;
} px_class;

typedef struct {
    int count;
    px_class *classes;
    PyObject *resolve;
    int resolved;
} px_classes;

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

/* The classes' Python classes, found once all are: whether they are. */
static int px_resolve(px_classes *t) {
    if (t == NULL || t->count == 0 || t->resolved) {
        return 1;
    }
    if (t->resolve == NULL) {
        return 0;
    }
    PyObject *found = PyObject_CallNoArgs(t->resolve);
    if (found == NULL) {
        PyErr_Clear();
        return 0;
    }
    int ok = PyTuple_Check(found) && PyTuple_GET_SIZE(found) == t->count;
    for (int i = 0; ok && i < t->count; i++) {
        ok = PyType_Check(PyTuple_GET_ITEM(found, i));
    }
    for (int i = 0; ok && i < t->count; i++) {
        px_class *c = &t->classes[i];
        PyObject **names = (PyObject **)PyMem_Calloc((size_t)(c->nfields + 1), sizeof(PyObject *));
        if (names == NULL) {
            ok = 0;
            break;
        }
        for (int f = 0; f < c->nfields; f++) {
            names[f] = PyUnicode_InternFromString(c->fields[f].name);
            if (names[f] == NULL) {
                PyErr_Clear();
                ok = 0;
                break;
            }
        }
        if (!ok) {
            for (int f = 0; f < c->nfields; f++) {
                Py_XDECREF(names[f]);
            }
            PyMem_Free(names);
            break;
        }
        PyObject *type = PyTuple_GET_ITEM(found, i);
        Py_INCREF(type);
        c->type = (PyTypeObject *)type;
        c->names = names;
    }
    Py_DECREF(found);
    t->resolved = ok;
    return ok;
}

/* -- one call's crossing ------------------------------------------------- */

/* A container or an object of the call: the Python object and its handle.
   `incoming` is one that came in with the arguments, which a call that
   writes copies back; `synced` is one copied out already. The object is
   held. */
typedef struct {
    PyObject *obj;
    int8_t *handle;
    const ppy_xs *spec;
    int incoming;
    int synced;
    /* Met while a parameter the call writes through came in: only those are
       copied back (`px_sync`). */
    int writable;
    /* A list of numbers' words as they came in, which tells the ones the
       call wrote from the ones it left (`px_rewrite_list`); NULL if none. */
    const int64_t *before;
    Py_ssize_t before_count;
} px_entry;

#define PX_INLINE 16

/* A call that writes through no parameter reads its containers as they
   were when it began, and nothing it does can change them: its lists, and
   the strings in them, are laid out in a block of memory the call owns
   (`px_chunk`) instead of one allocation per handle. A string is the
   Python string's own bytes, borrowed (header word 6 is 2, word 22 the
   object, which the call holds); a list's records sit in the block. Such a
   handle is marked in header word 19 (`PX_ARENA`, a word the collector only
   uses for the handles it tracks, and these are not on the thread's heap
   list) and starts at `PX_IMMORTAL` references more than a fresh handle,
   so nothing native code does lets it reach zero and be freed.

   When the call is done, the references are counted: a list no one holds
   any more lets go of what it holds, and anything still held past the call
   (a cached function's table kept it) has escaped. Then the whole block is
   kept for good, each list's records and each string's bytes copied into
   memory of their own, as a fresh handle would have them. */
#define PX_ARENA ((int64_t)0x70784152454e41LL)
#define PX_IMMORTAL ((int64_t)1 << 60)

typedef struct px_chunk {
    struct px_chunk *next;
    size_t used, room;
    int64_t words[];
} px_chunk;

/* The block a call takes, kept between calls (the GIL guards it). */
static px_chunk *px_spare = NULL;
/* Blocks whose handles escaped their call: kept, never freed. */
static px_chunk *px_kept_chunks = NULL;

typedef struct {
    px_classes *classes;
    int readonly, escaped;
    /* Whether the parameter coming in is one the call writes through, and
       whether one such met a container another parameter brought in, which
       then may be written too: every one is copied back. */
    int writing, shared;
    px_chunk *chunks;
    /* The arena handles the call made, in the order it made them. */
    int8_t **made;
    Py_ssize_t made_count, made_room;
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
    int8_t *made0[PX_INLINE * 4];
    /* Objects whose fields are still to be set, while `px_out` sets an
       object's (`px_deferred`). */
    struct px_pending *pending;
    Py_ssize_t pending_count, pending_room;
    int draining;
} ppy_cross;

/* An object made or found for a handle, its fields not yet set. */
typedef struct px_pending {
    PyObject *made;
    px_class *c;
    int8_t *handle;
    int fresh, rewrite;
} px_pending;

static void px_begin(ppy_cross *x, px_classes *classes, int readonly) {
    x->classes = classes;
    x->readonly = readonly;
    x->escaped = 0;
    x->writing = 1;
    x->shared = 0;
    x->chunks = NULL;
    x->made = x->made0;
    x->made_count = 0;
    x->made_room = PX_INLINE * 4;
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
    x->pending = NULL;
    x->pending_count = 0;
    x->pending_room = 0;
    x->draining = 0;
}

static void px_settle(ppy_cross *x);

/* Every reference the call held let go of, and the objects it held. */
static void px_end(ppy_cross *x) {
    for (Py_ssize_t i = 0; i < x->owned_count; i++) {
        ppy_rt.release(x->owned[i]);
    }
    x->owned_count = 0;
    if (x->made_count > 0) {
        px_settle(x);
    }
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

/* `words` words from the call's block, as they are, or NULL out of memory. */
static int64_t *px_alloc(ppy_cross *x, size_t words) {
    px_chunk *c = x->chunks;
    if (c == NULL || c->used + words > c->room) {
        size_t room = words > 8192 ? words : 8192;
        px_chunk *fresh = NULL;
        if (px_spare != NULL && px_spare->room >= room) {
            fresh = px_spare;
            px_spare = NULL;
        } else {
            fresh = (px_chunk *)PyMem_Malloc(sizeof(px_chunk) + room * 8);
            if (fresh == NULL) {
                return NULL;
            }
            fresh->room = room;
        }
        fresh->used = 0;
        fresh->next = x->chunks;
        x->chunks = fresh;
        c = fresh;
    }
    int64_t *found = c->words + c->used;
    c->used += words;
    return found;
}

static int px_made(ppy_cross *x, int8_t *handle) {
    if (x->made_count == x->made_room) {
        Py_ssize_t room = x->made_room * 2;
        int8_t **grown = (int8_t **)PyMem_Malloc((size_t)room * sizeof(int8_t *));
        if (grown == NULL) {
            return -1;
        }
        memcpy(grown, x->made, (size_t)x->made_count * sizeof(int8_t *));
        if (x->made != x->made0) {
            PyMem_Free(x->made);
        }
        x->made = grown;
        x->made_room = room;
    }
    x->made[x->made_count++] = handle;
    return 0;
}

static int px_arena_handle(const int8_t *handle) {
    return handle != NULL && ((const int64_t *)handle)[19] == PX_ARENA;
}

static int px_arena_list_handle(const int8_t *handle) {
    return px_arena_handle(handle) && ((const int64_t *)handle)[12] == 0;
}

/* The word of an arena list that holds the list it came from (`px_add`
   holds the reference). */
static int64_t *px_source(int8_t *handle) {
    int64_t *h = (int64_t *)handle;
    return h + 26 + 2 * (h[8] > 0 ? h[8] : 1);
}

/* A borrowed string: the bytes of `o`, an exact `str`, held for the call.
   0 done, -1 refused (a lone surrogate has no UTF-8), -2 out of memory. */
static int px_borrow_text(ppy_cross *x, PyObject *o, int64_t *word) {
    Py_ssize_t size = 0;
    const char *data;
    int ascii = PyUnicode_IS_ASCII(o);
    if (ascii) {
        data = (const char *)PyUnicode_DATA(o);
        size = PyUnicode_GET_LENGTH(o);
    } else {
        data = PyUnicode_AsUTF8AndSize(o, &size);
        if (data == NULL) {
            PyErr_Clear();
            return -1;
        }
    }
    int64_t *h = px_alloc(x, 25);
    if (h == NULL || px_made(x, (int8_t *)h) < 0) {
        PyErr_NoMemory();
        return -2;
    }
    memset(h, 0, 25 * 8);
    h[0] = (int64_t)size;
    h[1] = (int64_t)size / 8 + 1;
    h[2] = (int64_t)(intptr_t)data;
    h[3] = (int64_t)PyUnicode_GET_LENGTH(o);
    h[5] = ascii;
    h[6] = 2;
    h[8] = 1;
    h[11] = PX_IMMORTAL + 1;
    h[12] = 4;
    h[15] = 1;
    h[19] = PX_ARENA;
    Py_INCREF(o);
    h[22] = (int64_t)(intptr_t)o;
    *word = (int64_t)(intptr_t)h;
    return 0;
}

/* A list of `n` elements, `words` each, laid out in the call's block. */
static int8_t *px_arena_list(ppy_cross *x, Py_ssize_t n, int64_t words, int64_t floats,
                             int64_t handles) {
    int64_t w = words > 0 ? words : 1;
    /* The header, its scratch words, and the list it came from. */
    int64_t *h = px_alloc(x, (size_t)(26 + 2 * w + 1));
    int64_t room = n > 0 ? (int64_t)n : 1;
    int64_t *records = h != NULL ? px_alloc(x, (size_t)(room * w)) : NULL;
    if (records == NULL || px_made(x, (int8_t *)h) < 0) {
        return NULL;
    }
    /* The header's words zero; the records are each written as the list is
       filled, and the scratch words are the runtime's to write first. */
    memset(h, 0, 26 * 8);
    records[0] = 0;
    int64_t leaves = (int64_t)((uint64_t)handles >> 32);
    h[0] = (int64_t)n;
    h[1] = room;
    h[2] = (int64_t)(intptr_t)records;
    h[8] = words;
    h[9] = floats;
    h[10] = handles & 0xFFFFFFFF;
    h[11] = PX_IMMORTAL + 1;
    h[13] = (int64_t)((uint64_t)leaves << 48);
    h[14] = (int64_t)(intptr_t)(h + 26);
    h[15] = words;
    h[19] = PX_ARENA;
    return (int8_t *)h;
}

/* The call's blocks given back, or kept for good where `keep`. */
static void px_chunks_done(ppy_cross *x, int keep) {
    px_chunk *c = x->chunks;
    x->chunks = NULL;
    while (c != NULL) {
        px_chunk *next = c->next;
        if (keep) {
            c->next = px_kept_chunks;
            px_kept_chunks = c;
        } else if (px_spare == NULL || px_spare->room < c->room) {
            if (px_spare != NULL) {
                PyMem_Free(px_spare);
            }
            px_spare = c;
        } else {
            PyMem_Free(c);
        }
        c = next;
    }
}

/* Let go of the arena handles' objects and blocks; a call that did not
   answer leaves nothing that outlives it. */
static void px_close(ppy_cross *x) {
    for (Py_ssize_t i = 0; i < x->made_count; i++) {
        int64_t *h = (int64_t *)x->made[i];
        if (h[12] == 4 && h[22] != 0) {
            Py_DECREF((PyObject *)(intptr_t)h[22]);
        }
    }
    x->made_count = 0;
    if (x->made != x->made0) {
        PyMem_Free(x->made);
        x->made = x->made0;
    }
    px_chunks_done(x, 0);
}

/* `result`, once the arena is given back. */
static PyObject *ppy_closed(PyObject *result, ppy_cross *x) {
    px_close(x);
    return result;
}

/* Once the call's own references are let go of: each arena list no one
   holds lets go of what it holds, as its release would; one still held is
   one the call kept, and so is all it holds (`escaped`). The lists come in
   the order they were made, a list before what it holds. */
static void px_settle(ppy_cross *x) {
    for (Py_ssize_t i = 0; i < x->made_count; i++) {
        int64_t *h = (int64_t *)x->made[i];
        if (h[11] > PX_IMMORTAL) {
            x->escaped = 1;
            continue;
        }
        if (h[12] != 0 || h[10] == 0) {
            continue;
        }
        int64_t *records = (int64_t *)(intptr_t)h[2];
        for (int64_t at = 0; at < h[0]; at++) {
            int64_t *value = records + at * h[15];
            for (int64_t w = 0; w < h[8]; w++) {
                if (((h[10] >> w) & 1) && value[w] != 0) {
                    int8_t *child = (int8_t *)(intptr_t)value[w];
                    value[w] = 0;
                    if (px_arena_handle(child)) {
                        ((int64_t *)child)[11]--;
                    } else {
                        ppy_rt.release(child);
                    }
                }
            }
        }
    }
}

/* After a call that answered: the arena given back, or, where a handle
   escaped, kept for good. */
static void px_keep(ppy_cross *x) {
    int escaped = x->escaped;
    if (!escaped) {
        px_close(x);
        return;
    }
    /* Something kept a handle past the call: every one keeps its memory. */
    for (Py_ssize_t i = 0; i < x->made_count; i++) {
        int64_t *h = (int64_t *)x->made[i];
        if (h[12] == 4) {
            char *bytes = (char *)malloc((size_t)h[0] + 1);
            if (bytes == NULL) {
                Py_FatalError("out of memory keeping a string native code kept");
            }
            memcpy(bytes, (const void *)(intptr_t)h[2], (size_t)h[0] + 1);
            h[2] = (int64_t)(intptr_t)bytes;
            h[6] = 0;
            Py_DECREF((PyObject *)(intptr_t)h[22]);
            h[22] = 0;
        } else {
            size_t size = (size_t)(h[1] * (h[15] > 0 ? h[15] : 1)) * 8;
            void *records = malloc(size);
            if (records == NULL) {
                Py_FatalError("out of memory keeping a list native code kept");
            }
            memcpy(records, (const void *)(intptr_t)h[2], size);
            h[2] = (int64_t)(intptr_t)records;
        }
        h[19] = 0;
    }
    x->made_count = 0;
    if (x->made != x->made0) {
        PyMem_Free(x->made);
        x->made = x->made0;
    }
    px_chunks_done(x, 1);
}

static size_t px_hash(const void *pointer, Py_ssize_t slots) {
    /* The low bits of a product depend on the low bits alone, which objects
       of one size share: the high bits are mixed down first. */
    uint64_t h = (uint64_t)(uintptr_t)pointer;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return (size_t)h & (size_t)(slots - 1);
}

static void px_index(ppy_cross *x, Py_ssize_t at) {
    size_t i = px_hash(x->entries[at].obj, x->slots);
    while (x->by_obj[i] != 0) {
        i = (i + 1) & (size_t)(x->slots - 1);
    }
    x->by_obj[i] = at + 1;
    if (px_arena_list_handle(x->entries[at].handle)) {
        /* Found by the object it holds the address of (`px_out`). */
        return;
    }
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

/* A container or an object met for the first time: its index, or -1 out of
   memory. */
/* Room for `need` entries in all: 0, or -1 out of memory. */
static int px_room(ppy_cross *x, Py_ssize_t need) {
    if (need > x->room) {
        Py_ssize_t room = x->room * 2;
        while (room < need) {
            room *= 2;
        }
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
    if (need * 2 > x->slots) {
        Py_ssize_t slots = x->slots * 2;
        while (slots < need * 2) {
            slots *= 2;
        }
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
    return 0;
}

/* A container or an object met for the first time: its index, or -1 out of
   memory. */
static Py_ssize_t px_add(ppy_cross *x, PyObject *obj, int8_t *handle, const ppy_xs *spec,
                         int incoming) {
    if (px_room(x, x->count + 1) < 0) {
        return -1;
    }
    Py_ssize_t at = x->count++;
    px_entry *e = &x->entries[at];
    Py_INCREF(obj);
    e->obj = obj;
    e->handle = handle;
    e->spec = spec;
    e->incoming = incoming;
    e->synced = 0;
    e->writable = x->writing;
    e->before = NULL;
    e->before_count = 0;
    px_index(x, at);
    return at;
}

/* A container met again: one a written parameter reaches that came in with
   another may be written through it, and so may anything it holds. */
static void px_met(ppy_cross *x, px_entry *seen) {
    if (x->writing && !seen->writable) {
        x->shared = 1;
    }
}

/* -- classes ------------------------------------------------------------- */

static px_class *px_class_at(ppy_cross *x, int index) {
    return &x->classes->classes[index];
}

/* The class `type` is exactly, among those described: its index, or -1. */
static int px_class_of(ppy_cross *x, PyTypeObject *type) {
    px_classes *t = x->classes;
    for (int i = 0; t != NULL && i < t->count; i++) {
        if (t->classes[i].type == type) {
            return i;
        }
    }
    return -1;
}

static int px_class_by_tag(ppy_cross *x, int64_t tag) {
    px_classes *t = x->classes;
    for (int i = 0; t != NULL && i < t->count; i++) {
        if (!t->classes[i].record && t->classes[i].tag == tag) {
            return i;
        }
    }
    return -1;
}

/* Whether an instance of class `found` is an instance of class `wanted`. */
static int px_isa(ppy_cross *x, int found, int wanted) {
    px_class *c = px_class_at(x, found);
    for (int i = 0; i < c->nbases; i++) {
        if (c->bases[i] == wanted) {
            return 1;
        }
    }
    return 0;
}

/* An object's one record: where its fields' words are. */
static int64_t *px_record_of(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t stride = header[15] > 0 ? header[15] : 1;
    return (int64_t *)(intptr_t)header[2] + (header[3] % header[1]) * stride;
}

/* -- the layout of an element ------------------------------------------- */

static int64_t px_words(const ppy_xs *s) {
    return s->kind == PX_TUPLE || s->kind == PX_RECORD ? s->parts : 1;
}

static int64_t px_floats(const ppy_xs *s) {
    if (s->kind == PX_TUPLE || s->kind == PX_RECORD) {
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
    return s->kind == PX_STR || (s->kind >= PX_LIST && s->kind <= PX_OBJECT);
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

/* A field read as Python reads it, a new reference; NULL where it is missing. */
static PyObject *px_get(PyObject *o, PyObject *name) {
    PyObject *found = PyObject_GetAttr(o, name);
    if (found == NULL) {
        PyErr_Clear();
    }
    return found;
}

/* A value class's fields in its words, its class exactly the one declared. */
static int px_record_in(ppy_cross *x, PyObject *o, const ppy_xs *s, int64_t *words) {
    px_class *c = px_class_at(x, s->cls);
    if (c->type == NULL || Py_TYPE(o) != c->type) {
        return -1;
    }
    for (int f = 0; f < c->nfields; f++) {
        PyObject *item = px_get(o, c->names[f]);
        if (item == NULL) {
            return -1;
        }
        int done = px_scalar(item, s->part[f], words + c->fields[f].offset);
        Py_DECREF(item);
        if (done != 0) {
            return -1;
        }
    }
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
    case PX_RECORD:
        return px_record_in(x, o, s, words);
    case PX_STR: {
        if (!PyUnicode_CheckExact(o)) {
            return -1;
        }
        if (x->readonly) {
            return px_borrow_text(x, o, words);
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

/* Whether reading an element can run Python code (a property of an object). */
static int px_reads_attributes(const ppy_xs *s) {
    return s != NULL && (s->kind == PX_OBJECT || s->kind == PX_RECORD ||
                         (s->kind >= PX_LIST && s->kind <= PX_SET));
}

/* Let go of the handles placed in `count` elements' words so far. */
static void px_drop(const ppy_xs *s, const int64_t *words, Py_ssize_t count) {
    if (s == NULL || !px_reference(s)) {
        return;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        if (words[i] != 0) {
            ppy_rt.release((int8_t *)(intptr_t)words[i]);
        }
    }
}

static char kk_of(const ppy_xs *k) {
    return k->kind == PX_INT ? 'i' : k->kind == PX_FLOAT ? 'f' : 'b';
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
        if (v->kind == PX_BOOL) {
            PyObject **items = ((PyListObject *)o)->ob_item;
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject *item = items[i];
                if (item != Py_True && item != Py_False) {
                    return -1;
                }
                records[i] = item == Py_True;
            }
            return 0;
        }
        if (v->kind == PX_STR && x->readonly) {
            PyObject **items = ((PyListObject *)o)->ob_item;
            for (Py_ssize_t i = 0; i < n; i++) {
                if (!PyUnicode_CheckExact(items[i])) {
                    return -1;
                }
                int done = px_borrow_text(x, items[i], records + i);
                if (done != 0) {
                    return done;
                }
            }
            return 0;
        }
        if (x->readonly && v->kind >= PX_LIST && v->kind <= PX_SET) {
            /* No Python code runs while a call that holds no objects is
               filled: the list cannot change under the walk. */
            PyObject **items = ((PyListObject *)o)->ob_item;
            for (Py_ssize_t i = 0; i < n; i++) {
                int8_t *inner = NULL;
                int done = px_in(x, items[i], v, &inner);
                if (done != 0) {
                    /* The list is refused: what it took so far let go of. */
                    for (Py_ssize_t j = 0; j < i; j++) {
                        int8_t *taken = (int8_t *)(intptr_t)records[j];
                        if (px_arena_handle(taken)) {
                            ((int64_t *)taken)[11]--;
                        } else {
                            ppy_rt.release(taken);
                        }
                    }
                    ((int64_t *)handle)[0] = 0;
                    return done;
                }
                records[i] = (int64_t)(intptr_t)inner;
            }
            return 0;
        }
        if (v->kind == PX_INT || v->kind == PX_FLOAT) {
            /* Numbers, the common case, checked and copied in one loop. */
            char kind = v->kind == PX_INT ? 'i' : 'f';
            PyObject **items = ((PyListObject *)o)->ob_item;
            for (Py_ssize_t i = 0; i < n; i++) {
                if (px_scalar(items[i], kind, records + i) != 0) {
                    return -1;
                }
            }
            return 0;
        }
        for (Py_ssize_t i = 0; i < n; i++) {
            if (PyList_GET_SIZE(o) != n) {
                /* A property read along the way changed the list. */
                return -1;
            }
            PyObject *item = PyList_GET_ITEM(o, i);
            Py_INCREF(item);
            int done = px_word(x, item, v, records + i * w);
            Py_DECREF(item);
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
    Py_ssize_t made = 0, keys_made = 0;
    PyObject *key = NULL, *item = NULL;
    int numbers = (k->kind == PX_INT || k->kind == PX_FLOAT || k->kind == PX_BOOL) &&
                  (v == NULL || v->kind == PX_INT || v->kind == PX_FLOAT || v->kind == PX_BOOL);
    if (numbers && s->kind == PX_DICT) {
        /* Numbers to numbers: checked and copied in one walk. */
        char kk = k->kind == PX_INT ? 'i' : k->kind == PX_FLOAT ? 'f' : 'b';
        char vk = v->kind == PX_INT ? 'i' : v->kind == PX_FLOAT ? 'f' : 'b';
        Py_ssize_t position = 0;
        while (PyDict_Next(o, &position, &key, &item)) {
            if (keys_made == n || px_scalar(key, kk, keys + keys_made) != 0 ||
                px_scalar(item, vk, values + keys_made) != 0) {
                done = -1;
                break;
            }
            keys_made++;
        }
        made = keys_made;
    } else if (numbers) {
        /* A set of numbers: its members hash and compare without running
           Python code, so the walk cannot see the set change. */
        PyObject *walk = PyObject_GetIter(o);
        char kk = kk_of(k);
        if (walk == NULL) {
            done = -2;
        }
        while (done == 0 && (key = PyIter_Next(walk)) != NULL) {
            int bad = keys_made == n || px_scalar(key, kk, keys + keys_made) != 0;
            Py_DECREF(key);
            if (bad) {
                done = -1;
                break;
            }
            keys_made++;
        }
        if (done == 0 && PyErr_Occurred()) {
            done = -2;
        }
        Py_XDECREF(walk);
    } else if (s->kind == PX_DICT) {
        Py_ssize_t position = 0;
        int watch = px_reads_attributes(v);
        while (PyDict_Next(o, &position, &key, &item)) {
            if (keys_made == n) {
                done = -1;
                break;
            }
            done = px_word(x, key, k, keys + keys_made * kw);
            if (done != 0) {
                break;
            }
            keys_made++;
            Py_INCREF(item);
            done = px_word(x, item, v, values + made * w);
            Py_DECREF(item);
            if (done != 0) {
                break;
            }
            made++;
            if (watch && PyDict_GET_SIZE(o) != n) {
                done = -1;
                break;
            }
        }
    } else {
        PyObject *walk = PyObject_GetIter(o);
        if (walk == NULL) {
            done = -2;
        }
        while (done == 0 && (key = PyIter_Next(walk)) != NULL) {
            if (keys_made == n) {
                Py_DECREF(key);
                done = -1;
                break;
            }
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
    if (buffer != small) {
        PyMem_Free(buffer);
    }
    return done;
}

/* Whether two types are one: a container met again as another type has
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
    if ((a->kind == PX_OBJECT || a->kind == PX_RECORD) && a->cls != b->cls) {
        return 0;
    }
    return px_same_spec(a->key, b->key) && px_same_spec(a->value, b->value);
}

/* An instance of a project class as a handle to its record, the objects and
   containers its fields hold made native with it. */
static int px_object_in(ppy_cross *x, PyObject *o, const ppy_xs *s, int8_t **out) {
    if (o == Py_None) {
        if (!s->nullable) {
            return -1;
        }
        *out = NULL;
        return 0;
    }
    int found = px_class_of(x, Py_TYPE(o));
    if (found < 0 || !px_isa(x, found, s->cls)) {
        return -1;
    }
    px_entry *seen = px_by_obj(x, o);
    if (seen != NULL) {
        px_met(x, seen);
        ppy_rt.retain(seen->handle);
        *out = seen->handle;
        return 0;
    }
    px_class *c = px_class_at(x, found);
    if (c->record) {
        return -1;
    }
    int8_t *handle = ppy_rt.seq_new(1, c->words, c->floats, c->handles);
    ((int64_t *)handle)[4] = c->tag;
    if (px_add(x, o, handle, s, 1) < 0 || px_own(x, handle) < 0) {
        ppy_rt.release(handle);
        PyErr_NoMemory();
        return -2;
    }
    ppy_rt.retain(handle);
    int64_t *record = px_record_of(handle);
    for (int f = 0; f < c->nfields; f++) {
        PyObject *item = px_get(o, c->names[f]);
        if (item == NULL) {
            ppy_rt.release(handle);
            return -1;
        }
        int done = px_word(x, item, c->fields[f].spec, record + c->fields[f].offset);
        Py_DECREF(item);
        if (done != 0) {
            /* The handles placed so far are the record's, let go of with it. */
            ppy_rt.release(handle);
            return done;
        }
    }
    *out = handle;
    return 0;
}

static int px_in(ppy_cross *x, PyObject *o, const ppy_xs *s, int8_t **out) {
    if (s->kind == PX_OBJECT) {
        return px_object_in(x, o, s, out);
    }
    px_entry *seen = px_by_obj(x, o);
    if (seen != NULL) {
        if (!px_same_spec(seen->spec, s)) {
            return -1;
        }
        px_met(x, seen);
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
    if (x->readonly && s->kind == PX_LIST) {
        /* Laid out in the call's block: nothing to let go of but its elements. */
        int8_t *made = px_arena_list(x, PyList_GET_SIZE(o), words, floats, handles);
        if (made == NULL || px_add(x, o, made, s, 1) < 0 ||
            (v != NULL && v->kind >= PX_LIST && v->kind <= PX_SET &&
             px_room(x, x->count + PyList_GET_SIZE(o)) < 0)) {
            PyErr_NoMemory();
            return -2;
        }
        *px_source(made) = (int64_t)(intptr_t)o;
        int done = px_fill(x, made, o, s);
        if (done != 0) {
            return done;
        }
        *out = made;
        return 0;
    }
    int8_t *handle = s->kind == PX_LIST
                         ? ppy_rt.seq_new((int64_t)PyList_GET_SIZE(o), words, floats, handles)
                         : ppy_rt.map_new(px_words(s->key), words, floats, handles);
    if (s->kind != PX_LIST && s->key->kind == PX_STR) {
        ppy_rt.text_keys(handle, 1);
    }
    /* The call holds its own reference to every handle it made, so none is
       freed, and its address given to something new, while the call runs. */
    Py_ssize_t at = px_add(x, o, handle, s, 1);
    if (at < 0 || px_own(x, handle) < 0) {
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
    if (x->writing && s->kind == PX_LIST && PyList_GET_SIZE(o) > 0 &&
        (v->kind == PX_INT || v->kind == PX_FLOAT || v->kind == PX_BOOL)) {
        /* The words as they came in: copying back sets only what changed. */
        size_t n = (size_t)PyList_GET_SIZE(o);
        int64_t *before = px_alloc(x, n);
        if (before != NULL) {
            memcpy(before, (const void *)(intptr_t)((int64_t *)handle)[2], n * 8);
            x->entries[at].before = before;
            x->entries[at].before_count = (Py_ssize_t)n;
        }
    }
    *out = handle;
    return 0;
}

/* A top-level argument: a handle the call owns (NULL for a `None` object). */
static int px_argument(ppy_cross *x, PyObject *o, const ppy_xs *s, int8_t **out) {
    int done = px_in(x, o, s, out);
    if (done == 0 && *out != NULL && px_own(x, *out) < 0) {
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

/* An instance made without running `__init__` (native code ran it), as
   `object.__new__(cls)` makes it, with the checks that call makes. */
static PyObject *px_blank(PyTypeObject *type) {
    static PyObject *make = NULL;
    if (make == NULL) {
        make = PyObject_GetAttrString((PyObject *)&PyBaseObject_Type, "__new__");
        if (make == NULL) {
            return NULL;
        }
    }
    return PyObject_CallOneArg(make, (PyObject *)type);
}

/* Whether `old` already is the value of these words: then it stays, and so
   does its identity. Numbers, strings, tuples of numbers, and value classes
   are asked; a container or an object is the same object anyway. */
static int px_holds(ppy_cross *x, PyObject *old, const int64_t *words, const ppy_xs *s) {
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
    case PX_RECORD: {
        px_class *c = px_class_at(x, s->cls);
        if (c->type == NULL || Py_TYPE(old) != c->type) {
            return 0;
        }
        for (int f = 0; f < c->nfields; f++) {
            PyObject *part = PyObject_GenericGetAttr(old, c->names[f]);
            int64_t mine = 0;
            if (part == NULL) {
                PyErr_Clear();
                return 0;
            }
            int same = px_scalar(part, s->part[f], &mine) == 0 &&
                       mine == words[c->fields[f].offset];
            Py_DECREF(part);
            if (!same) {
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
        if (header[19] == PX_ARENA && (PyObject *)(intptr_t)header[22] == old) {
            return 1;
        }
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

/* A value class's instance from its fields' words. */
static PyObject *px_record_out(ppy_cross *x, const int64_t *words, const ppy_xs *s) {
    px_class *c = px_class_at(x, s->cls);
    if (c->type == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "native code returned a value of a class it did not describe");
        return NULL;
    }
    PyObject *made = px_blank(c->type);
    if (made == NULL) {
        return NULL;
    }
    for (int f = 0; f < c->nfields; f++) {
        PyObject *part = px_scalar_value(words[c->fields[f].offset], s->part[f]);
        if (part == NULL || PyObject_GenericSetAttr(made, c->names[f], part) < 0) {
            Py_XDECREF(part);
            Py_DECREF(made);
            return NULL;
        }
        Py_DECREF(part);
    }
    return made;
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
    case PX_RECORD:
        return px_record_out(x, words, s);
    case PX_STR: {
        const int64_t *header = (const int64_t *)(intptr_t)words[0];
        if (header[19] == PX_ARENA && header[22] != 0) {
            /* A borrowed string goes back as the string it is. */
            return Py_NewRef((PyObject *)(intptr_t)header[22]);
        }
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
    if (old != NULL && px_holds(x, old, words, s)) {
        Py_INCREF(old);
        return old;
    }
    return px_value(x, words, s, rewrite);
}

static int px_rewrite_list(ppy_cross *x, PyObject *made, const int64_t *values, Py_ssize_t n,
                           const ppy_xs *v, int64_t w, int rewrite, const int64_t *before,
                           Py_ssize_t was) {
    if (before != NULL && was == n && PyList_GET_SIZE(made) == n &&
        (v->kind == PX_INT || v->kind == PX_FLOAT || v->kind == PX_BOOL)) {
        /* The words the call left as they came in leave their elements as
           they are; the others are set, each in its place. */
        char kind = v->kind == PX_INT ? 'i' : v->kind == PX_FLOAT ? 'f' : 'b';
        Py_ssize_t i = 0;
        while (i < n) {
            if (values[i] == before[i]) {
                i++;
                continue;
            }
            PyObject *item = px_scalar_value(values[i], kind);
            if (item == NULL) {
                return -1;
            }
            PyList_SetItem(made, i, item);
            i++;
        }
        return 0;
    }
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
            if (PyList_GET_SIZE(made) != n) {
                break;
            }
            PyObject *old = PyList_GET_ITEM(made, i);
            Py_INCREF(old);
            PyObject *item = px_kept(x, old, values + i * w, v, rewrite);
            if (item == NULL) {
                Py_DECREF(old);
                return -1;
            }
            if (item == old || PyList_GET_SIZE(made) != n) {
                Py_DECREF(old);
                Py_DECREF(item);
                continue;
            }
            Py_DECREF(old);
            PyList_SetItem(made, i, item);
        }
        if (PyList_GET_SIZE(made) == n) {
            return 0;
        }
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
        while (same && i < n && PyDict_Next(made, &position, &key, &item)) {
            if (!px_holds(x, key, keys + i * kw, k)) {
                same = 0;
                break;
            }
            Py_INCREF(key);
            PyList_SET_ITEM(old_keys, i, key);
            i++;
        }
        same = same && i == n;
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
        Py_XINCREF(old);
        PyObject *item = px_kept(x, old, values + i * w, v, rewrite);
        if (item == NULL) {
            Py_XDECREF(old);
            Py_DECREF(key);
            Py_XDECREF(old_keys);
            return -1;
        }
        int done = item == old ? 0 : PyDict_SetItem(made, key, item);
        Py_XDECREF(old);
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

/* An object's fields set to what its record holds, each only where it changed. */
static int px_rewrite_object(ppy_cross *x, PyObject *made, px_class *c, int8_t *handle,
                             int fresh, int rewrite) {
    int64_t *record = px_record_of(handle);
    for (int f = 0; f < c->nfields; f++) {
        const px_field *field = &c->fields[f];
        PyObject *old = NULL;
        if (!fresh) {
            old = PyObject_GenericGetAttr(made, c->names[f]);
            if (old == NULL) {
                PyErr_Clear();
            }
        }
        PyObject *value = NULL;
        if (field->spec->kind == PX_OBJECT && record[field->offset] == 0) {
            value = Py_NewRef(Py_None);
        } else {
            value = px_kept(x, old, record + field->offset, field->spec, rewrite);
        }
        if (value == NULL) {
            Py_XDECREF(old);
            return -1;
        }
        int done = value == old ? 0 : PyObject_GenericSetAttr(made, c->names[f], value);
        Py_XDECREF(old);
        Py_DECREF(value);
        if (done < 0) {
            return -1;
        }
    }
    return 0;
}

/* An object's fields set later, by the `px_out` setting the fields of the
   object that reaches it: a linked list a million nodes long is then not a
   million C frames deep. Making an instance (`object.__new__`) and setting
   its attributes run no code of the program, so the order the objects are
   filled in cannot be seen. Holds a reference to `made`. */
static int px_defer(ppy_cross *x, PyObject *made, px_class *c, int8_t *handle, int fresh,
                    int rewrite) {
    if (x->pending_count == x->pending_room) {
        Py_ssize_t room = x->pending_room > 0 ? x->pending_room * 2 : 64;
        px_pending *grown =
            (px_pending *)PyMem_Realloc(x->pending, (size_t)room * sizeof(px_pending));
        if (grown == NULL) {
            return -1;
        }
        x->pending = grown;
        x->pending_room = room;
    }
    px_pending *p = &x->pending[x->pending_count++];
    p->made = Py_NewRef(made);
    p->c = c;
    p->handle = handle;
    p->fresh = fresh;
    p->rewrite = rewrite;
    return 0;
}

/* Set the fields of `made`, and of every object that sets deferred. */
static int px_deferred(ppy_cross *x, PyObject *made, px_class *c, int8_t *handle, int fresh,
                       int rewrite) {
    x->draining = 1;
    int done = px_rewrite_object(x, made, c, handle, fresh, rewrite);
    while (x->pending_count > 0) {
        px_pending p = x->pending[--x->pending_count];
        if (done == 0) {
            done = px_rewrite_object(x, p.made, p.c, p.handle, p.fresh, p.rewrite);
        }
        Py_DECREF(p.made);
    }
    x->draining = 0;
    PyMem_Free(x->pending);
    x->pending = NULL;
    x->pending_room = 0;
    return done;
}

/* The Python object of a handle, a new reference: the one it came from, its
   contents set again when `rewrite`, or a new container or instance. */
static PyObject *px_out(ppy_cross *x, int8_t *handle, const ppy_xs *s, int rewrite) {
    if (handle == NULL) {
        /* A null object handle is `None`. */
        return Py_NewRef(Py_None);
    }
    if (px_arena_list_handle(handle)) {
        /* A list read, not written: the list it came from. */
        return Py_NewRef((PyObject *)(intptr_t)*px_source(handle));
    }
    px_entry *known = px_by_handle(x, handle);
    if (known != NULL && (!rewrite || known->synced)) {
        Py_INCREF(known->obj);
        return known->obj;
    }
    PyObject *made;
    int fresh = known == NULL;
    px_class *c = NULL;
    if (s->kind == PX_OBJECT) {
        int found = px_class_by_tag(x, ((int64_t *)handle)[4]);
        c = found >= 0 ? px_class_at(x, found) : NULL;
        if (c == NULL || c->type == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                            "native code returned an object of a class it did not describe");
            return NULL;
        }
    }
    if (known != NULL) {
        made = known->obj;
        known->synced = 1;
        Py_INCREF(made);
    } else {
        made = c != NULL                ? px_blank(c->type)
               : s->kind == PX_LIST ? PyList_New(0)
               : s->kind == PX_DICT ? PyDict_New()
                                    : PySet_New(NULL);
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
    if (c != NULL) {
        if (x->draining) {
            if (px_defer(x, made, c, handle, fresh, rewrite) < 0) {
                Py_DECREF(made);
                return PyErr_NoMemory();
            }
            return made;
        }
        if (px_deferred(x, made, c, handle, fresh, rewrite) < 0) {
            Py_DECREF(made);
            return NULL;
        }
        return made;
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
        const int64_t *before = known != NULL ? known->before : NULL;
        Py_ssize_t was = before != NULL ? known->before_count : -1;
        done = px_rewrite_list(x, made, values, n, v, w, rewrite, before, was);
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

/* After a call that wrote through a parameter: every container and object
   that came in with such a parameter copied back, the ones the call no
   longer reaches too; where a written parameter reached what another brought
   in, every one. */
static int px_sync(ppy_cross *x) {
    for (Py_ssize_t i = 0; i < x->count; i++) {
        if (!x->entries[i].incoming || x->entries[i].synced ||
            (!x->entries[i].writable && !x->shared)) {
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
    if (handle != NULL && px_own(x, handle) < 0) {
        ppy_rt.release(handle);
        return PyErr_NoMemory();
    }
    return px_out(x, handle, s, 0);
}
