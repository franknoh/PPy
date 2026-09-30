/* Effects of native code under `ppy run`: output, and calls into Python.

   Native code reaches Python through one hook, a C function the boundary
   installs once per runtime (`ppy_io_set_hook`, `ppy_runtime/effects.py`):

     hook(op, a, b, c) -> 0, or -1 where Python raised

     op 1  write `b` bytes at `a` to stream `c` (1 `sys.stdout`, 2 `sys.stderr`)
     op 2  flush stream `a`
     op 3  call the Python callable named by the text at `a` (`b` bytes) with
           the arguments pushed so far; `c` is the kind of result wanted
     op 4  as 3, a method of the first argument, named by the text at `a`
     op 5  let go of the Python object numbered `a`

   A kind is 0 none, 1 int, 2 float, 3 bool, 4 str (a handle), 5 a Python
   object (its number in the boundary's table). Arguments are pushed with
   `ppy_io_push`; the result is left in the thread's state for
   `ppy_io_result` to take.

   What a native call prints is held in the thread's buffer, a run of
   records `[stream: 1 byte][length: 8 bytes][bytes]`, until the call ends:
   the boundary writes it out when the call answers, and drops it when the
   call falls back, since Python then runs the call again and prints it
   itself. Anything that cannot be taken back (a flush, a call into Python)
   is a barrier: it writes out what is held first, then marks the call as
   crossed, and the lowering has made sure that nothing after a barrier
   falls back (`lowering/effects.py`). A native exception after a barrier is
   raised at the boundary as it is, rather than by running the call again.

   Python may call native code again from inside the hook. The thread's
   collections heap is parked for that time (`ppy_coll_park`), so a nested
   call that fails and sweeps what its thread made cannot free what the
   outer call still holds. */

/* The thread's state:

     [0] the buffer              [1] bytes held      [2] its capacity
     [3] where the open record's length is, or -1    [4] that record's stream
     [5] whether the call crossed a barrier
     [6] the kind of the last result     [7] its word
     [8] arguments pushed        [9] a Python exception is pending here
     [10] the pending exception's tag    [11] its number at the boundary
     [12] its name (malloc'd)    [13] its name's length
     [14] str() of it (malloc'd) [15] that text's length
     [16] a text result's bytes (malloc'd)  [17] their length
     [18..18+2*32) each argument's kind and word */
int64_t *ppy_io_state(void) {
#ifdef __cplusplus
    static thread_local int64_t state[18 + 64] = {0, 0, 0, -1};
#else
    static _Thread_local int64_t state[18 + 64] = {0, 0, 0, -1};
#endif
    return state;
}

int64_t *ppy_io_hook_slot(void) {
    static int64_t hook;
    return &hook;
}

void ppy_io_set_hook(int64_t hook) {
    *ppy_io_hook_slot() = hook;
}

/* Output held for the thread, `stream` 1 or 2. */
void ppy_io_put(int64_t stream, const int8_t *data, int64_t bytes) {
    int64_t *state = ppy_io_state();
    if (bytes <= 0) {
        return;
    }
    int64_t need = state[1] + bytes + 9;
    if (need > state[2]) {
        int64_t room = state[2] * 2 > need ? state[2] * 2 : need;
        room = room < 256 ? 256 : room;
        char *grown = (char *)realloc((void *)(intptr_t)state[0], (size_t)room);
        if (grown == NULL) {
            ppy_coll_fail();
        }
        state[0] = (int64_t)(intptr_t)grown;
        state[2] = room;
    }
    char *buffer = (char *)(intptr_t)state[0];
    if (state[3] < 0 || state[4] != stream) {
        int64_t none = 0;
        buffer[state[1]] = (char)stream;
        memcpy(buffer + state[1] + 1, &none, 8);
        state[3] = state[1] + 1;
        state[4] = stream;
        state[1] += 9;
    }
    memcpy(buffer + state[1], data, (size_t)bytes);
    state[1] += bytes;
    int64_t length;
    memcpy(&length, buffer + state[3], 8);
    length += bytes;
    memcpy(buffer + state[3], &length, 8);
}

/* `print`'s one write: a string's bytes, held. */
void ppy_io_print(int64_t stream, int8_t *text) {
    ppy_io_put(stream, ppy_str_data(text), ppy_str_bytes(text));
}

/* Drop what is held: the call fell back, and Python prints it again. */
void ppy_io_discard(void) {
    int64_t *state = ppy_io_state();
    state[1] = 0;
    state[3] = -1;
}

/* The heap set aside while Python runs, and put back after (see the top). */
void ppy_io_park(int64_t *saved) {
    int64_t *heap = ppy_coll_heap();
    saved[0] = heap[0];
    saved[1] = heap[1];
    saved[2] = heap[5];
    saved[3] = heap[8];
    heap[0] = 0;
    heap[1] = 0;
    heap[5] = 0;
    heap[8] = 0;
}

void ppy_io_unpark(const int64_t *saved) {
    int64_t *heap = ppy_coll_heap();
    if (heap[5]) {
        /* A nested call failed and nothing was made since: what it left goes now. */
        ppy_coll_sweep();
    }
    for (int64_t list = 0; list < 2; list++) {
        /* What the nested calls left, in front of what the outer call holds. */
        int64_t *last = NULL;
        for (int64_t *at = ppy_coll_seen(heap[list]); at != NULL; at = ppy_coll_seen(at[17])) {
            last = at;
        }
        if (last == NULL) {
            heap[list] = saved[list];
            continue;
        }
        last[17] = saved[list];
        int64_t *first = ppy_coll_seen(saved[list]);
        if (first != NULL) {
            first[16] = ppy_coll_hide(last);
        }
    }
    heap[5] = saved[2];
    heap[8] = saved[3];
}

/* The hook, with the heap parked around it. */
int64_t ppy_io_hook(int64_t op, int64_t a, int64_t b, int64_t c) {
    int64_t hook = *ppy_io_hook_slot();
    if (hook == 0) {
        return -1;
    }
    int64_t saved[4];
    ppy_io_park(saved);
    int64_t done = ((int64_t (*)(int64_t, int64_t, int64_t, int64_t))(intptr_t)hook)(op, a, b, c);
    ppy_io_unpark(saved);
    return done;
}

/* Write out what is held, in order. Answers -1 where a write raised; what
   followed it is dropped, as the prints after a raising one never ran. */
int64_t ppy_io_commit(void) {
    int64_t *state = ppy_io_state();
    /* Taken off the thread first: Python may print from native code again. */
    char *buffer = (char *)(intptr_t)state[0];
    int64_t held = state[1];
    int64_t capacity = state[2];
    state[0] = 0;
    state[1] = 0;
    state[2] = 0;
    state[3] = -1;
    int64_t done = 0;
    for (int64_t at = 0; at < held && done == 0;) {
        int64_t length;
        memcpy(&length, buffer + at + 1, 8);
        done = ppy_io_hook(1, (int64_t)(intptr_t)(buffer + at + 9), length, buffer[at]);
        at += 9 + length;
    }
    if (state[0] == 0) {
        /* Nothing was printed meanwhile: the block is kept, empty. */
        state[0] = (int64_t)(intptr_t)buffer;
        state[2] = capacity;
    } else {
        free(buffer);
    }
    return done;
}

/* A barrier: from here the call cannot be run again, so what it held is written. */
int64_t ppy_io_barrier(void) {
    ppy_io_state()[5] = 1;
    return ppy_io_commit();
}

/* `print(..., flush=True)`: written and flushed now. */
int64_t ppy_io_flush(int64_t stream) {
    if (ppy_io_barrier() != 0) {
        return -1;
    }
    return ppy_io_hook(2, stream, 0, 0);
}

/* The boundary's: a native call starts. Answers whether the call around it
   had crossed, which `ppy_io_leave` puts back. */
int64_t ppy_io_enter(void) {
    int64_t *state = ppy_io_state();
    int64_t crossed = state[5];
    state[5] = 0;
    return crossed;
}

/* The boundary's: the call ended. Answers whether it crossed a barrier. */
int64_t ppy_io_leave(int64_t outer) {
    int64_t *state = ppy_io_state();
    int64_t crossed = state[5];
    state[5] = outer;
    return crossed;
}

void ppy_io_push(int64_t kind, int64_t word) {
    int64_t *state = ppy_io_state();
    int64_t at = state[8];
    if (at < 32) {
        state[18 + 2 * at] = kind;
        state[19 + 2 * at] = word;
    }
    state[8] = at + 1;
}

void ppy_io_push_float(double value) {
    int64_t word;
    memcpy(&word, &value, 8);
    ppy_io_push(2, word);
}

/* A string argument, borrowed for the call. */
void ppy_io_push_text(int8_t *text) {
    ppy_io_push(4, (int64_t)(intptr_t)text);
}

/* Called from the hook: the thread's exception pending, as CPython raised it.
   Copies the texts. */
void ppy_io_pending(int64_t tag, int64_t number, const int8_t *name, int64_t name_bytes,
                    const int8_t *text, int64_t text_bytes) {
    int64_t *state = ppy_io_state();
    free((void *)(intptr_t)state[12]);
    free((void *)(intptr_t)state[14]);
    char *kept_name = (char *)malloc((size_t)name_bytes + 1);
    char *kept_text = (char *)malloc((size_t)text_bytes + 1);
    if (kept_name == NULL || kept_text == NULL) {
        ppy_coll_fail();
    }
    memcpy(kept_name, name, (size_t)name_bytes);
    memcpy(kept_text, text, (size_t)text_bytes);
    state[9] = 1;
    state[10] = tag;
    state[11] = number;
    state[12] = (int64_t)(intptr_t)kept_name;
    state[13] = name_bytes;
    state[14] = (int64_t)(intptr_t)kept_text;
    state[15] = text_bytes;
}

/* Called from the hook: a text result's bytes, copied. */
void ppy_io_answer_text(const int8_t *data, int64_t bytes) {
    int64_t *state = ppy_io_state();
    free((void *)(intptr_t)state[16]);
    char *kept = (char *)malloc((size_t)bytes + 1);
    if (kept == NULL) {
        ppy_coll_fail();
    }
    memcpy(kept, data, (size_t)bytes);
    state[16] = (int64_t)(intptr_t)kept;
    state[17] = bytes;
}

/* Called from the hook: any other result. */
void ppy_io_answer(int64_t kind, int64_t word) {
    int64_t *state = ppy_io_state();
    state[6] = kind;
    state[7] = word;
}

/* The exception Python raised in the hook, made native and pending: its
   class's tag (a builtin's, or the nearest builtin base's), its name, and
   str() of it; flags bit 1 marks it as Python's own, numbered in bits 8 and
   up, which the boundary raises as the very object. */
void ppy_io_raise_pending(void) {
    int64_t *state = ppy_io_state();
    state[9] = 0;
    int8_t *name = ppy_str_new((const int8_t *)(intptr_t)state[12], state[13]);
    int8_t *text = ppy_str_new((const int8_t *)(intptr_t)state[14], state[15]);
    ppy_exc_raise(ppy_exc_make(state[10], name, text, 1 | 2 | (state[11] << 8)));
}

/* A call into Python (a barrier): the callable named by `name`, with what was
   pushed, wanting a result of `kind`. Answers 0, or -1 with the exception
   pending natively. */
int64_t ppy_io_call(const int8_t *name, int64_t bytes, int64_t kind, int64_t method) {
    int64_t *state = ppy_io_state();
    if (ppy_io_barrier() != 0) {
        state[8] = 0;
        ppy_io_raise_pending();
        return -1;
    }
    int64_t done = ppy_io_hook(method ? 4 : 3, (int64_t)(intptr_t)name, bytes, kind);
    state[8] = 0;
    if (done != 0) {
        ppy_io_raise_pending();
        return -1;
    }
    return 0;
}

/* What the call answered: the word of a number, a flag, or an object. */
int64_t ppy_io_result(void) {
    return ppy_io_state()[7];
}

/* A string result: the caller's new reference. */
int8_t *ppy_io_result_text(void) {
    int64_t *state = ppy_io_state();
    return ppy_str_new((const int8_t *)(intptr_t)state[16], state[17]);
}

double ppy_io_result_float(void) {
    double value;
    memcpy(&value, &ppy_io_state()[7], 8);
    return value;
}

/* `print(..., flush=True)` and a failed flush's exception, raised natively. */
int64_t ppy_io_flush_or_raise(int64_t stream) {
    if (ppy_io_flush(stream) != 0) {
        ppy_io_raise_pending();
        return -1;
    }
    return 0;
}

/* A Python object native code let go of (a file a `with` closed). */
void ppy_io_forget(int64_t object) {
    ppy_io_hook(5, object, 0, 0);
}
