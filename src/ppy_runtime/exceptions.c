/* Exceptions raised and caught in native code.

   An exception is an object of the collections runtime: a one-record
   sequence of four words,

     [0] its class's tag        [1] its class's name, a string
     [2] str() of it, a string  [3] flags: bit 0 set where [2] is what
                                    CPython's str() says (a guard that could
                                    not spell a key leaves it clear)

   One exception at a time is pending on a thread: raised and not yet
   caught. A native function it leaves returns the raised status; a caller
   whose call sits in a `try` looks at the pending exception's tag, and
   one that does not returns the raised status too. Under `ppy run`,
   Python sees the raised status at the boundary and runs the call again as
   Python, which raises the exception itself; a standalone program's `main`
   prints it and exits. */

int64_t *ppy_exc_slot(void) {
#ifdef __cplusplus
    static thread_local int64_t pending;
#else
    static _Thread_local int64_t pending;
#endif
    return &pending;
}

/* A new exception; takes the caller's references to `name` and `message`. */
int8_t *ppy_exc_make(int64_t tag, int8_t *name, int8_t *message, int64_t flags) {
    /* Words 1 and 2 are handles, and strings: leaves the collector skips. */
    int8_t *made = ppy_seq_new(1, 4, 0, 6 | ((int64_t)6 << 32));
    int64_t *record = (int64_t *)ppy_seq_at(made, 0);
    record[0] = tag;
    record[1] = (int64_t)(intptr_t)name;
    record[2] = (int64_t)(intptr_t)message;
    record[3] = flags;
    return made;
}

int64_t *ppy_exc_record(int8_t *exception) {
    return (int64_t *)ppy_seq_at(exception, 0);
}

/* Raise `exception`, taking the caller's reference to it. */
void ppy_exc_raise(int8_t *exception) {
    int64_t *slot = ppy_exc_slot();
    int8_t *old = (int8_t *)(intptr_t)*slot;
    *slot = (int64_t)(intptr_t)exception;
    if (old != NULL) {
        ppy_coll_release(old);
    }
}

/* The pending exception's tag, or 0 where none is pending. */
int64_t ppy_exc_pending_tag(void) {
    int8_t *pending = (int8_t *)(intptr_t)*ppy_exc_slot();
    return pending == NULL ? 0 : ppy_exc_record(pending)[0];
}

/* The pending exception, handed to the caller, which catches it. */
int8_t *ppy_exc_take(void) {
    int64_t *slot = ppy_exc_slot();
    int8_t *pending = (int8_t *)(intptr_t)*slot;
    *slot = 0;
    return pending;
}

/* The runtime frees every handle a failed call left; the pending exception
   is one of them, so the slot forgets it without letting go. */
void ppy_exc_forget(void) {
    *ppy_exc_slot() = 0;
}

int64_t ppy_exc_tag(int8_t *exception) {
    return ppy_exc_record(exception)[0];
}

/* str() of an exception: a reference the caller owns. */
int8_t *ppy_exc_str(int8_t *exception) {
    int8_t *message = (int8_t *)(intptr_t)ppy_exc_record(exception)[2];
    ppy_coll_retain(message);
    return message;
}

/* Whether str() of an exception is CPython's own. */
int64_t ppy_exc_known(int8_t *exception) {
    return ppy_exc_record(exception)[3] & 1;
}

/* What CPython prints last for an exception nothing caught: its class's
   name, then `: ` and str() of it where that is not empty. */
void ppy_exc_report(void) {
    int8_t *pending = ppy_exc_take();
    if (pending == NULL) {
        fputs("RuntimeError: a native guard failed with no Python to fall back to\n", stderr);
        return;
    }
    int64_t *record = ppy_exc_record(pending);
    int8_t *name = (int8_t *)(intptr_t)record[1];
    int8_t *message = (int8_t *)(intptr_t)record[2];
    fwrite(ppy_str_raw(name), 1, (size_t)ppy_str_bytes(name), stderr);
    if (ppy_str_bytes(message) > 0) {
        fputs(": ", stderr);
        fwrite(ppy_str_raw(message), 1, (size_t)ppy_str_bytes(message), stderr);
    }
    fputc('\n', stderr);
    fflush(stderr);
    ppy_coll_release(pending);
}
