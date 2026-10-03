/* Effects in native code, at a generated wrapper (`ppy_runtime/effects.py`).

   A native function that prints, reads, or calls into Python holds its
   output until the call ends. Around the call, the wrapper does what
   `Effects` does on the Python side, in C on the common path: it enters and
   leaves the call (`ppy_io_enter`, `ppy_io_leave`), writes out what an
   answered call held (`ppy_io_commit`), and drops what a call that fell
   back held, so that Python prints it when it runs the call again. What is
   rare goes back to `Effects`: a call that raised after a barrier, a write
   that raised, and forgetting the objects and exceptions the calls left
   once the outermost one is done.

   The binder fills `ppy_io` (`ppy_effects`) with the runtime's functions and
   the `Effects` object's methods. */

typedef struct {
    int64_t (*enter)(void);
    int64_t (*leave)(int64_t);
    int64_t (*commit)(void);
    void (*discard)(void);
    int64_t *(*calls)(void);
    PyObject *crossed;
    PyObject *failed;
    PyObject *tidy;
} ppy_io_fns;

static ppy_io_fns ppy_io;
static int ppy_io_ready = 0;

/* `ppy_effects(enter, leave, commit, discard, calls, crossed, failed, tidy)`. */
static PyObject *ppy_effects(PyObject *self, PyObject *args) {
    unsigned long long a[5];
    PyObject *crossed, *failed, *tidy;
    (void)self;
    if (!PyArg_ParseTuple(args, "KKKKKOOO", &a[0], &a[1], &a[2], &a[3], &a[4], &crossed, &failed,
                          &tidy)) {
        return NULL;
    }
    for (int i = 0; i < 5; i++) {
        if (a[i] == 0) {
            ppy_io_ready = 0;
            Py_RETURN_FALSE;
        }
    }
    *(void **)(&ppy_io.enter) = (void *)(uintptr_t)a[0];
    *(void **)(&ppy_io.leave) = (void *)(uintptr_t)a[1];
    *(void **)(&ppy_io.commit) = (void *)(uintptr_t)a[2];
    *(void **)(&ppy_io.discard) = (void *)(uintptr_t)a[3];
    *(void **)(&ppy_io.calls) = (void *)(uintptr_t)a[4];
    Py_INCREF(crossed);
    Py_INCREF(failed);
    Py_INCREF(tidy);
    Py_XDECREF(ppy_io.crossed);
    Py_XDECREF(ppy_io.failed);
    Py_XDECREF(ppy_io.tidy);
    ppy_io.crossed = crossed;
    ppy_io.failed = failed;
    ppy_io.tidy = tidy;
    ppy_io_ready = 1;
    Py_RETURN_TRUE;
}

/* After a call: once the outermost is done, what the calls left in Python is
   forgotten, where one of them reached Python for it. */
static void ppy_io_settle(void) {
    int64_t *calls = ppy_io.calls();
    if (calls[0] == 0 && calls[1] != 0) {
        PyObject *done = PyObject_CallNoArgs(ppy_io.tidy);
        if (done == NULL) {
            PyErr_Clear();
        }
        Py_XDECREF(done);
    }
}

/* Raise what `Effects` says a call that crossed a barrier raised. */
static PyObject *ppy_io_raise(PyObject *made) {
    if (made == NULL) {
        return NULL;
    }
    if (PyExceptionInstance_Check(made)) {
        PyErr_SetObject((PyObject *)Py_TYPE(made), made);
    } else {
        PyErr_SetString(PyExc_RuntimeError, "PPy: the boundary lost a native exception");
    }
    Py_DECREF(made);
    return NULL;
}

static PyObject *ppy_io_crossed(int status, const char *qualname) {
    PyObject *made = PyObject_CallFunction(ppy_io.crossed, "is", status, qualname);
    return ppy_io_raise(made);
}

/* An answered call's output written out; `result` handed back, or what a
   write raised. A result that failed to convert keeps its error. */
static PyObject *ppy_io_commit_result(PyObject *result) {
    PyObject *type, *value, *traceback;
    if (result == NULL) {
        PyErr_Fetch(&type, &value, &traceback);
    }
    int64_t written = ppy_io.commit();
    if (written != 0) {
        PyObject *made = PyObject_CallNoArgs(ppy_io.failed);
        if (result == NULL) {
            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(traceback);
        }
        Py_XDECREF(result);
        return ppy_io_raise(made);
    }
    if (result == NULL) {
        PyErr_Restore(type, value, traceback);
    }
    ppy_io_settle();
    return result;
}
