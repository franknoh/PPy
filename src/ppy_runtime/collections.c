/* The collections runtime: every `ppy` collection over words.

   A collection is a handle, the address of a twenty-one-word header:

     [0] length          [1] capacity (records)   [2] records
     [3] family word a   [4] family word b        [5] family word c
     [6] family word d   [7] family word e
     [8] value words     [9] float mask           [10] handle mask
     [11] references     [12] family              [13] key words; bits 32-47
                                                       which of them are strings,
                                                       bits 48-63 which value words
     [14] scratch        [15] record stride (words)
     [16] previous on the heap list               [17] next on the heap list
     [18] the heap that holds it                  [19] collector: references
     [20] collector: state

   An element or a value is `value words` eight-byte words: an int64_t or a
   double each (a set bit in the float mask), or a collection handle (a set
   bit in the handle mask). A key is `key words` int64_t. The code calling
   these functions reads and writes the words by address, typed; what is
   here is memory, order, and ownership.

   Families and their records:

     0 sequence (Vec, Deque, Heap)  [value]                       a: first index
     1 linked list                  [value][prev][next][alive]    a: head b: tail
                                                                  c: free d: made
     2 hash map and set             [key][value][alive]           a: used b: index
                                                                  c: index size d: version
     3 tree map and set             [key][value][left][right]     a: root b: free
                                    [priority][next free]         c: made d: version
                                                                  e: generator
     4 string                       see strings.c

   A key word that is a string (a bit of word 13 above bit 32) holds a
   string handle: keys hash and compare by the text, and the collection
   holds a reference to each key it keeps. A value word that is a handle is
   ordered, where it is ordered at all, as a string: nothing else a handle
   points at has an order.

   A value word that is a string is a leaf (bits 48-63 of word 13): a
   string holds no handles, so it cannot be in a cycle, and a collection
   whose handles are all leaves is not one the collector walks. `handles`
   as `ppy_coll_make` takes it carries the leaves above bit 32.

   An object of a class is a sequence of one record, its fields.

   Every handle is on a list of the handles its thread made (see
   `ppy_coll_heap`): the collector walks it to find cycles reference counts
   cannot free, and the handles a failed call left behind are freed from it.
   The links are stored hidden (`ppy_coll_hide`), so that a leak checker
   still sees a handle nothing else points at as leaked.

   Nothing is checked here: the caller guards an index, an empty pop, a
   missing key, which is what lets a failed check fall back to Python under
   `ppy run` and stop a standalone binary. */

void ppy_coll_fail(void) {
    fflush(stdout);
    fputs("MemoryError\n", stderr);
    exit(1);
}

int8_t *ppy_coll_none(void) {
    return NULL;
}

/* A pointer as a word a leak checker does not read as one, and back. */
int64_t ppy_coll_hide(const void *pointer) {
    if (pointer == NULL) {
        return 0;
    }
    return (int64_t)((uint64_t)(uintptr_t)pointer ^ 0x5bd1e9955bd1e995ULL);
}

int64_t *ppy_coll_seen(int64_t word) {
    if (word == 0) {
        return NULL;
    }
    return (int64_t *)(uintptr_t)((uint64_t)word ^ 0x5bd1e9955bd1e995ULL);
}

/* This thread's heap: what it made and what the collector knows of it.

     [0] handles that may hold handles (hidden)   [1] handles that cannot
     [2] handles live                             [3] holders made since the
                                                      last collection
     [4] holders live after the last collection   [5] a call failed since
     [6] collecting now                           [7] holders live

   A holder is a handle whose values include handles: only holders can be
   in a cycle, so only they are walked. */
int64_t *ppy_coll_heap(void) {
#ifdef __cplusplus
    static thread_local int64_t heap[8];
#else
    static _Thread_local int64_t heap[8];
#endif
    return heap;
}

/* Whether a handle's values may reach another holder: handles that are not
   strings. */
int64_t ppy_coll_holds(const int64_t *header) {
    int64_t leaves = (int64_t)((uint64_t)header[13] >> 48);
    return (header[10] & ~leaves) != 0;
}

void ppy_coll_track(int64_t *header) {
    int64_t *heap = ppy_coll_heap();
    int64_t list = ppy_coll_holds(header) ? 0 : 1;
    int64_t *first = ppy_coll_seen(heap[list]);
    header[16] = 0;
    header[17] = heap[list];
    header[18] = ppy_coll_hide(heap);
    if (first != NULL) {
        first[16] = ppy_coll_hide(header);
    }
    heap[list] = ppy_coll_hide(header);
    heap[2]++;
    if (list == 0) {
        heap[3]++;
        heap[7]++;
    }
}

void ppy_coll_untrack(int64_t *header) {
    int64_t *heap = ppy_coll_seen(header[18]);
    int64_t list = ppy_coll_holds(header) ? 0 : 1;
    int64_t *before = ppy_coll_seen(header[16]);
    int64_t *after = ppy_coll_seen(header[17]);
    if (before != NULL) {
        before[17] = header[17];
    } else {
        heap[list] = header[17];
    }
    if (after != NULL) {
        after[16] = header[16];
    }
    heap[2]--;
    if (list == 0) {
        heap[7]--;
    }
}

/* The memory of a handle, without letting go of what it holds. */
void ppy_coll_free(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_coll_untrack(header);
    if (header[12] == 2) {
        free((void *)(intptr_t)header[4]);
    }
    if (header[12] != 4 || !header[6]) {
        /* A string's bytes may sit in its header's own block (strings.c). */
        free((void *)(intptr_t)header[2]);
    }
    free((void *)(intptr_t)header[14]);
    free(header);
}

/* A native call failed and fell back: none of the handles this thread made
   is reachable any more, since no handle outlives the call that made it.
   The failed frames return without making anything, so the first handle
   made after that frees them all. */
void ppy_coll_failed(void) {
    ppy_coll_heap()[5] = 1;
}

void ppy_coll_sweep(void) {
    int64_t *heap = ppy_coll_heap();
    heap[5] = 0;
    for (int64_t list = 0; list < 2; list++) {
        int64_t *header = ppy_coll_seen(heap[list]);
        while (header != NULL) {
            int64_t *after = ppy_coll_seen(header[17]);
            ppy_coll_free((int8_t *)header);
            header = after;
        }
    }
    heap[3] = 0;
    heap[4] = 0;
}

/* How many handles this thread holds: what the tests count leaks by. */
int64_t ppy_coll_live_handles(void) {
    return ppy_coll_heap()[2];
}

int64_t *ppy_coll_record(int8_t *handle, int64_t index) {
    int64_t *header = (int64_t *)handle;
    return (int64_t *)(intptr_t)header[2] + index * header[15];
}

/* Room for one more record: the capacity doubles. A sequence is laid out
   from its first element again; the other families keep their indices. */
void ppy_coll_grow(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t capacity = header[1] * 2;
    int64_t stride = header[15] > 0 ? header[15] : 1;
    int64_t *old = (int64_t *)(intptr_t)header[2];
    int64_t *room = (int64_t *)calloc((size_t)(capacity * stride), 8);
    if (room == NULL) {
        ppy_coll_fail();
    }
    if (header[12] == 0) {
        for (int64_t i = 0; i < header[0]; i++) {
            memcpy(room + i * stride, old + ((header[3] + i) % header[1]) * stride,
                   (size_t)(stride * 8));
        }
        header[3] = 0;
    } else {
        memcpy(room, old, (size_t)(header[1] * stride * 8));
    }
    free(old);
    header[1] = capacity;
    header[2] = (int64_t)(intptr_t)room;
}

int64_t ppy_coll_len(int8_t *handle) {
    return ((int64_t *)handle)[0];
}

int64_t ppy_coll_field(int8_t *handle, int64_t field) {
    return ((int64_t *)handle)[field];
}

int8_t *ppy_coll_scratch(int8_t *handle) {
    return (int8_t *)(intptr_t)((int64_t *)handle)[14];
}

void ppy_coll_retain(int8_t *handle) {
    if (handle != NULL) {
        ((int64_t *)handle)[11]++;
    }
}

/* Mark the key words in `mask` as strings. */
void ppy_coll_text_keys(int8_t *handle, int64_t mask) {
    ((int64_t *)handle)[13] |= mask << 32;
}

int64_t ppy_coll_key_text(int8_t *handle) {
    return (int64_t)(((uint64_t)((int64_t *)handle)[13] >> 32) & 0xFFFF);
}

/* Take (1) or drop (-1) a reference to each string word of a key. */
void ppy_coll_hold_key(int8_t *handle, int64_t *key, int64_t delta) {
    int64_t text = ppy_coll_key_text(handle);
    for (int64_t w = 0; text != 0 && w < 32; w++) {
        if ((text >> w) & 1) {
            if (delta > 0) {
                ppy_coll_retain((int8_t *)(intptr_t)key[w]);
            } else {
                ppy_coll_release((int8_t *)(intptr_t)key[w]);
            }
        }
    }
}

/* Let go of the strings the live keys hold. */
void ppy_coll_release_keys(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t keys = header[13] & 0xFFFFFFFF;
    if (ppy_coll_key_text(handle) == 0 || (header[12] != 2 && header[12] != 3)) {
        return;
    }
    for (int64_t i = 0; i < header[1]; i++) {
        int64_t *value = ppy_coll_live(handle, i);
        if (value != NULL) {
            ppy_coll_hold_key(handle, value - keys, -1);
        }
    }
}

/* Is record `index` in use: a sequence's are the first `length` from its
   start, the others carry a flag. */
int64_t *ppy_coll_live(int8_t *handle, int64_t index) {
    int64_t *header = (int64_t *)handle;
    int64_t family = header[12];
    if (family == 0) {
        if (index >= header[0]) {
            return NULL;
        }
        return ppy_coll_record(handle, (header[3] + index) % header[1]);
    }
    int64_t made = family == 1 ? header[6] : family == 2 ? header[3] : header[5];
    if (index >= made) {
        return NULL;
    }
    int64_t *record = ppy_coll_record(handle, index);
    int64_t words = header[8];
    int64_t keys = (header[13] & 0xFFFFFFFF);
    if (family == 1) {
        return record[words + 2] ? record : NULL;
    }
    if (family == 2) {
        return record[keys + words] ? record + keys : NULL;
    }
    return record[keys + words + 2] >= 0 ? record + keys : NULL;
}

void ppy_coll_release(int8_t *handle) {
    if (handle == NULL) {
        return;
    }
    int64_t *header = (int64_t *)handle;
    if (--header[11] > 0) {
        return;
    }
    ppy_coll_release_keys(handle);
    if (header[10] != 0) {
        int64_t count = header[12] == 0 ? header[0] : header[1];
        for (int64_t i = 0; i < count; i++) {
            int64_t *value = ppy_coll_live(handle, i);
            for (int64_t w = 0; value != NULL && w < header[8]; w++) {
                if ((header[10] >> w) & 1) {
                    ppy_coll_release((int8_t *)(intptr_t)value[w]);
                }
            }
        }
    }
    ppy_coll_free(handle);
}

/* The collector: trial deletion over this thread's holders, as CPython's
   `gc` does it. Each holder's references, less those from other holders,
   are the ones from outside: a name, a native frame, a holder of another
   thread. What those reach is kept; what is left is only held by itself, a
   cycle, and is freed. Returns how many handles it freed. */
int64_t ppy_coll_collect(void) {
    int64_t *heap = ppy_coll_heap();
    if (heap[6]) {
        return 0;
    }
    heap[6] = 1;
    int64_t count = 0;
    for (int64_t *h = ppy_coll_seen(heap[0]); h != NULL; h = ppy_coll_seen(h[17])) {
        h[19] = h[11];
        h[20] = 1;
        count++;
    }
    int64_t **stack = (int64_t **)malloc((size_t)(count > 0 ? count : 1) * sizeof(int64_t *));
    if (stack == NULL) {
        ppy_coll_fail();
    }
    for (int64_t *h = ppy_coll_seen(heap[0]); h != NULL; h = ppy_coll_seen(h[17])) {
        int64_t records = h[12] == 0 ? h[0] : h[1];
        for (int64_t i = 0; i < records; i++) {
            int64_t *value = ppy_coll_live((int8_t *)h, i);
            for (int64_t w = 0; value != NULL && w < h[8]; w++) {
                int64_t *child = (int64_t *)(intptr_t)value[w];
                if (((h[10] >> w) & 1) && child != NULL && child[20] == 1) {
                    child[19]--;
                }
            }
        }
    }
    int64_t top = 0;
    for (int64_t *h = ppy_coll_seen(heap[0]); h != NULL; h = ppy_coll_seen(h[17])) {
        if (h[20] == 1 && h[19] > 0) {
            h[20] = 2;
            stack[top++] = h;
        }
    }
    while (top > 0) {
        int64_t *h = stack[--top];
        int64_t records = h[12] == 0 ? h[0] : h[1];
        for (int64_t i = 0; i < records; i++) {
            int64_t *value = ppy_coll_live((int8_t *)h, i);
            for (int64_t w = 0; value != NULL && w < h[8]; w++) {
                int64_t *child = (int64_t *)(intptr_t)value[w];
                if (((h[10] >> w) & 1) && child != NULL && child[20] == 1) {
                    child[20] = 2;
                    stack[top++] = child;
                }
            }
        }
    }
    int64_t garbage = 0;
    for (int64_t *h = ppy_coll_seen(heap[0]); h != NULL; h = ppy_coll_seen(h[17])) {
        if (h[20] == 1) {
            h[20] = 3;
            stack[garbage++] = h;
        } else {
            h[20] = 0;
        }
    }
    /* Break every cycle first: a value that is garbage loses the reference
       without being freed yet, anything else is let go of as usual (it is
       reachable, or holds no handles, so nothing garbage is freed twice). */
    for (int64_t g = 0; g < garbage; g++) {
        int64_t *h = stack[g];
        int64_t records = h[12] == 0 ? h[0] : h[1];
        for (int64_t i = 0; i < records; i++) {
            int64_t *value = ppy_coll_live((int8_t *)h, i);
            for (int64_t w = 0; value != NULL && w < h[8]; w++) {
                int64_t *child = (int64_t *)(intptr_t)value[w];
                if (!((h[10] >> w) & 1) || child == NULL) {
                    continue;
                }
                value[w] = 0;
                if (child[20] == 3) {
                    child[11]--;
                } else {
                    ppy_coll_release((int8_t *)child);
                }
            }
        }
    }
    for (int64_t g = 0; g < garbage; g++) {
        /* Keys are strings or numbers, never in a cycle: let go as usual. */
        ppy_coll_release_keys((int8_t *)stack[g]);
        ppy_coll_free((int8_t *)stack[g]);
    }
    free(stack);
    heap[3] = 0;
    heap[4] = heap[7];
    heap[6] = 0;
    return garbage;
}

/* Let go of the collections the values hold, where the values are handles. */
void ppy_coll_release_values(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_coll_release_keys(handle);
    if (header[10] == 0) {
        return;
    }
    int64_t count = header[12] == 0 ? header[0] : header[1];
    for (int64_t i = 0; i < count; i++) {
        int64_t *value = ppy_coll_live(handle, i);
        for (int64_t w = 0; value != NULL && w < header[8]; w++) {
            if ((header[10] >> w) & 1) {
                ppy_coll_release((int8_t *)(intptr_t)value[w]);
            }
        }
    }
}

int8_t *ppy_coll_make(int64_t family, int64_t keys, int64_t words, int64_t floats,
                      int64_t handles, int64_t stride, int64_t capacity) {
    int64_t *heap = ppy_coll_heap();
    if (heap[5]) {
        ppy_coll_sweep();
    }
    int64_t leaves = (int64_t)((uint64_t)handles >> 32);
    handles &= 0xFFFFFFFF;
    if ((handles & ~leaves) != 0 && heap[3] >= 700 + heap[4] && !heap[6]) {
        ppy_coll_collect();
    }
    if (family == 4) {
        /* A string keeps its bytes right after its header, in one block, and
           says so in word 6 until it outgrows them (strings.c). */
        int64_t room = capacity > 0 ? capacity : 1;
        int64_t *text = (int64_t *)calloc((size_t)(21 + room), sizeof(int64_t));
        if (text == NULL) {
            ppy_coll_fail();
        }
        text[1] = room;
        text[2] = (int64_t)(intptr_t)(text + 21);
        text[6] = 1;
        text[8] = words;
        text[11] = 1;
        text[12] = 4;
        text[15] = stride;
        ppy_coll_track(text);
        return (int8_t *)text;
    }
    int64_t *header = (int64_t *)calloc(21, sizeof(int64_t));
    int64_t room = capacity > 0 ? capacity : 1;
    int64_t *records = (int64_t *)calloc((size_t)(room * (stride > 0 ? stride : 1)), 8);
    int64_t spare = 2 * (words > keys ? (words > 0 ? words : 1) : keys);
    int64_t *scratch = (int64_t *)calloc((size_t)spare, 8);
    if (header == NULL || records == NULL || scratch == NULL) {
        ppy_coll_fail();
    }
    header[1] = room;
    header[2] = (int64_t)(intptr_t)records;
    header[8] = words;
    header[9] = floats;
    header[10] = handles;
    header[11] = 1;
    header[12] = family;
    header[13] = keys | (int64_t)((uint64_t)leaves << 48);
    header[14] = (int64_t)(intptr_t)scratch;
    header[15] = stride;
    ppy_coll_track(header);
    return (int8_t *)header;
}

/* Python's `<` over tuples of numbers: the first words that differ decide. */
int64_t ppy_coll_less(const int64_t *a, const int64_t *b, int64_t words, int64_t floats,
                      int64_t handles) {
    for (int64_t w = 0; w < words; w++) {
        if ((handles >> w) & 1) {
            int64_t order = ppy_str_order((int8_t *)(intptr_t)a[w], (int8_t *)(intptr_t)b[w]);
            if (order == 0) {
                continue;
            }
            return order < 0;
        }
        if ((floats >> w) & 1) {
            double x, y;
            memcpy(&x, &a[w], 8);
            memcpy(&y, &b[w], 8);
            if (x == y) {
                continue;
            }
            return x < y;
        }
        if (a[w] == b[w]) {
            continue;
        }
        return a[w] < b[w];
    }
    return 0;
}

/* -- sequences: Vec, Deque, Heap ---------------------------------------- */

int8_t *ppy_seq_new(int64_t count, int64_t words, int64_t floats, int64_t handles) {
    int8_t *handle = ppy_coll_make(0, 0, words, floats, handles, words, count);
    ((int64_t *)handle)[0] = count;
    return handle;
}

int8_t *ppy_seq_at(int8_t *handle, int64_t index) {
    int64_t *header = (int64_t *)handle;
    return (int8_t *)ppy_coll_record(handle, (header[3] + index) % header[1]);
}

int8_t *ppy_seq_push_back(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    if (header[0] == header[1]) {
        ppy_coll_grow(handle);
    }
    return ppy_seq_at(handle, header[0]++);
}

int8_t *ppy_seq_push_front(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    if (header[0] == header[1]) {
        ppy_coll_grow(handle);
    }
    header[3] = (header[3] + header[1] - 1) % header[1];
    header[0]++;
    return ppy_seq_at(handle, 0);
}

int8_t *ppy_seq_pop_back(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    header[0]--;
    return ppy_seq_at(handle, header[0]);
}

int8_t *ppy_seq_pop_front(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int8_t *first = ppy_seq_at(handle, 0);
    header[3] = (header[3] + 1) % header[1];
    header[0]--;
    return first;
}

void ppy_seq_clear(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_coll_release_values(handle);
    header[0] = 0;
    header[3] = 0;
}

void ppy_seq_reverse(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *spare = (int64_t *)(intptr_t)header[14];
    for (int64_t i = 0, j = header[0] - 1; i < j; i++, j--) {
        int8_t *low = ppy_seq_at(handle, i);
        int8_t *high = ppy_seq_at(handle, j);
        memcpy(spare, low, (size_t)(words * 8));
        memcpy(low, high, (size_t)(words * 8));
        memcpy(high, spare, (size_t)(words * 8));
    }
}

/* A stable merge sort, as Python's sort is stable: equal elements keep their
   order, which is visible where equal is not identical (0.0 and -0.0). */
void ppy_seq_sort(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    int64_t words = header[8];
    if (n < 2) {
        return;
    }
    int64_t *items = (int64_t *)calloc((size_t)(n * words), 8);
    int64_t *spare = (int64_t *)calloc((size_t)(n * words), 8);
    if (items == NULL || spare == NULL) {
        ppy_coll_fail();
    }
    for (int64_t i = 0; i < n; i++) {
        memcpy(items + i * words, ppy_seq_at(handle, i), (size_t)(words * 8));
    }
    for (int64_t width = 1; width < n; width *= 2) {
        for (int64_t low = 0; low < n; low += 2 * width) {
            int64_t middle = low + width < n ? low + width : n;
            int64_t high = low + 2 * width < n ? low + 2 * width : n;
            int64_t i = low, j = middle, k = low;
            while (i < middle && j < high) {
                int take = (int)ppy_coll_less(items + j * words, items + i * words, words, header[9],
                                                 header[10]);
                int64_t from = take ? j++ : i++;
                memcpy(spare + (k++) * words, items + from * words, (size_t)(words * 8));
            }
            while (i < middle) {
                memcpy(spare + (k++) * words, items + (i++) * words, (size_t)(words * 8));
            }
            while (j < high) {
                memcpy(spare + (k++) * words, items + (j++) * words, (size_t)(words * 8));
            }
        }
        int64_t *swap = items;
        items = spare;
        spare = swap;
    }
    free((void *)(intptr_t)header[2]);
    free(spare);
    header[1] = n;
    header[2] = (int64_t)(intptr_t)items;
    header[3] = 0;
}

/* The heap's order: `a` comes out before `b`. */
int64_t ppy_heap_before(int8_t *handle, const int64_t *a, const int64_t *b, int64_t max) {
    int64_t *header = (int64_t *)handle;
    return max ? ppy_coll_less(b, a, header[8], header[9], header[10])
               : ppy_coll_less(a, b, header[8], header[9], header[10]);
}

/* The value in the scratch words, sifted up into place. */
void ppy_heap_push(int8_t *handle, int64_t max) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *value = (int64_t *)(intptr_t)header[14];
    ppy_seq_push_back(handle);
    int64_t i = header[0] - 1;
    while (i > 0) {
        int64_t parent = (i - 1) / 2;
        int64_t *above = ppy_coll_record(handle, parent);
        if (!ppy_heap_before(handle, value, above, max)) {
            break;
        }
        memcpy(ppy_coll_record(handle, i), above, (size_t)(words * 8));
        i = parent;
    }
    memcpy(ppy_coll_record(handle, i), value, (size_t)(words * 8));
}

/* The first element out, into the scratch words, which are what this returns. */
int8_t *ppy_heap_pop(int8_t *handle, int64_t max) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *top = (int64_t *)(intptr_t)header[14];
    int64_t *last = top + (words > (header[13] & 0xFFFFFFFF) ? words : (header[13] & 0xFFFFFFFF));
    memcpy(top, ppy_coll_record(handle, 0), (size_t)(words * 8));
    header[0]--;
    int64_t n = header[0];
    if (n > 0) {
        memcpy(last, ppy_coll_record(handle, n), (size_t)(words * 8));
        int64_t i = 0;
        while (1) {
            int64_t child = 2 * i + 1;
            if (child >= n) {
                break;
            }
            if (child + 1 < n
                && ppy_heap_before(handle, ppy_coll_record(handle, child + 1),
                                   ppy_coll_record(handle, child), max)) {
                child++;
            }
            if (!ppy_heap_before(handle, ppy_coll_record(handle, child), last, max)) {
                break;
            }
            memcpy(ppy_coll_record(handle, i), ppy_coll_record(handle, child), (size_t)(words * 8));
            i = child;
        }
        memcpy(ppy_coll_record(handle, i), last, (size_t)(words * 8));
    }
    return (int8_t *)top;
}

/* -- LinkedList ----------------------------------------------------------- */

int8_t *ppy_list_new(int64_t words, int64_t floats, int64_t handles) {
    int8_t *handle = ppy_coll_make(1, 0, words, floats, handles, words + 3, 4);
    int64_t *header = (int64_t *)handle;
    header[3] = -1;
    header[4] = -1;
    header[5] = -1;
    return handle;
}

/* A new node linked between `before` and `after`, its value all zero; its id.
   The last freed id is the next one given out, as the reference gives them. */
int64_t ppy_list_node(int8_t *handle, int64_t before, int64_t after) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t node = header[5];
    if (node >= 0) {
        header[5] = ppy_coll_record(handle, node)[words + 1];
    } else {
        if (header[6] == header[1]) {
            ppy_coll_grow(handle);
        }
        node = header[6]++;
    }
    int64_t *record = ppy_coll_record(handle, node);
    memset(record, 0, (size_t)(words * 8));
    record[words] = before;
    record[words + 1] = after;
    record[words + 2] = 1;
    if (before == -1) {
        header[3] = node;
    } else {
        ppy_coll_record(handle, before)[words + 1] = node;
    }
    if (after == -1) {
        header[4] = node;
    } else {
        ppy_coll_record(handle, after)[words] = node;
    }
    header[0]++;
    return node;
}

int8_t *ppy_list_at(int8_t *handle, int64_t node) {
    return (int8_t *)ppy_coll_record(handle, node);
}

int64_t ppy_list_valid(int8_t *handle, int64_t node) {
    int64_t *header = (int64_t *)handle;
    if (node < 0 || node >= header[6]) {
        return 0;
    }
    return ppy_coll_record(handle, node)[header[8] + 2] ? 1 : 0;
}

/* `node` out of the chain and onto the free stack; its value stays readable
   until the id is given out again. */
void ppy_list_unlink(int8_t *handle, int64_t node) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *record = ppy_coll_record(handle, node);
    int64_t before = record[words];
    int64_t after = record[words + 1];
    if (before == -1) {
        header[3] = after;
    } else {
        ppy_coll_record(handle, before)[words + 1] = after;
    }
    if (after == -1) {
        header[4] = before;
    } else {
        ppy_coll_record(handle, after)[words] = before;
    }
    record[words] = -1;
    record[words + 1] = header[5];
    record[words + 2] = 0;
    header[5] = node;
    header[0]--;
}

/* 0 the previous node, 1 the next, 2 whether it is in the list. */
int64_t ppy_list_step(int8_t *handle, int64_t node, int64_t field) {
    int64_t *header = (int64_t *)handle;
    return ppy_coll_record(handle, node)[header[8] + field];
}

/* The node after `node` while walking, or -1: a removed node has none, as
   the reference's walk ends at one. */
int64_t ppy_list_after(int8_t *handle, int64_t node) {
    if (!ppy_list_valid(handle, node)) {
        return -1;
    }
    return ppy_list_step(handle, node, 1);
}

void ppy_list_clear(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_coll_release_values(handle);
    header[0] = 0;
    header[3] = -1;
    header[4] = -1;
    header[5] = -1;
    header[6] = 0;
}

/* -- HashMap and HashSet: entries in insertion order, an index over them -- */

int64_t ppy_map_hash(int8_t *handle, const int64_t *key, int64_t mask) {
    int64_t keys = ((int64_t *)handle)[13] & 0xFFFFFFFF;
    return ppy_str_key_hash(key, keys, ppy_coll_key_text(handle)) & mask;
}

void ppy_map_reindex(int8_t *handle, int64_t size) {
    int64_t *header = (int64_t *)handle;
    free((void *)(intptr_t)header[4]);
    int64_t *index = (int64_t *)malloc((size_t)size * sizeof(int64_t));
    if (index == NULL) {
        ppy_coll_fail();
    }
    for (int64_t i = 0; i < size; i++) {
        index[i] = -1;
    }
    int64_t keys = (header[13] & 0xFFFFFFFF);
    int64_t alive = keys + header[8];
    for (int64_t e = 0; e < header[3]; e++) {
        int64_t *record = ppy_coll_record(handle, e);
        if (record[alive]) {
            int64_t i = ppy_map_hash(handle, record, size - 1);
            while (index[i] != -1) {
                i = (i + 1) & (size - 1);
            }
            index[i] = e;
        }
    }
    header[4] = (int64_t)(intptr_t)index;
    header[5] = size;
}

int8_t *ppy_map_new(int64_t keys, int64_t words, int64_t floats, int64_t handles) {
    int8_t *handle = ppy_coll_make(2, keys, words, floats, handles, keys + words + 1, 8);
    ppy_map_reindex(handle, 16);
    return handle;
}

/* Where `key` sits in the index, or -1. */
int64_t ppy_map_slot(int8_t *handle, const int8_t *key) {
    int64_t *header = (int64_t *)handle;
    const int64_t *wanted = (const int64_t *)key;
    int64_t *index = (int64_t *)(intptr_t)header[4];
    int64_t keys = (header[13] & 0xFFFFFFFF);
    int64_t mask = header[5] - 1;
    int64_t text = ppy_coll_key_text(handle);
    int64_t i = ppy_map_hash(handle, wanted, mask);
    while (index[i] != -1) {
        int64_t e = index[i];
        if (e >= 0 && ppy_str_key_order(ppy_coll_record(handle, e), wanted, keys, text) == 0) {
            return i;
        }
        i = (i + 1) & mask;
    }
    return -1;
}

int64_t ppy_map_find(int8_t *handle, const int8_t *key) {
    int64_t at = ppy_map_slot(handle, key);
    return at < 0 ? -1 : ((int64_t *)(intptr_t)((int64_t *)handle)[4])[at];
}

/* The entry for `key`, made with an all-zero value where there was none:
   room for it by dropping removed entries or doubling, then its slot. */
int64_t ppy_map_put(int8_t *handle, const int8_t *key) {
    int64_t *header = (int64_t *)handle;
    int64_t found = ppy_map_find(handle, key);
    if (found >= 0) {
        return found;
    }
    int64_t keys = (header[13] & 0xFFFFFFFF);
    int64_t stride = header[15];
    int64_t alive = keys + header[8];
    if (header[3] == header[1]) {
        int64_t kept = 0;
        for (int64_t e = 0; e < header[3]; e++) {
            int64_t *record = ppy_coll_record(handle, e);
            if (record[alive]) {
                memmove(ppy_coll_record(handle, kept), record, (size_t)(stride * 8));
                kept++;
            }
        }
        header[3] = kept;
        if (kept * 2 > header[1]) {
            ppy_coll_grow(handle);
        }
        ppy_map_reindex(handle, header[1] * 2);
    }
    int64_t e = header[3]++;
    int64_t *record = ppy_coll_record(handle, e);
    memset(record, 0, (size_t)(stride * 8));
    memcpy(record, key, (size_t)(keys * 8));
    ppy_coll_hold_key(handle, record, 1);
    record[alive] = 1;
    int64_t *index = (int64_t *)(intptr_t)header[4];
    int64_t mask = header[5] - 1;
    int64_t i = ppy_map_hash(handle, record, mask);
    while (index[i] >= 0) {
        i = (i + 1) & mask;
    }
    index[i] = e;
    header[0]++;
    header[6]++;
    return e;
}

int8_t *ppy_map_value_at(int8_t *handle, int64_t entry) {
    return (int8_t *)(ppy_coll_record(handle, entry) + (((int64_t *)handle)[13] & 0xFFFFFFFF));
}

int8_t *ppy_map_key_at(int8_t *handle, int64_t entry) {
    return (int8_t *)ppy_coll_record(handle, entry);
}

int64_t ppy_map_alive(int8_t *handle, int64_t entry) {
    int64_t *header = (int64_t *)handle;
    return ppy_coll_record(handle, entry)[(header[13] & 0xFFFFFFFF) + header[8]];
}

/* `key` out; the entry, whose value stays readable until the next insertion,
   or -1. */
int64_t ppy_map_remove(int8_t *handle, const int8_t *key) {
    int64_t *header = (int64_t *)handle;
    int64_t at = ppy_map_slot(handle, key);
    if (at < 0) {
        return -1;
    }
    int64_t *index = (int64_t *)(intptr_t)header[4];
    int64_t e = index[at];
    index[at] = -2;
    ppy_coll_record(handle, e)[(header[13] & 0xFFFFFFFF) + header[8]] = 0;
    ppy_coll_hold_key(handle, ppy_coll_record(handle, e), -1);
    header[0]--;
    header[6]++;
    return e;
}

void ppy_map_clear(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_coll_release_values(handle);
    header[0] = 0;
    header[3] = 0;
    header[6]++;
    ppy_map_reindex(handle, header[5]);
}

/* -- TreeMap and TreeSet: a treap over an array of nodes ---------------------- */

int8_t *ppy_tree_new(int64_t keys, int64_t words, int64_t floats, int64_t handles) {
    int8_t *handle = ppy_coll_make(3, keys, words, floats, handles, keys + words + 4, 8);
    int64_t *header = (int64_t *)handle;
    header[3] = -1;
    header[4] = -1;
    header[7] = (int64_t)0x2545F4914F6CDD1DULL;
    return handle;
}

int64_t ppy_tree_order(int8_t *handle, const int64_t *a, const int64_t *b) {
    int64_t keys = ((int64_t *)handle)[13] & 0xFFFFFFFF;
    return ppy_str_key_order(a, b, keys, ppy_coll_key_text(handle));
}

/* The link words of a node: 0 left, 1 right, 2 priority, 3 next free. */
int64_t *ppy_tree_link(int8_t *handle, int64_t node, int64_t field) {
    int64_t *header = (int64_t *)handle;
    return ppy_coll_record(handle, node) + (header[13] & 0xFFFFFFFF) + header[8] + field;
}

void ppy_tree_split(int8_t *handle, int64_t node, const int64_t *key, int64_t *low,
                    int64_t *high) {
    if (node == -1) {
        *low = -1;
        *high = -1;
        return;
    }
    if (ppy_tree_order(handle, ppy_coll_record(handle, node), key) < 0) {
        ppy_tree_split(handle, *ppy_tree_link(handle, node, 1), key,
                       ppy_tree_link(handle, node, 1), high);
        *low = node;
    } else {
        ppy_tree_split(handle, *ppy_tree_link(handle, node, 0), key, low,
                       ppy_tree_link(handle, node, 0));
        *high = node;
    }
}

int64_t ppy_tree_merge(int8_t *handle, int64_t low, int64_t high) {
    if (low == -1) {
        return high;
    }
    if (high == -1) {
        return low;
    }
    if (*ppy_tree_link(handle, low, 2) > *ppy_tree_link(handle, high, 2)) {
        *ppy_tree_link(handle, low, 1) = ppy_tree_merge(handle, *ppy_tree_link(handle, low, 1), high);
        return low;
    }
    *ppy_tree_link(handle, high, 0) = ppy_tree_merge(handle, low, *ppy_tree_link(handle, high, 0));
    return high;
}

int64_t ppy_tree_erase(int8_t *handle, int64_t node, const int64_t *key) {
    int64_t *header = (int64_t *)handle;
    if (node == -1) {
        return -1;
    }
    int64_t order = ppy_tree_order(handle, key, ppy_coll_record(handle, node));
    if (order == 0) {
        int64_t joined = ppy_tree_merge(handle, *ppy_tree_link(handle, node, 0),
                                        *ppy_tree_link(handle, node, 1));
        *ppy_tree_link(handle, node, 2) = -1;
        *ppy_tree_link(handle, node, 3) = header[4];
        header[4] = node;
        return joined;
    }
    int64_t side = order < 0 ? 0 : 1;
    *ppy_tree_link(handle, node, side) = ppy_tree_erase(handle, *ppy_tree_link(handle, node, side), key);
    return node;
}

int64_t ppy_tree_find(int8_t *handle, const int8_t *key) {
    int64_t *header = (int64_t *)handle;
    const int64_t *wanted = (const int64_t *)key;
    int64_t node = header[3];
    while (node != -1) {
        int64_t order = ppy_tree_order(handle, wanted, ppy_coll_record(handle, node));
        if (order == 0) {
            return node;
        }
        node = *ppy_tree_link(handle, node, order < 0 ? 0 : 1);
    }
    return -1;
}

/* The node for `key`, made with an all-zero value where there was none. */
int64_t ppy_tree_put(int8_t *handle, const int8_t *key) {
    int64_t *header = (int64_t *)handle;
    int64_t found = ppy_tree_find(handle, key);
    if (found >= 0) {
        return found;
    }
    int64_t node = header[4];
    if (node >= 0) {
        header[4] = *ppy_tree_link(handle, node, 3);
    } else {
        if (header[5] == header[1]) {
            ppy_coll_grow(handle);
        }
        node = header[5]++;
    }
    uint64_t x = (uint64_t)header[7];
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    header[7] = (int64_t)x;
    int64_t *record = ppy_coll_record(handle, node);
    memset(record, 0, (size_t)(header[15] * 8));
    memcpy(record, key, (size_t)((header[13] & 0xFFFFFFFF) * 8));
    ppy_coll_hold_key(handle, record, 1);
    *ppy_tree_link(handle, node, 0) = -1;
    *ppy_tree_link(handle, node, 1) = -1;
    *ppy_tree_link(handle, node, 2) = (int64_t)(x >> 1);
    int64_t low = -1;
    int64_t high = -1;
    ppy_tree_split(handle, header[3], record, &low, &high);
    header[3] = ppy_tree_merge(handle, ppy_tree_merge(handle, low, node), high);
    header[0]++;
    header[6]++;
    return node;
}

/* `key` out; its node, whose value stays readable until the next insertion,
   or -1. */
int64_t ppy_tree_remove(int8_t *handle, const int8_t *key) {
    int64_t *header = (int64_t *)handle;
    int64_t node = ppy_tree_find(handle, key);
    if (node < 0) {
        return -1;
    }
    header[3] = ppy_tree_erase(handle, header[3], (const int64_t *)key);
    ppy_coll_hold_key(handle, ppy_coll_record(handle, node), -1);
    header[0]--;
    header[6]++;
    return node;
}

/* The nearest key's node: 0 at most `key`, 1 at least, 2 below, 3 above. */
int64_t ppy_tree_bound(int8_t *handle, const int8_t *key, int64_t mode) {
    int64_t *header = (int64_t *)handle;
    const int64_t *wanted = (const int64_t *)key;
    int64_t node = header[3];
    int64_t best = -1;
    while (node != -1) {
        int64_t order = ppy_tree_order(handle, ppy_coll_record(handle, node), wanted);
        int take = mode == 0 ? order <= 0 : mode == 1 ? order >= 0 : mode == 2 ? order < 0 : order > 0;
        if (take) {
            best = node;
        }
        int right = mode == 0 || mode == 2 ? take : !take;
        node = *ppy_tree_link(handle, node, right ? 1 : 0);
    }
    return best;
}

/* The first node (0) or the last (1), or -1. */
int64_t ppy_tree_end(int8_t *handle, int64_t last) {
    int64_t *header = (int64_t *)handle;
    int64_t node = header[3];
    while (node != -1 && *ppy_tree_link(handle, node, last) != -1) {
        node = *ppy_tree_link(handle, node, last);
    }
    return node;
}

int8_t *ppy_tree_key_at(int8_t *handle, int64_t node) {
    return (int8_t *)ppy_coll_record(handle, node);
}

int8_t *ppy_tree_value_at(int8_t *handle, int64_t node) {
    return (int8_t *)(ppy_coll_record(handle, node) + (((int64_t *)handle)[13] & 0xFFFFFFFF));
}

void ppy_tree_clear(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_coll_release_values(handle);
    header[0] = 0;
    header[3] = -1;
    header[4] = -1;
    header[5] = 0;
    header[6]++;
}

/* -- whole collections: equality, search, copies, and bulk moves ---------------

   Equality is Python's: a double compares as a double, so 0.0 equals -0.0,
   and a collection value compares by what it holds. CPython compares an
   element with itself as equal before asking `==`, which a NaN notices: the
   same NaN object is equal to itself and a different one is not. Native
   memory has no objects to tell apart, so a comparison that meets a NaN
   answers -1, and the caller hands the question back to Python. */

/* A map's or a set's entry for `key`, whichever family it is: its record's
   index, or -1. */
int64_t ppy_coll_find_key(int8_t *handle, const int8_t *key) {
    return ((int64_t *)handle)[12] == 2 ? ppy_map_find(handle, key) : ppy_tree_find(handle, key);
}

/* The value words of entry `entry` of a map, whichever family it is. */
int64_t *ppy_coll_value_words(int8_t *handle, int64_t entry) {
    return ppy_coll_record(handle, entry) + ((int64_t *)handle)[13];
}

/* The position after `at` in walking order (-1 at the end), and the first
   position (`at` -1): a sequence counts, a list follows links, a map skips
   removed entries, and a tree finds the next key. */
int64_t ppy_coll_step(int8_t *handle, int64_t at) {
    int64_t *header = (int64_t *)handle;
    int64_t family = header[12];
    if (family == 0) {
        return at + 1 < header[0] ? at + 1 : -1;
    }
    if (family == 1) {
        return at < 0 ? header[3] : ppy_list_after(handle, at);
    }
    if (family == 2) {
        for (int64_t e = at + 1; e < header[3]; e++) {
            if (ppy_map_alive(handle, e)) {
                return e;
            }
        }
        return -1;
    }
    if (at < 0) {
        return ppy_tree_end(handle, 0);
    }
    return ppy_tree_bound(handle, (const int8_t *)ppy_coll_record(handle, at), 3);
}

/* Where a walk's current element or key starts: `at` from `ppy_coll_step`. */
int64_t *ppy_coll_at(int8_t *handle, int64_t at) {
    if (((int64_t *)handle)[12] == 0) {
        return (int64_t *)ppy_seq_at(handle, at);
    }
    return ppy_coll_record(handle, at);
}

/* Equality in one function, so that it may call itself for a collection
   inside a collection: the value words `x` and `y` (`words` long, with their
   masks) where `x` is given, and otherwise the collections `a` and `b` whole.
   1 equal, 0 not, -1 undecided (a NaN). */
int64_t ppy_coll_compare(const int64_t *x, const int64_t *y, int64_t words, int64_t floats,
                         int64_t handles, int8_t *a, int8_t *b) {
    if (x != NULL) {
        for (int64_t w = 0; w < words; w++) {
            if ((floats >> w) & 1) {
                double p, q;
                memcpy(&p, &x[w], 8);
                memcpy(&q, &y[w], 8);
                if (p != p || q != q) {
                    return -1;
                }
                if (p != q) {
                    return 0;
                }
            } else if ((handles >> w) & 1) {
                int64_t inner = ppy_coll_compare(NULL, NULL, 0, 0, 0, (int8_t *)(intptr_t)x[w],
                                                 (int8_t *)(intptr_t)y[w]);
                if (inner != 1) {
                    return inner;
                }
            } else if (x[w] != y[w]) {
                return 0;
            }
        }
        return 1;
    }
    if (a == b) {
        return 1;
    }
    if (a == NULL || b == NULL) {
        return 0;
    }
    int64_t *ha = (int64_t *)a;
    int64_t *hb = (int64_t *)b;
    if (ha[0] != hb[0]) {
        return 0;
    }
    if (ha[12] <= 1) {
        int64_t i = ppy_coll_step(a, -1);
        int64_t j = ppy_coll_step(b, -1);
        while (i >= 0 && j >= 0) {
            int64_t same = ppy_coll_compare(ppy_coll_at(a, i), ppy_coll_at(b, j), ha[8], ha[9],
                                            ha[10], NULL, NULL);
            if (same != 1) {
                return same;
            }
            i = ppy_coll_step(a, i);
            j = ppy_coll_step(b, j);
        }
        return 1;
    }
    for (int64_t e = ppy_coll_step(a, -1); e >= 0; e = ppy_coll_step(a, e)) {
        int64_t other = ppy_coll_find_key(b, (const int8_t *)ppy_coll_record(a, e));
        if (other < 0) {
            return 0;
        }
        int64_t same = ppy_coll_compare(ppy_coll_value_words(a, e), ppy_coll_value_words(b, other),
                                        ha[8], ha[9], ha[10], NULL, NULL);
        if (same != 1) {
            return same;
        }
    }
    return 1;
}

int64_t ppy_coll_equal(int8_t *a, int8_t *b) {
    return ppy_coll_compare(NULL, NULL, 0, 0, 0, a, b);
}

int64_t ppy_coll_same(const int64_t *x, const int64_t *y, int64_t words, int64_t floats,
                      int64_t handles) {
    return ppy_coll_compare(x, y, words, floats, handles, NULL, NULL);
}

/* The first position from `start` (in walking order) holding `value`, or -1
   where there is none, or -2 where a NaN left it undecided. */
int64_t ppy_coll_find_value(int8_t *handle, const int8_t *value, int64_t start) {
    int64_t *header = (int64_t *)handle;
    int64_t index = 0;
    for (int64_t at = ppy_coll_step(handle, -1); at >= 0; at = ppy_coll_step(handle, at), index++) {
        if (index < start) {
            continue;
        }
        int64_t same = ppy_coll_same(ppy_coll_at(handle, at), (const int64_t *)value, header[8],
                                     header[9], header[10]);
        if (same != 0) {
            return same == 1 ? index : -2;
        }
    }
    return -1;
}

/* How many elements hold `value`, or -1 where a NaN left it undecided. */
int64_t ppy_coll_count_value(int8_t *handle, const int8_t *value) {
    int64_t *header = (int64_t *)handle;
    int64_t count = 0;
    for (int64_t at = ppy_coll_step(handle, -1); at >= 0; at = ppy_coll_step(handle, at)) {
        int64_t same = ppy_coll_same(ppy_coll_at(handle, at), (const int64_t *)value, header[8],
                                     header[9], header[10]);
        if (same < 0) {
            return -1;
        }
        count += same;
    }
    return count;
}

/* Take a reference to each collection the value words hold. */
void ppy_coll_retain_words(int8_t *handle, const int8_t *value) {
    int64_t *header = (int64_t *)handle;
    const int64_t *words = (const int64_t *)value;
    for (int64_t w = 0; w < header[8]; w++) {
        if ((header[10] >> w) & 1) {
            ppy_coll_retain((int8_t *)(intptr_t)words[w]);
        }
    }
}

/* Let go of each collection the value words hold. */
void ppy_coll_release_words(int8_t *handle, const int8_t *value) {
    int64_t *header = (int64_t *)handle;
    const int64_t *words = (const int64_t *)value;
    for (int64_t w = 0; w < header[8]; w++) {
        if ((header[10] >> w) & 1) {
            ppy_coll_release((int8_t *)(intptr_t)words[w]);
        }
    }
}

/* Python's `<` over two values' words, from native code: `a` before `b`. */
int64_t ppy_coll_before(const int8_t *a, const int8_t *b, int64_t words, int64_t floats) {
    return ppy_coll_less((const int64_t *)a, (const int64_t *)b, words, floats);
}

/* A new collection of the same type and contents; the collections it holds
   are shared, each with one more reference. */
int8_t *ppy_coll_copy(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t stride = header[15] > 0 ? header[15] : 1;
    int8_t *made = ppy_coll_make(header[12], header[13], header[8], header[9], header[10],
                                 header[15], header[1]);
    int64_t *copy = (int64_t *)made;
    memcpy((void *)(intptr_t)copy[2], (void *)(intptr_t)header[2],
           (size_t)(header[1] * stride * 8));
    copy[0] = header[0];
    for (int64_t w = 3; w <= 7; w++) {
        copy[w] = header[w];
    }
    if (header[12] == 2) {
        int64_t *index = (int64_t *)malloc((size_t)header[5] * sizeof(int64_t));
        if (index == NULL) {
            ppy_coll_fail();
        }
        memcpy(index, (void *)(intptr_t)header[4], (size_t)header[5] * sizeof(int64_t));
        copy[4] = (int64_t)(intptr_t)index;
    }
    if (header[10] != 0) {
        int64_t count = header[12] == 0 ? header[0] : header[1];
        for (int64_t i = 0; i < count; i++) {
            int64_t *value = ppy_coll_live(made, i);
            if (value != NULL) {
                ppy_coll_retain_words(made, (const int8_t *)value);
            }
        }
    }
    return made;
}

/* -- sequences, beyond the ends ----------------------------------------------- */

/* Room for one element before position `index` (0 to the length), the rest
   moved up by one: the new slot, all zero. */
int8_t *ppy_seq_insert(int8_t *handle, int64_t index) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    ppy_seq_push_back(handle);
    for (int64_t i = header[0] - 1; i > index; i--) {
        memcpy(ppy_seq_at(handle, i), ppy_seq_at(handle, i - 1), (size_t)(words * 8));
    }
    int8_t *slot = ppy_seq_at(handle, index);
    memset(slot, 0, (size_t)(words * 8));
    return slot;
}

/* The element at `index` out, the rest moved down by one: its words, in the
   scratch words, which are what this returns. */
int8_t *ppy_seq_erase(int8_t *handle, int64_t index) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *taken = (int64_t *)(intptr_t)header[14];
    memcpy(taken, ppy_seq_at(handle, index), (size_t)(words * 8));
    for (int64_t i = index; i + 1 < header[0]; i++) {
        memcpy(ppy_seq_at(handle, i), ppy_seq_at(handle, i + 1), (size_t)(words * 8));
    }
    header[0]--;
    return (int8_t *)taken;
}

/* Every element of `other` (a sequence or a list) at the end, each collection
   one more reference; `other` may be the collection itself. */
void ppy_seq_extend(int8_t *handle, int8_t *other) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t count = ((int64_t *)other)[0];
    int64_t at = ppy_coll_step(other, -1);
    for (int64_t n = 0; n < count && at >= 0; n++) {
        int64_t *from = ppy_coll_at(other, at);
        at = ppy_coll_step(other, at);
        int64_t *to = (int64_t *)ppy_seq_push_back(handle);
        from = other == handle ? ppy_coll_at(other, n) : from;
        memcpy(to, from, (size_t)(words * 8));
        ppy_coll_retain_words(handle, (const int8_t *)to);
    }
}

/* A list's slice, `start:stop:step` with each bound given or not (`given`
   bit 0 the start, bit 1 the stop): a new sequence, its collections shared. */
int8_t *ppy_seq_slice(int8_t *handle, int64_t start, int64_t stop, int64_t step, int64_t given) {
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    if (step < -INT64_MAX) {
        step = -INT64_MAX;
    }
    if (!(given & 1)) {
        start = step < 0 ? INT64_MAX : 0;
    }
    if (!(given & 2)) {
        stop = step < 0 ? INT64_MIN : INT64_MAX;
    }
    if (start < 0) {
        start += n;
        if (start < 0) {
            start = step < 0 ? -1 : 0;
        }
    } else if (start >= n) {
        start = step < 0 ? n - 1 : n;
    }
    if (stop < 0) {
        stop = stop < -n ? -1 : stop + n;
        if (stop < 0) {
            stop = step < 0 ? -1 : 0;
        }
    } else if (stop >= n) {
        stop = step < 0 ? n - 1 : n;
    }
    int64_t count = 0;
    if (step < 0 && stop < start) {
        count = (start - stop - 1) / (-step) + 1;
    } else if (step > 0 && start < stop) {
        count = (stop - start - 1) / step + 1;
    }
    int8_t *made = ppy_seq_new(0, header[8], header[9], header[10]);
    for (int64_t i = 0, at = start; i < count; i++, at += step) {
        int64_t *to = (int64_t *)ppy_seq_push_back(made);
        memcpy(to, ppy_seq_at(handle, at), (size_t)(header[8] * 8));
        ppy_coll_retain_words(made, (const int8_t *)to);
    }
    return made;
}

/* `a + b`: a new sequence of `a`'s elements and then `b`'s. */
int8_t *ppy_seq_concat(int8_t *a, int8_t *b) {
    int8_t *made = ppy_coll_copy(a);
    ppy_seq_extend(made, b);
    return made;
}

/* Deque's `rotate`: the last `steps` elements to the front (`steps` < 0, the
   first to the back). */
void ppy_seq_rotate(int8_t *handle, int64_t steps) {
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    if (n <= 1) {
        return;
    }
    int64_t shift = steps % n;
    if (shift < 0) {
        shift += n;
    }
    if (shift == 0) {
        return;
    }
    int64_t words = header[8];
    int64_t *items = (int64_t *)calloc((size_t)(n * words), 8);
    if (items == NULL) {
        ppy_coll_fail();
    }
    for (int64_t i = 0; i < n; i++) {
        memcpy(items + ((i + shift) % n) * words, ppy_seq_at(handle, i), (size_t)(words * 8));
    }
    free((void *)(intptr_t)header[2]);
    header[1] = n;
    header[2] = (int64_t)(intptr_t)items;
    header[3] = 0;
}

/* A stable merge sort on the first `compare` words of each element, smallest
   first or (`descending`) largest first; equal elements keep their order
   either way, as `list.sort(reverse=True)` keeps them. */
void ppy_seq_sort_by(int8_t *handle, int64_t compare, int64_t descending) {
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    int64_t words = header[8];
    if (n < 2) {
        return;
    }
    int64_t *items = (int64_t *)calloc((size_t)(n * words), 8);
    int64_t *spare = (int64_t *)calloc((size_t)(n * words), 8);
    if (items == NULL || spare == NULL) {
        ppy_coll_fail();
    }
    for (int64_t i = 0; i < n; i++) {
        memcpy(items + i * words, ppy_seq_at(handle, i), (size_t)(words * 8));
    }
    for (int64_t width = 1; width < n; width *= 2) {
        for (int64_t low = 0; low < n; low += 2 * width) {
            int64_t middle = low + width < n ? low + width : n;
            int64_t high = low + 2 * width < n ? low + 2 * width : n;
            int64_t i = low, j = middle, k = low;
            while (i < middle && j < high) {
                const int64_t *left = items + i * words;
                const int64_t *right = items + j * words;
                int take = descending ? (int)ppy_coll_less(left, right, compare, header[9])
                                      : (int)ppy_coll_less(right, left, compare, header[9]);
                int64_t from = take ? j++ : i++;
                memcpy(spare + (k++) * words, items + from * words, (size_t)(words * 8));
            }
            while (i < middle) {
                memcpy(spare + (k++) * words, items + (i++) * words, (size_t)(words * 8));
            }
            while (j < high) {
                memcpy(spare + (k++) * words, items + (j++) * words, (size_t)(words * 8));
            }
        }
        int64_t *swap = items;
        items = spare;
        spare = swap;
    }
    free((void *)(intptr_t)header[2]);
    free(spare);
    header[1] = n;
    header[2] = (int64_t)(intptr_t)items;
    header[3] = 0;
}

/* The elements put in the order `order` gives: its element `i` carries, at
   word `word`, the position the element `i` comes from. */
void ppy_seq_permute(int8_t *handle, int8_t *order, int64_t word) {
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    int64_t words = header[8];
    if (n < 2) {
        return;
    }
    int64_t *items = (int64_t *)calloc((size_t)(n * words), 8);
    if (items == NULL) {
        ppy_coll_fail();
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t from = ((int64_t *)ppy_seq_at(order, i))[word];
        memcpy(items + i * words, ppy_seq_at(handle, from), (size_t)(words * 8));
    }
    free((void *)(intptr_t)header[2]);
    header[1] = n;
    header[2] = (int64_t)(intptr_t)items;
    header[3] = 0;
}

/* -- heaps, beyond one push and one pop ----------------------------------------- */

/* `value` moved down from position `i`, past every child that comes out
   before it. */
void ppy_heap_sift(int8_t *handle, int64_t i, const int64_t *value, int64_t max) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t n = header[0];
    while (1) {
        int64_t child = 2 * i + 1;
        if (child >= n) {
            break;
        }
        if (child + 1 < n
            && ppy_heap_before(handle, ppy_coll_record(handle, child + 1),
                               ppy_coll_record(handle, child), max)) {
            child++;
        }
        if (!ppy_heap_before(handle, ppy_coll_record(handle, child), value, max)) {
            break;
        }
        memcpy(ppy_coll_record(handle, i), ppy_coll_record(handle, child), (size_t)(words * 8));
        i = child;
    }
    memcpy(ppy_coll_record(handle, i), value, (size_t)(words * 8));
}

/* `pushpop` (`replace` 0) or `replace` (1) with the value in the scratch
   words: the element that comes out, in the scratch words. `pushpop` hands
   the value straight back when it would come out first. */
int8_t *ppy_heap_exchange(int8_t *handle, int64_t max, int64_t replace) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *value = (int64_t *)(intptr_t)header[14];
    int64_t *out = value + words;
    if (!replace
        && (header[0] == 0 || !ppy_heap_before(handle, ppy_coll_record(handle, 0), value, max))) {
        return (int8_t *)value;
    }
    memcpy(out, ppy_coll_record(handle, 0), (size_t)(words * 8));
    ppy_heap_sift(handle, 0, value, max);
    return (int8_t *)out;
}

/* The elements arranged into a heap, from the last parent up, as `heapq.heapify`. */
void ppy_heap_heapify(int8_t *handle, int64_t max) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t *value = (int64_t *)(intptr_t)header[14];
    if (header[3] != 0) {
        ppy_seq_rotate(handle, 0);
    }
    for (int64_t i = header[0] / 2 - 1; i >= 0; i--) {
        memcpy(value, ppy_coll_record(handle, i), (size_t)(words * 8));
        ppy_heap_sift(handle, i, value, max);
    }
}

/* -- lists, maps, and trees, walked and combined ---------------------------------- */

/* The node before `node` while walking backwards, or -1. */
int64_t ppy_list_before(int8_t *handle, int64_t node) {
    if (!ppy_list_valid(handle, node)) {
        return -1;
    }
    return ppy_list_step(handle, node, 0);
}

/* The last live entry below `at`, or -1: a map walked backwards. */
int64_t ppy_map_back(int8_t *handle, int64_t at) {
    for (int64_t e = at - 1; e >= 0; e--) {
        if (ppy_map_alive(handle, e)) {
            return e;
        }
    }
    return -1;
}

/* Whether node `node` exists and its key is below `key`. */
int64_t ppy_tree_below(int8_t *handle, int64_t node, const int8_t *key) {
    if (node < 0) {
        return 0;
    }
    int64_t *header = (int64_t *)handle;
    return ppy_tree_order(ppy_coll_record(handle, node), (const int64_t *)key, header[13]) < 0;
}

/* An entry for `key` in a map or a set of either family: its index. */
int64_t ppy_coll_put_key(int8_t *handle, const int8_t *key) {
    return ((int64_t *)handle)[12] == 2 ? ppy_map_put(handle, key) : ppy_tree_put(handle, key);
}

/* Every entry of `other` put in `handle`, in `other`'s order: a value
   replaced lets go of what it held, a value stored takes a reference. */
void ppy_coll_update(int8_t *handle, int8_t *other) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t count = ((int64_t *)other)[0];
    int64_t *keys = (int64_t *)calloc((size_t)(count * header[13] + 1), 8);
    int64_t *values = (int64_t *)calloc((size_t)(count * words + 1), 8);
    if (keys == NULL || values == NULL) {
        ppy_coll_fail();
    }
    int64_t n = 0;
    for (int64_t e = ppy_coll_step(other, -1); e >= 0; e = ppy_coll_step(other, e), n++) {
        memcpy(keys + n * header[13], ppy_coll_record(other, e), (size_t)(header[13] * 8));
        memcpy(values + n * words, ppy_coll_value_words(other, e), (size_t)(words * 8));
        ppy_coll_retain_words(handle, (const int8_t *)(values + n * words));
    }
    for (int64_t i = 0; i < n; i++) {
        const int8_t *key = (const int8_t *)(keys + i * header[13]);
        int64_t found = ppy_coll_find_key(handle, key);
        if (found >= 0) {
            ppy_coll_release_words(handle, (const int8_t *)ppy_coll_value_words(handle, found));
        } else {
            found = ppy_coll_put_key(handle, key);
        }
        memcpy(ppy_coll_value_words(handle, found), values + i * words, (size_t)(words * 8));
    }
    free(keys);
    free(values);
}

/* `a | b` (0), `a & b` (1), `a - b` (2), `a ^ b` (3): a new set of `a`'s
   family, `a`'s keys first in `a`'s order and then `b`'s. */
int8_t *ppy_set_combine(int8_t *a, int8_t *b, int64_t op) {
    int64_t *header = (int64_t *)a;
    int64_t keys = header[13];
    int8_t *made = header[12] == 2 ? ppy_map_new(keys, 0, 0, 0) : ppy_tree_new(keys, 0, 0, 0);
    for (int64_t e = ppy_coll_step(a, -1); e >= 0; e = ppy_coll_step(a, e)) {
        const int8_t *key = (const int8_t *)ppy_coll_record(a, e);
        int64_t in_b = ppy_coll_find_key(b, key) >= 0;
        if ((in_b && (op == 0 || op == 1)) || (!in_b && op != 1)) {
            ppy_coll_put_key(made, key);
        }
    }
    if (op == 0 || op == 3) {
        for (int64_t e = ppy_coll_step(b, -1); e >= 0; e = ppy_coll_step(b, e)) {
            const int8_t *key = (const int8_t *)ppy_coll_record(b, e);
            if (ppy_coll_find_key(a, key) < 0) {
                ppy_coll_put_key(made, key);
            }
        }
    }
    return made;
}

/* `a.issubset(b)` (0), `a.issuperset(b)` (1), `a.isdisjoint(b)` (2). */
int64_t ppy_set_relation(int8_t *a, int8_t *b, int64_t op) {
    int8_t *walked = op == 1 ? b : a;
    int8_t *asked = op == 1 ? a : b;
    for (int64_t e = ppy_coll_step(walked, -1); e >= 0; e = ppy_coll_step(walked, e)) {
        int64_t found = ppy_coll_find_key(asked, (const int8_t *)ppy_coll_record(walked, e)) >= 0;
        if (op == 2 ? found : !found) {
            return 0;
        }
    }
    return 1;
}

/* -- the Python boundary: whole collections in one call --------------------------- */

/* `count` elements, `words` each, appended to a sequence from `values`. */
void ppy_seq_push_many(int8_t *handle, const int8_t *values, int64_t count) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    for (int64_t i = 0; i < count; i++) {
        memcpy(ppy_seq_push_back(handle), values + i * words * 8, (size_t)(words * 8));
    }
}

/* `count` entries put in a map or a set of either family: keys from `keys`
   and values from `values`, `key words` and `value words` each. */
void ppy_coll_put_many(int8_t *handle, const int8_t *keys, const int8_t *values, int64_t count) {
    int64_t *header = (int64_t *)handle;
    int64_t key_words = header[13];
    int64_t words = header[8];
    for (int64_t i = 0; i < count; i++) {
        int64_t entry = ppy_coll_put_key(handle, keys + i * key_words * 8);
        if (words > 0) {
            memcpy(ppy_coll_value_words(handle, entry), values + i * words * 8,
                   (size_t)(words * 8));
        }
    }
}

/* Every element (or key and value) in walking order, copied out: keys into
   `keys` and elements or values into `values`, `key words` and `value words`
   each. The count is the collection's length. */
void ppy_coll_copy_out(int8_t *handle, int8_t *keys, int8_t *values) {
    int64_t *header = (int64_t *)handle;
    int64_t key_words = header[13];
    int64_t words = header[8];
    int64_t family = header[12];
    int64_t n = 0;
    for (int64_t at = ppy_coll_step(handle, -1); at >= 0; at = ppy_coll_step(handle, at), n++) {
        if (family <= 1) {
            memcpy(values + n * words * 8, ppy_coll_at(handle, at), (size_t)(words * 8));
            continue;
        }
        memcpy(keys + n * key_words * 8, ppy_coll_record(handle, at), (size_t)(key_words * 8));
        if (words > 0) {
            memcpy(values + n * words * 8, ppy_coll_value_words(handle, at), (size_t)(words * 8));
        }
    }
}
