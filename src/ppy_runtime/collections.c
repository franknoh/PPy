/* The collections runtime: every `ppy` collection over words.

   A collection is a handle, the address of a sixteen-word header:

     [0] length          [1] capacity (records)   [2] records
     [3] family word a   [4] family word b        [5] family word c
     [6] family word d   [7] family word e
     [8] value words     [9] float mask           [10] handle mask
     [11] references     [12] family              [13] key words, and above
                                                       bit 32 which of them are strings
     [14] scratch        [15] record stride (words)

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

   Nothing is checked here: the caller guards an index, an empty pop, a
   missing key, which is what lets a failed check fall back to Python under
   `ppy run` and stop a standalone binary. */

void ppy_coll_fail(void) {
    fputs("ppy: MemoryError: a collection could not grow\n", stderr);
    exit(1);
}

int8_t *ppy_coll_none(void) {
    return NULL;
}

int8_t *ppy_coll_make(int64_t family, int64_t keys, int64_t words, int64_t floats,
                      int64_t handles, int64_t stride, int64_t capacity) {
    if (family == 4) {
        /* A string keeps its bytes right after its header, in one block, and
           says so in word 6 until it outgrows them (strings.c). */
        int64_t room = capacity > 0 ? capacity : 1;
        int64_t *text = (int64_t *)calloc((size_t)(16 + room), sizeof(int64_t));
        if (text == NULL) {
            ppy_coll_fail();
        }
        text[1] = room;
        text[2] = (int64_t)(intptr_t)(text + 16);
        text[6] = 1;
        text[8] = words;
        text[11] = 1;
        text[12] = 4;
        text[15] = stride;
        return (int8_t *)text;
    }
    int64_t *header = (int64_t *)calloc(16, sizeof(int64_t));
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
    header[13] = keys;
    header[14] = (int64_t)(intptr_t)scratch;
    header[15] = stride;
    return (int8_t *)header;
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
    return (int64_t)((uint64_t)((int64_t *)handle)[13] >> 32);
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
    if (header[12] == 4) {
        if (!header[6]) {
            free((void *)(intptr_t)header[2]);
        }
        free(header);
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
    if (header[12] == 2) {
        free((void *)(intptr_t)header[4]);
    }
    free((void *)(intptr_t)header[2]);
    free((void *)(intptr_t)header[14]);
    free(header);
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
