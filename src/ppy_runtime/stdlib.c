/* `heapq` and `bisect` over a list, as CPython's `_heapq` and `_bisect` do
   them: the same comparisons (`<` only, through `ppy_coll_ordered`, which
   is a class's `__lt__` where the elements have one), in the same order, so
   a list comes out as CPython leaves it. Elements move as their words; the
   references they hold move with them. The lowering guards what CPython
   raises for (an empty heap, a negative `lo`). */

/* Element `i`'s words, `words` of them, swapped with element `j`'s. */
void ppy_heapq_swap(int8_t *handle, int64_t i, int64_t j) {
    int64_t words = ((int64_t *)handle)[8];
    int64_t spare[64];
    int8_t *a = ppy_seq_at(handle, i);
    int8_t *b = ppy_seq_at(handle, j);
    memcpy(spare, a, (size_t)(words * 8));
    memcpy(a, b, (size_t)(words * 8));
    memcpy(b, spare, (size_t)(words * 8));
}

/* `_siftdown(heap, startpos, pos)`: the element at `pos` moved up past every
   parent it goes before. */
void ppy_heapq_sift_down(int8_t *handle, int64_t start, int64_t pos) {
    int64_t words = ((int64_t *)handle)[8];
    int64_t item[64];
    memcpy(item, ppy_seq_at(handle, pos), (size_t)(words * 8));
    while (pos > start) {
        int64_t parent = (pos - 1) >> 1;
        int8_t *above = ppy_seq_at(handle, parent);
        if (!ppy_coll_ordered(handle, item, (const int64_t *)above)) {
            break;
        }
        memcpy(ppy_seq_at(handle, pos), above, (size_t)(words * 8));
        pos = parent;
    }
    memcpy(ppy_seq_at(handle, pos), item, (size_t)(words * 8));
}

/* `_siftup(heap, pos)`: the smaller child moved up until a leaf, then the
   element at `pos` put there and moved back up. */
void ppy_heapq_sift_up(int8_t *handle, int64_t pos) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t end = header[0];
    int64_t start = pos;
    int64_t item[64];
    memcpy(item, ppy_seq_at(handle, pos), (size_t)(words * 8));
    int64_t child = 2 * pos + 1;
    while (child < end) {
        int64_t right = child + 1;
        if (right < end
            && !ppy_coll_ordered(handle, (const int64_t *)ppy_seq_at(handle, child),
                                 (const int64_t *)ppy_seq_at(handle, right))) {
            child = right;
        }
        memcpy(ppy_seq_at(handle, pos), ppy_seq_at(handle, child), (size_t)(words * 8));
        pos = child;
        child = 2 * pos + 1;
    }
    memcpy(ppy_seq_at(handle, pos), item, (size_t)(words * 8));
    ppy_heapq_sift_down(handle, start, pos);
}

/* `heappush`, after the element was appended. */
void ppy_heapq_push(int8_t *handle) {
    ppy_heapq_sift_down(handle, 0, ((int64_t *)handle)[0] - 1);
}

/* `heappop` of a heap that is not empty: the address of the smallest
   element's words, which the caller now owns and reads at once. */
int8_t *ppy_heapq_pop(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    if (header[0] > 1) {
        ppy_heapq_swap(handle, 0, header[0] - 1);
    }
    int8_t *last = ppy_seq_pop_back(handle);
    if (header[0] > 0) {
        ppy_heapq_sift_up(handle, 0);
    }
    return last;
}

/* `heapreplace(heap, item)`, with the item appended: the old smallest out,
   the item in its place and sifted. The heap was not empty before. */
int8_t *ppy_heapq_replace(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    ppy_heapq_swap(handle, 0, header[0] - 1);
    int8_t *last = ppy_seq_pop_back(handle);
    ppy_heapq_sift_up(handle, 0);
    return last;
}

/* `heappushpop(heap, item)`, with the item appended: where the smallest goes
   before the item, the two change places and the item is sifted; the one
   out is at the address returned. */
int8_t *ppy_heapq_pushpop(int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    int moved = 0;
    if (n > 1
        && ppy_coll_ordered(handle, (const int64_t *)ppy_seq_at(handle, 0),
                            (const int64_t *)ppy_seq_at(handle, n - 1))) {
        ppy_heapq_swap(handle, 0, n - 1);
        moved = 1;
    }
    int8_t *last = ppy_seq_pop_back(handle);
    if (moved) {
        ppy_heapq_sift_up(handle, 0);
    }
    return last;
}

/* `heapify`: every parent from the last up, sifted. The order CPython's
   cache-friendly version takes for a long list sifts each parent after its
   children too, which leaves the same list. */
void ppy_heapq_heapify(int8_t *handle) {
    int64_t n = ((int64_t *)handle)[0];
    for (int64_t i = n / 2 - 1; i >= 0; i--) {
        ppy_heapq_sift_up(handle, i);
    }
}

/* `nsmallest(n, xs)` and `nlargest(n, xs)`: what `sorted(xs)[:n]` and
   `sorted(xs, reverse=True)[:n]` give, which is what CPython documents them
   as and returns. A new list. */
int8_t *ppy_heapq_select(int8_t *handle, int64_t n, int64_t largest) {
    int8_t *made = ppy_coll_copy(handle);
    if (largest) {
        ppy_seq_reverse(made);
    }
    ppy_seq_sort(made);
    if (largest) {
        ppy_seq_reverse(made);
    }
    int64_t *header = (int64_t *)made;
    int64_t keep = n < 0 ? 0 : n;
    while (header[0] > keep) {
        ppy_coll_release_words(made, ppy_seq_pop_back(made));
    }
    return made;
}

/* `bisect_left` (`right` 0) or `bisect_right` of the value at `x`, between
   `lo` and `hi`. */
int64_t ppy_bisect(int8_t *handle, const int64_t *x, int64_t lo, int64_t hi, int64_t right) {
    while (lo < hi) {
        int64_t mid = (int64_t)(((uint64_t)lo + (uint64_t)hi) / 2);
        const int64_t *at = (const int64_t *)ppy_seq_at(handle, mid);
        if (right ? ppy_coll_ordered(handle, x, at) : !ppy_coll_ordered(handle, at, x)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
