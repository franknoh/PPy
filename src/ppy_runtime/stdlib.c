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

/* -- `math` beyond the machine's instructions ----------------------------------

   The integer functions answer -1 where the exact result is past a word
   (every result they have is 0 or more), and the caller falls back. */

/* `gcd(a, b)`. */
int64_t ppy_math_gcd(int64_t a, int64_t b) {
    uint64_t x = a < 0 ? (uint64_t)0 - (uint64_t)a : (uint64_t)a;
    uint64_t y = b < 0 ? (uint64_t)0 - (uint64_t)b : (uint64_t)b;
    while (y != 0) {
        uint64_t t = x % y;
        x = y;
        y = t;
    }
    return x > (uint64_t)INT64_MAX ? -1 : (int64_t)x;
}

/* `lcm(a, b)`: `abs(a // gcd(a, b) * b)`. */
int64_t ppy_math_lcm(int64_t a, int64_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    int64_t g = ppy_math_gcd(a, b);
    if (g < 0) {
        return -1;
    }
    __int128 made = ((__int128)a / g) * (__int128)b;
    if (made < 0) {
        made = -made;
    }
    return made > INT64_MAX ? -1 : (int64_t)made;
}

/* `isqrt(n)` for n >= 0: the largest r with r * r <= n. */
int64_t ppy_math_isqrt(int64_t n) {
    uint64_t r = (uint64_t)sqrt((double)n);
    while ((unsigned __int128)r * r > (unsigned __int128)n) {
        r--;
    }
    while ((unsigned __int128)(r + 1) * (r + 1) <= (unsigned __int128)n) {
        r++;
    }
    return (int64_t)r;
}

/* `comb(n, k)` for n, k >= 0. */
int64_t ppy_math_comb(int64_t n, int64_t k) {
    if (k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }
    unsigned __int128 made = 1;
    for (int64_t i = 0; i < k; i++) {
        /* C(n, i + 1) = C(n, i) * (n - i) / (i + 1), exact at each step. */
        made = made * (unsigned __int128)(n - i) / (unsigned __int128)(i + 1);
        if (made > (unsigned __int128)INT64_MAX) {
            return -1;
        }
    }
    return (int64_t)made;
}

/* `perm(n, k)` for n, k >= 0. */
int64_t ppy_math_perm(int64_t n, int64_t k) {
    if (k > n) {
        return 0;
    }
    unsigned __int128 made = 1;
    for (int64_t i = 0; i < k; i++) {
        made *= (unsigned __int128)(n - i);
        if (made > (unsigned __int128)INT64_MAX) {
            return -1;
        }
    }
    return (int64_t)made;
}

/* What the last of these functions to fail said: 1 an integer past a word
   or an intermediate overflow, 2 `-inf + inf` in `fsum`. */
int64_t *ppy_math_cell(void) {
    static int64_t cell[1];
    return cell;
}

/* The failure since the last ask, or 0; asking clears it. */
int64_t ppy_math_fault(void) {
    int64_t *cell = ppy_math_cell();
    int64_t fault = cell[0];
    cell[0] = 0;
    return fault;
}

/* The product of a list's ints from 1. */
int64_t ppy_math_prod_ints(int8_t *handle) {
    int64_t n = ((int64_t *)handle)[0];
    int64_t made = 1;
    for (int64_t i = 0; i < n; i++) {
        if (__builtin_mul_overflow(made, *(int64_t *)ppy_seq_at(handle, i), &made)) {
            ppy_math_cell()[0] = 1;
            return 0;
        }
    }
    return made;
}

/* The product of a list's floats, left to right from 1.0. */
double ppy_math_prod_floats(int8_t *handle) {
    int64_t n = ((int64_t *)handle)[0];
    double made = 1.0;
    for (int64_t i = 0; i < n; i++) {
        made *= *(double *)ppy_seq_at(handle, i);
    }
    return made;
}

/* `fsum` of a list's floats (or ints, `integers`), by CPython's algorithm:
   Shewchuk's exact partials, a correction for half-even rounding across
   them, and the special values summed apart; a failure is left in
   `ppy_math_cell`. */
double ppy_math_fsum(int8_t *handle, int64_t integers) {
    int64_t count = ((int64_t *)handle)[0];
    int64_t room = 32, n = 0;
    double stack[32];
    double *p = stack;
    double special_sum = 0.0, inf_sum = 0.0, hi = 0.0, lo = 0.0;
    for (int64_t index = 0; index < count; index++) {
        int8_t *at = ppy_seq_at(handle, index);
        double x = integers ? (double)*(int64_t *)at : *(double *)at;
        double xsave = x;
        int64_t i = 0;
        for (int64_t j = 0; j < n; j++) {
            double y = p[j];
            if (fabs(x) < fabs(y)) {
                double t = x;
                x = y;
                y = t;
            }
            hi = x + y;
            double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) {
                p[i++] = lo;
            }
            x = hi;
        }
        n = i;
        if (x != 0.0) {
            if (!(x - x == 0.0)) {
                if (xsave - xsave == 0.0) {
                    ppy_math_cell()[0] = 1;
                    if (p != stack) {
                        free(p);
                    }
                    return 0.0;
                }
                if (xsave == INFINITY || xsave == -INFINITY) {
                    inf_sum += xsave;
                }
                special_sum += xsave;
                n = 0;
            } else {
                if (n >= room) {
                    double *grown = (double *)malloc((size_t)(2 * room) * sizeof(double));
                    if (grown == NULL) {
                        ppy_coll_fail();
                    }
                    memcpy(grown, p, (size_t)n * sizeof(double));
                    if (p != stack) {
                        free(p);
                    }
                    p = grown;
                    room *= 2;
                }
                p[n++] = x;
            }
        }
    }
    if (special_sum != 0.0) {
        if (p != stack) {
            free(p);
        }
        if (inf_sum != inf_sum) {
            ppy_math_cell()[0] = 2;
            return 0.0;
        }
        return special_sum;
    }
    hi = 0.0;
    if (n > 0) {
        hi = p[--n];
        while (n > 0) {
            double x = hi;
            double y = p[--n];
            hi = x + y;
            double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) {
                break;
            }
        }
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            double y = lo * 2.0;
            double x = hi + y;
            double yr = x - hi;
            if (y == yr) {
                hi = x;
            }
        }
    }
    if (p != stack) {
        free(p);
    }
    return hi;
}

/* `isclose(a, b, rel_tol=..., abs_tol=...)`, the tolerances checked by the caller. */
int64_t ppy_math_isclose(double a, double b, double rel_tol, double abs_tol) {
    if (a == b) {
        return 1;
    }
    if (a == INFINITY || a == -INFINITY || b == INFINITY || b == -INFINITY) {
        return 0;
    }
    double diff = fabs(b - a);
    return ((diff <= fabs(rel_tol * b)) || (diff <= fabs(rel_tol * a))) || (diff <= abs_tol);
}

/* A double and the error of it, for `hypot`'s exact arithmetic. */
void ppy_math_dl_mul(double x, double y, double *hi, double *lo) {
    double z = x * y;
    *hi = z;
    *lo = fma(x, y, -z);
}

/* `hypot` of the magnitudes in `values` (`count` of them, the largest `max`,
   `nan` whether any is a NaN): CPython's `vector_norm`, scaled, squared,
   and summed without loss, then corrected once. */
double ppy_math_norm(double *values, int64_t count, double max, int64_t nan) {
    if (max == INFINITY) {
        return max;
    }
    if (nan) {
        return NAN;
    }
    if (max == 0.0 || count <= 1) {
        return max;
    }
    int max_e;
    frexp(max, &max_e);
    if (max_e < -1023) {
        for (int64_t i = 0; i < count; i++) {
            values[i] /= DBL_MIN;
        }
        return DBL_MIN * ppy_math_norm(values, count, max / DBL_MIN, nan);
    }
    double scale = ldexp(1.0, -max_e);
    double csum = 1.0, frac1 = 0.0, frac2 = 0.0, hi, lo;
    for (int64_t i = 0; i < count; i++) {
        double x = values[i] * scale;
        ppy_math_dl_mul(x, x, &hi, &lo);
        double sum = csum + hi;
        double error = (csum - sum) + hi;
        csum = sum;
        frac1 += lo;
        frac2 += error;
    }
    double h = sqrt(csum - 1.0 + (frac1 + frac2));
    ppy_math_dl_mul(-h, h, &hi, &lo);
    double sum = csum + hi;
    double error = (csum - sum) + hi;
    csum = sum;
    frac1 += lo;
    frac2 += error;
    double x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    return h / scale;
}

/* `hypot(x, y)`. */
double ppy_math_hypot2(double x, double y) {
    double values[2] = {fabs(x), fabs(y)};
    double max = 0.0;
    int64_t nan = 0;
    for (int64_t i = 0; i < 2; i++) {
        nan |= values[i] != values[i];
        if (values[i] > max) {
            max = values[i];
        }
    }
    return ppy_math_norm(values, 2, max, nan);
}

/* `hypot(x, y, z)`. */
double ppy_math_hypot3(double x, double y, double z) {
    double values[3] = {fabs(x), fabs(y), fabs(z)};
    double max = 0.0;
    int64_t nan = 0;
    for (int64_t i = 0; i < 3; i++) {
        nan |= values[i] != values[i];
        if (values[i] > max) {
            max = values[i];
        }
    }
    return ppy_math_norm(values, 3, max, nan);
}

/* `hypot` of a list of floats, or `dist` of two (`other` not NULL; the
   lengths checked by the caller). */
double ppy_math_hypot_list(int8_t *handle, int8_t *other) {
    int64_t n = ((int64_t *)handle)[0];
    double stack[16];
    double *values = n <= 16 ? stack : (double *)malloc((size_t)n * sizeof(double));
    if (values == NULL) {
        ppy_coll_fail();
    }
    double max = 0.0;
    int64_t nan = 0;
    for (int64_t i = 0; i < n; i++) {
        double x = *(double *)ppy_seq_at(handle, i);
        if (other != NULL) {
            x -= *(double *)ppy_seq_at(other, i);
        }
        x = fabs(x);
        values[i] = x;
        nan |= x != x;
        if (x > max) {
            max = x;
        }
    }
    double made = ppy_math_norm(values, n, max, nan);
    if (values != stack) {
        free(values);
    }
    return made;
}

/* One of libm's functions by number, as `math` calls it: 0 asin, 1 acos,
   2 atan, 3 sinh, 4 cosh, 5 tanh, 6 asinh, 7 acosh, 8 atanh, 9 log1p,
   10 expm1, 11 erf, 12 erfc, 13 cbrt, 14 exp2, 15 fabs, 16 tan. */
double ppy_math_unary(int64_t which, double x) {
    switch (which) {
    case 0: return asin(x);
    case 1: return acos(x);
    case 2: return atan(x);
    case 3: return sinh(x);
    case 4: return cosh(x);
    case 5: return tanh(x);
    case 6: return asinh(x);
    case 7: return acosh(x);
    case 8: return atanh(x);
    case 9: return log1p(x);
    case 10: return expm1(x);
    case 11: return erf(x);
    case 12: return erfc(x);
    case 13: return cbrt(x);
    case 14: return exp2(x);
    case 16: return tan(x);
    default: return fabs(x);
    }
}

/* Two-argument ones: 0 atan2(y, x), 1 copysign, 2 fmod. */
double ppy_math_binary(int64_t which, double x, double y) {
    switch (which) {
    case 0: return atan2(x, y);
    case 1: return copysign(x, y);
    default:
        /* `fmod(x, inf)` is x for a finite x, as C says too. */
        return fmod(x, y);
    }
}

/* -- `itertools`, made into a list where a loop or a call consumes it ----------

   `made` is the result, made empty by the caller with the element layout the
   checker gave the iterator's items; each item's words are the inputs'
   element words side by side, and the collections they hold are shared, one
   reference more each. The orders are CPython's: lexicographic by position
   in the inputs. */

/* One item from the elements at `positions` of `sources` (one source per
   position where `sources` has several, else the one). */
void ppy_iter_emit(int8_t *made, int8_t **sources, int64_t each, const int64_t *positions,
                   int64_t count) {
    int8_t *slot = ppy_seq_push_back(made);
    int64_t offset = 0;
    for (int64_t i = 0; i < count; i++) {
        int8_t *source = sources[each ? i : 0];
        int64_t words = ((int64_t *)source)[8];
        memcpy(slot + offset * 8, ppy_seq_at(source, positions[i]), (size_t)(words * 8));
        offset += words;
    }
    ppy_coll_retain_words(made, slot);
}

/* `product(a, b, ...)` of up to four lists. */
void ppy_iter_product(int8_t *made, int64_t count, int8_t *a, int8_t *b, int8_t *c, int8_t *d) {
    int8_t *sources[4] = {a, b, c, d};
    int64_t positions[4] = {0, 0, 0, 0};
    for (int64_t i = 0; i < count; i++) {
        if (((int64_t *)sources[i])[0] == 0) {
            return;
        }
    }
    for (;;) {
        ppy_iter_emit(made, sources, 1, positions, count);
        int64_t i = count - 1;
        while (i >= 0) {
            positions[i]++;
            if (positions[i] < ((int64_t *)sources[i])[0]) {
                break;
            }
            positions[i] = 0;
            i--;
        }
        if (i < 0) {
            return;
        }
    }
}

/* `permutations(xs, r)`, `combinations(xs, r)` (`kind` 1), and
   `combinations_with_replacement(xs, r)` (`kind` 2), r at least 0. Past 16
   positions (or 64 elements to arrange) it fails into `ppy_math_cell`. */
void ppy_iter_choose(int8_t *made, int8_t *source, int64_t r, int64_t kind) {
    int64_t n = ((int64_t *)source)[0];
    int64_t positions[16];
    if ((kind != 2 && r > n) || (kind == 2 && n == 0 && r > 0)) {
        return;
    }
    if (r > 16 || (kind == 0 && n > 64)) {
        ppy_math_cell()[0] = 1;
        return;
    }
    int8_t *sources[1] = {source};
    if (kind == 0) {
        /* Every arrangement of r distinct positions, smallest first. */
        int64_t used[64] = {0};
        int64_t depth = 0;
        int64_t next[17];
        next[0] = 0;
        if (r == 0) {
            ppy_iter_emit(made, sources, 0, positions, 0);
            return;
        }
        while (depth >= 0) {
            if (depth == r) {
                ppy_iter_emit(made, sources, 0, positions, r);
                depth--;
                used[positions[depth]] = 0;
                continue;
            }
            int64_t p = next[depth];
            while (p < n && used[p]) {
                p++;
            }
            if (p >= n) {
                depth--;
                if (depth >= 0) {
                    used[positions[depth]] = 0;
                }
                continue;
            }
            positions[depth] = p;
            used[p] = 1;
            next[depth] = p + 1;
            depth++;
            next[depth] = 0;
        }
        return;
    }
    for (int64_t i = 0; i < r; i++) {
        positions[i] = kind == 1 ? i : 0;
    }
    for (;;) {
        ppy_iter_emit(made, sources, 0, positions, r);
        int64_t i = r - 1;
        if (kind == 1) {
            while (i >= 0 && positions[i] == i + n - r) {
                i--;
            }
            if (i < 0) {
                return;
            }
            positions[i]++;
            for (int64_t j = i + 1; j < r; j++) {
                positions[j] = positions[j - 1] + 1;
            }
        } else {
            while (i >= 0 && positions[i] == n - 1) {
                i--;
            }
            if (i < 0) {
                return;
            }
            int64_t v = positions[i] + 1;
            for (int64_t j = i; j < r; j++) {
                positions[j] = v;
            }
        }
    }
}

/* `pairwise(xs)`. */
void ppy_iter_pairwise(int8_t *made, int8_t *source) {
    int64_t n = ((int64_t *)source)[0];
    int8_t *sources[1] = {source};
    for (int64_t i = 0; i + 1 < n; i++) {
        int64_t positions[2] = {i, i + 1};
        ppy_iter_emit(made, sources, 0, positions, 2);
    }
}

/* `chain(a, b, ...)` of up to four lists of one element type. */
void ppy_iter_chain(int8_t *made, int64_t count, int8_t *a, int8_t *b, int8_t *c, int8_t *d) {
    int8_t *sources[4] = {a, b, c, d};
    for (int64_t i = 0; i < count; i++) {
        int64_t n = ((int64_t *)sources[i])[0];
        for (int64_t j = 0; j < n; j++) {
            ppy_iter_emit(made, &sources[i], 0, &j, 1);
        }
    }
}

/* `islice(xs, start, stop, step)` of a list, the arguments checked. */
void ppy_iter_islice(int8_t *made, int8_t *source, int64_t start, int64_t stop, int64_t step) {
    int64_t n = ((int64_t *)source)[0];
    int8_t *sources[1] = {source};
    if (stop > n) {
        stop = n;
    }
    for (int64_t i = start; i < stop; i += step) {
        ppy_iter_emit(made, sources, 0, &i, 1);
    }
}

/* `accumulate(xs)` of ints (`floats` 0) or floats with `+`, or with `max`
   (`how` 1) or `min` (2), from `initial` where `has_initial`. The ints fail
   past a word, into `ppy_math_cell`. */
void ppy_iter_accumulate(int8_t *made, int8_t *source, int64_t floats, int64_t how,
                         int64_t has_initial, int64_t initial_int, double initial_float) {
    int64_t n = ((int64_t *)source)[0];
    int64_t total = initial_int;
    if (floats) {
        memcpy(&total, &initial_float, 8);
    }
    int64_t started = has_initial;
    if (has_initial) {
        *(int64_t *)ppy_seq_push_back(made) = total;
    }
    for (int64_t i = 0; i < n; i++) {
        int64_t word = *(int64_t *)ppy_seq_at(source, i);
        if (!started) {
            total = word;
            started = 1;
        } else if (floats) {
            double x, y;
            memcpy(&x, &total, 8);
            memcpy(&y, &word, 8);
            double z = how == 0 ? x + y : how == 1 ? (y > x ? y : x) : (y < x ? y : x);
            memcpy(&total, &z, 8);
        } else if (how == 0) {
            if (__builtin_add_overflow(total, word, &total)) {
                ppy_math_cell()[0] = 1;
                return;
            }
        } else {
            total = how == 1 ? (word > total ? word : total) : (word < total ? word : total);
        }
        *(int64_t *)ppy_seq_push_back(made) = total;
    }
}

/* `repeat(x, n)`: the one element at `value`, n times. */
void ppy_iter_repeat(int8_t *made, const int8_t *value, int64_t n) {
    int64_t words = ((int64_t *)made)[8];
    for (int64_t i = 0; i < n; i++) {
        int8_t *slot = ppy_seq_push_back(made);
        memcpy(slot, value, (size_t)(words * 8));
        ppy_coll_retain_words(made, slot);
    }
}

/* `range(start, stop, step)` into the empty `list[int]` `made`, step nonzero. */
void ppy_iter_range(int8_t *made, int64_t start, int64_t stop, int64_t step) {
    int64_t n = ppy_random_range_len(start, stop, step);
    for (int64_t i = 0; i < n; i++) {
        *(int64_t *)ppy_seq_push_back(made) = (int64_t)((__int128)start + (__int128)step * i);
    }
}
