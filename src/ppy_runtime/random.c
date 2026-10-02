/* The random runtime: CPython's Mersenne Twister, and `random`'s algorithms
   on top of it, draw for draw.

   A state is CPython's own layout, `RandomObject` without its object header:

     int32_t index;  uint32_t state[624];

   Under `ppy run` the state native code draws from is the one inside
   `random._inst`, the object Python's module-level `random` functions use,
   bound once at load (`ppy_random_bind`), so interleaved Python and native
   draws give CPython's sequence. A standalone binary keeps its own, seeded
   from the operating system on first use as CPython seeds an unseeded
   generator. The algorithms (`_randbelow`, `randrange`, `shuffle`,
   `sample`, `choices`, the distributions) are `Lib/random.py`'s, line for
   line; the lowering calls these, and guards what CPython raises for. */

/* Where the state lives: [0] the bound state's address, or 0; [1] whether
   the process-wide fallback below was seeded; [2] whether a seed call ran
   since the last ask (Python's `seed` also forgets `gauss_next`); [3]
   whether `gauss` holds a value back, and [4] its bits, in a standalone
   binary (under `ppy run` that value is Python's, and `gauss` stays there). */
int64_t *ppy_random_cell(void) {
    static int64_t cell[5];
    return cell;
}

/* `_random.Random.seed`'s `init_genrand`. */
void ppy_random_init_genrand(int8_t *state, int64_t seed) {
    uint32_t *mt = (uint32_t *)(state + 4);
    mt[0] = (uint32_t)seed;
    for (int64_t i = 1; i < 624; i++) {
        mt[i] = (uint32_t)(1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i);
    }
    *(int32_t *)state = 624;
}

/* `init_by_array`, as CPython seeds from an integer's 32-bit words. */
void ppy_random_init_by_array(int8_t *state, const uint32_t *key, int64_t length) {
    uint32_t *mt = (uint32_t *)(state + 4);
    ppy_random_init_genrand(state, 19650218);
    int64_t i = 1, j = 0;
    int64_t k = 624 > length ? 624 : length;
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + key[j] + (uint32_t)j;
        i++;
        j++;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
        if (j >= length) {
            j = 0;
        }
    }
    for (k = 623; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - (uint32_t)i;
        i++;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
    }
    mt[0] = 0x80000000U;
}

/* `random.seed(a)` of an integer: its absolute value's 32-bit words, least
   significant first, at least one. */
void ppy_random_seed_int(int8_t *state, int64_t a) {
    uint64_t n = a < 0 ? (uint64_t)0 - (uint64_t)a : (uint64_t)a;
    uint32_t key[2];
    key[0] = (uint32_t)n;
    key[1] = (uint32_t)(n >> 32);
    ppy_random_init_by_array(state, key, key[1] ? 2 : 1);
}

/* An unseeded generator, as CPython seeds one: 624 words from the operating
   system, or the clock where it has none. */
void ppy_random_seed_system(int8_t *state) {
    uint32_t key[624];
    size_t got = 0;
    FILE *source = fopen("/dev/urandom", "rb");
    if (source != NULL) {
        got = fread(key, sizeof key[0], 624, source);
        fclose(source);
    }
    if (got != 624) {
        uint64_t mix = (uint64_t)time(NULL) ^ ((uint64_t)clock() << 32) ^ (uint64_t)(uintptr_t)key;
        for (int64_t i = 0; i < 624; i++) {
            mix ^= mix << 13;
            mix ^= mix >> 7;
            mix ^= mix << 17;
            key[i] = (uint32_t)(mix >> 16);
        }
    }
    ppy_random_init_by_array(state, key, 624);
}

/* The state native code draws from: the bound one, or this process's own. */
int8_t *ppy_random_state(void) {
    int64_t *cell = ppy_random_cell();
    if (cell[0] != 0) {
        return (int8_t *)(intptr_t)cell[0];
    }
    static int32_t own[625];
    if (!cell[1]) {
        ppy_random_seed_system((int8_t *)own);
        cell[1] = 1;
    }
    return (int8_t *)own;
}

/* `random.seed(a)` of an integer (`given`), or `random.seed()`: the state,
   and the value `gauss` held back, forgotten. */
void ppy_random_reseed(int8_t *state, int64_t given, int64_t a) {
    if (given) {
        ppy_random_seed_int(state, a);
    } else {
        ppy_random_seed_system(state);
    }
    int64_t *cell = ppy_random_cell();
    cell[2] = 1;
    cell[3] = 0;
}

/* Draw from the state at `address`: `random._inst`'s, handed over by the
   Python side. */
void ppy_random_bind(int64_t address) {
    ppy_random_cell()[0] = address;
}

/* Whether a seed ran since the last ask; asking clears it. */
int64_t ppy_random_reseeded(void) {
    int64_t *cell = ppy_random_cell();
    int64_t was = cell[2];
    cell[2] = 0;
    return was;
}

/* `genrand_uint32`. */
int64_t ppy_random_u32(int8_t *state) {
    static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    int32_t *index = (int32_t *)state;
    uint32_t *mt = (uint32_t *)(state + 4);
    uint32_t y;
    if (*index >= 624) {
        int64_t kk;
        for (kk = 0; kk < 624 - 397; kk++) {
            y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
            mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; kk < 623; kk++) {
            y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
            mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        y = (mt[623] & 0x80000000U) | (mt[0] & 0x7fffffffU);
        mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 0x1U];
        *index = 0;
    }
    y = mt[(*index)++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return (int64_t)y;
}

/* `random()`: 53 bits, as CPython builds them. */
double ppy_random_double(int8_t *state) {
    uint32_t a = (uint32_t)ppy_random_u32(state) >> 5;
    uint32_t b = (uint32_t)ppy_random_u32(state) >> 6;
    return ((double)a * 67108864.0 + (double)b) * (1.0 / 9007199254740992.0);
}

/* `getrandbits(k)` for 0 <= k <= 63: words least significant first, the
   last one shifted down to what is left. */
int64_t ppy_random_bits(int8_t *state, int64_t k) {
    if (k <= 0) {
        return 0;
    }
    if (k <= 32) {
        return (int64_t)((uint32_t)ppy_random_u32(state) >> (32 - k));
    }
    uint64_t low = (uint32_t)ppy_random_u32(state);
    uint64_t high = (uint32_t)ppy_random_u32(state) >> (64 - k);
    return (int64_t)(low | (high << 32));
}

/* `_randbelow(n)` for n > 0: rejection sampling over `n.bit_length()` bits. */
int64_t ppy_random_below(int8_t *state, int64_t n) {
    int64_t k = 0;
    for (uint64_t v = (uint64_t)n; v; v >>= 1) {
        k++;
    }
    int64_t r = ppy_random_bits(state, k);
    while (r >= n) {
        r = ppy_random_bits(state, k);
    }
    return r;
}

/* `shuffle(x)` over a sequence: Fisher-Yates from the end, `_randbelow(i + 1)`. */
void ppy_random_shuffle(int8_t *state, int8_t *handle) {
    int64_t *header = (int64_t *)handle;
    int64_t words = header[8];
    int64_t n = header[0];
    int64_t swap[64];
    for (int64_t i = n - 1; i >= 1; i--) {
        int64_t j = ppy_random_below(state, i + 1);
        if (j == i) {
            continue;
        }
        int64_t *a = (int64_t *)ppy_seq_at(handle, i);
        int64_t *b = (int64_t *)ppy_seq_at(handle, j);
        memcpy(swap, a, (size_t)(words * 8));
        memcpy(a, b, (size_t)(words * 8));
        memcpy(b, swap, (size_t)(words * 8));
    }
}

/* The positions `sample(population, k)` takes from a population of `n`, in
   selection order, as a `Vec[int]`: a pool swapped from the end where the
   population is small, a set of the positions taken otherwise. The caller
   has checked 0 <= k <= n. */
int8_t *ppy_random_sample_positions(int8_t *state, int64_t n, int64_t k) {
    int8_t *made = ppy_seq_new(0, 1, 0, 0);
    int64_t setsize = 21;
    if (k > 5) {
        setsize += (int64_t)pow(4.0, ceil(log((double)(k * 3)) / log(4.0)));
    }
    if (n <= setsize) {
        int64_t *pool = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
        if (pool == NULL) {
            ppy_coll_fail();
        }
        for (int64_t i = 0; i < n; i++) {
            pool[i] = i;
        }
        for (int64_t i = 0; i < k; i++) {
            int64_t j = ppy_random_below(state, n - i);
            *(int64_t *)ppy_seq_push_back(made) = pool[j];
            pool[j] = pool[n - i - 1];
        }
        free(pool);
        return made;
    }
    /* An open-addressed set of the positions taken, twice as big as k. */
    int64_t size = 16;
    while (size < 2 * k + 2) {
        size *= 2;
    }
    int64_t *taken = (int64_t *)malloc((size_t)size * sizeof(int64_t));
    if (taken == NULL) {
        ppy_coll_fail();
    }
    for (int64_t s = 0; s < size; s++) {
        taken[s] = -1;
    }
    for (int64_t i = 0; i < k; i++) {
        int64_t j;
        int64_t slot;
        for (;;) {
            j = ppy_random_below(state, n);
            slot = (int64_t)(((uint64_t)j * 0x9E3779B97F4A7C15ULL) >> 20) & (size - 1);
            while (taken[slot] != -1 && taken[slot] != j) {
                slot = (slot + 1) & (size - 1);
            }
            if (taken[slot] == -1) {
                break;
            }
        }
        taken[slot] = j;
        *(int64_t *)ppy_seq_push_back(made) = j;
    }
    free(taken);
    return made;
}

/* `normalvariate(mu, sigma)`: Kinderman and Monahan's ratio method. */
double ppy_random_normal(int8_t *state, double mu, double sigma) {
    double magic = 4.0 * exp(-0.5) / sqrt(2.0);
    double z;
    for (;;) {
        double u1 = ppy_random_double(state);
        double u2 = 1.0 - ppy_random_double(state);
        z = magic * (u1 - 0.5) / u2;
        double zz = z * z / 4.0;
        if (zz <= -log(u2)) {
            break;
        }
    }
    return mu + z * sigma;
}

/* How many values `range(0, width, step)` holds, for `randrange`: step
   nonzero; 0 or less where it is empty. */
int64_t ppy_random_range_count(int64_t width, int64_t step) {
    __int128 w = width, t = step, n;
    __int128 top = step > 0 ? w + t - 1 : w + t + 1;
    n = top / t;
    if ((top % t != 0) && ((top < 0) != (t < 0))) {
        n -= 1;
    }
    if (n > INT64_MAX) {
        return INT64_MAX;
    }
    return n < 0 ? 0 : (int64_t)n;
}

/* `randrange`'s pick: `start + step * _randbelow(n)`, which lies in the range. */
int64_t ppy_random_range_pick(int8_t *state, int64_t start, int64_t n, int64_t step) {
    __int128 r = ppy_random_below(state, n);
    return (int64_t)((__int128)start + (__int128)step * r);
}

/* `len(range(start, stop, step))`, step nonzero. */
int64_t ppy_random_range_len(int64_t start, int64_t stop, int64_t step) {
    __int128 a = start, b = stop, t = step;
    if (t > 0 ? a >= b : a <= b) {
        return 0;
    }
    __int128 n = t > 0 ? (b - a - 1) / t + 1 : (a - b - 1) / (-t) + 1;
    return n > INT64_MAX ? INT64_MAX : (int64_t)n;
}

/* Positions into a range: each `p` becomes `start + step * p`. */
void ppy_random_affine(int8_t *positions, int64_t start, int64_t step) {
    int64_t n = ((int64_t *)positions)[0];
    for (int64_t i = 0; i < n; i++) {
        int64_t *word = (int64_t *)ppy_seq_at(positions, i);
        *word = (int64_t)((__int128)start + (__int128)step * *word);
    }
}

/* A new list of `population`'s elements at `positions`, in their order: the
   collections they hold shared, each with one more reference. */
int8_t *ppy_random_gather(int8_t *population, int8_t *positions) {
    int64_t *header = (int64_t *)population;
    int64_t k = ((int64_t *)positions)[0];
    /* Word 13 carries the leaves above bit 48; `ppy_coll_make` takes them
       above bit 32 of the handle mask. */
    int64_t leaves = (int64_t)(((uint64_t)header[13] >> 48) << 32);
    int8_t *made = ppy_coll_make(0, header[13] & 0xFFFFFFFF, header[8], header[9],
                                 header[10] | leaves, header[15], k > 0 ? k : 1);
    ppy_coll_inherit(made, population);
    for (int64_t i = 0; i < k; i++) {
        int64_t at = *(int64_t *)ppy_seq_at(positions, i);
        int8_t *slot = ppy_seq_push_back(made);
        memcpy(slot, ppy_seq_at(population, at), (size_t)(header[8] * 8));
        ppy_coll_retain_words(made, slot);
    }
    return made;
}

/* `choices(population, k=k)`: `floor(random() * n)`, k times, as positions. */
int8_t *ppy_random_choice_positions(int8_t *state, int64_t n, int64_t k) {
    int8_t *made = ppy_seq_new(0, 1, 0, 0);
    double size = (double)n;
    for (int64_t i = 0; i < k; i++) {
        *(int64_t *)ppy_seq_push_back(made) = (int64_t)floor(ppy_random_double(state) * size);
    }
    return made;
}

/* The weights' running total as `choices` builds it (`accumulate`, or the
   cumulative weights as given), each as a double, into `into`. */
void ppy_random_cumulative(int8_t *weights, int64_t cumulative, int64_t integers,
                           double *into) {
    int64_t n = ((int64_t *)weights)[0];
    int64_t whole = 0;
    double part = 0.0;
    for (int64_t i = 0; i < n; i++) {
        int8_t *at = ppy_seq_at(weights, i);
        if (integers) {
            int64_t w = *(int64_t *)at;
            whole = cumulative ? w : whole + w;
            into[i] = (double)whole;
        } else {
            double w = *(double *)at;
            part = cumulative ? w : part + w;
            into[i] = part;
        }
    }
}

/* `cum_weights[-1] + 0.0`: the total `choices` draws below. */
double ppy_random_weights_total(int8_t *weights, int64_t cumulative, int64_t integers) {
    int64_t n = ((int64_t *)weights)[0];
    if (n == 0) {
        return 0.0;
    }
    if (cumulative) {
        int8_t *last = ppy_seq_at(weights, n - 1);
        return integers ? (double)*(int64_t *)last : *(double *)last;
    }
    int64_t whole = 0;
    double part = 0.0;
    for (int64_t i = 0; i < n; i++) {
        int8_t *at = ppy_seq_at(weights, i);
        if (integers) {
            whole += *(int64_t *)at;
        } else {
            part += *(double *)at;
        }
    }
    return integers ? (double)whole : part;
}

/* What `choices` says of a total: 0 fine, 1 not above zero, 2 not finite. */
int64_t ppy_random_total_fault(double total) {
    if (total <= 0.0) {
        return 1;
    }
    return total - total == 0.0 ? 0 : 2; /* an infinity or a NaN is not */
}

/* `choices(population, weights, k=k)`: `bisect(cum, random() * total, 0,
   n - 1)`, k times, as positions. */
int8_t *ppy_random_weighted_positions(int8_t *state, int8_t *weights, int64_t cumulative,
                                      int64_t integers, int64_t k, double total) {
    int64_t n = ((int64_t *)weights)[0];
    double *cum = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (cum == NULL) {
        ppy_coll_fail();
    }
    ppy_random_cumulative(weights, cumulative, integers, cum);
    int8_t *made = ppy_seq_new(0, 1, 0, 0);
    for (int64_t i = 0; i < k; i++) {
        double x = ppy_random_double(state) * total;
        int64_t lo = 0, hi = n - 1;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (x < cum[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        *(int64_t *)ppy_seq_push_back(made) = lo;
    }
    free(cum);
    return made;
}

/* `triangular(low, high, mode)`; `has_mode` 0 is `mode=None`. */
double ppy_random_triangular(int8_t *state, double low, double high, double mode,
                             int64_t has_mode) {
    double u = ppy_random_double(state);
    double c = 0.5;
    if (has_mode) {
        if (high - low == 0.0) {
            return low;
        }
        c = (mode - low) / (high - low);
    }
    if (u > c) {
        u = 1.0 - u;
        c = 1.0 - c;
        double swap = low;
        low = high;
        high = swap;
    }
    return low + (high - low) * sqrt(u * c);
}

/* `expovariate(lambd)`. It draws before it divides, as Python does: with
   `lambd` zero the caller raises after the draw. */
double ppy_random_expo(int8_t *state, double lambd) {
    double drawn = -log(1.0 - ppy_random_double(state));
    return lambd != 0.0 ? drawn / lambd : 0.0;
}

/* `gauss(mu, sigma)`, with the value it holds back, in a standalone binary. */
double ppy_random_gauss(int8_t *state, double mu, double sigma) {
    return ppy_random_gauss_held(state, mu, sigma, (int8_t *)(ppy_random_cell() + 3));
}

/* `gauss`, the value it holds back in `held`: [0] whether there is one,
   [1] its bits. */
double ppy_random_gauss_held(int8_t *state, double mu, double sigma, int8_t *where) {
    int64_t *held = (int64_t *)where;
    double z;
    if (held[0]) {
        memcpy(&z, &held[1], sizeof z);
        held[0] = 0;
    } else {
        double x2pi = ppy_random_double(state) * 6.283185307179586;
        double g2rad = sqrt(-2.0 * log(1.0 - ppy_random_double(state)));
        z = cos(x2pi) * g2rad;
        double next = sin(x2pi) * g2rad;
        memcpy(&held[1], &next, sizeof next);
        held[0] = 1;
    }
    return mu + z * sigma;
}

/* A `random.Random` of its own: a sequence of 316 words, the state in the
   first 2500 bytes, what `gauss` holds back in words 313 and 314, and 0 in
   word 315 (`ppy_random_external`), seeded from `seed` (`given`) or from
   the operating system. */
int8_t *ppy_random_new(int64_t given, int64_t seed) {
    int8_t *made = ppy_seq_new(316, 1, 0, 0);
    ppy_random_seed_instance(made, given, seed);
    return made;
}

/* `r.seed(a)`, `r.seed()`: the state, and what `gauss` held back, forgotten. */
void ppy_random_seed_instance(int8_t *made, int64_t given, int64_t seed) {
    int8_t *state = ppy_seq_at(made, 0);
    if (given) {
        ppy_random_seed_int(state, seed);
    } else {
        ppy_random_seed_system(state);
    }
    ((int64_t *)state)[313] = 0;
}

/* A `random.Random` Python lends: the same sequence, its word 315 the
   address of the object's own state, which draws move in place. */
int8_t *ppy_random_external(int64_t address) {
    int8_t *made = ppy_seq_new(316, 1, 0, 0);
    ((int64_t *)ppy_seq_at(made, 0))[315] = address;
    return made;
}

/* Whether an instance is one Python lent: its `gauss` and `seed` touch
   `gauss_next`, which Python keeps. */
int64_t ppy_random_instance_lent(int8_t *made) {
    return ((int64_t *)ppy_seq_at(made, 0))[315] != 0;
}

/* An instance's state, and where `gauss` holds back a value. */
int8_t *ppy_random_instance_state(int8_t *made) {
    int64_t lent = ((int64_t *)ppy_seq_at(made, 0))[315];
    return lent ? (int8_t *)(intptr_t)lent : ppy_seq_at(made, 0);
}

int8_t *ppy_random_instance_held(int8_t *made) {
    return (int8_t *)((int64_t *)ppy_seq_at(made, 0) + 313);
}

/* `paretovariate(alpha)`: `(1 - random()) ** (-1 / alpha)`, drawn before the
   division as `ppy_random_expo` is. */
double ppy_random_pareto(int8_t *state, double alpha) {
    double u = 1.0 - ppy_random_double(state);
    return alpha != 0.0 ? pow(u, -1.0 / alpha) : 0.0;
}

/* `weibullvariate(alpha, beta)`, drawn before the division. */
double ppy_random_weibull(int8_t *state, double alpha, double beta) {
    double u = 1.0 - ppy_random_double(state);
    return beta != 0.0 ? alpha * pow(-log(u), 1.0 / beta) : 0.0;
}

/* `gammavariate(alpha, beta)`, both above zero: `random.py`'s three cases. */
double ppy_random_gamma(int8_t *state, double alpha, double beta) {
    if (alpha > 1.0) {
        double ainv = sqrt(2.0 * alpha - 1.0);
        double bbb = alpha - log(4.0);
        double ccc = alpha + ainv;
        double sg_magic = 1.0 + log(4.5);
        for (;;) {
            double u1 = ppy_random_double(state);
            if (!(1e-7 < u1 && u1 < 0.9999999)) {
                continue;
            }
            double u2 = 1.0 - ppy_random_double(state);
            double v = log(u1 / (1.0 - u1)) / ainv;
            double x = alpha * exp(v);
            double z = u1 * u1 * u2;
            double r = bbb + ccc * v - x;
            if (r + sg_magic - 4.5 * z >= 0.0 || r >= log(z)) {
                return x * beta;
            }
        }
    }
    if (alpha == 1.0) {
        return -log(1.0 - ppy_random_double(state)) * beta;
    }
    double e = 2.718281828459045;
    double x;
    for (;;) {
        double u = ppy_random_double(state);
        double b = (e + alpha) / e;
        double p = b * u;
        if (p <= 1.0) {
            x = pow(p, 1.0 / alpha);
        } else {
            x = -log((b - p) / alpha);
        }
        double u1 = ppy_random_double(state);
        if (p > 1.0) {
            if (u1 <= pow(x, alpha - 1.0)) {
                break;
            }
        } else if (u1 <= exp(-x)) {
            break;
        }
    }
    return x * beta;
}

/* `betavariate(alpha, beta)`, alpha above zero. A NaN where the second
   `gammavariate` would raise, after the first one drew. */
double ppy_random_beta(int8_t *state, double alpha, double beta) {
    double y = ppy_random_gamma(state, alpha, 1.0);
    if (y != 0.0) {
        if (!(beta > 0.0)) {
            return NAN;
        }
        return y / (y + ppy_random_gamma(state, beta, 1.0));
    }
    return 0.0;
}
