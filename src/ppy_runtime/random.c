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
   since the last ask (Python's `seed` also forgets `gauss_next`). */
int64_t *ppy_random_cell(void) {
    static int64_t cell[3];
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
    ppy_random_cell()[2] = 1;
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

/* `choices(population, cum_weights=..., k)` for one pick: `bisect_right` of
   `random() * total` over the cumulative weights, below `hi`. */
int64_t ppy_random_pick_weighted(int8_t *state, const double *cumulative, int64_t n,
                                 double total) {
    double x = ppy_random_double(state) * total;
    int64_t lo = 0, hi = n - 1;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (x < cumulative[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
