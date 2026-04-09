#define _POSIX_C_SOURCE 200809L
/* string_search.c — SSE4.2 string instructions and byte scanning
 *
 * Compile: gcc -O2 -mavx512f -mavx512bw -mavx512vl -mfma \
 *              -o string_search string_search.c
 *
 * SSE4.2 added dedicated string comparison instructions:
 *   PCMPISTRM / PCMPISTRI  — implicit-length (null-terminated) strings
 *   PCMPESTRM / PCMPESTRI  — explicit-length strings
 *
 * These process 16 bytes per instruction and support:
 *   - Find character in string
 *   - Character class membership (set of chars)
 *   - Substring search
 *   - String comparison
 *
 * Real-world users: glibc memchr, strlen, strchr; Go runtime; various parsers.
 *
 * AVX2/AVX-512 approach: use _mm256_cmpeq_epi8 / _mm512_cmpeq_epi8_mask.
 * Simpler API, processes 32/64 bytes at a time.
 *
 * This file demonstrates:
 *   1. Hand-rolled AVX2 strlen and strchr (most practical)
 *   2. SSE4.2 PCMPISTR* for character class search
 *   3. Comparison of approaches
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

/* ── strlen implementations ─────────────────────────────────────────────── */

/* Scalar reference */
size_t strlen_scalar(const char *s) {
    const char *p = s;
    while (*p) p++;
    return (size_t)(p - s);
}

/* AVX2: check 32 bytes at a time for null bytes */
size_t strlen_avx2(const char *s) {
    __m256i zero = _mm256_setzero_si256();

    /* Align to 32-byte boundary to avoid page crossing issues */
    uintptr_t addr = (uintptr_t)s;
    uintptr_t aligned = addr & ~(uintptr_t)31;
    int offset = (int)(addr - aligned);

    const __m256i *p = (const __m256i*)aligned;

    /* First chunk: mask off bytes before the string starts */
    __m256i chunk = _mm256_load_si256(p);
    __m256i eq    = _mm256_cmpeq_epi8(chunk, zero);
    int mask = _mm256_movemask_epi8(eq) >> offset;  /* ignore bytes before s */
    if (mask) return (size_t)(__builtin_ctz(mask));

    p++;
    /* Subsequent 32-byte aligned chunks */
    while (1) {
        chunk = _mm256_load_si256(p);
        eq    = _mm256_cmpeq_epi8(chunk, zero);
        mask  = _mm256_movemask_epi8(eq);
        if (mask) {
            return (size_t)((const char*)p - s) + __builtin_ctz(mask) - offset;
        }
        p++;
    }
}

/* ── strchr / memchr implementations ────────────────────────────────────── */

/* AVX2 strchr: find first occurrence of c in s (null-terminated) */
const char *strchr_avx2(const char *s, char c) {
    __m256i needle = _mm256_set1_epi8(c);
    __m256i zero   = _mm256_setzero_si256();

    uintptr_t addr    = (uintptr_t)s;
    uintptr_t aligned = addr & ~(uintptr_t)31;
    int offset = (int)(addr - aligned);

    const __m256i *p = (const __m256i*)aligned;
    __m256i chunk = _mm256_load_si256(p);

    int found_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle)) >> offset;
    int null_mask  = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, zero))  >> offset;

    while (!(found_mask | null_mask)) {
        p++;
        chunk      = _mm256_load_si256(p);
        found_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle));
        null_mask  = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, zero));
    }

    const char *base = s + ((const char*)p - (const char*)(uintptr_t)aligned) - offset;
    (void)base;

    /* Prefer the found_mask if it appears before the null */
    if (found_mask && (__builtin_ctz(found_mask) < __builtin_ctz(null_mask | 0x80000000u))) {
        int pos = __builtin_ctz(found_mask);
        const char *result = (const char*)p - offset + pos;
        /* Verify it's at or after s */
        if (result >= s) return result;
    }
    return (null_mask && c == '\0') ? (const char*)p - offset + __builtin_ctz(null_mask) : NULL;
}

/* AVX-512: find byte in buffer (explicit length) — 64 bytes at a time */
const char *memchr_avx512(const char *buf, char c, size_t len) {
    __m512i needle = _mm512_set1_epi8(c);
    size_t i = 0;

    for (; i + 64 <= len; i += 64) {
        __m512i chunk = _mm512_loadu_si512(buf + i);
        __mmask64 eq  = _mm512_cmpeq_epi8_mask(chunk, needle);
        if (eq) return buf + i + __builtin_ctzll(eq);
    }

    /* Tail with masked load */
    if (i < len) {
        int tail = (int)(len - i);
        __mmask64 k = (__mmask64)(tail == 64 ? ~0ULL : (1ULL << tail) - 1ULL);
        __m512i chunk = _mm512_maskz_loadu_epi8(k, buf + i);
        __mmask64 eq  = _mm512_mask_cmpeq_epi8_mask(k, chunk, needle);
        if (eq) return buf + i + __builtin_ctzll(eq);
    }

    return NULL;
}

/* ── SSE4.2 PCMPISTR*: specialized string instructions ──────────────────── */
/*
 * _mm_cmpistri(a, b, imm8):
 *   a     = reference string (up to 16 chars, null-terminated)
 *   b     = data string (up to 16 chars, null-terminated)
 *   imm8  = mode flags (see below)
 *   return: index of first match (or 16 if none)
 *
 * The mode flags (_SIDD_*) control:
 *   Data type: _SIDD_UBYTE_OPS (uint8), _SIDD_UWORD_OPS (uint16)
 *   Comparison:
 *     _SIDD_CMP_EQUAL_ANY    — b[i] == any char in a (character class membership)
 *     _SIDD_CMP_RANGES       — a has pairs [lo,hi]: is b[i] in any range?
 *     _SIDD_CMP_EQUAL_EACH   — a[i] == b[i] (strcmp-style)
 *     _SIDD_CMP_EQUAL_ORDERED — is a a substring of b?
 *   Polarity: _SIDD_POSITIVE_POLARITY or _SIDD_NEGATIVE_POLARITY
 *   Output: _SIDD_LEAST_SIGNIFICANT (first match) or _SIDD_MOST_SIGNIFICANT (last)
 *
 * Useful helpers that read CPU flags set by pcmpistr*:
 *   _mm_cmpistrc(a,b,imm) — carry flag: result < 16 (found before end of b)
 *   _mm_cmpistrz(a,b,imm) — zero flag: null found in b (end of string)
 *   _mm_cmpistrs(a,b,imm) — sign flag: null found in a
 *   _mm_cmpistra(a,b,imm) — above flag: not at end and not found
 */

/* Find first character from a set (character class membership).
 * Example: find first whitespace, first non-printable, etc. */
int find_char_in_set_sse42(const char *str, const char *charset) {
    /* Load charset into xmm (up to 16 chars) */
    __m128i chars = _mm_loadu_si128((const __m128i*)charset);

    int mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
               _SIDD_POSITIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT;

    for (int i = 0; ; i += 16) {
        __m128i data = _mm_loadu_si128((const __m128i*)(str + i));

        /* Check if any char in data is in charset */
        if (_mm_cmpistrc(chars, data, mode)) {
            return i + _mm_cmpistri(chars, data, mode);
        }
        /* Check if we hit end of string */
        if (_mm_cmpistrz(chars, data, mode)) {
            return -1;  /* not found */
        }
    }
}

/* Find characters NOT in a set (inverse) */
int find_char_not_in_set_sse42(const char *str, const char *charset) {
    __m128i chars = _mm_loadu_si128((const __m128i*)charset);

    int mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
               _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT;

    for (int i = 0; ; i += 16) {
        __m128i data = _mm_loadu_si128((const __m128i*)(str + i));
        if (_mm_cmpistrc(chars, data, mode)) {
            return i + _mm_cmpistri(chars, data, mode);
        }
        if (_mm_cmpistrz(chars, data, mode)) return -1;
    }
}

/* strspn equivalent: count leading chars in set */
size_t strspn_sse42(const char *str, const char *accept) {
    __m128i chars = _mm_loadu_si128((const __m128i*)accept);
    /* Find first char NOT in accept set */
    int mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
               _SIDD_NEGATIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT;

    for (int i = 0; ; i += 16) {
        __m128i data = _mm_loadu_si128((const __m128i*)(str + i));
        if (_mm_cmpistrc(chars, data, mode)) {
            return (size_t)(i + _mm_cmpistri(chars, data, mode));
        }
        if (_mm_cmpistrz(chars, data, mode)) return strlen(str);
    }
}

/* ── Count byte frequency using AVX2 ────────────────────────────────────── */
/* Count occurrences of each byte value in a buffer */
void count_bytes_avx2(const uint8_t *buf, size_t len, uint32_t counts[256]) {
    memset(counts, 0, 256 * sizeof(uint32_t));

    /* Process 32 bytes at a time, accumulate into 8-bit counters,
     * flush to 32-bit counts every 255 iterations to avoid overflow */
    __m256i acc[256/32]; /* 8 vectors of 32 × uint8 */
    memset(acc, 0, sizeof(acc));

    size_t flush_at = 255;
    size_t i = 0;

    while (i < len) {
        size_t chunk_end = i + flush_at * 32;
        if (chunk_end > len) chunk_end = len;

        /* Accumulate in uint8 (will count up to 255 per byte value per vector) */
        /* For each byte position j in [0,32), acc[j/32][j%32] counts buf[i+j] */
        /* This is a simplified version; a full histogram uses a transpose */
        for (; i < chunk_end; i++) {
            counts[buf[i]]++;  /* scalar fallback — full histogram is complex */
        }
    }
    /* Note: A proper SIMD histogram using shuffle-based popcount is complex.
     * See Muła's algorithm for vectorized histograms. This is left as an exercise. */
}

/* ── Benchmark ───────────────────────────────────────────────────────────── */
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define BENCH_REPS 1000

void benchmark_strlen(void) {
    printf("\n=== strlen benchmark ===\n");
    int LEN = 1 << 20;  /* 1 MB string */

    char *s = aligned_alloc(64, LEN + 64);
    memset(s, 'A', LEN);
    s[LEN] = '\0';

    volatile size_t sink = 0;
    double t0, t1;

    t0 = now();
    for (int r = 0; r < BENCH_REPS; r++) sink += strlen_scalar(s);
    t1 = now();
    printf("scalar:   %.3f ms  %.1f GB/s\n",
           (t1-t0)/BENCH_REPS*1000,
           (double)LEN/(t1-t0)*BENCH_REPS/1e9);

    t0 = now();
    for (int r = 0; r < BENCH_REPS; r++) sink += strlen_avx2(s);
    t1 = now();
    printf("AVX2:     %.3f ms  %.1f GB/s\n",
           (t1-t0)/BENCH_REPS*1000,
           (double)LEN/(t1-t0)*BENCH_REPS/1e9);

    t0 = now();
    for (int r = 0; r < BENCH_REPS; r++) sink += strlen(s);
    t1 = now();
    printf("libc:     %.3f ms  %.1f GB/s  (glibc uses AVX2 or AVX-512 internally)\n",
           (t1-t0)/BENCH_REPS*1000,
           (double)LEN/(t1-t0)*BENCH_REPS/1e9);

    free(s);
}

void benchmark_memchr(void) {
    printf("\n=== memchr benchmark ===\n");
    size_t LEN = 1 << 20;

    char *buf = aligned_alloc(64, LEN + 64);
    memset(buf, 'A', LEN);
    buf[LEN - 1] = 'Z';  /* needle at the very end */

    volatile const char *sink = NULL;
    double t0, t1;

    t0 = now();
    for (int r = 0; r < BENCH_REPS; r++) sink = (const char*)memchr(buf, 'Z', LEN);
    t1 = now();
    printf("libc memchr:       %.3f ms  %.1f GB/s\n",
           (t1-t0)/BENCH_REPS*1000, LEN/(t1-t0)*BENCH_REPS/1e9);

    t0 = now();
    for (int r = 0; r < BENCH_REPS; r++) sink = memchr_avx512(buf, 'Z', LEN);
    t1 = now();
    printf("AVX-512 memchr:    %.3f ms  %.1f GB/s\n",
           (t1-t0)/BENCH_REPS*1000, LEN/(t1-t0)*BENCH_REPS/1e9);

    (void)sink;
    free(buf);
}

int main(void) {
    printf("=== SSE4.2 string instructions demo ===\n\n");

    /* strlen */
    {
        char buf[64] = "Hello, SIMD world! This string is 37 characters.";
        size_t n1 = strlen_scalar(buf);
        size_t n2 = strlen_avx2(buf);
        printf("strlen scalar: %zu\n", n1);
        printf("strlen avx2:   %zu\n", n2);
        assert(n1 == n2);
    }

    /* Character class search */
    {
        const char *str = "Hello, World! This has spaces and punctuation.";
        /* Find first whitespace */
        char whitespace[16] = " \t\n\r";  /* 4 chars + null-pad to 16 */
        memset(whitespace + 4, 0, 12);

        int pos = find_char_in_set_sse42(str, whitespace);
        printf("\nFirst whitespace in \"%s\"\n  at index %d (char '%c')\n",
               str, pos, str[pos]);

        /* Find first non-letter using charset of valid letters */
        char letters[16];
        memset(letters, 0, 16);
        /* For ranges, use _SIDD_CMP_RANGES mode: pairs (lo,hi) */
        /* Here we'll just demonstrate with specific chars */
        int pos2 = find_char_in_set_sse42(str, whitespace);
        printf("First space: index %d\n", pos2);
    }

    /* strspn */
    {
        const char *s = "   \t  hello";
        char ws[16] = " \t";
        memset(ws + 2, 0, 14);
        size_t leading = strspn_sse42(s, ws);
        printf("\nLeading whitespace count: %zu\n", leading);
        assert(leading == 6);
    }

    benchmark_strlen();
    benchmark_memchr();

    printf("\nKey takeaways:\n");
    printf("  - AVX2 strlen processes 32 bytes per iteration vs 1 byte scalar\n");
    printf("  - AVX-512 memchr processes 64 bytes per masked load\n");
    printf("  - glibc uses similar techniques internally; don't rewrite unless profiling\n");
    printf("    shows it's actually a bottleneck\n");
    printf("  - SSE4.2 pcmpistr* is useful for character class and substring search\n");
    printf("    but requires care: only handles 16 chars, null-terminated semantics\n");

    return 0;
}
