/* hello_sse.c — Introduction to SSE intrinsics
 *
 * Compile: gcc -O2 -msse4.2 -o hello_sse hello_sse.c
 *
 * SSE (Streaming SIMD Extensions) adds 8 128-bit XMM registers to x86-64.
 * Instead of processing one float at a time, we process 4 simultaneously.
 *
 * Header: <immintrin.h> includes everything.
 * You can also include targeted headers:
 *   <xmmintrin.h>  → SSE
 *   <emmintrin.h>  → SSE2
 *   <pmmintrin.h>  → SSE3
 *   <tmmintrin.h>  → SSSE3
 *   <smmintrin.h>  → SSE4.1
 *   <nmmintrin.h>  → SSE4.2
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#define N 1024

/* ── Scalar reference ─────────────────────────────────────────────────────── */

void scalar_add(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ── SSE intrinsics ───────────────────────────────────────────────────────── */

/*
 * Naming convention:  _mm_{op}_{suffix}
 *   ps  = packed single  (4 × float32)
 *   pd  = packed double  (2 × float64)
 *   ss  = scalar single  (operates on element 0 only)
 *   sd  = scalar double  (operates on element 0 only)
 *   epi32 = packed 32-bit signed integers  (epi8, epi16, epi64, ...)
 *   epu8  = packed 8-bit unsigned integers (epu16, ...)
 */

void sse_add(const float *a, const float *b, float *c, int n) {
    int i = 0;

    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);   /* load 4 floats — 'u' = unaligned OK */
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);    /* lane-wise add: [a0+b0, a1+b1, a2+b2, a3+b3] */
        _mm_storeu_ps(c + i, vc);
    }

    /* scalar tail: handle remainder when n is not a multiple of 4 */
    for (; i < n; i++)
        c[i] = a[i] + b[i];
}

/*
 * Aligned variant.
 * _mm_load_ps requires the pointer to be 16-byte aligned.
 * On modern CPUs (Skylake and later) the performance difference between
 * aligned and unaligned load/store is negligible for hot-cache data.
 * It matters more for large streaming workloads.
 */
void sse_add_aligned(const float * __restrict__ a,
                     const float * __restrict__ b,
                     float * __restrict__ c, int n) {
    /* n must be a multiple of 4; arrays must be 16-byte aligned */
    for (int i = 0; i < n; i += 4) {
        __m128 va = _mm_load_ps(a + i);
        __m128 vb = _mm_load_ps(b + i);
        _mm_store_ps(c + i, _mm_add_ps(va, vb));
    }
}

/* ── Common SSE operations ────────────────────────────────────────────────── */

void demo_basic_ops(void) {
    printf("\n=== Basic SSE ops ===\n");

    /* Set/splat operations */
    __m128 zeros = _mm_setzero_ps();                       /* {0, 0, 0, 0} */
    __m128 ones  = _mm_set1_ps(1.0f);                      /* {1, 1, 1, 1} */

    /* _mm_set_ps arguments are in REVERSE order: element 3, 2, 1, 0 */
    __m128 v = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);        /* {1, 2, 3, 4} in memory */

    /* Arithmetic */
    __m128 added  = _mm_add_ps(v, ones);   /* {2, 3, 4, 5} */
    __m128 mulled = _mm_mul_ps(v, v);      /* {1, 4, 9, 16} */
    __m128 sqrts  = _mm_sqrt_ps(v);        /* {1, 1.41, 1.73, 2} */

    /* Fast approximate reciprocal sqrt — ~12 bits of precision */
    __m128 rsqrt  = _mm_rsqrt_ps(v);       /* ≈ {1, 0.707, 0.577, 0.5} */

    float out[4];
    _mm_storeu_ps(out, added);
    printf("add  v + 1:   %.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);

    _mm_storeu_ps(out, mulled);
    printf("mul  v * v:   %.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);

    _mm_storeu_ps(out, sqrts);
    printf("sqrt v:       %.3f %.3f %.3f %.3f\n", out[0], out[1], out[2], out[3]);

    (void)zeros; (void)rsqrt;
}

/* ── Comparison and masking ───────────────────────────────────────────────── */
/*
 * SSE comparisons don't return a scalar bool — they return a mask vector.
 * Each lane is either all-ones (0xFFFFFFFF) if true, or all-zeros if false.
 * You then use bitwise ops (_mm_and_ps, _mm_andnot_ps) or SSE4.1 blend
 * to select between values.
 */
void demo_masking(void) {
    printf("\n=== Comparison and masking ===\n");

    __m128 v     = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  /* {1, 2, 3, 4} */
    __m128 thresh = _mm_set1_ps(2.5f);

    /* Compare: each lane → all-ones if v > thresh, all-zeros if not */
    __m128 mask  = _mm_cmpgt_ps(v, thresh);   /* {0, 0, 0xFFFF..., 0xFFFF...} */

    /* Select: keep v where v > 2.5, else zero */
    __m128 filtered = _mm_and_ps(v, mask);     /* {0, 0, 3, 4} */

    /* SSE4.1: blend — cleaner than and/andnot */
    __m128 replacement = _mm_set1_ps(-1.0f);
    /* blendv: take from 'replacement' where mask[i] has high bit set, else from 'v' */
    __m128 blended = _mm_blendv_ps(v, replacement, mask); /* {1, 2, -1, -1} */

    /* movemask: collapse the high bit of each 4-byte lane to a 4-bit integer */
    int bits = _mm_movemask_ps(mask);  /* 0b1100 = 12, bits 2 and 3 set */

    float out[4];
    _mm_storeu_ps(out, filtered);
    printf("filtered (>2.5 or 0): %.1f %.1f %.1f %.1f\n",
           out[0], out[1], out[2], out[3]);

    _mm_storeu_ps(out, blended);
    printf("blended  (>2.5 → -1): %.1f %.1f %.1f %.1f\n",
           out[0], out[1], out[2], out[3]);

    printf("movemask bits: 0x%x  (bit i = lane i > 2.5)\n", bits);
}

/* ── Horizontal operations ────────────────────────────────────────────────── */
/*
 * Most SSE ops are vertical (lane-wise between two vectors).
 * Horizontal ops work across lanes of a single vector.
 * They are generally slower — avoid in hot loops if possible.
 */
float sse_hsum(__m128 v) {
    /* SSE3: haddps adds adjacent pairs */
    __m128 h1 = _mm_hadd_ps(v, v);   /* {v0+v1, v2+v3, v0+v1, v2+v3} */
    __m128 h2 = _mm_hadd_ps(h1, h1); /* {v0+v1+v2+v3, ...} */
    return _mm_cvtss_f32(h2);         /* extract lane 0 as scalar */
}

/* Alternative using shuffle — often preferred to hadd on older CPUs */
float sse_hsum_shuffle(__m128 v) {
    /* movehl: move high two floats to low position */
    __m128 high = _mm_movehl_ps(v, v);    /* {v2, v3, v2, v3} */
    __m128 sum2 = _mm_add_ps(v, high);    /* {v0+v2, v1+v3, ...} */
    /* shuffle to get v1+v3 into lane 0 */
    __m128 shuf = _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(1,1,1,1));
    __m128 sum1 = _mm_add_ss(sum2, shuf); /* lane 0 = (v0+v2) + (v1+v3) */
    return _mm_cvtss_f32(sum1);
}

void demo_hsum(void) {
    printf("\n=== Horizontal sum ===\n");
    __m128 v = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
    printf("hadd sum:    %f\n", sse_hsum(v));          /* should be 10 */
    printf("shuffle sum: %f\n", sse_hsum_shuffle(v));  /* should be 10 */
}

/* ── SSE2: integer SIMD ───────────────────────────────────────────────────── */
void demo_integer(void) {
    printf("\n=== SSE2 integer ops ===\n");

    /* 4 × int32 */
    __m128i a32 = _mm_set_epi32(4, 3, 2, 1);
    __m128i b32 = _mm_set1_epi32(10);
    __m128i c32 = _mm_add_epi32(a32, b32);  /* {11, 12, 13, 14} */

    int32_t out32[4];
    _mm_storeu_si128((__m128i*)out32, c32);
    printf("int32 add:  %d %d %d %d\n", out32[0], out32[1], out32[2], out32[3]);

    /* 16 × int8 — note: _mm_set_epi8 args are also reversed */
    __m128i bytes = _mm_set_epi8(15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0);
    /* Saturating add: clamps to [0..255] for unsigned, [-128..127] for signed */
    __m128i saturated = _mm_adds_epi8(bytes, _mm_set1_epi8(100));

    int8_t out8[16];
    _mm_storeu_si128((__m128i*)out8, saturated);
    printf("saturated adds (0+100, 1+100, ..., 27+100 clamped to 127): ");
    for (int i = 0; i < 16; i++) printf("%d ", out8[i]);
    printf("\n");

    /* Shift: _mm_slli_epi32 = shift left logical each 32-bit lane */
    __m128i shifted = _mm_slli_epi32(a32, 2);  /* multiply by 4 */
    _mm_storeu_si128((__m128i*)out32, shifted);
    printf("sll by 2 (×4): %d %d %d %d\n", out32[0], out32[1], out32[2], out32[3]);
}

/* ── SSE4.1: extras ───────────────────────────────────────────────────────── */
void demo_sse41(void) {
    printf("\n=== SSE4.1 extras ===\n");

    __m128 v = _mm_set_ps(4.7f, -3.2f, 2.9f, -1.1f);

    /* Floor/ceil without scalar roundtrip */
    __m128 floored = _mm_floor_ps(v);
    __m128 ceiled  = _mm_ceil_ps(v);

    float out[4];
    _mm_storeu_ps(out, floored);
    printf("floor: %.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);
    _mm_storeu_ps(out, ceiled);
    printf("ceil:  %.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);

    /* _mm_blend_ps: select lanes with an immediate (compile-time) mask
     * Bit i = 0 → take from a; bit i = 1 → take from b */
    __m128 a = _mm_set_ps(40.0f, 30.0f, 20.0f, 10.0f);
    __m128 b = _mm_set_ps(400.0f, 300.0f, 200.0f, 100.0f);
    __m128 blend = _mm_blend_ps(a, b, 0b1010);  /* lanes 1,3 from b; 0,2 from a */
    _mm_storeu_ps(out, blend);
    printf("blend (0b1010): %.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);
    /* Expected: 10, 200, 30, 400 */

    /* Integer extract/insert */
    __m128i vi = _mm_set_epi32(40, 30, 20, 10);
    int lane2 = _mm_extract_epi32(vi, 2);     /* SSE4.1: extract lane 2 without store */
    printf("extract lane 2: %d\n", lane2);    /* 30 */
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    /* Allocate 16-byte-aligned arrays for the aligned variant */
    float *a, *b, *c_scalar, *c_sse;
    posix_memalign((void**)&a,        16, N * sizeof(float));
    posix_memalign((void**)&b,        16, N * sizeof(float));
    posix_memalign((void**)&c_scalar, 16, N * sizeof(float));
    posix_memalign((void**)&c_sse,    16, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(1000 - i);
    }

    scalar_add(a, b, c_scalar, N);
    sse_add(a, b, c_sse, N);

    for (int i = 0; i < N; i++)
        assert(c_scalar[i] == c_sse[i]);
    printf("Array add: scalar and SSE match (%d elements, all %.1f).\n",
           N, c_sse[0]);

    demo_basic_ops();
    demo_masking();
    demo_hsum();
    demo_integer();
    demo_sse41();

    free(a); free(b); free(c_scalar); free(c_sse);
    return 0;
}
