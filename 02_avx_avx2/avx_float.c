/* avx_float.c — AVX and AVX2 floating-point SIMD
 *
 * Compile: gcc -O2 -mavx2 -mfma -o avx_float avx_float.c
 *
 * AVX (2011): 256-bit YMM registers, 8 × float32 or 4 × float64.
 *             Mostly float operations; integer ops still 128-bit.
 *             Key additions: 3-operand (non-destructive) form for most ops,
 *             _mm256_permute* for lane-crossing shuffles.
 *
 * AVX2 (2013): Extends integer operations to full 256-bit.
 *              Adds gather loads, vperm2i128, and more.
 *
 * FMA3 (with AVX2): VFMADD231PS and friends — fused multiply-add in one
 *                   instruction with one rounding error instead of two.
 *
 * Important: upper YMM bits and the VZEROUPPER instruction
 *   Mixing SSE (XMM) and AVX (YMM) without VZEROUPPER causes a large
 *   performance penalty on Intel Skylake-era CPUs (AVX-SSE transition penalty).
 *   GCC handles this automatically when using intrinsics.
 *   Always compile all translation units with -mavx2 or use -march=native.
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#define N 1024

/* ── Basic 256-bit float ops ─────────────────────────────────────────────── */
void demo_avx_basics(void) {
    printf("=== AVX basics: __m256 (8 × float32) ===\n");

    /* Set: arguments are in reverse order (element 7 first, element 0 last) */
    __m256 v = _mm256_set_ps(8,7,6,5, 4,3,2,1);   /* {1,2,3,4,5,6,7,8} */
    __m256 ones = _mm256_set1_ps(1.0f);
    __m256 zeros = _mm256_setzero_ps();

    __m256 added = _mm256_add_ps(v, ones);
    __m256 mulled = _mm256_mul_ps(v, v);
    __m256 sqrts = _mm256_sqrt_ps(v);

    float out[8];
    _mm256_storeu_ps(out, added);
    printf("v + 1:  "); for (int i=0;i<8;i++) printf("%.1f ",out[i]); printf("\n");

    _mm256_storeu_ps(out, mulled);
    printf("v * v:  "); for (int i=0;i<8;i++) printf("%.1f ",out[i]); printf("\n");

    _mm256_storeu_ps(out, sqrts);
    printf("sqrt v: "); for (int i=0;i<8;i++) printf("%.2f ",out[i]); printf("\n");

    /* AVX has a 3-operand form: c = a op b (non-destructive, unlike SSE) */
    /* This is actually the default for all AVX intrinsics */
    (void)zeros;
}

/* ── Comparison and masking in AVX ───────────────────────────────────────── */
void demo_avx_masking(void) {
    printf("\n=== AVX masking (8-wide) ===\n");

    __m256 a = _mm256_set_ps(8,7,6,5, 4,3,2,1);
    __m256 thresh = _mm256_set1_ps(4.5f);

    /* Compare: returns all-ones or all-zeros per lane */
    __m256 mask = _mm256_cmp_ps(a, thresh, _CMP_GT_OS); /* ordered, signaling */
    /* _CMP predicates: _CMP_LT_OS, _CMP_LE_OS, _CMP_EQ_OQ, _CMP_GT_OS, etc. */

    /* movemask: 8-bit integer where bit i = sign bit of lane i */
    int bits = _mm256_movemask_ps(mask);  /* 0b11110000 = 0xF0 */
    printf("mask bits (lanes > 4.5): 0x%02x  (bits 4-7 set)\n", bits);

    /* blendv: select from a or b per-lane based on sign bit of mask */
    __m256 negone = _mm256_set1_ps(-1.0f);
    __m256 blended = _mm256_blendv_ps(a, negone, mask); /* >4.5 → -1 */
    float out[8];
    _mm256_storeu_ps(out, blended);
    printf("blendv (>4.5 → -1.0):   ");
    for (int i=0;i<8;i++) printf("%.1f ", out[i]);
    printf("\n");

    /* blend with immediate: _mm256_blend_ps(a, b, imm8)
     * bit i=1 → lane i from b, bit i=0 → from a */
    __m256 b = _mm256_set1_ps(99.0f);
    __m256 blend_imm = _mm256_blend_ps(a, b, 0b00110011); /* lanes 0,1,4,5 from b */
    _mm256_storeu_ps(out, blend_imm);
    printf("blend imm (0b00110011): ");
    for (int i=0;i<8;i++) printf("%.0f ", out[i]);
    printf("\n");
}

/* ── FMA: Fused Multiply-Add ─────────────────────────────────────────────── */
/*
 * FMA computes a*b + c in a single instruction with one rounding error.
 * Without FMA: mul then add = two instructions, two roundings.
 * Benefits:
 *   1. Higher throughput: one instruction instead of two
 *   2. Better precision: one rounding preserves more bits
 *   3. Critical for numerically stable algorithms
 *
 * Three forms depending on which operand accumulates:
 *   _mm256_fmadd_ps(a, b, c)  = a*b + c      (c is the addend)
 *   _mm256_fmsub_ps(a, b, c)  = a*b - c
 *   _mm256_fnmadd_ps(a, b, c) = -(a*b) + c   (negate product)
 *
 * The 132/213/231 variants control which register is overwritten:
 *   _mm256_fmadd132_ps(a, b, c) = a*c + b  (overwrites a)
 *   _mm256_fmadd213_ps(a, b, c) = a*b + c  (overwrites a) ← most common
 *   _mm256_fmadd231_ps(a, b, c) = b*c + a  (overwrites a, adds to a)
 */
void demo_fma(void) {
    printf("\n=== FMA (Fused Multiply-Add) ===\n");

    __m256 a = _mm256_set1_ps(2.0f);
    __m256 b = _mm256_set1_ps(3.0f);
    __m256 c = _mm256_set1_ps(1.0f);

    /* a*b + c = 2*3 + 1 = 7 */
    __m256 fma_result = _mm256_fmadd_ps(a, b, c);

    /* Without FMA (two ops, two roundings) */
    __m256 mul_result = _mm256_mul_ps(a, b);
    __m256 add_result = _mm256_add_ps(mul_result, c);

    float fma_out[8], add_out[8];
    _mm256_storeu_ps(fma_out, fma_result);
    _mm256_storeu_ps(add_out, add_result);

    printf("fmadd(2,3,1): %.6f  (single rounding)\n", fma_out[0]);
    printf("mul+add:      %.6f  (two roundings — may differ for large inputs)\n", add_out[0]);

    /* Precision difference: Kahan summation style example */
    /* Near-cancellation: (1e8 + 1) - 1e8 should be 1.0 */
    __m256 big = _mm256_set1_ps(1e8f);
    __m256 one = _mm256_set1_ps(1.0f);

    /* (big + one) - big: with floats, big+1 may round to big, giving 0 */
    __m256 naive = _mm256_sub_ps(_mm256_add_ps(big, one), big);

    /* FMA: big*1 + 1 is exact for this case, then subtract big */
    /* (This is illustrative — FMA's real precision benefit is in dot products) */
    float out[8];
    _mm256_storeu_ps(out, naive);
    printf("\nNear-cancellation: (1e8+1)-1e8 = %.6f (may lose precision)\n", out[0]);
}

/* ── Permute and shuffle: moving data between lanes ─────────────────────── */
/*
 * AVX treats the 256-bit register as two 128-bit "lanes". Many ops
 * that shuffle within a lane don't cross the 128-bit boundary.
 * Lane-crossing operations are more expensive.
 */
void demo_permute(void) {
    printf("\n=== Permute and shuffle ===\n");

    __m256 v = _mm256_set_ps(8,7,6,5, 4,3,2,1);  /* {1,2,3,4,5,6,7,8} */
    float out[8];

    /* _mm256_permute_ps: shuffle within each 128-bit lane using 8-bit immediate
     * imm8 = two 2-bit selectors per lane (4 elements per 128-bit lane)
     * _MM_SHUFFLE(z,y,x,w) = w goes to output[0], x to [1], y to [2], z to [3]
     * Does NOT cross the 128-bit lane boundary. */
    __m256 perm_within = _mm256_permute_ps(v, _MM_SHUFFLE(0,1,2,3));
    _mm256_storeu_ps(out, perm_within);
    printf("permute (reverse within lane): ");
    for (int i=0;i<8;i++) printf("%.0f ",out[i]);
    printf("\n");  /* {4,3,2,1,8,7,6,5} */

    /* _mm256_permutevar8x32_ps (AVX2): fully general 8-element permutation
     * Can cross the 128-bit lane boundary. Useful for transpositions. */
    __m256i idx = _mm256_set_epi32(0,1,2,3,4,5,6,7);  /* reverse all 8 */
    __m256 perm_cross = _mm256_permutevar8x32_ps(v, idx);
    _mm256_storeu_ps(out, perm_cross);
    printf("permutevar8x32 (full reverse): ");
    for (int i=0;i<8;i++) printf("%.0f ",out[i]);
    printf("\n");  /* {8,7,6,5,4,3,2,1} */

    /* _mm256_permute2f128_ps: swap or select 128-bit halves
     * imm8 selects which half of src1/src2 goes to output's low/high halves */
    __m256 a = _mm256_set_ps(8,7,6,5, 4,3,2,1);
    __m256 b = _mm256_set_ps(80,70,60,50, 40,30,20,10);
    /* 0x21 = low half from b's high, high half from a's low */
    __m256 swapped = _mm256_permute2f128_ps(a, b, 0x21);
    _mm256_storeu_ps(out, swapped);
    printf("permute2f128 (cross-lane mix): ");
    for (int i=0;i<8;i++) printf("%.0f ",out[i]);
    printf("\n");  /* {40,30,20,10, 80,70,60,50} ... wait: low=b.high={80,70,60,50} */
}

/* ── Horizontal reduction: summing 8 floats to one ───────────────────────── */
/*
 * Reductions require multiple shuffles to reduce across lanes.
 * This is the standard pattern for summing a __m256.
 */
float avx_hsum(__m256 v) {
    /* Step 1: add the two 128-bit halves */
    __m128 lo = _mm256_castps256_ps128(v);       /* low 128 bits */
    __m128 hi = _mm256_extractf128_ps(v, 1);     /* high 128 bits */
    __m128 sum128 = _mm_add_ps(lo, hi);          /* {v0+v4, v1+v5, v2+v6, v3+v7} */

    /* Step 2: reduce the 4-wide sum (SSE hadd) */
    __m128 h1 = _mm_hadd_ps(sum128, sum128);     /* {v0+v4+v1+v5, v2+v6+v3+v7, ...} */
    __m128 h2 = _mm_hadd_ps(h1, h1);            /* {total, total, total, total} */
    return _mm_cvtss_f32(h2);
}

/* ── 256-bit double precision ───────────────────────────────────────────────*/
void demo_avx_double(void) {
    printf("\n=== AVX double precision (__m256d: 4 × float64) ===\n");

    __m256d a = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
    __m256d b = _mm256_set1_pd(0.5);

    __m256d result = _mm256_mul_pd(a, b);  /* {0.5, 1.0, 1.5, 2.0} */

    double out[4];
    _mm256_storeu_pd(out, result);
    printf("4 × f64 mul: ");
    for (int i=0;i<4;i++) printf("%.2f ", out[i]);
    printf("\n");

    /* Same API: _mm256_add_pd, _mm256_sqrt_pd, _mm256_fmadd_pd, etc. */
}

/* ── AVX2: 256-bit integer SIMD preview ─────────────────────────────────── */
/* Full coverage in avx2_integer.c — quick preview here */
void demo_avx2_integer_preview(void) {
    printf("\n=== AVX2 integer preview (__m256i: 32 × i8 or 8 × i32) ===\n");

    /* 8 × int32 */
    __m256i a = _mm256_set_epi32(8,7,6,5,4,3,2,1);
    __m256i b = _mm256_set1_epi32(100);
    __m256i c = _mm256_add_epi32(a, b);

    int32_t out[8];
    _mm256_storeu_si256((__m256i*)out, c);
    printf("8 × i32 add: ");
    for (int i=0;i<8;i++) printf("%d ",out[i]);
    printf("\n");
}

/* ── VZEROUPPER / VZEROALL ───────────────────────────────────────────────── */
/*
 * On Intel CPUs before Ice Lake, mixing 128-bit (XMM) and 256-bit (YMM)
 * operations in the same program incurs a penalty because the CPU must
 * save/restore the upper 128 bits.
 *
 * VZEROUPPER zeroes the upper 128 bits of all YMM registers, eliminating
 * the penalty. GCC inserts this automatically at function boundaries when
 * compiling with -mavx.
 *
 * You'll see this in compiler-generated asm. In intrinsics code, do NOT
 * call _mm256_zeroupper() manually — trust the compiler.
 *
 * The penalty doesn't exist on AMD CPUs or Intel Ice Lake+.
 */

int main(void) {
    demo_avx_basics();
    demo_avx_masking();
    demo_fma();
    demo_permute();
    demo_avx_double();
    demo_avx2_integer_preview();

    /* Horizontal sum demo */
    printf("\n=== Horizontal reduction ===\n");
    __m256 v = _mm256_set_ps(8,7,6,5, 4,3,2,1);
    printf("sum of {1..8} = %.1f  (expected 36.0)\n", avx_hsum(v));

    return 0;
}
