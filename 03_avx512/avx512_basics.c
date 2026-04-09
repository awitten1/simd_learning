/* avx512_basics.c — AVX-512 fundamentals and masking
 *
 * Compile: gcc -O2 -mavx512f -mavx512bw -mavx512dq -mavx512vl \
 *              -mavx512vnni -o avx512_basics avx512_basics.c
 *
 * AVX-512 introduced in Intel Skylake-X (2017), AMD Zen4 (2022).
 * Your CPU (Ryzen 9 9900X) supports the full AVX-512 suite.
 *
 * Key additions over AVX2:
 *   1. ZMM registers: 512-bit, holds 16×float32 or 8×float64
 *   2. Extended register set: 32 ZMM registers (vs 16 YMM)
 *   3. Opmask registers (k0–k7): 1 bit per SIMD lane
 *      - Merge masking:  _mm512_mask_*   — unmasked lanes copy from src
 *      - Zero masking:   _mm512_maskz_*  — unmasked lanes become 0
 *   4. New instructions: compress/expand, scatter, ternlog, VNNI, BF16, ...
 *   5. Embedded rounding control per-instruction
 *
 * The opmask registers are the MOST important new feature.
 * They enable conditional SIMD without computing both branches.
 *
 * Historical note on AVX-512 extensions:
 *   AVX512F   = Foundation (required; ZMM, k-registers, basic ops)
 *   AVX512BW  = Byte and Word (i8/i16 in 512-bit)
 *   AVX512DQ  = Doubleword and Quadword (i32/i64 extras, float<->int)
 *   AVX512VL  = Vector Length (use k-masks with 128/256-bit too)
 *   AVX512VNNI = Vector Neural Network Instructions (int8 dot product)
 *   AVX512BF16 = BFloat16 support
 *   AVX512VBMI = More byte manipulation
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>

/* ── Basic 512-bit float ops ─────────────────────────────────────────────── */
void demo_basics(void) {
    printf("=== AVX-512 basics: __m512 (16 × float32) ===\n");

    /* _mm512_set_ps: 16 arguments, reversed order as always */
    __m512 v = _mm512_set_ps(
        16,15,14,13, 12,11,10,9,
         8, 7, 6, 5,  4, 3, 2,1
    );
    __m512 ones = _mm512_set1_ps(1.0f);

    __m512 added  = _mm512_add_ps(v, ones);
    __m512 mulled = _mm512_mul_ps(v, v);
    __m512 sqrts  = _mm512_sqrt_ps(v);

    float out[16];
    _mm512_storeu_ps(out, added);
    printf("v+1:  "); for(int i=0;i<16;i++) printf("%.0f ",out[i]); printf("\n");
    _mm512_storeu_ps(out, sqrts);
    printf("sqrt: "); for(int i=0;i<8;i++) printf("%.2f ",out[i]);
    printf("...\n");

    /* Reduction: sum all 16 lanes to a scalar */
    float total = _mm512_reduce_add_ps(v);  /* AVX-512DQ */
    printf("sum {1..16} = %.0f  (expected 136)\n", total);
}

/* ── Opmask registers: the key AVX-512 feature ───────────────────────────── */
/*
 * k0–k7 are 64-bit registers (but used as 16-bit, 32-bit, or 64-bit masks
 * depending on operation width).
 *
 * For 16 × float32 (ZMM): 16-bit mask in k-register (__mmask16)
 * For 32 × int16  (ZMM): 32-bit mask in k-register (__mmask32)
 * For 64 × int8   (ZMM): 64-bit mask in k-register (__mmask64)
 *
 * Two masking modes:
 *   Merge masking: _mm512_mask_op_ps(src, k, a, b)
 *     lane i = (k[i] == 1) ? (a op b)[i] : src[i]
 *
 *   Zero masking: _mm512_maskz_op_ps(k, a, b)
 *     lane i = (k[i] == 1) ? (a op b)[i] : 0.0f
 */
void demo_masking(void) {
    printf("\n=== AVX-512 opmask registers ===\n");

    __m512 a = _mm512_set_ps(
        16,15,14,13, 12,11,10,9,
         8, 7, 6, 5,  4, 3, 2,1
    );
    __m512 b = _mm512_set1_ps(100.0f);

    /* Compare: returns a k-register (not a vector!) */
    __mmask16 k = _mm512_cmp_ps_mask(a, _mm512_set1_ps(8.0f), _CMP_GT_OS);
    /* k = bitmask where a[i] > 8.0: bits 8-15 set = 0xFF00 */
    printf("mask (a > 8):  0x%04x  (bits 8-15 set)\n", (unsigned)k);

    /* Zero masking: add b where k is set, else 0 */
    __m512 zero_masked = _mm512_maskz_add_ps(k, a, b);
    float out[16];
    _mm512_storeu_ps(out, zero_masked);
    printf("maskz_add:     ");
    for (int i=0;i<16;i++) printf("%.0f ", out[i]);
    printf("\n");  /* {0,0,0,0,0,0,0,0, 109,110,...,116} */

    /* Merge masking: add b where k is set, else keep a */
    __m512 merge_masked = _mm512_mask_add_ps(a, k, a, b);
    _mm512_storeu_ps(out, merge_masked);
    printf("mask_add:      ");
    for (int i=0;i<16;i++) printf("%.0f ", out[i]);
    printf("\n");  /* {1,2,...,8, 109,110,...,116} */

    /* Masked load: load from memory only where k is set */
    float src_mem[16];
    for (int i=0;i<16;i++) src_mem[i] = (float)(i * 10);
    __m512 passthrough = _mm512_set1_ps(-99.0f);
    __m512 masked_load = _mm512_mask_loadu_ps(passthrough, k, src_mem);
    _mm512_storeu_ps(out, masked_load);
    printf("mask_loadu:    ");
    for (int i=0;i<16;i++) printf("%.0f ", out[i]);
    printf("\n");  /* {-99,...,-99, 80,90,...,150} */

    /* Masked store: write only selected lanes */
    float dst[16];
    memset(dst, 0, sizeof(dst));
    _mm512_mask_storeu_ps(dst, k, a);
    printf("mask_storeu:   ");
    for (int i=0;i<16;i++) printf("%.0f ", dst[i]);
    printf("\n");  /* {0,0,...,0, 9,10,...,16} */
}

/* ── Mask arithmetic ─────────────────────────────────────────────────────── */
void demo_mask_ops(void) {
    printf("\n=== Mask register operations ===\n");

    __m512 v = _mm512_set_ps(16,15,14,13,12,11,10,9, 8,7,6,5,4,3,2,1);

    __mmask16 gt8  = _mm512_cmp_ps_mask(v, _mm512_set1_ps(8.0f),  _CMP_GT_OS);
    __mmask16 lt12 = _mm512_cmp_ps_mask(v, _mm512_set1_ps(12.0f), _CMP_LT_OS);

    /* Boolean ops on masks — single instructions */
    __mmask16 between = _kand_mask16(gt8, lt12);  /* 8 < v < 12 */
    __mmask16 outside = _kor_mask16(_knot_mask16(gt8), _knot_mask16(lt12));
    __mmask16 either  = _kor_mask16(gt8, lt12);
    __mmask16 xored   = _kxor_mask16(gt8, lt12);

    printf("v > 8:       0x%04x  (%d lanes)\n", (unsigned)gt8,
           __builtin_popcount(gt8));
    printf("v < 12:      0x%04x  (%d lanes)\n", (unsigned)lt12,
           __builtin_popcount(lt12));
    printf("8 < v < 12:  0x%04x  (%d lanes, values 9,10,11)\n",
           (unsigned)between, __builtin_popcount(between));
    (void)outside; (void)either; (void)xored;

    /* Convert mask to integer (useful for branching) */
    if (_ktestz_mask16_u8(between, between)) {
        printf("No elements in range.\n");
    } else {
        printf("Found elements in (8,12).\n");
    }
}

/* ── Compress and Expand: filter/scatter with masks ─────────────────────── */
/*
 * vpcompressd/vpcompressps: pack selected lanes to the low positions.
 * vpexpandd/vpexpandps: scatter low positions to selected lanes.
 *
 * These are AVX-512F additions — no equivalent in SSE/AVX.
 *
 * Use case: partition or filter without a scalar fallback.
 */
void demo_compress_expand(void) {
    printf("\n=== Compress and Expand ===\n");

    float data[16];
    for (int i=0;i<16;i++) data[i] = (float)(i+1);
    __m512 v = _mm512_loadu_ps(data);

    /* Keep only elements > 8 */
    __mmask16 k = _mm512_cmp_ps_mask(v, _mm512_set1_ps(8.0f), _CMP_GT_OS);

    /* Compress: pack selected lanes to low positions */
    __m512 compressed = _mm512_maskz_compress_ps(k, v);
    float out[16];
    _mm512_storeu_ps(out, compressed);
    int count = __builtin_popcount(k);
    printf("compress (>8): ");
    for(int i=0;i<count;i++) printf("%.0f ", out[i]);
    printf("  (rest unspecified)\n");  /* 9 10 11 12 13 14 15 16 */

    /* Expand: spread low-position elements to selected lanes */
    float fill[16] = {100,200,300,400,500,600,700,800, 0,0,0,0,0,0,0,0};
    __m512 src_expand = _mm512_loadu_ps(fill);  /* only first 8 used */
    __m512 base = _mm512_setzero_ps();
    __m512 expanded = _mm512_mask_expand_ps(base, k, src_expand);
    _mm512_storeu_ps(out, expanded);
    printf("expand to (>8 lanes): ");
    for(int i=0;i<16;i++) printf("%.0f ", out[i]);
    printf("\n");  /* {0,0,...,0,100,200,...,800} for the upper 8 lanes */
}

/* ── AVX-512 VNNI: integer dot products for ML ───────────────────────────── */
/*
 * VNNI = Vector Neural Network Instructions (AVX512VNNI / AVX_VNNI)
 *
 * VPDPBUSD: for each int32 accumulator lane i:
 *   acc[i] += (uint8)a[4i+0] * (int8)b[4i+0]
 *           + (uint8)a[4i+1] * (int8)b[4i+1]
 *           + (uint8)a[4i+2] * (int8)b[4i+2]
 *           + (uint8)a[4i+3] * (int8)b[4i+3]
 *
 * In one 512-bit operation: 16 int32 accumulators, each accumulating
 * a dot product of 4 int8 pairs = 64 multiply-accumulates per instruction!
 *
 * This is how quantized ML models (int8 weights) run efficiently on x86.
 * A single ZMM VNNI op does what 64 scalar MACs do.
 *
 * Note: operand 'a' is uint8 (unsigned activations after ReLU),
 *       operand 'b' is int8 (signed weights).
 */
void demo_vnni(void) {
    printf("\n=== AVX-512 VNNI: int8 dot product ===\n");

    /* 64 uint8 activations (a) and 64 int8 weights (b) */
    uint8_t a_data[64];
    int8_t  b_data[64];

    for (int i = 0; i < 64; i++) {
        a_data[i] = 1;       /* all ones for easy verification */
        b_data[i] = (int8_t)(i % 4 == 0 ? 1 : 0);  /* 1 at positions 0,4,8,...,60 */
    }

    /* Load as raw bytes */
    __m512i a = _mm512_loadu_si512(a_data);
    __m512i b = _mm512_loadu_si512(b_data);

    /* Accumulate into zeroed int32 vector */
    __m512i acc = _mm512_setzero_si512();
    acc = _mm512_dpbusd_epi32(acc, a, b);
    /* Each int32 lane accumulates 4 consecutive uint8*int8 products.
     * a_data all 1s, b_data has pattern {1,0,0,0} repeated.
     * So each group of 4: 1*1 + 1*0 + 1*0 + 1*0 = 1.
     * All 16 int32 accumulators should be 1. */

    int32_t out[16];
    _mm512_storeu_si512(out, acc);
    printf("VNNI acc (each should be 1): ");
    for (int i=0;i<16;i++) printf("%d ", out[i]);
    printf("\n");

    /* Real use: int8 matrix multiply kernel
     * Inner loop over K: acc += A_row * B_col (both int8, acc int32)
     * Then convert acc to float and apply scale factor */
    printf("\nVNNI computes 64 int8 MACs per instruction.\n");
    printf("Used in quantized (int8) neural network inference.\n");
    printf("Throughput: 0.5 cycles per instruction on Zen4 → ~128 MACs/cycle.\n");
}

/* ── Ternary logic: VPTERNLOGD ───────────────────────────────────────────── */
/*
 * _mm512_ternarylogic_epi32(a, b, c, imm8):
 * Computes any 3-input boolean function in a single instruction.
 * imm8 is an 8-bit truth table: for each bit pattern (c[i],b[i],a[i]):
 *   bit 0: f(0,0,0), bit 1: f(0,0,1), ..., bit 7: f(1,1,1)
 *
 * Examples:
 *   imm8 = 0x96: XOR of all three  (a XOR b XOR c)
 *   imm8 = 0xE8: majority function (at least 2 of 3 bits set)
 *   imm8 = 0xFE: OR of all three
 *   imm8 = 0x80: AND of all three
 *
 * Why useful: bitfield manipulation, bit-twiddling hacks, popcount tricks.
 */
void demo_ternlog(void) {
    printf("\n=== VPTERNLOGD: 3-input logic ===\n");

    __m512i a = _mm512_set1_epi32(0x0F0F0F0F);
    __m512i b = _mm512_set1_epi32(0x00FF00FF);
    __m512i c = _mm512_set1_epi32(0x0000FFFF);

    /* XOR of all three: a^b^c */
    __m512i xor3 = _mm512_ternarylogic_epi32(a, b, c, 0x96);
    int32_t v;
    _mm_storeu_si32(&v, _mm512_castsi512_si128(xor3));
    printf("a XOR b XOR c = 0x%08x  (expected 0x0FF00FF0)\n", (unsigned)v);

    /* Majority: true if at least 2 of 3 bits set */
    __m512i maj = _mm512_ternarylogic_epi32(a, b, c, 0xE8);
    _mm_storeu_si32(&v, _mm512_castsi512_si128(maj));
    printf("majority(a,b,c) = 0x%08x\n", (unsigned)v);

    /* Combined AND-NOT-OR: (a AND b) OR (NOT c) — replaces two instructions */
    __m512i combo = _mm512_ternarylogic_epi32(a, b, c, 0x3F);
    /* Without ternlog: _mm512_or_si512(_mm512_and_si512(a,b), _mm512_andnot_si512(c, ones)) */
    _mm_storeu_si32(&v, _mm512_castsi512_si128(combo));
    printf("(a AND b) OR (NOT c) = 0x%08x\n", (unsigned)v);
}

/* ── Loop tail handling: the key masking use case ────────────────────────── */
/*
 * Processing N elements where N is not a multiple of 16.
 * Without masking: you need a separate scalar loop for the tail.
 * With masking: handle the tail in the same vectorized path.
 */
void process_with_tail(const float *a, const float *b, float *c, int n) {
    int i = 0;

    /* Full 16-wide iterations */
    for (; i <= n - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }

    /* Tail: remaining elements using a mask — no separate scalar loop needed */
    int tail = n - i;
    if (tail > 0) {
        /* Mask: bits 0..(tail-1) set */
        __mmask16 k = (__mmask16)((1u << tail) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(k, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(k, b + i);
        _mm512_mask_storeu_ps(c + i, k, _mm512_add_ps(va, vb));
    }
}

void demo_tail_handling(void) {
    printf("\n=== Tail handling with masks ===\n");

    /* N = 19: 16 full + 3 tail */
    int n = 19;
    float a[32], b[32], c[32];
    memset(c, 0, sizeof(c));
    for (int i=0;i<n;i++) { a[i]=(float)i; b[i]=100.0f; }

    process_with_tail(a, b, c, n);

    printf("c[0..18] = ");
    for (int i=0;i<n;i++) printf("%.0f ", c[i]);
    printf("\n");
    printf("c[19] (untouched) = %.0f\n", c[19]);

    /* Verify */
    for (int i=0;i<n;i++) assert(c[i] == a[i] + 100.0f);
    assert(c[19] == 0.0f);
    printf("Verified: tail correctly processed, no out-of-bounds write.\n");
}

int main(void) {
    demo_basics();
    demo_masking();
    demo_mask_ops();
    demo_compress_expand();
    demo_vnni();
    demo_ternlog();
    demo_tail_handling();
    return 0;
}
