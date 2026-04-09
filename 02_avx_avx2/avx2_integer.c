/* avx2_integer.c — AVX2 256-bit integer SIMD
 *
 * Compile: gcc -O2 -mavx2 -o avx2_integer avx2_integer.c
 *
 * AVX2 extends integer SIMD to the full 256-bit width.
 * SSE2 had 128-bit integer ops (_mm_* with __m128i).
 * AVX2 adds 256-bit integer ops (_mm256_* with __m256i).
 *
 * Integer lane widths in __m256i:
 *   32 × int8 (epi8 / epu8)
 *   16 × int16 (epi16 / epu16)
 *    8 × int32 (epi32)
 *    4 × int64 (epi64)
 *
 * Notable limitations vs float:
 *   - No integer division instruction (use multiply-shift trick or scalar)
 *   - No 32×32→32 multiply before AVX2 at 256-bit (SSE4.1 had _mm_mullo_epi32)
 *   - Horizontal ops are limited (vphadd exists but is slow)
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

/* ── Basic integer ops ────────────────────────────────────────────────────── */
void demo_basic_integer(void) {
    printf("=== AVX2 basic integer ops ===\n");

    /* 8 × int32 */
    __m256i a = _mm256_set_epi32(8,7,6,5, 4,3,2,1);
    __m256i b = _mm256_set1_epi32(10);

    __m256i added   = _mm256_add_epi32(a, b);   /* {11,12,13,14,15,16,17,18} */
    __m256i subbed  = _mm256_sub_epi32(a, b);   /* {-9,-8,...,-2} */
    __m256i mulled  = _mm256_mullo_epi32(a, b); /* low 32 bits of 32×32 product */

    /* min/max for signed integers */
    __m256i mn = _mm256_min_epi32(a, b);        /* element-wise min */
    __m256i mx = _mm256_max_epi32(a, b);

    int32_t out[8];
    _mm256_storeu_si256((__m256i*)out, added);
    printf("add  (×32): "); for(int i=0;i<8;i++) printf("%d ",out[i]); printf("\n");
    _mm256_storeu_si256((__m256i*)out, mulled);
    printf("mullo(×32): "); for(int i=0;i<8;i++) printf("%d ",out[i]); printf("\n");
    _mm256_storeu_si256((__m256i*)out, mn);
    printf("min:        "); for(int i=0;i<8;i++) printf("%d ",out[i]); printf("\n");
    (void)subbed; (void)mx;

    /* 32 × int8 */
    printf("\n32 × int8:\n");
    __m256i bytes = _mm256_set1_epi8(127);
    __m256i one8  = _mm256_set1_epi8(1);
    __m256i sat   = _mm256_adds_epi8(bytes, one8); /* saturating: stays at 127 */
    __m256i wrap  = _mm256_add_epi8(bytes, one8);  /* wrapping: 127+1 = -128 */

    int8_t out8[32];
    _mm256_storeu_si256((__m256i*)out8, sat);
    printf("saturating 127+1 = %d  (stays at 127)\n", out8[0]);
    _mm256_storeu_si256((__m256i*)out8, wrap);
    printf("wrapping   127+1 = %d  (wraps to -128)\n", out8[0]);
}

/* ── Shifts ───────────────────────────────────────────────────────────────── */
void demo_shifts(void) {
    printf("\n=== Shifts ===\n");

    __m256i v = _mm256_set_epi32(256,128,64,32, 16,8,4,2);

    /* Logical shift right (fills with zeros) */
    __m256i srl = _mm256_srli_epi32(v, 1);  /* divide by 2 for positive */

    /* Arithmetic shift right (fills with sign bit) */
    __m256i neg = _mm256_set1_epi32(-128);
    __m256i sra = _mm256_srai_epi32(neg, 2); /* -128 >> 2 = -32 */

    /* Variable shift (AVX2): each lane shifted by its own count */
    __m256i counts = _mm256_set_epi32(7,6,5,4, 3,2,1,0);
    __m256i base   = _mm256_set1_epi32(1);
    __m256i vshl   = _mm256_sllv_epi32(base, counts); /* 1<<0, 1<<1, ..., 1<<7 */

    int32_t out[8];
    _mm256_storeu_si256((__m256i*)out, srl);
    printf("srli by 1:    "); for(int i=0;i<8;i++) printf("%d ",out[i]); printf("\n");

    _mm256_storeu_si256((__m256i*)out, vshl);
    printf("sllv (1<<i):  "); for(int i=0;i<8;i++) printf("%d ",out[i]); printf("\n");

    int32_t sra_val;
    _mm_storeu_si32(&sra_val, _mm256_castsi256_si128(sra));
    printf("srai -128>>2: %d  (expected -32)\n", sra_val);
}

/* ── Comparison and masking ───────────────────────────────────────────────── */
void demo_comparison(void) {
    printf("\n=== Integer comparison ===\n");

    __m256i a = _mm256_set_epi32(10, 5, 3, 8, 1, 9, 2, 7);
    __m256i b = _mm256_set1_epi32(5);

    /* Returns all-ones or all-zeros per lane */
    __m256i eq = _mm256_cmpeq_epi32(a, b);   /* where a == 5 */
    __m256i gt = _mm256_cmpgt_epi32(a, b);   /* where a > 5  (signed) */
    /* Note: no cmplt_epi32 — swap operands and use cmpgt */

    /* movemask: 8-bit integer with one bit per 32-bit lane's sign bit */
    int eq_bits = _mm256_movemask_epi8(eq);  /* 4 bits per lane (epi32 = 4 bytes) */
    /* Easier: extract per-lane using conversion to float mask */
    /* Or use vptest */

    int32_t out[8];
    _mm256_storeu_si256((__m256i*)out, gt);
    printf("a > 5: ");
    for(int i=0;i<8;i++) printf("%s ", out[i] ? "T" : "F");
    printf("  (a={7,2,9,1,8,3,5,10})\n");

    /* Count set lanes via popcount on movemask */
    int gt_bits = _mm256_movemask_epi8(gt);
    printf("lanes > 5: %d  (popcount of 32-bit movemask)\n",
           __builtin_popcount(gt_bits) / 4); /* 4 bytes per 32-bit lane */
    (void)eq_bits;
}

/* ── Byte shuffle: vpshufb ───────────────────────────────────────────────── */
/*
 * _mm256_shuffle_epi8 (VPSHUFB): rearrange bytes within each 128-bit lane.
 * The index register selects which source byte goes to each destination byte.
 * If index[i] has bit 7 set, output[i] = 0.
 *
 * This is extremely powerful for:
 *   - Byte-level data transformation (RGBA → BGRA)
 *   - Table lookup with 16-entry tables (one per lane)
 *   - Reversing byte order
 *   - Extract specific fields from packed structs
 */
void demo_pshufb(void) {
    printf("\n=== VPSHUFB: byte shuffle ===\n");

    /* Byte-reverse a sequence (big-endian ↔ little-endian) */
    uint8_t data_arr[32];
    for (int i = 0; i < 32; i++) data_arr[i] = (uint8_t)i;

    __m256i data = _mm256_loadu_si256((const __m256i*)data_arr);

    /* Reverse bytes within each 128-bit lane */
    /* index 15,14,...,0 in each lane */
    __m256i rev_idx = _mm256_set_epi8(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,   /* high lane reversed */
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15    /* low lane reversed */
    );
    __m256i reversed = _mm256_shuffle_epi8(data, rev_idx);

    uint8_t out[32];
    _mm256_storeu_si256((__m256i*)out, reversed);
    printf("byte-reversed (per lane): ");
    for(int i=0;i<32;i++) printf("%2d ",out[i]); printf("\n");
    /* Expected: 15,14,...,0, 31,30,...,16 */

    /* RGB → BGR swap using pshufb (common image processing pattern) */
    /* Assume 4 pixels of RGB (3 bytes each) packed = 12 bytes, padded to 16 */
    uint8_t rgb[16] = {255,0,0, 0,255,0, 0,0,255, 128,64,32, 0,0,0,0};
    __m128i rgb_v = _mm_loadu_si128((const __m128i*)rgb);

    /* Swap R and B: for each RGB triple at offset i: {i+2, i+1, i+0} */
    __m128i bgr_idx = _mm_set_epi8(
        -1,-1,-1,-1,            /* padding bytes (bit 7 set → zero output) */
        11,10, 9,               /* pixel 3: B,G,R */
         8, 7, 6,               /* pixel 2: B,G,R */
         5, 4, 3,               /* pixel 1: B,G,R */
         2, 1, 0                /* pixel 0: B,G,R */
    );
    __m128i bgr_v = _mm_shuffle_epi8(rgb_v, bgr_idx);

    uint8_t bgr[16];
    _mm_storeu_si128((__m128i*)bgr, bgr_v);
    printf("RGB→BGR pixel 0: R=%d G=%d B=%d → B=%d G=%d R=%d\n",
           rgb[0], rgb[1], rgb[2], bgr[0], bgr[1], bgr[2]);
}

/* ── Pack and unpack ─────────────────────────────────────────────────────── */
/*
 * Pack: convert wider integers to narrower (with saturation or truncation)
 * Unpack: interleave and widen integers
 *
 * Common use: converting int32 result back to int8 after computation
 * (e.g., after multiplying int8 values promoted to int16/int32)
 */
void demo_pack_unpack(void) {
    printf("\n=== Pack and unpack ===\n");

    /* Pack int32 → int16 with signed saturation */
    __m256i big = _mm256_set_epi32(200, 100, 50, 1000, -500, -100, 32767, -32768);
    /* Need two int32 vectors to pack into one int16 vector */
    __m256i zeros = _mm256_setzero_si256();
    __m256i packed16 = _mm256_packs_epi32(big, zeros); /* saturate to [-32768, 32767] */

    int16_t out16[16];
    _mm256_storeu_si256((__m256i*)out16, packed16);
    printf("pack i32→i16 (saturated): ");
    /* AVX2 pack interleaves within 128-bit lanes: big[0..3], zeros[0..3], big[4..7], zeros[4..7] */
    for(int i=0;i<8;i++) printf("%d ", out16[i]); printf("\n");

    /* Unpack (interleave) for widening */
    __m256i bytes = _mm256_set1_epi8(100);
    __m256i zero8 = _mm256_setzero_si256();
    /* unpacklo: interleave low 8 bytes of each 128-bit lane with zeros → zero-extended to int16 */
    __m256i widen = _mm256_unpacklo_epi8(bytes, zero8);

    int16_t out_widen[16];
    _mm256_storeu_si256((__m256i*)out_widen, widen);
    printf("unpack i8→i16 (zero extend): %d %d %d %d ...\n",
           out_widen[0], out_widen[1], out_widen[2], out_widen[3]);
}

/* ── Pattern: fast byte search using _mm256_cmpeq_epi8 ──────────────────── */
/*
 * Search for a byte value in a 32-byte chunk. Returns the index of the first
 * occurrence, or -1. Used by libc memchr and strlen implementations.
 */
int find_byte_avx2(const uint8_t *buf, uint8_t needle, int len) {
    __m256i target = _mm256_set1_epi8((int8_t)needle);

    int i = 0;
    for (; i <= len - 32; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i*)(buf + i));
        __m256i eq    = _mm256_cmpeq_epi8(chunk, target);
        int mask      = _mm256_movemask_epi8(eq);
        if (mask != 0)
            return i + __builtin_ctz(mask); /* ctz = count trailing zeros = index of first set bit */
    }
    /* scalar tail */
    for (; i < len; i++)
        if (buf[i] == needle) return i;
    return -1;
}

void demo_byte_search(void) {
    printf("\n=== Fast byte search ===\n");

    uint8_t buf[128];
    memset(buf, 'A', sizeof(buf));
    buf[73] = 'X';
    buf[100] = 'X';

    int pos = find_byte_avx2(buf, 'X', 128);
    printf("First 'X' at index %d  (expected 73)\n", pos);
    assert(pos == 73);

    pos = find_byte_avx2(buf, 'Z', 128);
    printf("'Z' not found: %d  (expected -1)\n", pos);
}

/* ── Integer → float conversion ──────────────────────────────────────────── */
void demo_conversion(void) {
    printf("\n=== Integer ↔ float conversion ===\n");

    __m256i ints = _mm256_set_epi32(8,7,6,5, 4,3,2,1);

    /* int32 → float32 */
    __m256 floats = _mm256_cvtepi32_ps(ints);
    float fout[8];
    _mm256_storeu_ps(fout, floats);
    printf("i32→f32: "); for(int i=0;i<8;i++) printf("%.1f ",fout[i]); printf("\n");

    /* float32 → int32 (truncate toward zero) */
    __m256 fvals = _mm256_set_ps(7.9f, -2.3f, 3.1f, 4.8f, 0.5f, -1.0f, 2.7f, -0.1f);
    __m256i truncated = _mm256_cvttps_epi32(fvals);  /* truncate */
    __m256i rounded   = _mm256_cvtps_epi32(fvals);   /* round to nearest-even */

    int32_t iout[8];
    _mm256_storeu_si256((__m256i*)iout, truncated);
    printf("f32→i32 (trunc): "); for(int i=0;i<8;i++) printf("%d ",iout[i]); printf("\n");
    _mm256_storeu_si256((__m256i*)iout, rounded);
    printf("f32→i32 (round): "); for(int i=0;i<8;i++) printf("%d ",iout[i]); printf("\n");

    /* int8 sign-extend to int32 (useful before multiply) */
    int8_t small[32];
    for (int i = 0; i < 32; i++) small[i] = (int8_t)(i - 16);
    __m128i s8 = _mm_loadu_si128((const __m128i*)small); /* load first 16 bytes */
    __m256i s32 = _mm256_cvtepi8_epi32(s8); /* sign-extend 8 × int8 to 8 × int32 */
    _mm256_storeu_si256((__m256i*)iout, s32);
    printf("i8→i32 (sign ext): "); for(int i=0;i<8;i++) printf("%d ",iout[i]); printf("\n");
}

int main(void) {
    demo_basic_integer();
    demo_shifts();
    demo_comparison();
    demo_pshufb();
    demo_pack_unpack();
    demo_byte_search();
    demo_conversion();
    return 0;
}
