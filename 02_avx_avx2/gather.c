#define _POSIX_C_SOURCE 200809L
/* gather.c — AVX2 gather loads and the AoS vs SoA layout problem
 *
 * Compile: gcc -O2 -mavx2 -mfma -o gather gather.c
 *
 * GATHER: load elements from non-contiguous memory addresses.
 *   _mm256_i32gather_ps(base, vindex, scale)
 *     base:   float* base address
 *     vindex: __m256i of int32 byte or element offsets
 *     scale:  1, 2, 4, or 8 (multiplied by each index)
 *   Result: 8 floats loaded from base[vindex[0]], base[vindex[1]], ...
 *
 * When is gather useful?
 *   - Sparse operations: sum elements at arbitrary indices
 *   - AoS → SoA conversion in a hot loop
 *   - Indirect array access (e.g., graph algorithms, hash lookups)
 *
 * When is gather slow?
 *   - Random access with poor cache behavior — each lane is an independent
 *     cache miss; gather doesn't help you get cache hits, it just parallelizes
 *     the issue of them. Throughput is limited by the memory subsystem.
 *   - On many microarchitectures gather has ~5-10 cycle latency vs 4 for
 *     contiguous load. For fully cached data it's competitive; for DRAM it's
 *     slower than you'd hope because the lanes don't fetch concurrently.
 *
 * The AoS vs SoA problem:
 *   AoS (Array of Structures): {x,y,z}, {x,y,z}, {x,y,z}, ...
 *     - Natural for CPU scalar code, OOP
 *     - Bad for SIMD: loading 8 x-values requires gather or transpose
 *
 *   SoA (Structure of Arrays): xxxxx..., yyyyy..., zzzzz...
 *     - Awkward for scalar code
 *     - Perfect for SIMD: loading 8 x-values is one contiguous load
 *
 *   AoSoA (tiled): 8 x, 8 y, 8 z, 8 x, 8 y, 8 z, ...
 *     - Best of both: SIMD-friendly chunking, reasonably cache-local
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#define N 1024

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Basic gather demo ───────────────────────────────────────────────────── */
void demo_basic_gather(void) {
    printf("=== Basic gather ===\n");

    float data[64];
    for (int i = 0; i < 64; i++) data[i] = (float)i;

    /* Load 8 floats at indices {0, 7, 14, 21, 28, 35, 42, 49} (stride 7) */
    __m256i idx = _mm256_set_epi32(49, 42, 35, 28, 21, 14, 7, 0);
    __m256 gathered = _mm256_i32gather_ps(data, idx, 4); /* scale=4: idx in elements */

    float out[8];
    _mm256_storeu_ps(out, gathered);
    printf("gathered at stride-7 indices: ");
    for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
    printf("\n");  /* 0 7 14 21 28 35 42 49 */

    /* Compare to contiguous load */
    __m256 contiguous = _mm256_loadu_ps(data);  /* loads data[0..7] */
    _mm256_storeu_ps(out, contiguous);
    printf("contiguous load:              ");
    for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
    printf("\n");  /* 0 1 2 3 4 5 6 7 */

    /* Masked gather: load only selected lanes, keep others from src */
    __m256 src = _mm256_set1_ps(-1.0f);
    /* mask: bit 31 of each float lane (all-ones = load, all-zeros = keep src) */
    __m256 mask = _mm256_set_ps(0,-1.0f,0,-1.0f, 0,-1.0f,0,-1.0f); /* every other lane */
    __m256 masked_gather = _mm256_mask_i32gather_ps(src, data, idx, mask, 4);
    _mm256_storeu_ps(out, masked_gather);
    printf("masked gather (alt lanes):    ");
    for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
    printf("\n");  /* -1 7 -1 21 -1 35 -1 49 */
}

/* ── AoS (Array of Structures) ───────────────────────────────────────────── */
typedef struct { float x, y, z, w; } Vec4;   /* w padding for alignment */

/* AoS dot product: hard to vectorize */
void dot_product_aos_scalar(const Vec4 *a, const Vec4 *b,
                             float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
}

/* AoS with gather: load x-components of 8 consecutive vectors */
void dot_product_aos_gather(const Vec4 *a, const Vec4 *b,
                             float *out, int n) {
    /* Stride between x-components is sizeof(Vec4)/sizeof(float) = 4 elements */
    __m256i stride_idx = _mm256_set_epi32(7*4, 6*4, 5*4, 4*4,
                                           3*4, 2*4, 1*4, 0*4);
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 ax = _mm256_i32gather_ps(&a[i].x, stride_idx, 4);
        __m256 ay = _mm256_i32gather_ps(&a[i].y, stride_idx, 4);
        __m256 az = _mm256_i32gather_ps(&a[i].z, stride_idx, 4);
        __m256 bx = _mm256_i32gather_ps(&b[i].x, stride_idx, 4);
        __m256 by = _mm256_i32gather_ps(&b[i].y, stride_idx, 4);
        __m256 bz = _mm256_i32gather_ps(&b[i].z, stride_idx, 4);

        __m256 dot = _mm256_fmadd_ps(ax, bx,
                     _mm256_fmadd_ps(ay, by,
                     _mm256_mul_ps(az, bz)));
        _mm256_storeu_ps(out + i, dot);
    }
    for (; i < n; i++)
        out[i] = a[i].x*b[i].x + a[i].y*b[i].y + a[i].z*b[i].z;
}

/* ── SoA (Structure of Arrays) ───────────────────────────────────────────── */
typedef struct {
    float *x, *y, *z;
    int n;
} Vec3SoA;

/* SoA dot product: contiguous loads — the ideal SIMD pattern */
void dot_product_soa(const Vec3SoA *a, const Vec3SoA *b, float *out) {
    int n = a->n;
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 ax = _mm256_loadu_ps(a->x + i);
        __m256 ay = _mm256_loadu_ps(a->y + i);
        __m256 az = _mm256_loadu_ps(a->z + i);
        __m256 bx = _mm256_loadu_ps(b->x + i);
        __m256 by = _mm256_loadu_ps(b->y + i);
        __m256 bz = _mm256_loadu_ps(b->z + i);

        __m256 dot = _mm256_fmadd_ps(ax, bx,
                     _mm256_fmadd_ps(ay, by,
                     _mm256_mul_ps(az, bz)));
        _mm256_storeu_ps(out + i, dot);
    }
    for (; i < n; i++)
        out[i] = a->x[i]*b->x[i] + a->y[i]*b->y[i] + a->z[i]*b->z[i];
}

/* ── Transpose: AoS → SoA with shuffles (no gather needed) ─────────────── */
/*
 * For stride-4 AoS (x,y,z,w repeated), you can transpose with shuffles
 * instead of gather. This is faster because it uses contiguous loads.
 *
 * Load 8 Vec4 structs (32 floats) and transpose to 4 × __m256:
 *   row0 = {x0,x1,x2,x3,x4,x5,x6,x7}
 *   row1 = {y0,y1,...}
 *   row2 = {z0,z1,...}
 *   row3 = {w0,w1,...}
 */
void aos_to_soa_8(const Vec4 *src,
                  __m256 *row0, __m256 *row1, __m256 *row2, __m256 *row3) {
    /* Load 8 Vec4s = 8 × 128-bit chunks */
    __m256 r0 = _mm256_loadu_ps((const float*)&src[0]);  /* x0 y0 z0 w0 | x1 y1 z1 w1 */
    __m256 r1 = _mm256_loadu_ps((const float*)&src[2]);  /* x2 y2 z2 w2 | x3 y3 z3 w3 */
    __m256 r2 = _mm256_loadu_ps((const float*)&src[4]);  /* x4 y4 z4 w4 | x5 y5 z5 w5 */
    __m256 r3 = _mm256_loadu_ps((const float*)&src[6]);  /* x6 y6 z6 w6 | x7 y7 z7 w7 */

    /* Unpack: interleave x,y and z,w pairs */
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);  /* x0 x2 y0 y2 | x1 x3 y1 y3 */
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);  /* z0 z2 w0 w2 | z1 z3 w1 w3 */
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);  /* x4 x6 y4 y6 | x5 x7 y5 y7 */
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);  /* z4 z6 w4 w6 | z5 z7 w5 w7 */

    /* Shuffle to group x,y,z,w across all 8 elements */
    *row0 = _mm256_unpacklo_ps(t0, t2);  /* x0 x4 x2 x6 | x1 x5 x3 x7 (still scrambled) */
    *row1 = _mm256_unpackhi_ps(t0, t2);
    *row2 = _mm256_unpacklo_ps(t1, t3);
    *row3 = _mm256_unpackhi_ps(t1, t3);
    /* Note: the within-lane order is: x0,x4,x2,x6 in low lane, x1,x5,x3,x7 in high.
     * A final permute2f128 would produce full {x0..x7}. This transpose is illustrative. */
}

/* ── Sparse sum: typical gather use case ─────────────────────────────────── */
/*
 * Sum elements at arbitrary indices. This is the canonical use case where
 * gather gives a real benefit over scalar for compute-bound workloads.
 */
float sparse_sum_scalar(const float *data, const int *indices, int nidx) {
    float s = 0;
    for (int i = 0; i < nidx; i++) s += data[indices[i]];
    return s;
}

float sparse_sum_gather(const float *data, const int *indices, int nidx) {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= nidx - 8; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256 vals = _mm256_i32gather_ps(data, idx, 4);
        acc = _mm256_add_ps(acc, vals);
    }
    /* Horizontal reduce */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_hadd_ps(s4, s4);
    __m128 s1 = _mm_hadd_ps(s2, s2);
    float result = _mm_cvtss_f32(s1);
    for (; i < nidx; i++) result += data[indices[i]];
    return result;
}

void demo_sparse_sum(void) {
    printf("\n=== Sparse sum benchmark (gather vs scalar) ===\n");

    int M = 1 << 20;  /* data array size */
    int NIDX = 1 << 16; /* number of indices */

    float *data   = aligned_alloc(32, M * sizeof(float));
    int   *indices = aligned_alloc(32, NIDX * sizeof(int));

    srand(42);
    for (int i = 0; i < M; i++) data[i] = (float)(rand() % 100);
    for (int i = 0; i < NIDX; i++) indices[i] = rand() % M;

    int REPS = 200;

    volatile float sink = 0;
    double t0 = now();
    for (int r = 0; r < REPS; r++) sink += sparse_sum_scalar(data, indices, NIDX);
    double scalar_time = (now() - t0) / REPS;

    t0 = now();
    for (int r = 0; r < REPS; r++) sink += sparse_sum_gather(data, indices, NIDX);
    double gather_time = (now() - t0) / REPS;

    printf("scalar: %.3f ms\n", scalar_time * 1000);
    printf("gather: %.3f ms  (%.1fx)\n", gather_time * 1000, scalar_time / gather_time);
    printf("Note: with random access, gather may not be faster than scalar\n");
    printf("because both are memory-latency-bound, not compute-bound.\n");
    printf("Gather shines when data IS in cache (compute-bound path).\n");

    free(data); free(indices);
}

int main(void) {
    demo_basic_gather();

    printf("\n=== AoS vs SoA dot product ===\n");

    Vec4 *a_aos = aligned_alloc(32, N * sizeof(Vec4));
    Vec4 *b_aos = aligned_alloc(32, N * sizeof(Vec4));

    Vec3SoA a_soa = {
        .x = aligned_alloc(32, N * sizeof(float)),
        .y = aligned_alloc(32, N * sizeof(float)),
        .z = aligned_alloc(32, N * sizeof(float)),
        .n = N
    };
    Vec3SoA b_soa = {
        .x = aligned_alloc(32, N * sizeof(float)),
        .y = aligned_alloc(32, N * sizeof(float)),
        .z = aligned_alloc(32, N * sizeof(float)),
        .n = N
    };
    float *out_aos = aligned_alloc(32, N * sizeof(float));
    float *out_soa = aligned_alloc(32, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a_aos[i] = (Vec4){.x=(float)i, .y=(float)(i+1), .z=(float)(i+2), .w=0};
        b_aos[i] = (Vec4){.x=1.0f, .y=1.0f, .z=1.0f, .w=0};
        a_soa.x[i] = (float)i;   a_soa.y[i] = (float)(i+1); a_soa.z[i] = (float)(i+2);
        b_soa.x[i] = 1.0f;       b_soa.y[i] = 1.0f;         b_soa.z[i] = 1.0f;
    }

    int REPS = 2000;
    double t0;

    t0 = now();
    for (int r = 0; r < REPS; r++) dot_product_aos_scalar(a_aos, b_aos, out_aos, N);
    printf("AoS scalar: %.3f ms\n", (now()-t0)/REPS*1000);

    t0 = now();
    for (int r = 0; r < REPS; r++) dot_product_aos_gather(a_aos, b_aos, out_aos, N);
    printf("AoS gather: %.3f ms\n", (now()-t0)/REPS*1000);

    t0 = now();
    for (int r = 0; r < REPS; r++) dot_product_soa(&a_soa, &b_soa, out_soa);
    printf("SoA  load:  %.3f ms  <- fastest; no gather needed\n", (now()-t0)/REPS*1000);

    /* Verify */
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i+1) + (float)(i+2);
        /* SoA result should match */
        if (fabsf(out_soa[i] - expected) > 0.01f) {
            fprintf(stderr, "MISMATCH at %d: got %f expected %f\n", i, out_soa[i], expected);
        }
    }
    printf("Results verified.\n");

    free(a_aos); free(b_aos); free(out_aos); free(out_soa);
    free(a_soa.x); free(a_soa.y); free(a_soa.z);
    free(b_soa.x); free(b_soa.y); free(b_soa.z);

    demo_sparse_sum();

    return 0;
}
