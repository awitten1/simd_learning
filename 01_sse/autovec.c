/* autovec.c — Teaching the compiler to auto-vectorize
 *
 * Compile: gcc -O2 -mavx2 -mfma -o autovec autovec.c
 *
 * Key commands:
 *   make autovec_info   — see which loops GCC vectorized (and which it missed)
 *
 *   Manually:
 *   gcc -O2 -mavx2 -fopt-info-vec-optimized -c autovec.c -o /dev/null
 *   gcc -O2 -mavx2 -fopt-info-vec-missed    -c autovec.c -o /dev/null
 *
 *   See the assembly:
 *   gcc -O2 -mavx2 -S -masm=intel autovec.c -o autovec.s
 *   (look for vmovups, vaddps, vfmadd231ps, etc.)
 *
 * Autovectorization rules:
 *   DOES vectorize:
 *     - Simple loops with no iteration-to-iteration dependencies
 *     - __restrict__ pointers (no aliasing)
 *     - Countable loops (compiler knows trip count, or can check)
 *     - Calls to vectorizable math functions (with -ffast-math sometimes)
 *
 *   DOES NOT vectorize:
 *     - Loop-carried dependencies: result[i] depends on result[i-1]
 *     - Aliased pointers (output overlaps input — compiler must be conservative)
 *     - Unpredictable branches per element
 *     - Function calls the compiler can't inline or vectorize
 */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N (1 << 22)   /* 4M floats */

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── 1. This loop vectorizes ─────────────────────────────────────────────── */
/*
 * Three conditions met:
 *   a) No aliasing: __restrict__ tells the compiler a, b, c don't overlap
 *   b) No dependency between iterations
 *   c) Compiler can determine trip count from 'n'
 */
void vec_add(const float * __restrict__ a,
             const float * __restrict__ b,
             float * __restrict__ c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* Without __restrict__: the compiler must assume a/b/c could overlap and
 * often falls back to scalar or adds a runtime alias check. */
void vec_add_aliased(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ── 2. FMA loop vectorizes ──────────────────────────────────────────────── */
/* SAXPY: single-precision a*x + y  (classic BLAS level-1 operation) */
void saxpy(float alpha, const float * __restrict__ x,
           float * __restrict__ y, int n) {
    for (int i = 0; i < n; i++)
        y[i] = alpha * x[i] + y[i];
    /* GCC will emit vfmadd231ps with -mavx2 -mfma */
}

/* ── 3. Reduction vectorizes (with careful codegen) ─────────────────────── */
float sum_array(const float * __restrict__ a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += a[i];
    /* GCC vectorizes this to 8 partial sums in a YMM register, then
     * reduces horizontally. The result can differ from strict scalar due
     * to floating-point associativity. Use -ffast-math to allow this
     * reordering, or -fassociative-math specifically. */
    return s;
}

/* ── 4. This loop does NOT vectorize: loop-carried dependency ────────────── */
/*
 * result[i] = result[i-1] + a[i]
 * Each iteration depends on the result of the previous — serial chain.
 * This is a prefix sum / scan. SIMD implementations exist but require
 * algorithmic changes, not just compiler hints.
 */
void prefix_sum(const float * __restrict__ a, float * __restrict__ out, int n) {
    out[0] = a[0];
    for (int i = 1; i < n; i++)
        out[i] = out[i - 1] + a[i];
    /* GCC will NOT vectorize this. Run `make autovec_info` to confirm. */
}

/* ── 5. This loop does NOT vectorize: function call ──────────────────────── */
/* logf() is not inlined by default without -ffast-math. With -ffast-math
 * GCC may use its vectorized libm (libmvec). Without it: scalar. */
void log_array(const float * __restrict__ a, float * __restrict__ out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = logf(a[i]);
    /* Try recompiling with -ffast-math and check the asm for vlog_avx */
}

/* ── 6. Alignment hint: can help the compiler generate cleaner code ─────── */
/*
 * __builtin_assume_aligned tells the compiler the pointer is aligned.
 * With 32-byte alignment + AVX, the compiler can emit aligned vmovaps
 * instead of unaligned vmovups. On Skylake+ the difference is small,
 * but it can avoid a runtime alignment check in the loop prologue.
 */
void vec_add_aligned_hint(const float * __restrict__ a_,
                          const float * __restrict__ b_,
                          float * __restrict__ c_, int n) {
    const float *a = __builtin_assume_aligned(a_, 32);
    const float *b = __builtin_assume_aligned(b_, 32);
    float       *c = __builtin_assume_aligned(c_, 32);
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* Equivalent: use the __attribute__((aligned(32))) on the declaration
 * or C11 alignas(32) on local arrays. */

/* ── 7. Unrolling helps: multiple accumulators for reductions ────────────── */
/*
 * A single accumulator in a reduction creates a dependency chain.
 * FMA has ~4 cycle latency. One accumulator = one FMA per 4 cycles.
 * Four accumulators = four independent chains = four FMAs per 4 cycles.
 */
float sum_unrolled(const float * __restrict__ a, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        s0 += a[i + 0];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    for (; i < n; i++) s0 += a[i];
    return s0 + s1 + s2 + s3;
    /* GCC vectorizes this. With 8 accumulators you fully utilize AVX throughput. */
}

/* ── 8. Type width matters: int8 multiplies differently than float ────────── */
/*
 * For int16 multiplication, SSE2 gives 8-wide.
 * For int8, there's no direct 16-wide int8 multiply in SSE/AVX2
 * (AVX-512 VNNI adds that). The common trick: unpack int8 to int16,
 * multiply int16, pack back. Compilers know this idiom.
 */
void int16_scale(const int16_t * __restrict__ a,
                 int16_t * __restrict__ out, int16_t scalar, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)(a[i] * scalar);
    /* GCC vectorizes this with pmullw (packed multiply low word) */
}

/* ── Benchmark: vectorized vs scalar ─────────────────────────────────────── */
/*
 * To compare, compile two versions:
 *   gcc -O2 -mavx2 -mfma          autovec.c -o autovec_vec
 *   gcc -O2 -fno-tree-vectorize    autovec.c -o autovec_scalar
 */
int main(void) {
    float *a, *b, *c;
    posix_memalign((void**)&a, 32, N * sizeof(float));
    posix_memalign((void**)&b, 32, N * sizeof(float));
    posix_memalign((void**)&c, 32, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = (float)i * 0.001f + 0.1f;
        b[i] = (float)(N - i) * 0.001f;
    }

    printf("=== Autovectorization demo ===\n");
    printf("N = %d floats = %.1f MB each array\n\n",
           N, N * sizeof(float) / 1e6);

    /* vec_add */
    {
        double t0 = now();
        for (int r = 0; r < 20; r++) vec_add(a, b, c, N);
        double elapsed = (now() - t0) / 20;
        double gb = 3.0 * N * sizeof(float) / 1e9;
        printf("vec_add (__restrict__):      %.2f ms, %.1f GB/s\n",
               elapsed * 1000, gb / elapsed);
    }

    /* vec_add_aliased */
    {
        double t0 = now();
        for (int r = 0; r < 20; r++) vec_add_aliased(a, b, c, N);
        double elapsed = (now() - t0) / 20;
        double gb = 3.0 * N * sizeof(float) / 1e9;
        printf("vec_add (no restrict):       %.2f ms, %.1f GB/s\n",
               elapsed * 1000, gb / elapsed);
    }

    /* saxpy */
    {
        double t0 = now();
        for (int r = 0; r < 20; r++) saxpy(2.5f, a, b, N);
        double elapsed = (now() - t0) / 20;
        double gb = 3.0 * N * sizeof(float) / 1e9;
        printf("saxpy (FMA expected):        %.2f ms, %.1f GB/s\n",
               elapsed * 1000, gb / elapsed);
    }

    /* sum */
    {
        double t0 = now();
        volatile float sink = 0;
        for (int r = 0; r < 20; r++) sink += sum_array(a, N);
        double elapsed = (now() - t0) / 20;
        double gb = N * sizeof(float) / 1e9;
        printf("sum_array (reduction):       %.2f ms, %.1f GB/s\n",
               elapsed * 1000, gb / elapsed);
    }

    /* sum unrolled */
    {
        double t0 = now();
        volatile float sink = 0;
        for (int r = 0; r < 20; r++) sink += sum_unrolled(a, N);
        double elapsed = (now() - t0) / 20;
        double gb = N * sizeof(float) / 1e9;
        printf("sum_unrolled (4 accum):      %.2f ms, %.1f GB/s\n",
               elapsed * 1000, gb / elapsed);
    }

    /* prefix_sum — NOT vectorized */
    {
        float *out;
        posix_memalign((void**)&out, 32, N * sizeof(float));
        double t0 = now();
        for (int r = 0; r < 5; r++) prefix_sum(a, out, N);
        double elapsed = (now() - t0) / 5;
        double gb = 2.0 * N * sizeof(float) / 1e9;
        printf("prefix_sum (serial dep):     %.2f ms, %.1f GB/s  <-- not vectorized\n",
               elapsed * 1000, gb / elapsed);
        free(out);
    }

    printf("\nRun: make autovec_info  to see the compiler's vectorization report.\n");
    printf("Run: make asm           to inspect the generated assembly.\n");
    printf("Tip: grep for 'ymm' or 'zmm' in the .s file to confirm vectorization.\n");

    free(a); free(b); free(c);
    return 0;
}
