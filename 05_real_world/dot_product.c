#define _POSIX_C_SOURCE 200809L
/* dot_product.c — Six implementations of dot product, with explanation
 *
 * Compile: gcc -O2 -mavx512f -mavx512dq -mavx512vl -mfma \
 *              -o dot_product dot_product.c -lm
 *
 * This is the "real-world" version: production-quality implementations
 * with correct tail handling, verified results, and usage guidance.
 *
 * Dot product is the canonical SIMD example because:
 *   1. It's simple to understand (sum of pairwise products)
 *   2. It exercises load, multiply, accumulate, and horizontal reduction
 *   3. The accumulator depth trick has dramatic, measurable effect
 *   4. It's the inner loop of matrix multiply, neural network layers,
 *      convolutions, and nearly every numerical algorithm
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

/* ── Horizontal sum helpers ──────────────────────────────────────────────── */
static inline float hsum128(__m128 v) {
    __m128 h = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(_mm_hadd_ps(h, h));
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum128(_mm_add_ps(lo, hi));
}

static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

/* ── Implementation 1: Scalar ────────────────────────────────────────────── */
float dot_v1_scalar(const float * __restrict__ a,
                    const float * __restrict__ b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ── Implementation 2: SSE, single accumulator ───────────────────────────── */
float dot_v2_sse(const float * __restrict__ a,
                 const float * __restrict__ b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i;
    for (i = 0; i <= n - 4; i += 4)
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i)));
    float s = hsum128(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ── Implementation 3: AVX2, FMA, single accumulator ───────────────────── */
float dot_v3_avx_fma(const float * __restrict__ a,
                     const float * __restrict__ b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float s = hsum256(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ── Implementation 4: AVX2, FMA, 4 accumulators ────────────────────────── */
/*
 * Breaking the FMA dependency chain.
 * The CPU has 2 FMA execution units (Zen4) with 4-cycle latency each.
 * With 1 accumulator: one FMA per 4 cycles.
 * With 4 accumulators: four independent chains → can issue one FMA per cycle.
 * With 8 accumulators: saturates both FMA units → one FMA per 0.5 cycles.
 *
 * Rule of thumb: num_accumulators ≥ fma_latency × fma_units_per_cycle
 *   = 4 × 2 = 8 for Zen4.
 */
float dot_v4_avx_4acc(const float * __restrict__ a,
                      const float * __restrict__ b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 32; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+0),  _mm256_loadu_ps(b+i+0),  acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0,acc1), _mm256_add_ps(acc2,acc3));
    float s = hsum256(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ── Implementation 5: AVX-512, FMA, 4 accumulators ─────────────────────── */
/*
 * 16-wide lanes + 4 accumulators.
 * Each iteration processes 4 × 16 = 64 floats.
 * At 2 FMA/cycle, throughput = 2 × 16 = 32 floats/cycle.
 * More accumulators (8) would saturate fully, but 4 is already close
 * because AVX-512 loads are wider and the throughput bound is memory.
 */
float dot_v5_avx512(const float * __restrict__ a,
                    const float * __restrict__ b, int n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    int i;
    for (i = 0; i <= n - 64; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+0),  _mm512_loadu_ps(b+i+0),  acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0,acc1), _mm512_add_ps(acc2,acc3));
    float s = hsum512(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ── Implementation 6: AVX-512 with masked tail ─────────────────────────── */
/*
 * Eliminates the scalar tail loop entirely using AVX-512 masking.
 * Clean, no branch for the tail, same code path for all cases.
 */
float dot_v6_avx512_masked(const float * __restrict__ a,
                            const float * __restrict__ b, int n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    int i;
    for (i = 0; i <= n - 64; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+0),  _mm512_loadu_ps(b+i+0),  acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }

    /* Handle remaining elements 16 at a time with masking */
    while (i < n) {
        int tail = n - i < 16 ? n - i : 16;
        __mmask16 k = (__mmask16)((1u << tail) - 1u);
        __m512 va = _mm512_maskz_loadu_ps(k, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(k, b + i);
        acc0 = _mm512_fmadd_ps(va, vb, acc0);
        i += 16;
    }

    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0,acc1), _mm512_add_ps(acc2,acc3));
    return hsum512(acc);
}

/* ── Verification and benchmark ──────────────────────────────────────────── */
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef float (*dot_fn)(const float*, const float*, int);

static double time_fn(dot_fn fn, const float *a, const float *b, int n) {
    volatile float sink = 0;
    double best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now();
        for (int r = 0; r < 100; r++) sink += fn(a, b, n);
        double elapsed = (now() - t0) / 100;
        if (elapsed < best) best = elapsed;
    }
    (void)sink;
    return best;
}

int main(void) {
    /* N with awkward tail to test tail handling */
    int n = (1 << 20) + 37;  /* ~1M + 37 elements */
    printf("N = %d  (not a multiple of 16, 32, or 64 — tests tail handling)\n\n", n);

    float *a = aligned_alloc(64, ((n+63)/64*64) * sizeof(float));
    float *b = aligned_alloc(64, ((n+63)/64*64) * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        b[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    /* Reference */
    float ref = dot_v1_scalar(a, b, n);

    struct { const char *name; dot_fn fn; } impls[] = {
        {"v1 scalar",               dot_v1_scalar      },
        {"v2 SSE",                  dot_v2_sse         },
        {"v3 AVX2+FMA 1 acc",       dot_v3_avx_fma     },
        {"v4 AVX2+FMA 4 acc",       dot_v4_avx_4acc    },
        {"v5 AVX-512 4 acc",        dot_v5_avx512      },
        {"v6 AVX-512 masked tail",  dot_v6_avx512_masked},
    };
    int nimpls = sizeof(impls) / sizeof(impls[0]);

    printf("%-28s  %8s  %8s  %6s  %s\n",
           "Implementation", "ms", "GFLOPS", "Speedup", "Result");
    printf("%-28s  %8s  %8s  %6s  %s\n",
           "---", "---", "---", "---", "---");

    double t_scalar = 0;
    for (int i = 0; i < nimpls; i++) {
        float result = impls[i].fn(a, b, n);
        double t = time_fn(impls[i].fn, a, b, n);
        double gflops = 2.0 * n / t / 1e9;

        if (i == 0) t_scalar = t;
        float relerr = fabsf(result - ref) / (fabsf(ref) + 1e-30f);

        printf("%-28s  %8.3f  %8.2f  %6.1fx  %.8f  (err %.1e)\n",
               impls[i].name, t*1000, gflops, t_scalar/t, result, relerr);
    }

    printf("\nNote: floating-point results differ slightly between implementations\n");
    printf("because FMA has one rounding instead of two, and accumulator order\n");
    printf("changes the final value. All are equally 'correct' per IEEE 754.\n");

    free(a); free(b);
    return 0;
}
