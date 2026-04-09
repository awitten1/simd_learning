#define _POSIX_C_SOURCE 200809L
/* bench_dot_product.c — Benchmarking SIMD dot product implementations
 *
 * Compile: gcc -O2 -mavx512f -mavx512dq -mavx512vl -mfma \
 *              -o bench_dot_product bench_dot_product.c -lm
 *
 * This benchmark demonstrates:
 *   1. Real speedups from wider SIMD (SSE→AVX→AVX-512)
 *   2. The critical importance of multiple accumulators for FP reductions
 *   3. The difference between compute-bound and memory-bound regimes
 *   4. How to write a reliable microbenchmark
 *
 * Key insight — Dependency chains and throughput vs latency:
 *
 *   VFMADD (FMA instruction) on Zen4:
 *     Latency:    4 cycles (result available after 4 cycles)
 *     Throughput: 0.5 cycles (can issue 2 per cycle)
 *
 *   With ONE accumulator:
 *     iteration i+1 must wait for iteration i to finish
 *     → one FMA per 4 cycles (limited by latency)
 *     → 4× slower than optimal
 *
 *   With EIGHT accumulators (8 independent chains):
 *     CPU can keep 8 FMAs in flight simultaneously
 *     → one FMA per 0.5 cycles (limited by throughput)
 *     → achieves peak compute
 *
 *   The lesson: for reductions, unroll with multiple accumulators.
 *   GCC does this automatically for simple loops; explicit unrolling helps.
 */
#include <immintrin.h>
#include <x86intrin.h>   /* __rdtsc, __rdtscp */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ── Timing ──────────────────────────────────────────────────────────────── */
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * RDTSC serialization pattern:
 *
 *   LFENCE prevents the CPU from issuing subsequent instructions until all
 *   prior memory operations complete. Combined with RDTSC/RDTSCP it gives
 *   a clean measurement window.
 *
 *   Before: LFENCE → RDTSC        (ensures prior work finishes first)
 *   After:  RDTSCP → LFENCE       (RDTSCP waits for prior insns to retire;
 *                                   LFENCE prevents subsequent insns from
 *                                   issuing before RDTSCP reads the counter)
 *
 * Note: RDTSC counts TSC ticks, not core cycles. On modern x86 the TSC
 * increments at a fixed rate equal to the nominal (base) CPU frequency,
 * regardless of turbo or power-saving states. So cycles ≈ TSC × (core_freq /
 * tsc_freq). For an unthrottled benchmark the approximation is close.
 */
static inline uint64_t tsc_start(void) {
    _mm_lfence();
    return __rdtsc();
}

static inline uint64_t tsc_end(void) {
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

typedef struct { double secs; double cycles; } bench_result_t;

/* Run 'fn' REPS times; report best (minimum) measurement over TRIALS trials.
 * Taking the minimum eliminates OS scheduling jitter. */
#define REPS 50
#define TRIALS 5

static bench_result_t bench_fn(void (*fn)(const float*, const float*, int, float*),
                                const float *a, const float *b, int n) {
    volatile float sink = 0;
    float result;
    double best_secs   = 1e18;
    double best_cycles = 1e18;

    for (int t = 0; t < TRIALS; t++) {
        uint64_t c0 = tsc_start();
        double   t0 = now();

        for (int r = 0; r < REPS; r++) {
            fn(a, b, n, &result);
            sink += result;
        }

        uint64_t c1 = tsc_end();
        double   t1 = now();

        double elapsed_secs   = (t1 - t0) / REPS;
        double elapsed_cycles = (double)(c1 - c0) / REPS;

        if (elapsed_secs < best_secs) {
            best_secs   = elapsed_secs;
            best_cycles = elapsed_cycles;
        }
    }
    (void)sink;
    return (bench_result_t){ best_secs, best_cycles };
}

/* ── Implementation 1: Scalar ────────────────────────────────────────────── */
void dot_scalar(const float *a, const float *b, int n, float *out) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    *out = s;
}

/* ── Implementation 2: Scalar with 4 accumulators ───────────────────────── */
void dot_scalar_4acc(const float *a, const float *b, int n, float *out) {
    float s0=0, s1=0, s2=0, s3=0;
    int i;
    for (i = 0; i <= n-4; i += 4) {
        s0 += a[i+0] * b[i+0];
        s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2];
        s3 += a[i+3] * b[i+3];
    }
    for (; i < n; i++) s0 += a[i]*b[i];
    *out = s0+s1+s2+s3;
}

/* ── Implementation 3: SSE (4-wide) with 2 accumulators ─────────────────── */
void dot_sse(const float *a, const float *b, int n, float *out) {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a+i),   _mm_loadu_ps(b+i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a+i+4), _mm_loadu_ps(b+i+4)));
    }
    __m128 acc = _mm_add_ps(acc0, acc1);
    for (; i < n; i++) acc = _mm_add_ss(acc, _mm_mul_ss(_mm_load_ss(a+i), _mm_load_ss(b+i)));
    /* Horizontal sum */
    __m128 h = _mm_hadd_ps(acc, acc);
    h = _mm_hadd_ps(h, h);
    *out = _mm_cvtss_f32(h);
}

/* ── Implementation 4: AVX2 (8-wide) — 1 accumulator ────────────────────── */
void dot_avx_1acc(const float *a, const float *b, int n, float *out) {
    __m256 acc = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);   /* acc += a*b */
    }
    /* Horizontal reduce */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; i++) result += a[i]*b[i];
    *out = result;
}

/* ── Implementation 5: AVX2 (8-wide) — 8 accumulators ──────────────────── */
/*
 * 8 independent accumulators break the FMA dependency chain.
 * The CPU can issue 2 FMAs per cycle; each has 4-cycle latency.
 * We need ≥8 independent ops in flight to saturate throughput.
 * 8 × 256-bit FMAs = 64 float MACs per cycle at peak.
 */
void dot_avx_8acc(const float *a, const float *b, int n, float *out) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 64; i += 64) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+0),  _mm256_loadu_ps(b+i+0),  acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+32), _mm256_loadu_ps(b+i+32), acc4);
        acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+40), _mm256_loadu_ps(b+i+40), acc5);
        acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+48), _mm256_loadu_ps(b+i+48), acc6);
        acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+56), _mm256_loadu_ps(b+i+56), acc7);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                              _mm256_add_ps(acc2, acc3)),
                                _mm256_add_ps(_mm256_add_ps(acc4, acc5),
                                              _mm256_add_ps(acc6, acc7)));
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; i++) result += a[i]*b[i];
    *out = result;
}

/* ── Implementation 6: AVX-512 (16-wide) — 8 accumulators ──────────────── */
/*
 * 8 × 512-bit accumulators = 8 × 16 = 128 float MACs per accumulator iteration.
 * With 2 FMAs per cycle: 2 × 16 = 32 floats per cycle at peak.
 * Peak throughput: 32 × clock_GHz GFLOPS (for FMA = 2 ops per multiply-add).
 */
void dot_avx512_8acc(const float *a, const float *b, int n, float *out) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();
    int i;
    for (i = 0; i <= n - 128; i += 128) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+0),   _mm512_loadu_ps(b+i+0),   acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16),  _mm512_loadu_ps(b+i+16),  acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32),  _mm512_loadu_ps(b+i+32),  acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48),  _mm512_loadu_ps(b+i+48),  acc3);
        acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+64),  _mm512_loadu_ps(b+i+64),  acc4);
        acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+80),  _mm512_loadu_ps(b+i+80),  acc5);
        acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+96),  _mm512_loadu_ps(b+i+96),  acc6);
        acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+112), _mm512_loadu_ps(b+i+112), acc7);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(acc0,acc1),
                                              _mm512_add_ps(acc2,acc3)),
                                _mm512_add_ps(_mm512_add_ps(acc4,acc5),
                                              _mm512_add_ps(acc6,acc7)));
    float result = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) result += a[i]*b[i];
    *out = result;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

static void print_result(const char *name, bench_result_t r, int n, double scalar_secs) {
    double gflops      = 2.0 * n / r.secs / 1e9;  /* 2 ops per MAC */
    double gb          = 2.0 * n * sizeof(float) / r.secs / 1e9;
    double cy_per_elem = r.cycles / n;
    printf("  %-35s %6.3f ms  %5.2f cy/e  %6.2f GFLOPS  %5.1f GB/s  %4.1fx\n",
           name, r.secs * 1000, cy_per_elem, gflops, gb, scalar_secs / r.secs);
}

int main(void) {
    /* Try multiple N values: L1-cached, L2-cached, LLC-cached, DRAM */
    int sizes[] = {
        4*1024,          /*  16 KB: fits in L1 cache (32 KB L1D on Zen4) */
        64*1024,         /* 256 KB: fits in L2 cache (1 MB L2 on Zen4) */
        4*1024*1024,     /*  16 MB: fits in L3 cache (64 MB L3 on 9900X) */
        32*1024*1024,    /* 128 MB: exceeds L3, goes to DRAM */
    };
    const char *size_names[] = {"L1 (16 KB)", "L2 (256 KB)", "L3 (16 MB)", "DRAM (128 MB)"};

    printf("Dot product benchmark: scalar vs SSE vs AVX2 vs AVX-512\n");
    printf("CPU: AMD Ryzen 9 9900X (Zen4)\n");
    printf("Compiler: gcc -O2 -mavx512f -mavx512dq -mfma\n\n");

    for (int si = 0; si < 4; si++) {
        int n = sizes[si];
        float *a = aligned_alloc(64, n * sizeof(float));
        float *b = aligned_alloc(64, n * sizeof(float));
        if (!a || !b) { fprintf(stderr, "OOM\n"); return 1; }

        for (int i = 0; i < n; i++) {
            a[i] = (float)i / (float)n;
            b[i] = 1.0f / ((float)i + 1.0f);
        }

        printf("=== %s: N=%d floats ===\n", size_names[si], n);
        printf("  %-35s %8s  %8s  %12s  %10s  %5s\n",
               "Implementation", "Time", "cy/elem", "GFLOPS", "Bandwidth", "Speedup");

        bench_result_t r_scalar = bench_fn(dot_scalar,      a, b, n);
        print_result("scalar (1 acc)",         r_scalar, n, r_scalar.secs);

        bench_result_t r_s4    = bench_fn(dot_scalar_4acc, a, b, n);
        print_result("scalar (4 acc)",         r_s4,    n, r_scalar.secs);

        bench_result_t r_sse   = bench_fn(dot_sse,         a, b, n);
        print_result("SSE    (4-wide, 2 acc)", r_sse,   n, r_scalar.secs);

        bench_result_t r_avx1  = bench_fn(dot_avx_1acc,   a, b, n);
        print_result("AVX2   (8-wide, 1 acc)", r_avx1,  n, r_scalar.secs);

        bench_result_t r_avx8  = bench_fn(dot_avx_8acc,   a, b, n);
        print_result("AVX2   (8-wide, 8 acc)", r_avx8,  n, r_scalar.secs);

        bench_result_t r_512   = bench_fn(dot_avx512_8acc, a, b, n);
        print_result("AVX-512(16-wide,8 acc)", r_512,   n, r_scalar.secs);

        printf("\n");

        /* Correctness check */
        float c_scalar, c_avx8, c_512;
        dot_scalar(a, b, n, &c_scalar);
        dot_avx_8acc(a, b, n, &c_avx8);
        dot_avx512_8acc(a, b, n, &c_512);

        float rel_avx8 = fabsf(c_avx8 - c_scalar) / fabsf(c_scalar + 1e-30f);
        float rel_512  = fabsf(c_512  - c_scalar) / fabsf(c_scalar + 1e-30f);
        printf("  Correctness: scalar=%.6g avx8=%.6g (err=%.2e) avx512=%.6g (err=%.2e)\n\n",
               c_scalar, c_avx8, rel_avx8, c_512, rel_512);

        free(a); free(b);
    }

    printf("Interpretation:\n");
    printf("  - L1 results:   compute-bound. Expect large speedup from wider SIMD + more accumulators.\n");
    printf("  - DRAM results: memory-bound. Speedup limited by memory bandwidth (all implementations\n");
    printf("                  hit the same bandwidth wall). Wide SIMD may not help much here.\n");
    printf("  - AVX2 1acc vs 8acc: the accumulator effect. Same register width, huge throughput difference.\n");
    printf("  - AVX-512 vs AVX2: wider registers help when compute-bound, less so when memory-bound.\n");
    return 0;
}
