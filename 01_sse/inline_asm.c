/* inline_asm.c — SSE via GCC extended inline assembly
 *
 * Compile: gcc -O2 -msse4.2 -o inline_asm inline_asm.c
 *
 * GCC extended inline assembly syntax:
 *
 *   asm volatile (
 *       "instructions"
 *       : output operands           <- what the asm writes
 *       : input operands            <- what the asm reads
 *       : clobber list              <- what the asm destroys
 *   );
 *
 * Operand constraints:
 *   "r"  = general-purpose register
 *   "m"  = memory location
 *   "x"  = SSE/XMM register
 *   "=r" = output to general-purpose register
 *   "=m" = output to memory
 *   "+r" = read-write general-purpose register
 *
 * Why learn this?
 *   - Understand what the CPU actually executes
 *   - Occasionally needed for exact instruction sequences
 *   - Helps reading compiler-generated asm output
 *
 * Why prefer intrinsics?
 *   - Compiler can schedule, optimize, and allocate registers around intrinsics
 *   - asm volatile is a black box: the compiler cannot look inside
 *   - Much less error-prone; no register name typos or clobber mistakes
 *
 * Note: All x86 asm in GCC uses AT&T syntax by default.
 *   - Source before destination: "addps %src, %dst"  (opposite of Intel docs)
 *   - Registers prefixed with %%: "%%xmm0"
 *   - Immediates prefixed with $
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#define N 1024

/* ── Example 1: load/add/store with positional operands ──────────────────── */
/*
 * %0, %1, %2 refer to operands in order: output first, then input.
 * "r" = compiler picks any general-purpose register for the pointer.
 * "xmm0", "xmm1" in the clobber list tell the compiler we used those.
 */
void sse_add_asm_positional(const float *a, const float *b, float *c, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __asm__ volatile (
            "movups (%1), %%xmm0\n\t"   /* xmm0 = a[i..i+3] */
            "movups (%2), %%xmm1\n\t"   /* xmm1 = b[i..i+3] */
            "addps  %%xmm1, %%xmm0\n\t" /* xmm0 += xmm1  (AT&T: src, dst) */
            "movups %%xmm0, (%0)\n\t"   /* c[i..i+3] = xmm0 */
            :                            /* no output operands (write through ptr) */
            : "r"(c + i),               /* %0: pointer to c[i] */
              "r"(a + i),               /* %1: pointer to a[i] */
              "r"(b + i)               /* %2: pointer to b[i] */
            : "xmm0", "xmm1", "memory" /* registers clobbered + memory written */
        );
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

/* ── Example 2: named operands (more readable) ───────────────────────────── */
void sse_add_asm_named(const float *a, const float *b, float *c, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __asm__ volatile (
            "movups (%[src_a]), %%xmm0\n\t"
            "movups (%[src_b]), %%xmm1\n\t"
            "addps  %%xmm1, %%xmm0\n\t"
            "movups %%xmm0, (%[dst])\n\t"
            :
            : [dst]   "r"(c + i),
              [src_a] "r"(a + i),
              [src_b] "r"(b + i)
            : "xmm0", "xmm1", "memory"
        );
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

/* ── Example 3: XMM register constraint ("x") ───────────────────────────── */
/*
 * When you let the compiler allocate XMM registers via "x" constraints,
 * you don't need to hardcode %%xmm0, %%xmm1, etc. and don't need to clobber them.
 * Use %[name] to reference the compiler-allocated register.
 *
 * This only works when the operands fit entirely in XMM registers.
 */
__m128 sse_mul_register(__m128 a, __m128 b) {
    __asm__ (
        "mulps %[b], %[a]\n\t"          /* AT&T: mulps src, dst  => dst *= src */
        : [a] "+x"(a)                   /* "+x" = read-write XMM register */
        : [b]  "x"(b)
    );
    return a;
}

/* ── Example 4: scalar asm with XMM ─────────────────────────────────────── */
/*
 * Extracting a scalar result from XMM.
 * "=m" means write output to a memory location.
 */
float sse_dot4_asm(const float a[4], const float b[4]) {
    float result;
    __asm__ volatile (
        "movups (%[a]), %%xmm0\n\t"      /* xmm0 = {a0, a1, a2, a3} */
        "movups (%[b]), %%xmm1\n\t"      /* xmm1 = {b0, b1, b2, b3} */
        "mulps  %%xmm1, %%xmm0\n\t"      /* xmm0 = {a0*b0, a1*b1, a2*b2, a3*b3} */
        /* horizontal sum using haddps (SSE3) */
        "haddps %%xmm0, %%xmm0\n\t"      /* xmm0 = {a0b0+a1b1, a2b2+a3b3, ...} */
        "haddps %%xmm0, %%xmm0\n\t"      /* xmm0[0] = a0b0+a1b1+a2b2+a3b3 */
        "movss  %%xmm0, %[result]\n\t"   /* store lane 0 to result */
        : [result] "=m"(result)
        : [a] "r"(a), [b] "r"(b)
        : "xmm0", "xmm1"
    );
    return result;
}

/* ── Example 5: CPUID to check what's supported ─────────────────────────── */
/*
 * CPUID is a special instruction you must write in asm — no intrinsic exists.
 * It returns CPU feature information in eax/ebx/ecx/edx.
 */
typedef struct {
    uint32_t eax, ebx, ecx, edx;
} cpuid_result_t;

cpuid_result_t cpuid(uint32_t leaf, uint32_t subleaf) {
    cpuid_result_t r;
    __asm__ volatile (
        "cpuid"
        : "=a"(r.eax), "=b"(r.ebx), "=c"(r.ecx), "=d"(r.edx)
        : "a"(leaf), "c"(subleaf)
    );
    return r;
}

void check_cpu_features(void) {
    printf("\n=== CPU feature detection via CPUID ===\n");

    cpuid_result_t r = cpuid(1, 0);
    printf("SSE:    %s\n", (r.edx >> 25 & 1) ? "yes" : "no");
    printf("SSE2:   %s\n", (r.edx >> 26 & 1) ? "yes" : "no");
    printf("SSE3:   %s\n", (r.ecx >>  0 & 1) ? "yes" : "no");
    printf("SSSE3:  %s\n", (r.ecx >>  9 & 1) ? "yes" : "no");
    printf("SSE4.1: %s\n", (r.ecx >> 19 & 1) ? "yes" : "no");
    printf("SSE4.2: %s\n", (r.ecx >> 20 & 1) ? "yes" : "no");
    printf("AVX:    %s\n", (r.ecx >> 28 & 1) ? "yes" : "no");
    printf("FMA:    %s\n", (r.ecx >> 12 & 1) ? "yes" : "no");

    r = cpuid(7, 0);
    printf("AVX2:   %s\n", (r.ebx >>  5 & 1) ? "yes" : "no");
    printf("AVX512F:%s\n", (r.ebx >> 16 & 1) ? "yes" : "no");
}

/* ── Inline asm vs intrinsics: a direct comparison ───────────────────────── */
void compare_approaches(void) {
    printf("\n=== Inline asm vs intrinsics ===\n");

    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};

    /* Intrinsic: clear intent, compiler can optimize context */
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 vc_intr = _mm_add_ps(va, vb);
    float result_intr[4];
    _mm_storeu_ps(result_intr, vc_intr);

    /* Inline asm: same operation, but opaque to optimizer */
    float result_asm[4];
    __asm__ volatile (
        "movups (%1), %%xmm0\n\t"
        "movups (%2), %%xmm1\n\t"
        "addps  %%xmm1, %%xmm0\n\t"
        "movups %%xmm0, (%0)\n\t"
        : : "r"(result_asm), "r"(a), "r"(b) : "xmm0", "xmm1", "memory"
    );

    printf("intrinsics: %.1f %.1f %.1f %.1f\n",
           result_intr[0], result_intr[1], result_intr[2], result_intr[3]);
    printf("inline asm: %.1f %.1f %.1f %.1f\n",
           result_asm[0],  result_asm[1],  result_asm[2],  result_asm[3]);

    printf("\nBoth give the same result. The difference:\n");
    printf("  Intrinsics: compiler can merge, reorder, CSE across the call.\n");
    printf("  Inline asm: asm volatile = fence; compiler works around it.\n");
    printf("  Recommendation: use intrinsics; reach for inline asm only when\n");
    printf("  you need exact instruction sequences (e.g., CPUID, RDTSC, CRC32).\n");
}

/* ── Useful instructions without intrinsics: RDTSC ──────────────────────── */
/*
 * RDTSC reads the timestamp counter — useful for microbenchmarking.
 * GCC provides __builtin_ia32_rdtsc() but the inline asm makes it explicit.
 */
uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ volatile (
        "rdtsc"
        : "=a"(lo), "=d"(hi)
    );
    return ((uint64_t)hi << 32) | lo;
}

uint64_t rdtscp(uint32_t *aux) {
    uint32_t lo, hi;
    __asm__ volatile (
        "rdtscp"
        : "=a"(lo), "=d"(hi), "=c"(*aux)
    );
    return ((uint64_t)hi << 32) | lo;
}

void demo_rdtsc(void) {
    printf("\n=== RDTSC timing ===\n");

    uint32_t aux;
    uint64_t t0 = rdtscp(&aux);
    /* some work */
    volatile float x = 0;
    for (int i = 0; i < 1000; i++) x += (float)i;
    uint64_t t1 = rdtscp(&aux);

    printf("loop took ~%llu TSC ticks\n", (unsigned long long)(t1 - t0));
    printf("(divide by CPU GHz to get nanoseconds)\n");
}

int main(void) {
    float *a, *b, *c1, *c2;
    posix_memalign((void**)&a,  16, N * sizeof(float));
    posix_memalign((void**)&b,  16, N * sizeof(float));
    posix_memalign((void**)&c1, 16, N * sizeof(float));
    posix_memalign((void**)&c2, 16, N * sizeof(float));

    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(N - i); }

    sse_add_asm_positional(a, b, c1, N);
    sse_add_asm_named(a, b, c2, N);

    for (int i = 0; i < N; i++) {
        assert(c1[i] == c2[i]);
        assert(c1[i] == a[i] + b[i]);
    }
    printf("Array add: positional and named asm variants match.\n");

    float da[4] = {1, 2, 3, 4};
    float db[4] = {1, 2, 3, 4};
    float dot = sse_dot4_asm(da, db);
    printf("dot([1,2,3,4],[1,2,3,4]) = %.1f  (expected 30.0)\n", dot);

    check_cpu_features();
    compare_approaches();
    demo_rdtsc();

    free(a); free(b); free(c1); free(c2);
    return 0;
}
