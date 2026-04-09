# x86 SIMD Learning Path

Hands-on tour of x86 SIMD from SSE through AVX-512, using three approaches:
inline assembly, intrinsics, and autovectorization.

**Your hardware: AMD Ryzen 9 9900X**
Supports: SSE4.2 · AVX · AVX2 · FMA · AVX-512 (F/BW/DQ/VL/CD/IFMA/VBMI/VBMI2/VNNI/BF16/BITALG/VPOPCNTDQ)

---

## The Three Approaches

| Approach | Portability | Compiler visibility | When to use |
|---|---|---|---|
| **Autovectorization** | Highest — just write loops | Compiler sees everything | Simple loops; free speedup |
| **Intrinsics** | Medium — header-based | Compiler sees, can optimize around | Most real SIMD code |
| **Inline assembly** | Lowest — exact ISA | Compiler treats as black box | Learning; exact instruction sequences |

Prefer intrinsics. Use inline asm only when you need to control the exact instructions emitted.
Autovectorization is your first line of defense — always check if the compiler does it for free.

---

## Learning Path

Work through the modules in order. Each builds on the previous.

### Module 1 — SSE: 128-bit SIMD (`01_sse/`)

SSE adds 8 XMM registers (128-bit each) to x86-64. One XMM register holds:
- 4 × float32  (`__m128`)
- 2 × float64  (`__m128d`)
- 16 × int8, 8 × int16, 4 × int32, 2 × int64  (`__m128i`)

| File | Teaches |
|---|---|
| `hello_sse.c` | First intrinsics program: float array ops, comparison masking, SSE4.1 extras |
| `inline_asm.c` | Same operations in GCC extended inline asm; shows why intrinsics usually win |
| `autovec.c` | What makes loops vectorize; `restrict`, alignment hints, dependency chains |

**Key exercise:** run `make autovec_info` to see GCC's vectorization report.

### Module 2 — AVX and AVX2: 256-bit SIMD (`02_avx_avx2/`)

AVX doubles register width to 256-bit (YMM). AVX2 extends integer ops to full 256-bit
and adds gather loads.

| File | Teaches |
|---|---|
| `avx_float.c` | `__m256` ops, FMA (`_mm256_fmadd_ps`), horizontal reduction |
| `avx2_integer.c` | 256-bit integer SIMD; `_mm256_cmpeq_epi8` byte-search pattern |
| `gather.c` | Non-contiguous loads; AoS vs SoA layout; when gather helps vs hurts |

**Key concept:** FMA computes `a*b + c` in one instruction with one rounding error
instead of two. Critical for BLAS, ML inference, signal processing.

### Module 3 — AVX-512: 512-bit SIMD + Masking (`03_avx512/`)

AVX-512 adds 32 ZMM registers (512-bit) and 8 opmask registers (k0–k7), enabling
true per-element conditional operations without computing both branches.

| File | Teaches |
|---|---|
| `avx512_basics.c` | 512-bit ops; masking with k-registers; VNNI int8 dot products |

**Key concept: opmask registers**
```c
// SSE/AVX style: compute both branches, blend results
__m256 result = _mm256_blendv_ps(else_val, then_val, mask);

// AVX-512 style: only compute selected lanes
__m512 result = _mm512_mask_add_ps(else_val, k, a, b);  // merge masking
__m512 result = _mm512_maskz_add_ps(k, a, b);           // zero masking
```

### Module 4 — Benchmarks: When Does SIMD Help? (`04_benchmarks/`)

| File | Teaches |
|---|---|
| `bench_dot_product.c` | Scalar vs SSE vs AVX vs AVX-512; the multiple-accumulator trick |

**Key lesson:** FMA has ~4 cycle latency but 0.5 cycle throughput. A single accumulator
creates a serial dependency chain. Use 8+ independent accumulators to saturate the units.

### Module 5 — Real-World Patterns (`05_real_world/`)

| File | Teaches |
|---|---|
| `dot_product.c` | Six implementations scalar→AVX-512+unrolled; accumulator depth explained |
| `string_search.c` | SSE4.2 string instructions (`_mm_cmpistri`); 16-byte-at-a-time scanning |

---

## Building

```bash
make all           # compile all modules
make bench         # compile and run benchmarks
make autovec_info  # show GCC vectorization report for autovec.c
make asm           # dump Intel-syntax assembly to *.s files
make clean
```

## Register and Type Quick Reference

```
Width    Register   Float      Double     Integer
128-bit  XMM        __m128     __m128d    __m128i
256-bit  YMM        __m256     __m256d    __m256i
512-bit  ZMM        __m512     __m512d    __m512i
```

## Intrinsic Naming Convention

```
_mm{width}_{op}_{suffix}
   |         |    +-- ps=packed float32, pd=packed float64
   |         |        epi8/16/32/64=signed int, epu8/16=unsigned int
   |         +-- add, sub, mul, div, fmadd, load, store, cmp, blend, ...
   +-- (blank)=128-bit, 256=256-bit, 512=512-bit
```

## When SIMD Helps

- Large arrays, uniform element-wise ops (add, multiply, compare, min/max)
- Reductions (sum, dot product, min, max) with careful horizontal reduction
- Image and signal processing — embarrassingly parallel
- String scanning — SSE4.2 processes 16 bytes per cycle

## When SIMD Does NOT Help

- Pointer-chasing structures (linked lists, trees, hash tables)
- Heavy data-dependent branching per element
- Arrays too small to amortize setup cost
- Loop-carried dependencies (prefix sum, recurrences)
- Irregular scatter writes (available in AVX-512 but rarely faster)

## References

- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Agner Fog's optimization manuals: https://agner.org/optimize/
- Per-instruction throughput/latency: https://uops.info/
