CC      = gcc
STD     = -std=c11
WARN    = -Wall -Wextra

# Per-module compiler flags. Each module enables only what it needs.
FLAGS_SSE   = $(STD) $(WARN) -O2 -msse4.2
FLAGS_AVX   = $(STD) $(WARN) -O2 -mavx2 -mfma
FLAGS_A512  = $(STD) $(WARN) -O2 -mavx512f -mavx512bw -mavx512dq -mavx512vl \
                              -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mfma

SSE_BINS  = 01_sse/hello_sse  01_sse/inline_asm  01_sse/autovec
AVX_BINS  = 02_avx_avx2/avx_float  02_avx_avx2/avx2_integer  02_avx_avx2/gather
A512_BINS = 03_avx512/avx512_basics
BENCH_BINS = 04_benchmarks/bench_dot_product
REAL_BINS  = 05_real_world/dot_product  05_real_world/string_search

ALL_BINS = $(SSE_BINS) $(AVX_BINS) $(A512_BINS) $(BENCH_BINS) $(REAL_BINS)

.PHONY: all bench autovec_info asm clean

all: $(ALL_BINS)

# Pattern rules — each directory gets compiled with its own flags
$(SSE_BINS): %: %.c
	$(CC) $(FLAGS_SSE) -o $@ $< -lm

$(AVX_BINS): %: %.c
	$(CC) $(FLAGS_AVX) -o $@ $<

$(A512_BINS): %: %.c
	$(CC) $(FLAGS_A512) -o $@ $<

$(BENCH_BINS): %: %.c
	$(CC) $(FLAGS_A512) -o $@ $< -lm

$(REAL_BINS): %: %.c
	$(CC) $(FLAGS_A512) -o $@ $< -lm

# Show what GCC vectorizes (and what it misses) in autovec.c
autovec_info: 01_sse/autovec.c
	@echo "======= Loops vectorized ======="
	$(CC) $(FLAGS_AVX) -fopt-info-vec-optimized -c $< -o /dev/null
	@echo ""
	@echo "======= Loops NOT vectorized ======="
	$(CC) $(FLAGS_AVX) -fopt-info-vec-missed -c $< -o /dev/null 2>&1 | head -40

# Dump Intel-syntax assembly for each module
asm: all
	@for f in 01_sse/*.c; do \
		$(CC) $(FLAGS_SSE) -S -masm=intel -o "$${f%.c}.s" $$f; \
		echo "wrote $${f%.c}.s"; \
	done
	@for f in 02_avx_avx2/*.c; do \
		$(CC) $(FLAGS_AVX) -S -masm=intel -o "$${f%.c}.s" $$f; \
		echo "wrote $${f%.c}.s"; \
	done
	@for f in 03_avx512/*.c; do \
		$(CC) $(FLAGS_A512) -S -masm=intel -o "$${f%.c}.s" $$f; \
		echo "wrote $${f%.c}.s"; \
	done

bench: $(BENCH_BINS)
	./04_benchmarks/bench_dot_product

clean:
	rm -f $(ALL_BINS)
	find . -name '*.s' -delete
