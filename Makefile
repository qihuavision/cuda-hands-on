# cuda-hands-on Makefile
# Usage: make run_01 / make run_04 / make run_all

CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
ARCH := -arch=sm_86
OPTFLAGS := -O3
CFLAGS := $(OPTFLAGS) $(ARCH) -Xcompiler -Wall

.PHONY: all run_01 run_02 run_03 run_04 run_05 run_06 run_07 run_08 run_all clean

all: run_all

# 01 Vector Add
01_vector_add/va: 01_vector_add/va.cu
	$(NVCC) $(CFLAGS) -o $@ $<
run_01: 01_vector_add/va
	./01_vector_add/va

# 02 Reduction
02_reduction/reduce: 02_reduction/reduce.cu
	$(NVCC) $(CFLAGS) -o $@ $<
run_02: 02_reduction/reduce
	./02_reduction/reduce

# 03 Transpose
03_transpose/transpose: 03_transpose/transpose.cu
	$(NVCC) $(CFLAGS) -o $@ $<
run_03: 03_transpose/transpose
	./03_transpose/transpose

# 04 GEMM ladder (compile all 6)
04_gemm/gemm_v%: 04_gemm/gemm_v%.cu
	$(NVCC) $(CFLAGS) -o $@ $<
run_04: 04_gemm/gemm_v1_naive 04_gemm/gemm_v2_coalesced 04_gemm/gemm_v3_smem \
        04_gemm/gemm_v4_register 04_gemm/gemm_v5_float4 04_gemm/gemm_v6_double_buffer
	@echo "=== v1 naive ==="     && ./04_gemm/gemm_v1_naive
	@echo "=== v2 coalesced ===" && ./04_gemm/gemm_v2_coalesced
	@echo "=== v3 smem ==="      && ./04_gemm/gemm_v3_smem
	@echo "=== v4 register ==="  && ./04_gemm/gemm_v4_register
	@echo "=== v5 float4 ==="    && ./04_gemm/gemm_v5_float4 || echo "(v5 is a skeleton — see README)"
	@echo "=== v6 double-buffer ===" && ./04_gemm/gemm_v6_double_buffer || echo "(v6 is a core snippet — see README)"

# 05 Softmax
05_softmax/softmax: 05_softmax/softmax.cu
	$(NVCC) $(CFLAGS) -o $@ $<
run_05: 05_softmax/softmax
	./05_softmax/softmax

# 06 Triton Softmax (Python)
run_06:
	python 06_softmax_triton/softmax_triton.py

# 07 Triton LayerNorm (Python)
run_07:
	python 07_layernorm/layernorm.py

# 08 FlashAttention (Python)
run_08:
	python 08_flash_attention/fa_forward.py
	python 08_flash_attention/fa_triton.py

run_all: run_01 run_02 run_03 run_04 run_05 run_06 run_07 run_08

clean:
	find . -name "va" -o -name "reduce" -o -name "transpose" -o -name "gemm_v*" -o -name "softmax" | xargs rm -f
