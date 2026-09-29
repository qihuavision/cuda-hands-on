# cuda-hands-on

RTX 3060 6GB (sm_86) 上手写的 8 个 CUDA/Triton kernel。

## 算子清单

| # | 算子 | 版本 | 指标 |
|---|------|------|------|
| 01 | Vector Add | naive / grid-stride | ~95% peak bandwidth |
| 02 | Reduction | 两级归约 (shuffle + fence) | 对拍 CPU |
| 03 | Transpose | naive / SMEM tile + padding | ~74% peak |
| 04 | GEMM | 六级优化阶梯 | 1.3% → ~55% cuBLAS |
| 05 | Softmax | 3-pass / online | 换算律验证 |
| 06 | Fused Softmax | Triton | 对拍 CUDA 版 |
| 07 | LayerNorm | Triton | max_err < 1e-3 |
| 08 | FlashAttention | Triton | max_err < 1e-3 vs SDPA |

## GEMM 六级

| 级 | 优化点 | % cuBLAS |
|----|--------|----------|
| v1 | naive | 1.3% |
| v2 | 合并访存 | ~8% |
| v3 | SMEM 分块 | ~18% |
| v4 | 寄存器分块 | ~35% |
| v5 | float4 向量化 | ~45% |
| v6 | cp.async 双缓冲 | ~55% |

## 环境

- GPU: RTX 3060 Laptop 6GB, sm_86, 336 GB/s
- CUDA 12.x (WSL2), Triton 3.x
- `nvcc -O3 -arch=sm_86`

## 运行

```bash
make run_01
make run_04
python 08_flash_attention/fa_triton.py
```

## 参考

- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

MIT
