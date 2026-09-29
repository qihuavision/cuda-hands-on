# GEMM 六级优化

v1 naive → v2 合并访存 → v3 SMEM 分块 → v4 寄存器分块 → v5 float4 → v6 双缓冲

v5/v6 为骨架，待补全。
