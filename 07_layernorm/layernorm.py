# layernorm_triton.py —— 手撕第7题：LayerNorm（Triton 版，两遍法在驻块上零额外访存）
import torch, triton, triton.language as tl, math

@triton.jit
def ln_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_cols,
              BLOCK: tl.constexpr):
    row = tl.program_id(0)                          # ① 一个 program = 一行（同题 6）
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row*n_cols + cols,
                mask=mask, other=0.).to(tl.float32)  # ② other=0：Σx 安全（加法单位元）
    mean = tl.sum(x, 0) / n_cols                     # ③ 统计量#1：μ（越界位贡献 0 ✓）
    xm = tl.where(mask, x - mean, 0.)                # ④ ★第二道门：偏差的越界位显式清零
                                                     #    （不清零则 (0−mean)² 虚增 σ²）
    var = tl.sum(xm*xm, 0) / n_cols                  # ⑤ 统计量#2：σ²（两遍法——驻块上算，零访存）
    rstd = 1.0 / tl.sqrt(var + 1e-5)                 # ⑥ ε=1e-5 防除零（fp16 世界的安全垫）
    y = xm * rstd                                    # ⑦ 标准化（越界位=0 不影响）
    g = tl.load(g_ptr + cols, mask=mask, other=1.)   # ⑧ γ：每行共享的列向量（L2 常驻）
    b = tl.load(b_ptr + cols, mask=mask, other=0.)   # ⑨ β：同上
    tl.store(out_ptr + row*n_cols + cols, y*g + b, mask=mask)  # ⑩ 仿射+写回

def ln_triton(x, g, b):
    out = torch.empty_like(x)
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)                # 官方口径：下一个 2 的幂 ≥ N
    ln_kernel[(M,)](x, g, b, out, N, BLOCK=BLOCK, num_warps=8)
    return out

# 对拍：官方教程的目标——跑赢 PyTorch 的 F.layer_norm
x = torch.randn(4096, 1024, device='cuda', dtype=torch.float16)
g = torch.randn(1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, device='cuda', dtype=torch.float16)
ref = torch.nn.functional.layer_norm(x.float(), (1024,), g.float(), b.float(), 1e-5).half()
out = ln_triton(x, g, b)
err = (ref.float() - out.float()).abs().max().item()
print(f"max err = {err:.2e}")    # ~1e-3（fp16 舍入级）
