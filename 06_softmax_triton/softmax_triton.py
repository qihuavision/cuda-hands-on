# softmax_triton.py —— 手撕第6题：fused softmax（Triton 版，2 访存/元素）
# pip install triton torch
import torch, triton, triton.language as tl, time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
    ],
    key=['n_cols'],        # 列数变了才重新搜
)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride,
                   BLOCK_SIZE: tl.constexpr):   # constexpr=编译期常量（每取值一份特化代码）
    row = tl.program_id(0)                      # ① 一个 program = 一行（外层并行）
    cols = tl.arange(0, BLOCK_SIZE)             # ② 块内向量索引 [0..BLOCK)
    mask = cols < n_cols                        # ③ 行尾掩码（越界位不参与）
    x = tl.load(x_ptr + row*stride + cols,      # ④ 读第1遍（唯一一遍）：整块进寄存器
                mask=mask, other=-float('inf')) #    other=-inf：越界位不污染 max
    m = tl.max(x, 0)                            # ⑤ 一行顶 CUDA 四段式（含 mask 位剔除）
    e = tl.exp(x - m)                           # ⑥ 广播减（每元素减同一 m）+ 指数
                                                #    exp(-inf)=0：mask 位自动零贡献
    l = tl.sum(e, 0)                            # ⑦ 分母（越界位贡献 0）
    y = e / l                                   # ⑧ 归一化——数据没离开寄存器，零回访！
    tl.store(out_ptr + row*stride + cols, y, mask=mask)  # ⑨ 写第1遍

def softmax_triton(x):
    out = torch.empty_like(x)
    rows, n = x.shape
    softmax_kernel[(rows,)](x, out, n, x.stride(0))
    return out

# ---- 对拍 + 基准 ----
x = torch.randn(4096, 1024, device='cuda') * 20      # ×20：考验 max 技巧（e^20 会爆）
assert torch.allclose(softmax_triton(x), torch.softmax(x, -1), atol=1e-5)

def bench(f, n=100):
    for _ in range(10): f(x); torch.cuda.synchronize()    # 预热（首跑含 autotune 搜索）
    t = time.perf_counter()
    for _ in range(n): f(x)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

t_tri = bench(softmax_triton)
t_tor = bench(lambda t: torch.softmax(t, -1))
gbps = 2*4096*1024*4 / (t_tri*1e-3) / 1e9               # 2 次/元素口径
print(f"triton: {t_tri:.3f} ms ({gbps:.0f} GB/s, {100*gbps/336:.0f}%peak) | torch: {t_tor:.3f} ms")
