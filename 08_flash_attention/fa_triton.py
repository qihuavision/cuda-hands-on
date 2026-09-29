# fa_triton.py —— 手撕第8题终镰：简化 FA 前向（Triton 版，单头非因果）
import torch, triton, triton.language as tl, math

@triton.jit
def fa_fwd(Q, K, V, O, N,
           D: tl.constexpr,      # 头维度（如 64）
           BM: tl.constexpr,     # Q 块高（行数）
           BN: tl.constexpr):    # K/V 块高（行数）
    pid = tl.program_id(0)                    # ① 我是谁：第几个 program（Q 块）
    offs_m = pid * BM + tl.arange(0, BM)      # ② 我负责的 Q 行号（64 个）
    offs_d = tl.arange(0, D)                  # ③ 头维度的列号（0..63）
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :])   # ④ 装入我的 Q 块（64×64）
    m_i = tl.full([BM], -math.inf, tl.float32)   # ⑤ 运行态 m：64 行各一个，初值 -∞
    l_i = tl.zeros([BM], tl.float32)              # ⑥ 运行态 ℓ：64 行各一个，初值 0
    acc = tl.zeros([BM, D], tl.float32)           # ⑦ 运行态 O：64×64 累加器，初值 0
    scale = 1.0 / math.sqrt(D)                    # ⑧ softmax 的温度系数
    for j in range(0, N, BN):                     # ⑨ 内层循环：K/V 块按序流过
        offs_n = j + tl.arange(0, BN)             # ⑩ 本块 K/V 的行号
        k = tl.load(K + offs_n[:, None] * D + offs_d[None, :])  # ⑪ 装 K 块（64×64）
        qk = tl.dot(q, tl.trans(k)) * scale       # ⑫ 块分数 S_ij（64×64），减 √d 在这
        m_blk = tl.max(qk, 1)                     # ⑬ 本块每行最大（axis=1 按行）
        m_new = tl.maximum(m_i, m_blk)            # ⑭ 换算律①：新基准
        alpha = tl.exp(m_i - m_new)               # ⑮ 汇率 e^(m_old−m_new)
        p = tl.exp(qk - m_new[:, None])           # ⑯ P̃（减新基准，指数安全）
        l_i = l_i * alpha + tl.sum(p, 1)          # ⑰ 换算律②：分母折算
        acc = acc * alpha[:, None]                # ⑱ 换算律③前半：老 O 乘汇率
        v = tl.load(V + offs_n[:, None] * D + offs_d[None, :])  # ⑲ 装 V 块
        acc += tl.dot(p.to(tl.float16), v)        # ⑳ 换算律③后半：加 P̃·V
        m_i = m_new                               # ㉑ 基准正式切换
    acc = acc / l_i[:, None]                      # ㉒ 归一化：扫完所有块才除
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], acc)    # ㉓ 写回输出

def fa_triton(Q, K, V, BM=64, BN=64):
    N, D = Q.shape
    O = torch.empty_like(Q)
    fa_fwd[(N // BM,)](Q, K, V, O, N, D=D, BM=BM, BN=BN, num_warps=4)
    return O
