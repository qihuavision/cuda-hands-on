# fa_reference.py —— FA 前向教学版（分块直译）+ 标准对照版
# 每一行右侧注释 = 这行在伪代码里的编号
import torch, math

def attn_standard(Q, K, V):
    """标准 attention：S/P 全量落地（就是我们要消灭的那种实现）"""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)              # N×N 落地
    P = torch.softmax(S, dim=-1)            # N×N 再落地
    return P @ V

def attn_flash_teaching(Q, K, V, Bc=64):
    """分块 + online softmax 直译版。
    注意：这版的 S_ij 仍在 Python 里生成（还没进 kernel），
    但换算律三式与真 FA 完全一致——数学层已就位。"""
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros(N, d)                   # ③ 全部输出
    m = torch.full((N,), -math.inf)         # ③ 每行运行态 m
    l = torch.zeros(N,)                     # ③ 每行运行态 ℓ
    for j in range(0, N, Bc):               # ④ 内层：K/V 块流过
        Kj = K[j:j+Bc]                      # ⑤ 取 K 块
        Vj = V[j:j+Bc]                      # ⑤ 取 V 块
        S = (Q @ Kj.T) * scale              # ⑥ 块分数 N×Bc（教学版全 Q 一起算）
        m_blk = S.max(dim=-1).values        # ⑦ 本块行最大
        m_new = torch.maximum(m, m_blk)     # ⑧ 换算律①
        alpha = torch.exp(m - m_new)        #    折算汇率 e^(m_old−m_new)
        P = torch.exp(S - m_new[:, None])   # ⑨ P̃（减新基准）
        l = l * alpha + P.sum(-1)           # ⑩ 换算律②
        O = O * alpha[:, None] + P @ Vj     # ⑪ 换算律③
        m = m_new                           # ⑫ 基准切换
    return O / l[:, None]                   # ⑬ 归一化

# ---- 对拍：随机数据下两种实现必须一致到浮点舍入 ----
torch.manual_seed(42)
Q, K, V = (torch.randn(256, 64) for _ in range(3))
a = attn_standard(Q, K, V)
b = attn_flash_teaching(Q, K, V)
err = (a - b).abs().max().item()
print(f"max err = {err:.2e}")   # 期望 ~1e-7（float32 舍入级）——这就是 exact 的代码证据
assert err < 1e-5
