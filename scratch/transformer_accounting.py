from math import pow

def rmsnorm(b, s, d) -> int:
    res = 0
    res += b * s * d # x.pow(2) *
    res += b * s * (d - 1) # mean(-1,) +
    res += b * s # mean(-1,) /
    res += b * s # +eps +
    res += b * s # sqrt
    res += b * s * d # x/rms /
    res += b * s * d # *g *
    return res # O(4BSD)

def rope(b, s, dk) -> int:
    res = 0
    res += b * s * dk / 2 * 6 # x_even * cos_vals - x_odd * sin_cached and x_even * sin_cached + x_odd * cos_vals
    return res # O(3BSD/h)

def sdpa(b, n, m, dk, dv) -> int:
    res = 0
    res += 2 * b * n * m * dk # q@k @
    res += b * n * m # / sqrt(dk) /
    res += 4 * b * n * m # softmax bnm- bnmexp bn(m-1)+ bnm/:3bnm+bn(m-1)≈4bnm
    res += 2 * b * n * m * dv # qk@v @
    return res # O(4BS^2D/h)

def mha(b, s, d, h) -> int:
    res = 0
    dk = dv = d // h
    res += 2 * d * h * dk * b * s # Q@x @
    res += 2 * d * h * dk * b * s # K@x @
    res += 2 * d * h * dv * b * s # V@x @
    res += 2 * h * rope(b, s, dk) # rope(q and k)
    res += h * sdpa(b, s, s, dk, dv) # heads
    res += 2 * b * s * d * h * dv # O@heads
    # print(f"  MHA: {res / 1e12:.3f} TFLOPs")
    return res # O(8BSD^2 + 4BS^2D)

def ffn(b, s, d, dff) -> int:
    res = 0
    res += 2 * b * s * d * dff # w1@x @
    res += 5 * b * s * dff # x * sigmoid(x)
    res += 2 * b * s * d * dff  # w3@x @
    res += b * s * dff # silu*w3 *
    res += 2 * b * s * d * dff # w2@silu
    # print(f"  FFN: {res / 1e12:.3f} TFLOPs")
    return res # O(6BSDF=16BSD^2) same as standard ffn: d -> 4d ->d, (2 * b * s * d * 4d) * 2 = 16BSD^2

def block(b, s, d, dff, h) -> int:
    res = 0
    res += rmsnorm(b, s, d) # ln1(x) O(4BSD)
    res += mha(b, s, d, h) # mha(x) O(8BSD^2 + 4BS^2D)
    res += b * s * d # add O(BSD)
    res += rmsnorm(b, s, d)  # ln2(x) O(4BSD)
    res += ffn(b, s, d, dff) # ffn(x) O(16BSD^2)
    res += b * s * d  # add O(BSD)
    # print(f"block: {res / 1e12:.3f} TFLOPs")
    return res # O(24BSD^2 + 4BS^2D)

def lm_head(b, s, d, v) -> int:
    res = 0
    res += 2 * b * s * d * v # linear@x @
    return res # O(2BSDV)

def lm(b, s, d, dff, h, v, l) -> int:
    res = 0
    res_block = l * block(b, s, d, dff, h)  # L 层
    res += res_block
    res += rmsnorm(b, s, d)  # ln_final
    res_head = lm_head(b, s, d, v)  # LM head
    res += res_head
    # print(f"LM total: {res / 1e12:.3f} TFLOPs  ({l} blocks: {res_block / 1e12:.3f} T  lm_head: {res_head / 1e12:.3f} T)")
    return res # O(24LBSD² + 4LBS²D + 2BSDV)

if __name__ == '__main__':
    B = 1
    T = 1024
    V = 50_257
    configs = [
        ("GPT-2 Small",  12,  768, 12, 1024),
        ("GPT-2 Medium", 24, 1024, 16, 1024),
        ("GPT-2 Large",  36, 1280, 20, 1024),
        ("GPT-2 XL",     48, 1600, 25, 1024),
        ("GPT-2 XL T=16K",48, 1600, 25, 16384),
    ]
    for name, L, D, H, T in configs:
        F = ((int(8/3 * D) + 63) // 64) * 64
        total = lm(B, T, D, F, H, V, L)
        blk = block(B, T, D, F, H)
        m = mha(B, T, D, H)
        f = ffn(B, T, D, F)
        print(f"\n=== {name} (L={L}, D={D}, H={H}, F={F}) ===")
        print(f"  MHA:      {m/1e12:.3f} TFLOPs ({m/blk*100:.0f}% of block)")
        print(f"  FFN:      {f/1e12:.3f} TFLOPs ({f/blk*100:.0f}% of block)")
        print(f"  block:    {blk/1e12:.3f} TFLOPs")
        print(f"  {L} blocks: {L*blk/1e12:.3f} TFLOPs ({L*blk/total*100:.0f}% of total)")
        print(f"  lm_head:  {lm_head(B,T,D,V)/1e12:.3f} TFLOPs ({lm_head(B,T,D,V)/total*100:.0f}% of total)")
        print(f"  Total:    {total/1e12:.3f} TFLOPs")

'''
(a)
  Token Embedding:  V × D  = 50,257 × 1,600 = 80.41M
  Per Block:
    attn QKV+O:      4 × D²        = 4 × 2,560,000 = 10.24M
    ffn W1+W2+W3:    3 × D × F     = 3 × 1,600 × 4,288 = 20.58M
    ln1+ln2:          2 × D         = 3,200
    subtotal:                                       ~30.83M/block
  48 Blocks:          48 × 30.83M  = 1,480M
  ln_final:           D             = 1,600
  LM Head:            V × D        = 50,257 × 1,600 = 80.41M
  Total params:       80.41 + 1480 + 0.0016 + 80.41 = 1,640.45M
  Memory (float32):   1.640B × 4 = 6.56 GB

(b)(c)
  ┌───────────┬────────┬────────────┬────────────┐
  │           │ TFLOPs │ % of block │ % of total │
  ├───────────┼────────┼────────────┼────────────┤
  │ MHA       │ 0.028  │ 40%        │ —          │
  ├───────────┼────────┼────────────┼────────────┤
  │ FFN       │ 0.042  │ 60%        │ —          │
  ├───────────┼────────┼────────────┼────────────┤
  │ block     │ 0.070  │ 100%       │ —          │
  ├───────────┼────────┼────────────┼────────────┤
  │ 48 blocks │ 3.361  │ —          │ 95.3%      │
  ├───────────┼────────┼────────────┼────────────┤
  │ lm_head   │ 0.165  │ —          │ 4.7%       │
  ├───────────┼────────┼────────────┼────────────┤
  │ Total     │ 3.526  │ —          │ 100%       │
  └───────────┴────────┴────────────┴────────────┘
(d)
  ┌─────────────────────┬─────────────┬──────────────┬────────────────────┐
  │        Model        │ Blocks rate │ LM Head rate │ FFN rate in Block  │
  ├─────────────────────┼─────────────┼──────────────┼────────────────────┤
  │ Small (12L, 768D)   │ 73%         │ 27%          │ 54%                │
  ├─────────────────────┼─────────────┼──────────────┼────────────────────┤
  │ Medium (24L, 1024D) │ 87%         │ 13%          │ 57%                │
  ├─────────────────────┼─────────────┼──────────────┼────────────────────┤
  │ Large (36L, 1280D)  │ 93%         │ 7%           │ 59%                │
  ├─────────────────────┼─────────────┼──────────────┼────────────────────┤
  │ XL (48L, 1600D)     │ 95%         │ 5%           │ 60%                │
  └─────────────────────┴─────────────┴──────────────┴────────────────────┘
  blocks 占比随模型增大而升高（73%→95%），原因是 blocks FLOPs ∝ L×D²，
  而 lm_head ∝ D×V。虽然 V 很大(50K)，但 L 和 D 同时增长，L×D² 增速远超 D×V。
  块内 FFN 占比略高于 MHA（54%→60%），因为 FFN matmul 项为 6BSDF，
  代入 F≈8/3D 后 ≈16BSD²，MHAD matmul 项约 8BSD²，FFN 系数更大。
(e)
  ┌──────────┬────────────────────────┬────────────────────────┬───────┐
  │          │         T=1024         │        T=16384         │Change │
  ├──────────┼────────────────────────┼────────────────────────┼───────┤
  │ MHA      │ 0.028 T (40% of block) │ 2.087 T (76% of block) │ ×74.5 │
  ├──────────┼────────────────────────┼────────────────────────┼───────┤
  │ FFN      │ 0.042 T (60% of block) │ 0.675 T (24% of block) │ ×16   │
  ├──────────┼────────────────────────┼────────────────────────┼───────┤
  │ Total    │ 3.526 T                │ 135.229 T              │ ×38.4 │
  ├──────────┼────────────────────────┼────────────────────────┼───────┤
  │ MHA rate │ 40%                    │ 76% ← Overtake FFN     │       │
  └──────────┴────────────────────────┴────────────────────────┴───────┘
原因：MHA 有 4BS²D 的 S² 项（QK^T + attn@V），T 从 1024→16384（16×）变成 256×。FFN 只有 16× 线性增长。大上下文时 MHA 绝对主导。
'''