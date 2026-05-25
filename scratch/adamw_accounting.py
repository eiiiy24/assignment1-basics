import math

# ===== (a) Memory =====
# Everything in float32 = 4 bytes
BYTES = 4

def params():
    """Trainable parameters."""
    return (
        V * D                    # token_embeddings
        + L * (
            4 * D * D             # attn QKV+O weights
            + D + D               # ln1 + ln2 weights
            + 3 * D * F           # ffn w1 w2 w3 (w2: DxF, not FxD? check)
        )
        + D                       # ln_final
        + V * D                   # lm_head
    )

def activations():
    """Peak activations saved per block (approximate). 8/3 D rounded up to F."""
    # Per block (one layer):
    #   - ln1 input/output: B*T*D
    #   - Q, K, V projections output: 3 * B*T*D
    #   - QK^T result: B*H*T*T
    #   - softmax output: B*H*T*T
    #   - attn * V output: B*H*T*D_h = B*T*D
    #   - O projection output: B*T*D
    #   - ln2 input/output: B*T*D
    #   - W1 output: B*T*F
    #   - SiLU output: B*T*F
    #   - W3 output: B*T*F
    #   - gate * w3 result: B*T*F
    #   - W2 output: B*T*D
    D_h = D // H
    per_block = (
        B*T*D           # ln1
        + 3*B*T*D       # Q, K, V after projections
        + B*H*T*T       # QK^T
        + B*H*T*T       # softmax scores
        + B*T*D         # attn * V
        + B*T*D         # O proj output
        + B*T*D         # ln2
        + B*T*F * 4     # W1, SiLU(gate), W3(silu*W3), gated output
        + B*T*D         # W2 output (needed for next layer's residual)
    )
    # Without activation checkpointing, ALL layers' activations are kept
    # for backward (PyTorch autograd saves every intermediate).
    return L * per_block + B*T*D + B*T*V  # L layers + final ln + lm_head logits

def gradients():
    """Same count as parameters."""
    return params()

def opt_state():
    """AdamW m + v, both same shape as theta."""
    return 2 * params()

def total_memory():
    P = params()
    A = activations()
    G = gradients()
    O = opt_state()
    return (P + A + G + O) * BYTES


# ===== (c) AdamW FLOPs =====
def adamw_flops():
    """FLOPs for one optimizer step (element-wise, per parameter)."""
    # per param: g ready (from backward), compute:
    #   alpha_t: 4 FLOPs (2 pow, 1 sqrt, 2 mult, 1 sub per step, shared - negligible)
    #   weight_decay: 1 mul + 1 sub = 2
    #   m update: 2 mul + 1 add = 3
    #   v update: 2 mul + 1 add = 3
    #   param update: 1 mul + 1 div + 1 sub = 3
    #   total per param ≈ 11
    return 11 * params()

# ===== (d) Training Time =====
def training_hours():
    forward = total_flops_forward  # from previous accounting
    per_step = 3 * forward         # forward + 2*backward
    steps = 400_000
    total = per_step * steps
    return total / (495e12 * 0.5) / 3600  # hours


if __name__ == '__main__':
    # GPT-2 XL
    B = 1
    T = 1024
    D = 1600
    H = 25
    V = 50257
    L = 48
    F = 4288

    # Forward FLOPs from earlier (just the matmul sum, per batch=1)
    # Simplified formula: L*(24*B*T*D*D + 4*B*T*T*D) + 2*B*T*D*V
    total_flops_forward = L*(24*B*T*D*D + 4*B*T*T*D) + 2*B*T*D*V

    P = params()
    mem_bytes = total_memory()
    gb = mem_bytes / 1e9

    print(f"=== (a) Peak Memory (B={B}) ===")
    print(f"  parameters:    {P / 1e9:.4f}B           ({P*BYTES/1e9:.2f} GB)")
    print(f"  activations:   {activations() / 1e9:.4f}B ({activations()*BYTES/1e9:.2f} GB)")
    print(f"  gradients:     {gradients() / 1e9:.4f}B ({gradients()*BYTES/1e9:.2f} GB)")
    print(f"  opt state:     {opt_state() / 1e9:.4f}B ({opt_state()*BYTES/1e9:.2f} GB)")
    print(f"  TOTAL:         {P + activations() + gradients() + opt_state():.1f} params  →  {gb:.2f} GB")

    # (b)
    B_var = 'B'
    P_gb = P * BYTES / 1e9
    A_per_B = activations() * BYTES / 1e9  # activations ∝ B
    G_per_B = gradients() * BYTES / 1e9
    # Actually params don't depend on B, but activations do
    # Peak = (P_all + A*B) * 4
    # A_single = activations_per_batch

    # Let's compute per batch
    B_test = 1
    A_one = activations()  # with B=1
    params_mem = P * BYTES / 1e9
    grad_mem = gradients() * BYTES / 1e9
    opt_mem = opt_state() * BYTES / 1e9
    act_mem_per_batch = A_one * BYTES / 1e9

    print(f"\n=== (b) Memory vs Batch Size ===")
    print(f"  constant (params+grads+opt): {params_mem + grad_mem + opt_mem:.2f} GB")
    print(f"  activations per batch:       {act_mem_per_batch:.4f} GB")
    for B_val in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        total_b = params_mem + grad_mem + opt_mem + act_mem_per_batch * B_val
        print(f"  B={B_val:4d}  peak={total_b:.2f} GB")
        if total_b > 80:
            print(f"           ^^^ exceeds 80 GB at B={B_val}")
            break

    print(f"\n=== (c) AdamW FLOPs per step ===")
    print(f"  {P / 1e9:.4f}B model parameters × ~11 FLOPs each")
    print(f"  {adamw_flops() / 1e9:.2f} GFLOPs per optimizer step (negligible vs forward matmul)")

    print(f"\n=== (d) Training Time ===")
    print(f"  Forward FLOPs: {total_flops_forward / 1e12:.2f} TFLOPs (B={B})")
    per_step_flops = 3 * total_flops_forward * 1024  # B=1024
    total_flops = per_step_flops * 400_000
    effective_tflops = 495e12 * 0.5
    hours = total_flops / effective_tflops / 3600
    print(f"  Per step (B=1024,3×): {per_step_flops / 1e18:.4f} EFLOPs")
    print(f"  Total (400K steps):   {total_flops / 1e18:.4f} EFLOPs")
    print(f"  Effective TFLOPs:     {effective_tflops / 1e12:.0f} TFLOPs")
    print(f"  Training time:        {hours:.1f} hours = {hours/24:.1f} days")
'''
=== (a) Peak Memory (B=1) ===
  parameters:    1.6405B           (6.56 GB)
  activations:   4.0419B (16.17 GB)
  gradients:     1.6405B (6.56 GB)
  opt state:     3.2809B (13.12 GB)
  TOTAL:         10603695872.0 params  →  42.41 GB

=== (b) Memory vs Batch Size ===
  constant (params+grads+opt): 26.25 GB
  activations per batch:       16.1675 GB
  B=   1  peak=42.41 GB
  B=   2  peak=58.58 GB
  B=   4  peak=90.92 GB
           ^^^ exceeds 80 GB at B=4

=== (c) AdamW FLOPs per step ===
  1.6405B model parameters × ~11 FLOPs each
  18.04 GFLOPs per optimizer step (negligible vs forward matmul)

=== (d) Training Time ===
  Forward FLOPs: 3.51 TFLOPs (B=1)
  Per step (B=1024,3×): 0.0108 EFLOPs
  Total (400K steps):   4309.0373 EFLOPs
  Effective TFLOPs:     248 TFLOPs
  Training time:        4836.2 hours = 201.5 days
'''
