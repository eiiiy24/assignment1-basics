import torch, numpy as np
from pathlib import Path
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import load_data
from cs336_basics.gradient_clipping import gradient_clip
from cs336_basics.learning_rate_schedule import get_cos_lr
from einops import rearrange
import wandb
import sys, platform

env_info = {
    "python": sys.version.split()[0],
    "pytorch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "cpu_cores": torch.get_num_threads(),
    "os": platform.platform(),
}

print("=== Experiment Environment ===")
for k, v in env_info.items():
    print(f"  {k}: {v}")
print("=" * 30, "\n")

PROJECT_DIR = Path(__file__).parent.parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = np.load(str(PROJECT_DIR / "output" / "ts_train_ids.npy"), mmap_mode='r')

vocab_size = 10_000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10_000
num_layers = 4
num_heads = 16
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
total_steps = 500
batch_size = 32
max_norm = 1.0
alpha_max = 5e-3
alpha_min = 0.0
T_w = 0.1 * total_steps
T_c = total_steps

for lr_val in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1, 2, 5, 10, 20, 50]:
    model = TransformerLM(512, 16, 1344, 10000, 10000, 256, 4, device=device, dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=lr_val, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    wandb.init(
        project="cs336",
        group="lr_sweep_with_clip",
        name=f"lr_{lr_val}",
        config={
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "batch_size": batch_size,
            "total_steps": total_steps,
            "alpha_max": alpha_max,
            "alpha_min": alpha_min,
            "T_w": T_w,
            "T_c": T_c,
            "max_norm": max_norm,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "total_tokens": batch_size * total_steps * context_length,
            "env": env_info
        })

    for step in range(1, total_steps + 1):
        x, y = load_data(dataset, batch_size, context_length, str(device))
        logits = rearrange(model(x), 'b t v -> (b t) v')
        loss = cross_entropy(logits, rearrange(y, 'b t -> (b t)'))
        loss.backward()
        gradient_clip(model.parameters(), max_norm)
        lr = get_cos_lr(step, lr_val, 0.0, T_w, T_c)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({
            "train/loss": loss.item(),
            "optim/lr": lr,
        }, step=step)
        if step % 50 == 0:
            print(f"  lr={lr_val} step={step:4d} loss={loss.item():.4f}")

    wandb.log({"final_loss": loss.item()})
    wandb.finish()
    print(f"lr={lr_val}: final_loss={loss.item():.4f}")
