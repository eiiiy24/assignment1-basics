import torch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import load_data
import numpy as np
from pathlib import Path
from cs336_basics.gradient_clipping import gradient_clip
from cs336_basics.learning_rate_schedule import get_cos_lr
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from einops import rearrange
from tqdm import tqdm
import wandb
import time

PROJECT_DIR = Path(__file__).parent.parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
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
total_steps = 40000
batch_size = 32
max_norm = 1.0
alpha_max = 2e-3
alpha_min = 0.0
T_w = 0.1 * total_steps
T_c = total_steps
run_name = "ablation_nope_lr2e-3"
wandb.init(
    project="cs336",
    group="ablation_nope",
    name=run_name,
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

    })

model = TransformerLM(d_model, num_heads, d_ff, rope_theta, vocab_size, context_length, num_layers, \
                      use_rope=False, device=device, dtype=dtype)
model = torch.compile(model)
optimizer = AdamW(model.parameters(), lr=alpha_max, betas=betas, eps=eps, weight_decay=weight_decay)
train_dataset = np.load(str(PROJECT_DIR / "output" / "ts_train_ids.npy"), mmap_mode='r')
valid_dataset = np.load(str(PROJECT_DIR / "output" / "ts_valid_ids.npy"), mmap_mode='r')

checkpoint_dir = PROJECT_DIR / "runs" / run_name / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

pbar = tqdm(range(1, total_steps + 1), desc="Training")
start_time = time.time()
for step in pbar:
    x, y = load_data(train_dataset, batch_size, context_length, str(device))
    logits = model(x)
    logits = rearrange(logits, 'b t v -> (b t) v')
    y = rearrange(y, 'b t -> (b t)')
    loss = cross_entropy(logits, y)
    loss.backward()
    gradient_clip(model.parameters(), max_norm)
    lr = get_cos_lr(step, alpha_max, alpha_min, T_w, T_c)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.step()
    optimizer.zero_grad()

    elapsed = time.time() - start_time
    tokens = step * batch_size * context_length
    wandb.log({
        "train/loss": loss.item(),
        "optim/lr": lr,
        "time/elapsed_sec": elapsed,
        "train/tokens": tokens,
        "time/tokens_per_sec": tokens / elapsed,
    }, step=step)
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for _ in range(10):
                vx, vy = load_data(valid_dataset, batch_size, context_length, str(device))
                vlogits = model(vx)
                vlogits = rearrange(vlogits, 'b t v -> (b t) v')
                vy = rearrange(vy, 'b t -> (b t)')
                total_val_loss += cross_entropy(vlogits, vy).item()
            val_loss = total_val_loss / 10
        wandb.log({
            "val/loss": val_loss,
            "time/elapsed_sec": elapsed,
            "train/tokens": tokens,
        }, step=step)
        save_checkpoint(model, optimizer, step, str(checkpoint_dir / f"checkpoint_{step}.pt"))
        # 只保留最近 5 个
        all_ckpts = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for old in all_ckpts[:-5]:
            old.unlink()
        model.train()
    pbar.set_postfix(step=f"{step}", lr=f"{lr}", loss=f"{loss.item()}")
