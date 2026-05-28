import torch, numpy as np, sys, time, platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import load_data
from cs336_basics.gradient_clipping import gradient_clip
from cs336_basics.learning_rate_schedule import get_cos_lr
from einops import rearrange
import wandb

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

context_length = 256
total_steps = 500
T_w = total_steps // 10
T_c = total_steps
lr = 2e-3

for B in [32, 48, 64, 96, 128, 256]:
    print(f"\n=== B={B} ===", flush=True)
    try:
        model = TransformerLM(512, 16, 1344, 10000, 10000, 256, 4, device=device, dtype=torch.float32)
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

        # wandb.init(project="cs336", group="batch_sweep_gpu", name=f"B_{B}_gpu", ...)
        wandb.init(
            project="cs336",
            group="batch_sweep_gpu",
            name=f"B_{B}_3080ti",
            config={"batch_size": B, "context_length": context_length, "lr": lr, "total_steps": total_steps,
                    "env": env_info},
        )

        t0 = time.time()
        for step in range(1, total_steps + 1):
            x, y = load_data(dataset, B, context_length, str(device))
            logits = rearrange(model(x), 'b t v -> (b t) v')
            loss = cross_entropy(logits, rearrange(y, 'b t -> (b t)'))
            loss.backward()
            gradient_clip(model.parameters(), 1.0)
            optimizer.param_groups[0]['lr'] = get_cos_lr(step, lr, 0.0, T_w, T_c)
            optimizer.step()
            optimizer.zero_grad()

            elapsed = time.time() - t0
            tokens = step * B * context_length
            wandb.log({
                "train/loss": loss.item(),
                "time/elapsed_sec": elapsed,
                "train/tokens": tokens,
                "time/tokens_per_sec": tokens / elapsed,
            }, step=step)

            if step % 50 == 0:
                print(f"  step {step:4d}: loss={loss.item():.4f}, tok/s={tokens/elapsed:.0f}", flush=True)

        wandb.log({"final_loss": loss.item()})
        wandb.finish()
        print(f"B={B}: final_loss={loss.item():.4f}", flush=True)

    except RuntimeError as e:
        print(f"B={B}: ERROR - {e}", flush=True)
        wandb.finish()
        if "out of memory" in str(e).lower():
            print(f"  -> OOM at B={B}, stopping sweep", flush=True)
            break

print("\nDone!")
