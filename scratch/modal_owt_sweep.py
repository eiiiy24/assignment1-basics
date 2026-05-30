import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(  # rebuild v2
        "torch~=2.11.0",
        "einops>=0.8",
        "jaxtyping>=0.3",
        "numpy>=2.4",
        "tqdm>=4.67",
        "wandb>=0.25",
    )
    .run_commands("git clone https://github.com/eiiiy24/assignment1-basics.git /app")
)

app = modal.App("cs336-owt")

def train():
    """
      Train on OWT for `steps` steps with given lr/batch_size.
      Returns final train loss.
      """
    import gc
    try:
        import sys
        sys.path.insert(0, "/app")
        import torch
        from cs336_basics.transformer_lm import TransformerLM
        from cs336_basics.adamw import AdamW
        from cs336_basics.cross_entropy import cross_entropy
        from cs336_basics.data_loading import load_data
        import numpy as np
        from cs336_basics.gradient_clipping import gradient_clip
        from cs336_basics.learning_rate_schedule import get_cos_lr
        from einops import rearrange
        import wandb
        import time

        torch.cuda.empty_cache()
        wandb.init(project="cs336", group="modal_sweep")
        lr = wandb.config.lr
        batch_size = wandb.config.batch_size
        wandb.run.name = f"lr_{wandb.config.lr:.1e}_bs_{wandb.config.batch_size}"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        context_length = 256
        total_steps = 500
        T_w = total_steps // 10
        T_c = total_steps

        model = TransformerLM(512, 16, 1344, 10000, 32000, 256, 4, \
                              device=device, dtype=dtype)
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        train_dataset = np.load("/data/owt_train_ids.npy", mmap_mode='r')

        t0 = time.time()
        for step in range(1, total_steps + 1):
            x, y = load_data(train_dataset, batch_size, context_length, str(device))
            logits = model(x)
            logits = rearrange(logits, 'b t v -> (b t) v')
            y = rearrange(y, 'b t -> (b t)')
            loss = cross_entropy(logits, y)
            loss.backward()
            gradient_clip(model.parameters(), 1.0)
            scheduled_lr = get_cos_lr(step, lr, 0.0, T_w, T_c)
            optimizer.param_groups[0]['lr'] = scheduled_lr
            optimizer.step()
            optimizer.zero_grad()

            elapsed = time.time() - t0
            tokens = step * batch_size * context_length
            wandb.log({"train/loss": loss.item(), "optim/lr": scheduled_lr, "tokens": tokens, "tokens_per_sec":
                tokens / elapsed}, step=step)

        wandb.log({"final_loss": loss.item()})
    except Exception as e:
        wandb.log({"error": str(e)})
        print(f"Run failed: {e}")
    finally:
        import torch
        wandb.finish()
        del model, optimizer, train_dataset
        gc.collect()  # 强制 Python 释放死引用
        torch.cuda.empty_cache()  # 清理 PyTorch 缓存
        torch.cuda.reset_peak_memory_stats()

@app.function(gpu="B200", image=image, secrets=[modal.Secret.from_name("WANDB_API_KEY")], volumes={"/data": modal.Volume.from_name("owt-data")})
def run_sweep():
    import wandb
    import os
    os.environ["WANDB_API_KEY"] = "your-api-key"
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "train/loss", "goal": "minimize"},
        "parameters": {
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 2e-2},
            "batch_size": {"values": [64, 96, 128, 192, 256, 384, 512]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="cs336")
    wandb.agent(sweep_id, function=train, count=30)

@app.local_entrypoint()
def main():
    run_sweep.remote()