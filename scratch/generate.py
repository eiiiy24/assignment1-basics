import torch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.checkpointing import load_checkpoint
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
from cs336_basics.adamw import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import decode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
vocab_size = 10_000
context_length = 256
d_model = 512
d_ff = 1344
rope_theta = 10_000
num_layers = 4
num_heads = 16

model = TransformerLM(d_model, num_heads, d_ff, rope_theta, vocab_size, context_length, num_layers,\
                      device=device, dtype=dtype)
optimizer = AdamW(model.parameters())
state = load_checkpoint(str(PROJECT_DIR / 'runs/ts_lr2e-3_b32_T256_327M/checkpoints' / 'checkpoint_40000.pt'), model, optimizer)
tokenizer = Tokenizer.from_files(
    str(PROJECT_DIR / 'output' / 'tinystories_vocab.json'),
    str(PROJECT_DIR / 'output' / 'tinystories_merges.json'),
    ["<|endoftext|>"]
)

prompt = "Once upon a time"
text = decode(model, tokenizer, prompt, max_tokens_num=256, temperature=0.5, top_p=0.95)
print(prompt + text)