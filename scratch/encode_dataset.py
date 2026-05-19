import numpy as np
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
# 为什么 uint16？ 因为 vocab ≤ 65535（10K 和 32K 都在 uint16 范围），用 uint16 比 int32 省一半存储。

story_tokenizer = Tokenizer.from_files(
    str(PROJECT_DIR / "output" / "tinystories_vocab.json"),
    str(PROJECT_DIR / "output" / "tinystories_merges.json"),
    ["<|endoftext|>"]
)
owt_tokenizer = Tokenizer.from_files(
    str(PROJECT_DIR / "output" / "owt_vocab.json"),
    str(PROJECT_DIR / "output" / "owt_merges.json"),
    ["<|endoftext|>"]
)

def encode_and_save(tokenizer, input_path, output_path):
    with open(input_path) as f:
        text = f.read()
    ids = tokenizer.encode(text)
    np.save(output_path, np.array(ids, dtype=np.uint16))

tasks = [
    (story_tokenizer, "TinyStoriesV2-GPT4-train.txt", "ts_train_ids.npy"),
    (story_tokenizer, "TinyStoriesV2-GPT4-valid.txt", "ts_valid_ids.npy"),
    (owt_tokenizer, "owt_train.txt", "owt_train_ids.npy"),
    (owt_tokenizer, "owt_valid.txt", "owt_valid_ids.npy"),
]
for tokenizer, input_file, output_file in tasks:
    input_path = str(PROJECT_DIR / "data" / input_file)
    output_path = str(PROJECT_DIR / "output" / output_file)
    encode_and_save(tokenizer, input_path, output_path)
