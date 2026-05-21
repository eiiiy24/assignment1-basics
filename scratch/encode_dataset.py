import numpy as np
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
from tqdm import tqdm
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
    (story_tokenizer, "TinyStoriesV2-GPT4-valid.txt", "ts_valid_ids.npy"),
    (owt_tokenizer, "owt_valid.txt", "owt_valid_ids.npy"),
    (story_tokenizer, "TinyStoriesV2-GPT4-train.txt", "ts_train_ids.npy"),
    (owt_tokenizer, "owt_train.txt", "owt_train_ids.npy"),
]
# for tokenizer, input_file, output_file in tasks:
#     input_path = str(PROJECT_DIR / "data" / input_file)
#     output_path = str(PROJECT_DIR / "output" / output_file)
#     encode_and_save(tokenizer, input_path, output_path)

for tokenizer, input_file, output_file in tasks:
    token_num = 0
    input_path = str(PROJECT_DIR / "data" / input_file)
    with open(input_path) as f:
        ids_iterator = tokenizer.encode_iterable(f)
        for token_id in tqdm(ids_iterator):
            token_num += 1
        print(f"input_path: {input_path}, token_num: {token_num}")

    # /home/cs336/assignment1-basics/.venv/bin/python /home/cs336/assignment1-basics/scratch/encode_dataset.py
    # 5465883it [00:22, 239552.67it/s]
    # input_path: /home/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt, token_num: 5465883
    # 100%|██████████| 5465883/5465883 [00:23<00:00, 233724.48it/s]
    # 66401098it [06:14, 177072.41it/s]
    # input_path: /home/cs336/assignment1-basics/data/owt_valid.txt, token_num: 66401098
    # 100%|██████████| 66401098/66401098 [05:59<00:00, 184575.52it/s]
    # 541229347it [37:11, 242526.72it/s]
    # input_path: /home/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt, token_num: 541229347
    # 100%|██████████| 541229347/541229347 [37:25<00:00, 241073.86it/s]
    # 2727120452it [4:24:14, 172004.94it/s]
    # input_path: /home/cs336/assignment1-basics/data/owt_train.txt, token_num: 2727120452
    # 100%|██████████| 2727120452/2727120452 [4:39:26<00:00, 162651.80it/s]

    with open(input_path) as f:
        ids_iterator = tokenizer.encode_iterable(f)
        output_path = str(PROJECT_DIR / "output" / output_file)
        arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_num,))
        pos = 0
        buffer = []
        max_len = 1_000_000
        for token_id in tqdm(ids_iterator, total=token_num):
            buffer.append(token_id)
            if len(buffer) >= max_len:
                chunk = np.array(buffer, dtype=np.uint16)
                arr[pos:pos + len(chunk)] = chunk
                pos += len(chunk)
                buffer = []
        if buffer:
            chunk = np.array(buffer, dtype=np.uint16)
            arr[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
            buffer = []
        arr.flush()
        assert pos == token_num