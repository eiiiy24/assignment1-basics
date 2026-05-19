import regex as re
import random
from cs336_basics.tokenizer import Tokenizer
import time

random.seed(42)
with open("../data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
    tiny_stories = f.read()
    all_stories = re.split(re.escape("<|endoftext|>"), tiny_stories)
    random.shuffle(all_stories)
    ten_stories = all_stories[:10]

with open("../data/owt_valid.txt", "r") as f:
    owts = f.read()
    all_owts = re.split(re.escape("<|endoftext|>"), owts)
    random.shuffle(all_owts)
    ten_owts = all_owts[:10]

story_tokenizer = Tokenizer.from_files(
    "../output/tinystories_vocab.json",
    "../output/tinystories_merges.json",
    ["<|endoftext|>"]
)

owt_tokenizer = Tokenizer.from_files(
    "../output/owt_vocab.json",
    "../output/owt_merges.json",
    ["<|endoftext|>"]
)

total_ids = 0
total_bytes = 0
for i, tiny_story in enumerate(ten_stories):
    tiny_story_bytes = tiny_story.encode("utf-8")
    ids = story_tokenizer.encode(tiny_story)
    compression_ratio = len(tiny_story_bytes) / len(ids)
    print(f"Compression ratio of story {i}: {compression_ratio:.2f} bytes / token")
    total_ids += len(ids)
    total_bytes += len(tiny_story_bytes)
print(f"Avg Compression ratio of tiny_stories: {total_bytes / total_ids:.2f} bytes / token")

total_ids = 0
total_bytes = 0
for i, owt in enumerate(ten_owts):
    owt_bytes = owt.encode("utf-8")
    ids = owt_tokenizer.encode(owt)
    compression_ratio = len(owt_bytes) / len(ids)
    print(f"Compression ratio of owt {i}: {compression_ratio:.2f} bytes / token")
    total_ids += len(ids)
    total_bytes += len(owt_bytes)
print(f"Avg Compression ratio of owts: {total_bytes / total_ids:.2f} bytes / token")

print("===================================================")
total_ids = 0
total_bytes = 0
for i, tiny_story in enumerate(ten_stories):
    tiny_story_bytes = tiny_story.encode("utf-8")
    ids = owt_tokenizer.encode(tiny_story)
    compression_ratio = len(tiny_story_bytes) / len(ids)
    print(f"Compression ratio of story {i} in owt_tokenizer: {compression_ratio:.2f} bytes / token")
    total_ids += len(ids)
    total_bytes += len(tiny_story_bytes)
print(f"Avg Compression ratio of tiny_stories in owt_tokenizer: {total_bytes / total_ids:.2f} bytes / token")

total_ids = 0
total_bytes = 0
for i, owt in enumerate(ten_owts):
    owt_bytes = owt.encode("utf-8")
    ids = story_tokenizer.encode(owt)
    compression_ratio = len(owt_bytes) / len(ids)
    print(f"Compression ratio of owt {i} in story_tokenizer: {compression_ratio:.2f} bytes / token")
    total_ids += len(ids)
    total_bytes += len(owt_bytes)
print(f"Avg Compression ratio of owts in story_tokenizer: {total_bytes / total_ids:.2f} bytes / token")

'''
平均值：
  ┌─────────────────────┬─────────┬─────────────┐
  │                     │ TS 文档 │  OWT 文档   │
  ├─────────────────────┼─────────┼─────────────┤
  │ TS tokenizer (10K)  │ 4.10    │ 3.12 ← 大跌 │
  ├─────────────────────┼─────────┼─────────────┤
  │ OWT tokenizer (32K) │ 4.02    │ 4.24        │
  └─────────────────────┴─────────┴─────────────┘

  两个结论：
  1. TS tokenizer 编 OWT 文档，压缩率从 4.24 → 3.12，跌 26%。因为 10K vocab 是童书训练出来的，OWT 里的网文词汇、URL、代码都没有对应的 merge  
  token，只能拆细。
  2. OWT tokenizer 编 TS 童书几乎不降（4.10 → 4.02），因为 32K vocab 够大，童书里的词基本都覆盖了。
  所以 tokenizer 和数据同域则好，跨域则差。
'''

print("===================================================")
total_bytes = 0
for story in all_stories:
    total_bytes += len(story.encode("utf-8"))

start = time.perf_counter()
for story in all_stories:
    ids = story_tokenizer.encode(story)
end = time.perf_counter()

print(f"Throughput of story_tokenizer: {total_bytes / (end - start):.2f} bytes / second")

total_bytes = 0
for owt in all_owts:
    total_bytes += len(owt.encode("utf-8"))

start = time.perf_counter()
for owt in all_owts:
    ids = owt_tokenizer.encode(owt)
end = time.perf_counter()

print(f"Throughput of owt_tokenizer: {total_bytes / (end - start):.2f} bytes / second")
'''
Throughput of story_tokenizer: 1104264.27 bytes / second
Throughput of owt_tokenizer: 863406.46 bytes / second
'''