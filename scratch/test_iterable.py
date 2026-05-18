"""Test that encode_iterable preserves tokens across chunk boundaries."""
import sys
sys.path.insert(0, "tests")
from adapters import get_tokenizer
from common import FIXTURES_PATH, gpt2_bytes_to_unicode
import json

# Build a GPT-2 tokenizer the same way tests do
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
with open(FIXTURES_PATH / "gpt2_vocab.json") as f:
    gpt2_vocab = json.load(f)
vocab = {idx: bytes([gpt2_byte_decoder[t] for t in item]) for item, idx in gpt2_vocab.items()}
merges = []  # GPT-2 merges are not needed for a fairness test — both versions use same no-merge setup
# Actually we need the real merges. Load them.
gpt2_merges = []
with open(FIXTURES_PATH / "gpt2_merges.txt") as f:
    for line in f:
        cleaned = line.rstrip()
        if cleaned and len(cleaned.split(" ")) == 2:
            merge_tokens = cleaned.split(" ")
        if len(merge_tokens) == 2:
            a_bytes = bytes([gpt2_byte_decoder[c] for c in merge_tokens[0]])
            b_bytes = bytes([gpt2_byte_decoder[c] for c in merge_tokens[1]])
            gpt2_merges.append((a_bytes, b_bytes))

tokenizer = get_tokenizer(vocab, gpt2_merges, special_tokens=["<|endoftext|>"])

# Chunks that split a GPT-2 pre-token across the boundary.
# " how" should be a single token, but the boundary breaks it.
chunks = ["Hello, ho", "w are you?"]
full_text = "Hello, how are you?"

correct = tokenizer.encode(full_text)
streamed = list(tokenizer.encode_iterable(iter(chunks)))

if correct == streamed:
    print("PASS — encode_iterable matches encode()")
else:
    print("FAIL — encode_iterable differs from encode()")
    print(f"  encode()  : {correct}")
    print(f"  iterable  : {streamed}")
    print(f"  first diff at index: ", end="")
    for i, (c, s) in enumerate(zip(correct, streamed)):
        if c != s:
            print(f"{i} (correct={c}, iterable={s})")
            break
