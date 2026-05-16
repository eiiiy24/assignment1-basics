import regex as re
from collections import Counter


# Assignment handout p.9: Problem train_bpe function.
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on a text corpus.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: Maximum final vocabulary size, including the initial byte
            vocabulary, merged tokens, and special tokens.
        special_tokens: Strings to add to the vocabulary. During training, treat
            them as hard boundaries that prevent merges across their spans, and
            exclude them from merge statistics.

    Returns:
        A tuple of:
        - vocab: Mapping from token ID to token bytes.
        - merges: Ordered BPE merges, where each item is the pair of byte tokens
            merged at that training step.
    """
    # Initialize the byte vocabulary.
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    num_merges = vocab_size - len(vocab)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # strip out all special tokens from corpus (or chunk, if using a parallel implementation).
    if special_tokens:
        # with careful use of re.escape since | may occur in the special tokens)
        special_regex = "|".join(re.escape(t) for t in special_tokens)
        train_segments = re.split(special_regex, text)
    else :
        train_segments = [text]

    GPT2_PAT = (
        r"""'(?:[sdmt]|ll|ve|re)"""  # English contractions: 's, 'd, 'm, 't, 'll, 've, 're.
        r"""| ?\p{L}+"""  # Optional leading space followed by one or more Unicode letters.
        r"""| ?\p{N}+"""  # Optional leading space followed by one or more Unicode numbers.
        r"""| ?[^\s\p{L}\p{N}]+"""  # Optional leading space followed by symbols/punctuation.
        r"""|\s+(?!\S)"""  # Whitespace run not followed by a non-whitespace character.
        r"""|\s+"""  # Any other whitespace run.
    )

    # Example: "cat" -> b"cat" -> (b"c", b"a", b"t").
    # For non-ASCII text, "你好" -> b"\xe4\xbd\xa0\xe5\xa5\xbd" -> (b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd').
    raw_counts = Counter()
    for segment in train_segments:
        words = re.findall(GPT2_PAT, segment)
        for word in words:
            word_bytes = word.encode("utf-8")
            key = tuple(bytes([b]) for b in word_bytes)
            raw_counts[key] += 1

    # Example: raw_counts = Counter({
    #     (b"a", b"b", b"c"): 10,
    #     (b"a", b"b", b"d"): 5,
    #     (b"x", b"y"): 7,
    # })

    merges = []
    for _ in range(num_merges):
        pair_counts = Counter()
        for word, count in raw_counts.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += count

        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token

        new_word_counts = Counter()
        for old_word, count in raw_counts.items():
            new_word = []
            i = 0
            while i < len(old_word):
                if i + 1 < len(old_word) and old_word[i] == best_pair[0] and old_word[i + 1] == best_pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            new_word = tuple(new_word)
            new_word_counts[new_word] += count
        raw_counts = new_word_counts

    return vocab, merges





