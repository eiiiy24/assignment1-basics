import regex as re
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import os
from tqdm import tqdm


def pre_tokenize_chunk(args):
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # strip out all special tokens from corpus (or chunk, if using a parallel implementation).
        if special_tokens:
            # with careful use of re.escape since | may occur in the special tokens)
            special_regex = "|".join(re.escape(t) for t in special_tokens)
            train_segments = re.split(special_regex, chunk)
        else:
            train_segments = [chunk]

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
            for match in re.finditer(GPT2_PAT, segment):
                word = match.group()
                word_bytes = word.encode("utf-8")
                key = tuple(bytes([b]) for b in word_bytes)
                raw_counts[key] += 1

        return raw_counts



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
    raw_counts = Counter()
    with open(input_path, "rb") as f:
        num_processes = os.cpu_count()
        # num_processes = 12
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        task_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            task_list.append((input_path, start, end, special_tokens))
        with Pool(num_processes) as pool:
            results = pool.imap_unordered(pre_tokenize_chunk, task_list)
            counts_list = list(tqdm(results, total=len(task_list), desc="Pre-tokenizing"))
        for counts in counts_list:
            raw_counts.update(counts)

    # Example: raw_counts = Counter({
    #     (b"a", b"b", b"c"): 10,
    #     (b"a", b"b", b"d"): 5,
    #     (b"x", b"y"): 7,
    # })

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, count in raw_counts.items():
        for pair in word_bytes_to_pairs(word):
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    merges = []
    for _ in tqdm(range(num_merges), desc="BPE merges"):
        # pair_counts = Counter()
        # for word, count in raw_counts.items():
        #     for pair in word_bytes_to_pairs(word):
        #         pair_counts[pair] += count

        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_token

        affected_words = list(pair_to_words[best_pair])
        for old_word in affected_words:
            count = raw_counts[old_word]
            # 重复 pair 会重复 remove，先统计 old_word 内部每个 pair 出现了几次
            old_word_pair_counts = Counter(word_bytes_to_pairs(old_word)) # (a, a, a, a, a) -> {(a, a): 4}
            for old_pair, pair_occurrences in old_word_pair_counts.items():
                pair_counts[old_pair] -= count * pair_occurrences
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                pair_to_words[old_pair].remove(old_word) # 该操作就只执行一次（因为 old_word_pair_counts 字典的 key 唯一）

            new_word = merge_word(old_word, best_pair)
            new_word_pair_counts = Counter(word_bytes_to_pairs(new_word))
            for new_pair, pair_occurrences in new_word_pair_counts.items():
                pair_counts[new_pair] += count * pair_occurrences
                pair_to_words[new_pair].add(new_word)

            # 这里应该累加而不是覆盖，例如{
            #     (b"a", b"b"): 2,
            #     (b"ab",): 3,
            # } -> (b"ab",): 5（累加） 而不是 (b"ab",): 2（覆盖）
            old_count = raw_counts.pop(old_word)
            raw_counts[new_word] += old_count

        # new_word_counts = Counter()
        # for old_word, count in raw_counts.items():
        #     new_word = merge_word(old_word, best_pair)
        #     new_word_counts[new_word] += count
        # raw_counts = new_word_counts

    return vocab, merges


# Merge non-overlapping occurrences from left to right, e.g. (a,a,a,a,a) -> (aa,aa,a).
def merge_word(
    word: tuple[bytes, ...],
    best_pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    merged_token = best_pair[0] + best_pair[1]
    res = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
            res.append(merged_token)
            i += 2
        else:
            res.append(word[i])
            i += 1
    return tuple(res)


# Example: (b"a", b"b", b"c") -> [(b"a", b"b"), (b"b", b"c")]
def word_bytes_to_pairs(
    word_bytes: tuple[bytes, ...],
) -> list[tuple[bytes, bytes]]:
    res = []
    for i in range(len(word_bytes) - 1):
        res.append((word_bytes[i], word_bytes[i + 1]))
    return res
