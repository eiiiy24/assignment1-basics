from collections import Counter, defaultdict
from cs336_basics.train_bpe import word_bytes_to_pairs, merge_word

def run_case(name, raw_counts):
    std_counts, std_merges, std_history = std(raw_counts.copy())
    unknown_counts, unknown_merges, unknown_history = unknown(raw_counts.copy())

    assert unknown_counts == std_counts, (
        f"{name}: raw_counts differ\n"
        f"std={std_counts}\n"
        f"unknown={unknown_counts}"
    )
    assert unknown_merges == std_merges, (
        f"{name}: merges differ\n"
        f"std={std_merges}\n"
        f"unknown={unknown_merges}"
    )
    assert unknown_history == std_history, (
        f"{name}: history differs\n"
        f"std={std_history}\n"
        f"unknown={unknown_history}"
    )

    print(f"\033[32mPASS\033[0m {name}")

def std(raw_counts):
    history = []
    merges = []
    for i in range(100):
        pair_counts = Counter()
        for word, count in raw_counts.items():
            for pair in word_bytes_to_pairs(word):
                pair_counts[pair] += count

        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)

        new_word_counts = Counter()
        for old_word, count in raw_counts.items():
            new_word = merge_word(old_word, best_pair)
            new_word_counts[new_word] += count
        raw_counts = new_word_counts

        print("====第{i}轮=====".format(i=i))
        print(pair_counts)
        print(raw_counts)
        history.append(pair_counts.copy())
        history.append(raw_counts.copy())
    print("merges =", merges)
    return raw_counts, merges, history


def unknown(raw_counts):
    history = []
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, count in raw_counts.items():
        for pair in word_bytes_to_pairs(word):
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    merges = []
    for i in range(100):
        if not pair_counts:
            break
        print("====第{i}轮=====".format(i=i))
        print(pair_counts)
        history.append(pair_counts.copy())
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)

        affected_words = list(pair_to_words[best_pair])
        for old_word in affected_words:
            count = raw_counts[old_word]
            # 重复 pair 会重复 remove，先统计 old_word 内部每个 pair 出现了几次
            old_word_pair_counts = Counter(word_bytes_to_pairs(old_word))
            for old_pair, pair_occurrences in old_word_pair_counts.items():
                pair_counts[old_pair] -= count * pair_occurrences
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                pair_to_words[old_pair].remove(old_word)

            new_word = merge_word(old_word, best_pair)
            new_pairs = word_bytes_to_pairs(new_word)
            for new_pair in new_pairs:
                pair_counts[new_pair] += count
                pair_to_words[new_pair].add(new_word)

            # 这里应该累加而不是覆盖，例如{
            #     (b"a", b"b"): 2,
            #     (b"ab",): 3,
            # } -> (b"ab",): 5（累加） 而不是 (b"ab",): 2（覆盖）
            old_count = raw_counts.pop(old_word)
            raw_counts[new_word] += old_count
        print(raw_counts)
        history.append(raw_counts.copy())
    print("merges =", merges)
    return raw_counts, merges, history

if __name__ == "__main__":
    cases = {
        "shared-ab": Counter({
            (b"a", b"b", b"c"): 10,
            (b"a", b"b", b"d"): 5,
            (b"x", b"y"): 7,
        }), # 测试基础多词共享 (a,b)。
        "repeated-aa": Counter({
            (b"a", b"a", b"a", b"a", b"a"): 1,
        }), # 测试重复 pair，避免 pair_to_words.remove 重复删除。
        "stale-xa-index": Counter({
            (b"x", b"a", b"b"): 1,
            (b"x", b"a", b"c"): 1,
        }), # 测试旧 pair (x,a) 仍在别的词里，不能错误删除索引。
        "merge-into-existing": Counter({
            (b"a", b"b"): 2,
            (b"ab",): 3,
        }), # 测试合并后 new word 已存在时必须累加，不能覆盖。
        "repeated-ab-with-count": Counter({
            (b"a", b"b", b"a", b"b", b"c"): 4,
        }), # 测试同一个 pair 在同一个 word 内出现多次，并且带频率。
        "single-token-word": Counter({
            (b"a",): 10,
            (b"b", b"c"): 2,
        }), # 它测试“长度为 1 的 word 不产生 pair”。
    }
    for name, raw_counts in cases.items():
        run_case(name, raw_counts)

    # for _, raw_counts in cases.items():
    #     std_counts, std_merges, std_history = std(raw_counts.copy())
    #     unknown_counts, unknown_merges, unknown_history = unknown(raw_counts.copy())
    #
    #     assert unknown_counts == std_counts
    #     assert unknown_merges == std_merges
    #     assert unknown_history == std_history
