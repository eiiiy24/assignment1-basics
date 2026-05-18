from cs336_basics.train_bpe import train_bpe
import json
from pathlib import Path
import threading, psutil, time

peak_mb = 0
stop_flag = False

def sample_memory():
    global peak_mb
    process = psutil.Process()
    while not stop_flag:
        total_rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak_mb = max(peak_mb, total_rss / 1024 / 1024)
        time.sleep(1)

if __name__ == '__main__':
    PROJECT_DIR = Path(__file__).parent.parent  # scratch/ 的父目录即 assignment1-basics/
    train_path = PROJECT_DIR / "data" / "owt_train.txt"
    start = time.perf_counter()
    # 开始训练前启动采样线程
    t = threading.Thread(target=sample_memory, daemon=True)
    t.start()
    vocab, merges = train_bpe(
        str(train_path),
        32_000,
        ['<|endoftext|>']
    )
    stop_flag = True
    t.join()
    print(f"Peak memory: {peak_mb:.2f} MB")
    end = time.perf_counter()
    print("Train Time Taken: " + str(end - start) + " seconds")

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))
    print("longest token id:", longest_id)
    print("longest token byte length:", len(longest_token))
    print("longest token bytes:", longest_token)
    print("longest token decoded:", longest_token.decode("utf-8", errors="replace"))

    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    # int key → str key, bytes value → list of int (0-255)
    json_vocab = {}
    for k, v in vocab.items():
        json_vocab[str(k)] = {
            "bytes": list(v),
            "text": v.decode("utf-8", errors="replace")
        }
    with open(output_dir / "owt_vocab.json", "w") as f:
        json.dump(json_vocab, f, ensure_ascii=False, indent=2)

    json_merges = []
    for a, b in merges:
        json_merges.append({
            "token1": {"bytes": list(a), "text": a.decode("utf-8", errors="replace")},
            "token2": {"bytes": list(b), "text": b.decode("utf-8", errors="replace")},
        })

    with open(output_dir / "owt_merges.json", "w") as f:
        json.dump(json_merges, f, ensure_ascii=False, indent=2)

    # 反序列化
    # with open(output_dir / "owt_vocab.json") as f:
    #     json_vocab = json.load(f)
    # vocab = {int(k): bytes(v["bytes"]) for k, v in json_vocab.items()}
    # with open(output_dir / "owt_merges.json") as f:
    #     json_merges = json.load(f)
    # merges = [(bytes(m["token1"]["bytes"]), bytes(m["token2"]["bytes"])) for m in json_merges]
