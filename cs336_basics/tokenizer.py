from collections.abc import Iterable, Iterator
import json
import regex as re


class Tokenizer(object):
    """A byte-level BPE tokenizer that encodes text into token IDs and decodes IDs back to text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally)
        a list of special tokens.

        Args:
            vocab: Mapping from integer token ID to token bytes.
            merges: Ordered list of BPE merges, each a tuple of bytes (token1, token2).
            special_tokens: Strings that will never be split into multiple tokens.
                Appended to vocab if not already present.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        # bytes → id
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.pair_to_rank = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Construct and return a Tokenizer from a serialized vocabulary and list of merges,
        in the same format output by BPE training.

        Args:
            vocab_filepath: Path to a JSON file containing the vocabulary.
            merges_filepath: Path to a JSON file containing the list of merges.
            special_tokens: Strings that will never be split into multiple tokens.

        Returns:
            A Tokenizer instance.
        """
        with open(vocab_filepath) as f:
            json_vocab = json.load(f)
        vocab = {int(k): bytes(v["bytes"]) for k, v in json_vocab.items()}

        with open(merges_filepath) as f:
            json_merges = json.load(f)
        merges = [(bytes(m["token1"]["bytes"]), bytes(m["token2"]["bytes"])) for m in json_merges]

        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Steps: pre-tokenize with the GPT-2 regex, split on special tokens,
        apply BPE merges in order of creation, then look up each merged token
        in the vocabulary.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        # Example. "hello<|endoftext|>world" -> [hello, <|endoftext|>, world]
        # 先匹配长的特殊 token，否则若有特殊 token 是另一特殊 token 的子词，可能会把长的特殊 token 拆掉
        if self.special_tokens:
            sort_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_regex = '(' + '|'.join(re.escape(t) for t in sort_tokens) + ')'
            segments = re.split(special_regex, text)
        else:
            segments = [text]

        ids = []
        for segment in segments:
            if segment in self.special_tokens:
                ids.append(self.token_to_id[segment.encode("utf-8")])
            else:
                GPT2_PAT = (
                    r"""'(?:[sdmt]|ll|ve|re)"""  # English contractions: 's, 'd, 'm, 't, 'll, 've, 're.
                    r"""| ?\p{L}+"""  # Optional leading space followed by one or more Unicode letters.
                    r"""| ?\p{N}+"""  # Optional leading space followed by one or more Unicode numbers.
                    r"""| ?[^\s\p{L}\p{N}]+"""  # Optional leading space followed by symbols/punctuation.
                    r"""|\s+(?!\S)"""  # Whitespace run not followed by a non-whitespace character.
                    r"""|\s+"""  # Any other whitespace run.
                )
                for m in re.finditer(GPT2_PAT, segment):
                    word = m.group()
                    ids.extend(self._encode_token(word))
        return ids

    def _encode_token(self, word: str) -> list[int]:
        if word in self.special_tokens:
            return [self.token_to_id[word.encode("utf-8")]]
        word_bytes = word.encode("utf-8")
        word_bytes_list = [bytes([b]) for b in word_bytes]
        if len(word_bytes_list) == 1:
            return [self.token_to_id[word_bytes_list[0]]]
        while len(word_bytes_list) >= 2:
            prior_pair = None
            min_rank = float("inf")
            for i in range(len(word_bytes_list) - 1):
                pair = (word_bytes_list[i], word_bytes_list[i + 1])
                if pair in self.pair_to_rank and self.pair_to_rank[pair] < min_rank:
                    min_rank = self.pair_to_rank[pair]
                    prior_pair = pair
            if prior_pair is None:
                break
            new_word_bytes_list = []
            i = 0
            while i < len(word_bytes_list):
                if i + 1 < len(word_bytes_list) and (word_bytes_list[i], word_bytes_list[i + 1]) == (prior_pair[0], prior_pair[1]):
                    new_word_bytes_list.append(prior_pair[0] + prior_pair[1])
                    i += 2
                else:
                    new_word_bytes_list.append(word_bytes_list[i])
                    i += 1
            word_bytes_list = new_word_bytes_list

        return [self.token_to_id[b] for b in word_bytes_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a file handle), lazily yield token IDs.

        Processes the input in manageable chunks for constant memory usage,
        ensuring tokens do not cross chunk boundaries.

        Args:
            iterable: An iterable yielding strings (e.g., lines from a file).

        Yields:
            Integer token IDs, one at a time.
        """
        # uv run python scratch/test_iterable.py 专门测试了以下第二种写法的正确性
        # for chunk in iterable:
        #     yield from self.encode(chunk) # 不能简写，可能有单词跨 chunk

        GPT2_PAT = (
            r"""'(?:[sdmt]|ll|ve|re)"""  # English contractions: 's, 'd, 'm, 't, 'll, 've, 're.
            r"""| ?\p{L}+"""  # Optional leading space followed by one or more Unicode letters.
            r"""| ?\p{N}+"""  # Optional leading space followed by one or more Unicode numbers.
            r"""| ?[^\s\p{L}\p{N}]+"""  # Optional leading space followed by symbols/punctuation.
            r"""|\s+(?!\S)"""  # Whitespace run not followed by a non-whitespace character.
            r"""|\s+"""  # Any other whitespace run.
        )

        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if self.special_tokens:
                sort_tokens = sorted(self.special_tokens, key=len, reverse=True)
                special_regex = '(' + '|'.join(re.escape(t) for t in sort_tokens) + ')'
                segments = re.split(special_regex, buffer)
            else:
                segments = [buffer]

            for segment in segments[:-1]:
                if segment in self.special_tokens:
                    yield self.token_to_id[segment.encode("utf-8")]
                else:
                    for m in re.finditer(GPT2_PAT, segment):
                        word = m.group()
                        for id in self._encode_token(word):
                            yield id
            # 保留最后一个可能被截断的 segment
            buffer = segments[-1]
        # 剩余的尾部
        if buffer:
            for segment in re.split(special_regex, buffer) if self.special_tokens else [buffer]:
                if segment in self.special_tokens:
                    yield self.token_to_id[segment.encode("utf-8")]
                else:
                    for m in re.finditer(GPT2_PAT, segment):
                        for id in self._encode_token(m.group()):
                            yield id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back into text.

        Looks up each ID's bytes in the vocabulary, concatenates them,
        and decodes the result as UTF-8. Malformed byte sequences are
        replaced with the Unicode replacement character U+FFFD.

        Args:
            ids: List of integer token IDs to decode.

        Returns:
            Decoded string.
        """
        decoded_bytes = b""
        for id in ids:
            decoded_bytes += self.vocab[id]
        # 最后一次性解码：中文 UTF-8 编码 "牛" = 3 字节 b'\xe7\x89\x9b'。如果 BPE 只合并了两个字节如 b'\xe7\x89'，单独 decode 这个 token 会失败
        # errors="replace" 处理所有 token 拼起来后仍然存在的非法字节。
        return decoded_bytes.decode("utf-8", errors="replace")