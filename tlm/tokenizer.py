import os
import pickle
import regex as re
from typing import Iterable, Iterator
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """
    Byte-level BPE tokenizer compatible with GPT-2-style pre-tokenization.
    Vocab maps token_id -> bytes. Merges are list[(bytes, bytes)] in creation order.
    """

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab, self.merges, self.special_tokens = vocab, merges, special_tokens or []
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.vocab_bytes_to_id = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        result = []
        # split chunks by special character
        segments = self.split_special_characters(text)
        # for each split chunk pretokenize spliting by regex pattern
        for segment in segments:
             if segment in self.special_tokens:
                 result.append(self.vocab_bytes_to_id[segment.encode("utf-8")])
                 continue
             for match in re.finditer(GPT2_PRETOKENIZE_PATTERN, segment):
                pretoken = match.group()
                pretoken_bytes = list(bytes([b]) for b in pretoken.encode("utf-8"))
                merged = self.apply_merge(pretoken_bytes)
                result.extend(self.vocab_bytes_to_id[piece] for piece in merged)
        return result 

        # for each split merge based on order of merges in the bpe training 
        # convert the merged bytes into the vocab ids 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
             for token_id in self.encode(string):
                  yield token_id
    
    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.vocab[tid] for tid in ids]
        all_bytes = b"".join(byte_chunks)
        return all_bytes.decode("utf-8", errors="replace")


    # helpers:
    def apply_merge(self, pretoken_bytes: list[bytes]) -> list[bytes]:
        res = pretoken_bytes
        while len(res) > 1:
            pairs = zip(res[:-1], res[1:])
            candidates = [p for p in pairs if p in self.merge_ranks]
            if not candidates:
                break
            best_pair = min(candidates, key=lambda p: self.merge_ranks[p])
            res = self.merge_pair(res, best_pair)
            

        return res

    def merge_pair(self, pretoken_bytes, best_pair):
        new_pretoken_bytes = []
        a, b = best_pair
        i = 0 
        L = len(pretoken_bytes)
        while i < L:
            if i < L - 1 and pretoken_bytes[i] == a and pretoken_bytes[i + 1] == b:
                            new_pretoken_bytes.append(a + b)
                            i += 2
            else:
                new_pretoken_bytes.append(pretoken_bytes[i])
                i += 1
        return new_pretoken_bytes
            
        
    
    def split_special_characters(self, text: str) -> list[str]: 
        if not self.special_tokens:
            return [text]
        ordered_specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(special) for special in ordered_specials)
        segments = re.split(f"({pattern})", text)
        return [seg for seg in segments if seg != ""]  


