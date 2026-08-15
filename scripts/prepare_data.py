import argparse
import multiprocessing as mp
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from tlm.pretokenization_example import find_chunk_boundaries
from tlm.tokenizer import Tokenizer

_worker_tokenizer: Tokenizer | None = None
_worker_vocab_path: str | None = None
_worker_merges_path: str | None = None


def _init_worker(vocab_path: str, merges_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])


def _tokenize_chunk(args: tuple[str, int, int]) -> list[int]:
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return _worker_tokenizer.encode(text)


def tokenize_file_parallel(input_path: Path, vocab_path: str, merges_path: str, output_path: Path, num_workers: int) -> None:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
    chunks = [(str(input_path), start, end) for start, end in pairwise(boundaries)]

    t0 = time.perf_counter()
    with mp.Pool(num_workers, initializer=_init_worker, initargs=(vocab_path, merges_path)) as pool:
        chunk_ids = pool.map(_tokenize_chunk, chunks)
    elapsed = time.perf_counter() - t0

    ids = [tid for chunk in chunk_ids for tid in chunk]
    array = np.array(ids, dtype=np.uint16)
    array.tofile(output_path)
    print(f"{input_path.name}: {len(ids):,} tokens -> {output_path} ({elapsed:.1f}s, {len(ids) / elapsed:,.0f} tok/s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid-input", default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab", default="workspace/tinystories_bpe_vocab.pkl")
    parser.add_argument("--merges", default="workspace/tinystories_bpe_merges.pkl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenize_file_parallel(Path(args.valid_input), args.vocab, args.merges, output_dir / "valid.bin", args.num_workers)
    tokenize_file_parallel(Path(args.train_input), args.vocab, args.merges, output_dir / "train.bin", args.num_workers)


if __name__ == "__main__":
    main()
