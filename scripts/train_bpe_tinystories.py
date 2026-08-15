import cProfile
import os
import pickle
import pstats
import resource
import time

from tlm.train_bpe import train_bpe


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    t0 = time.perf_counter()

    # --- profiled run ---
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    profiler.disable()
    t1 = time.perf_counter()

    # --- peak RSS since process start (Linux: KB, macOS: bytes) ---
    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        peak_rss_bytes = peak_rss_raw  # macOS reports bytes
    else:
        peak_rss_bytes = peak_rss_raw * 1024  # Linux reports KB
    mem_gb = peak_rss_bytes / (1024**3)

    # --- save profiling stats to file ---
    profile_path = os.path.join(output_dir, "train_bpe_profile.prof")
    profiler.dump_stats(profile_path)
    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    stats_txt_path = os.path.join(output_dir, "train_bpe_profile_top30.txt")
    with open(stats_txt_path, "w") as f:
        stats.stream = f
        stats.print_stats(30)

    # --- save vocab/merges ---
    vocab_path = os.path.join(output_dir, "tinystories_bpe_vocab.pkl")
    merges_path = os.path.join(output_dir, "tinystories_bpe_merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # --- longest token ---
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    elapsed_s = t1 - t0
    elapsed_min = elapsed_s / 60.0
    elapsed_hr = elapsed_s / 3600.0

    print(f"Saved vocab -> {vocab_path}")
    print(f"Saved merges -> {merges_path}")
    print(f"Saved profile (binary) -> {profile_path}")
    print(f"Saved profile (top 30, tottime) -> {stats_txt_path}")
    print(f"Elapsed: {elapsed_s:.2f}s ({elapsed_min:.2f} min, {elapsed_hr:.4f} hr)")
    print(f"Peak RSS: {mem_gb:.2f} GB")
    print(f"Longest token id={longest_id}, bytes_len={len(longest_bytes)}")
    print(f"Longest token (decoded): {longest_str!r}")


if __name__ == "__main__":
    main()