import regex as re
from collections import Counter, defaultdict
from tlm.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import heapq

class ReverseLex:
    __slots__ = ("pair",)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair
    def __eq__(self, other):
        return self.pair == other.pair
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocab(special_tokens: list[str]) -> dict:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab
    


def split_special_characters(text: str, special_tokens: list[str]) -> list[str]: 
    if special_tokens:
        ordered_specials = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(special) for special in ordered_specials)
        segments = re.split(pattern, text)
    else:
        segments = [text]

    return segments

def count_pretokens(segments: list[str]) -> Counter:
    pretoken_freqs = Counter()
    for segment in segments:
        for match in re.finditer(GPT2_PRETOKENIZE_PATTERN, segment):
            pretoken = match.group()
            if pretoken:
                pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                pretoken_freqs[pretoken_bytes] += 1
    return pretoken_freqs

def process_chunk(input_path: str, special_tokens: list[str], start: int, end: int) -> Counter:

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="ignore")
    segments = split_special_characters(text, special_tokens)
    return count_pretokens(segments)

def process_chunk_wrapper(args):
    return process_chunk(*args)

    
def pretokenize_parallel(input_path, special_tokens):
    num_processes = 15
    num_chunks = 60

    with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    chunk_ranges = zip(boundaries[:-1], boundaries[1:])
    tasks = [(input_path, special_tokens, start, end) for start, end in chunk_ranges]
    pretoken_counter_total = Counter()

    with Pool(processes=num_processes) as pool:
        for partial_counts in pool.imap_unordered(process_chunk_wrapper, tasks):
            pretoken_counter_total.update(partial_counts)

    return pretoken_counter_total




def init_pretoken_store(pretoken_freqs: Counter) -> dict:
    pretoken_store = {}
    pretok_id = 0
    for pretoken, count in pretoken_freqs.items():
        pretoken_store[pretok_id] = [pretoken, count]
        pretok_id += 1
    return pretoken_store
    
                   
def build_pair_stats(pretoken_store: dict) -> tuple[Counter, dict]:
    """ 
    Builds the pair_counts and the pair_to_ids
    """
    pair_counts = Counter()
    pair_to_ids = defaultdict(set)
    for pid, (pretoken, count) in pretoken_store.items():
        pairs = zip(pretoken[:-1], pretoken[1:])
        for pair in pairs:
            pair_to_ids[pair].add(pid)
            pair_counts[pair] += count
    return (pair_counts, pair_to_ids)


def init_heap(pair_counts):
    heap = [(-count, ReverseLex(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def get_best_pair(heap, pair_counts: Counter) -> tuple:
    while heap and pair_counts[heap[0][2]] != -heap[0][0]:
        heapq.heappop(heap)
    _, _, best_pair = heapq.heappop(heap)
    return best_pair

def word_pairs(word):
    """All adjacent pairs in a word, as a plain list (repeats included)."""
    return list(zip(word[:-1], word[1:]))


def apply_merge(best_pair, pretoken_store, pair_counts, pair_to_ids, heap):
    a, b = best_pair
    new_token = a + b

    affected_ids = pair_to_ids[best_pair]  # capture before mutating
    pair_to_ids.pop(best_pair, None)
    pair_counts.pop(best_pair, None)

    touched_pairs = set()  # every pair whose count changed, across all affected ids this call

    for pid in affected_ids:
        old_pretoken, count = pretoken_store[pid]

        # 1. remove old contribution
        for pair in word_pairs(old_pretoken):
            if pair != best_pair:
                pair_to_ids[pair].discard(pid)
                pair_counts[pair] -= count
                touched_pairs.add(pair)

        # 2. build new_pretoken (single pass merge)
        new_pretoken = []
        i = 0
        L = len(old_pretoken)
        while i < L:
            if i < L - 1 and old_pretoken[i] == a and old_pretoken[i + 1] == b:
                new_pretoken.append(new_token)
                i += 2
            else:
                new_pretoken.append(old_pretoken[i])
                i += 1
        new_pretoken = tuple(new_pretoken)

        # 3. add new contribution
        for pair in word_pairs(new_pretoken):
            pair_counts[pair] += count
            pair_to_ids[pair].add(pid)
            touched_pairs.add(pair)

        pretoken_store[pid] = [new_pretoken, count]

    for pair in touched_pairs:
        current_count = pair_counts.get(pair, 0)
        if current_count > 0:
            heapq.heappush(heap, (-current_count, ReverseLex(pair), pair))
        if current_count <= 0:
            pair_counts.pop(pair, None)  
from tqdm import tqdm

def run_bpe_merges(pretoken_freqs: Counter, vocab: dict, vocab_size: int) -> tuple[dict, list]:
    merges = []
    
    pretoken_store = init_pretoken_store(pretoken_freqs)
    pair_counts, pair_to_ids = build_pair_stats(pretoken_store)
    heap = init_heap(pair_counts)

    initial_vocab_size = len(vocab)
    with tqdm(total=vocab_size - initial_vocab_size, desc="BPE merges") as pbar:
        while len(vocab) < vocab_size:
            if not pair_counts:
                break
            best_pair = get_best_pair(heap, pair_counts)
            merges.append((best_pair[0], best_pair[1]))
            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            apply_merge(best_pair, pretoken_store, pair_counts, pair_to_ids, heap)
            pbar.update(1)

    return vocab, merges


def train_bpe(input_path, vocab_size, special_tokens):
    

    """Train a byte-pair encoding (BPE) tokenizer vocabulary and merge list."""
    # Initialize vocab
    merges = []
    pretoken_freq = pretokenize_parallel(input_path, special_tokens)
    vocab = init_vocab(special_tokens)
    
    return run_bpe_merges(pretoken_freq, vocab, vocab_size)



# old implementation 

def train_bpe_slow(input_path, vocab_size, special_tokens):
    """Train a byte-pair encoding (BPE) tokenizer vocabulary and merge list."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Initialize vocab
    pretoken_freqs = Counter()  # {tuple of bytes: count}
    vocab = {}  # token id -> bytes
    merges = []  # list[tuple[bytes, bytes]] representing (token1, token2) merges

    for i in range(256):
        vocab[i] = bytes([i])
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Split text on special tokens, then pre-tokenize on whitespace
    if special_tokens:
        ordered_specials = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(special) for special in ordered_specials)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    for chunk in chunks:
        for pretoken in chunk.split(" "):
            if pretoken:
                pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                pretoken_freqs[pretoken_bytes] += 1

    # Iteratively merge the most frequent adjacent byte pair
    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)
        for pretoken_bytes, count in pretoken_freqs.items():
            for left in range(len(pretoken_bytes) - 1):
                right = left + 1
                pair_counts[(pretoken_bytes[left], pretoken_bytes[right])] += count

        if not pair_counts:
            break

        best_pair, _ = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        first, second = best_pair
        merges.append((first, second))
        vocab[len(vocab)] = first + second

        # Apply the new merge across all pretokens
        new_pretoken_freqs = {}
        for pretoken_bytes, count in pretoken_freqs.items():
            merged_bytes = []
            i = 0
            while i < len(pretoken_bytes):
                if (
                    i < len(pretoken_bytes) - 1
                    and pretoken_bytes[i] == first
                    and pretoken_bytes[i + 1] == second
                ):
                    merged_bytes.append(first + second)
                    i += 2
                else:
                    merged_bytes.append(pretoken_bytes[i])
                    i += 1
            new_pretoken_freqs[tuple(merged_bytes)] = count

        pretoken_freqs = new_pretoken_freqs

    return vocab, merges
