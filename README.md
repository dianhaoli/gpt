# tlm

A ~25M-parameter decoder-only transformer, trained from scratch on TinyStories, using a custom byte-level BPE tokenizer. Everything is self-contained: no assignment scaffolding, no test harness to satisfy — just the model, the data pipeline, and the training/inference scripts.

## Repo layout

```
transformer/
├── pyproject.toml
├── workspace/
│   ├── tinystories_bpe_vocab.pkl     # trained tokenizer vocab (10,000 tokens)
│   └── tinystories_bpe_merges.pkl    # trained tokenizer merge rules
├── data/                              # you populate this (see "Getting data onto the box")
├── tlm/                                # the package
│   ├── tokenizer.py                   # BPE tokenizer: encode/decode
│   ├── train_bpe.py                   # trains a BPE vocab from raw text (already done — vocab is in workspace/)
│   ├── pretokenization_example.py     # find_chunk_boundaries, used to split files for parallel processing
│   ├── generate.py                    # sampling logic: temperature / top-k / top-p, KV-cache decode loop
│   ├── model/
│   │   ├── config.py                  # TransformerConfig dataclass
│   │   ├── rmsnorm.py                 # RMSNorm
│   │   ├── rope.py                    # rotary position embeddings
│   │   ├── attention.py               # causal self-attention (flash/SDPA), KV-cache support
│   │   ├── feedforward.py             # SwiGLU MLP
│   │   ├── block.py                   # one pre-norm transformer block
│   │   └── transformer.py             # TransformerLM: embed -> N blocks -> final norm -> lm_head
│   └── train/
│       ├── data.py                    # memmap dataset, get_batch, BatchPrefetcher (background loading thread)
│       ├── optim.py                   # AdamW with decay/no-decay param groups, cosine LR schedule
│       ├── checkpoint.py              # save/load/resume
│       └── train.py                   # the training loop (entry point: `python -m tlm.train.train`)
└── scripts/
    ├── prepare_data.py                # tokenizes TinyStories .txt -> train.bin / valid.bin (parallelized)
    ├── train_bpe_tinystories.py       # (re)trains the BPE vocab from scratch, if you ever need to
    └── generate.py                    # CLI: load a checkpoint, generate text
```

## How the pieces fit together

**1. Tokenizer (`tlm/tokenizer.py`).** Byte-level BPE, same idea as GPT-2's tokenizer. `workspace/tinystories_bpe_vocab.pkl` and `_merges.pkl` are already trained on TinyStories (10,000-token vocab) — you don't need to retrain unless you want a different vocab size or dataset. If you do, `scripts/train_bpe_tinystories.py` reruns training from `data/TinyStoriesV2-GPT4-train.txt`.

**2. Data prep (`scripts/prepare_data.py`).** Takes the raw TinyStories `.txt` files, splits each into chunks on `<|endoftext|>` boundaries (`find_chunk_boundaries`), tokenizes the chunks in parallel across CPU cores, and writes the resulting token ids as flat `uint16` arrays to `data/train.bin` / `data/valid.bin`. This only needs to run once per dataset/vocab combination.

**3. Model (`tlm/model/`).** A pre-norm, decoder-only transformer — architecturally closer to Llama than to the original GPT-2:
- RMSNorm instead of LayerNorm
- RoPE instead of learned/sinusoidal position embeddings
- SwiGLU feedforward instead of a plain ReLU/GELU MLP
- Causal self-attention via `F.scaled_dot_product_attention` (dispatches to a fused flash-attention kernel on CUDA)
- Tied input/output embeddings
- No biases anywhere

Default config: `d_model=512, n_layers=6, n_heads=8, d_ff=1536, context_length=512, vocab_size=10000` → **25.57M params**.

**4. Training (`tlm/train/train.py`).** Loads the `.bin` files as memmaps (so the dataset never has to fit in RAM), samples random contiguous windows for each batch, and trains with:
- AdamW, fused on CUDA, with weight decay only on 2D+ parameters (matmul weights and the embedding table), not on RMSNorm scales
- Cosine LR schedule with linear warmup
- bf16 autocast on CUDA (fp16 + gradient scaling as a fallback if bf16 isn't supported), fp32 elsewhere
- Gradient clipping (default max norm 1.0)
- Gradient accumulation (`--grad-accum-steps`) for effective batch sizes larger than what fits in memory
- `torch.compile`, applied automatically whenever a CUDA device is detected (disable with `--no-compile`)
- A background `BatchPrefetcher` thread that prepares the next batch while the GPU is still working on the current step
- Periodic validation loss + perplexity, periodic checkpointing, resumable via `--resume`
- Optional Weights & Biases logging (`--wandb`)

**5. Generation (`tlm/generate.py` / `scripts/generate.py`).** Autoregressive sampling with a KV cache (so each new token is an O(1) forward pass, not O(n)), temperature, top-k, and top-p, stopping at `<|endoftext|>`.

## Running this on an EC2 instance

### 1. Launch and connect

Pick a GPU instance — a `g5.xlarge` (A10G, 24GB) or `g6.xlarge` (L4) is plenty for a 25M-param model; you don't need anything bigger. Use the AWS Deep Learning AMI (Ubuntu) if you want CUDA/drivers preinstalled, which saves you a setup step.

```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>
```

### 2. Get the code and data onto the box

From your local machine:

```bash
rsync -avz --exclude .venv --exclude __pycache__ \
  /Users/dianhaoli/llms/transformer/ ubuntu@<instance-ip>:~/transformer/

rsync -avz \
  /Users/dianhaoli/llms/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt \
  /Users/dianhaoli/llms/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt \
  ubuntu@<instance-ip>:~/transformer/data/
```

The train file is ~2.2GB — this will take a few minutes depending on your connection. Alternatively, upload the files to S3 first and `aws s3 cp` them down on the instance, which is usually faster and resumable.

### 3. Install dependencies

```bash
ssh ubuntu@<instance-ip>
cd ~/transformer
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
```

Confirm the GPU is visible:

```bash
.venv/bin/python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Tokenize the dataset

```bash
.venv/bin/python3 scripts/prepare_data.py
```

This writes `data/train.bin` and `data/valid.bin` using the vocab already in `workspace/`. Uses all CPU cores by default (`--num-workers`); with a full 2.2GB train file expect roughly 10-20 minutes depending on core count.

### 5. Train

```bash
.venv/bin/python3 -m tlm.train.train \
  --max-steps 20000 \
  --batch-size 64 \
  --checkpoint-dir workspace/checkpoints \
  --wandb
```

All the model/optimization hyperparameters have sane defaults (see `tlm/train/train.py` for the full list) — you generally only need to set `--max-steps`, `--batch-size`, and whether you want `--wandb` logging. `torch.compile` and fused AdamW kick in automatically since you're on CUDA.

To run it detached so it survives your SSH session dropping:

```bash
nohup .venv/bin/python3 -m tlm.train.train --max-steps 20000 --wandb \
  > train.log 2>&1 &
disown
tail -f train.log
```

To resume after an interruption:

```bash
.venv/bin/python3 -m tlm.train.train --resume workspace/checkpoints/step_10000.pt --max-steps 20000
```

### 6. Generate from a checkpoint

```bash
.venv/bin/python3 scripts/generate.py \
  --checkpoint workspace/checkpoints/final.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```

### 7. Pull the trained checkpoint back down

```bash
rsync -avz ubuntu@<instance-ip>:~/transformer/workspace/checkpoints/final.pt ./workspace/checkpoints/
```

Don't forget to stop or terminate the instance when you're done training.
