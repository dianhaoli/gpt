"""Streamlit demo for the TinyStories transformer LM.

Run with:
    uv run streamlit run app.py
"""

import time
from collections.abc import Iterator

import streamlit as st
import torch

from tlm.generate import sample_token
from tlm.model import TransformerLM
from tlm.tokenizer import Tokenizer
from tlm.train.checkpoint import load_checkpoint, load_config
from tlm.train.train import pick_device

CHECKPOINT = "tlm-model/final.pt"
VOCAB = "tlm-model/tinystories_bpe_vocab.pkl"
MERGES = "tlm-model/tinystories_bpe_merges.pkl"
EOS = "<|endoftext|>"


@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint: str, vocab: str, merges: str):
    device = pick_device()
    tokenizer = Tokenizer.from_files(vocab, merges, special_tokens=[EOS])
    config = load_config(checkpoint)
    model = TransformerLM(config).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def stream_generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> Iterator[str]:
    eos_token_id = tokenizer.vocab_bytes_to_id[EOS.encode()]
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [eos_token_id]

    generated = list(prompt_ids)
    printed = tokenizer.decode(generated)
    yield printed

    idx = torch.tensor([generated], dtype=torch.long, device=device)
    logits, _, kv_caches = model(idx)

    for _ in range(max_new_tokens):
        next_token = sample_token(logits[:, -1, :], temperature, top_k, top_p)
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        # redecode the whole sequence each step so a token that's a partial
        # multi-byte UTF-8 sequence resolves once enough bytes arrive
        text = tokenizer.decode(generated)
        yield text[len(printed) :]
        printed = text

        if len(generated) >= model.config.context_length:
            break
        idx = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _, kv_caches = model(idx, kv_caches=kv_caches)


st.set_page_config(page_title="TinyStories LM", page_icon="📖", layout="centered")
st.title("📖 TinyStories LM")

model, tokenizer, device = load_model(CHECKPOINT, VOCAB, MERGES)
config = model.config

with st.sidebar:
    st.subheader("Sampling")
    max_new_tokens = st.slider("Max new tokens", 16, config.context_length, 200, step=16)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.8, step=0.05, help="0 = greedy decoding")
    use_top_k = st.checkbox("Top-k", value=True)
    top_k = st.slider("k", 1, 200, 50, disabled=not use_top_k)
    use_top_p = st.checkbox("Top-p (nucleus)", value=False)
    top_p = st.slider("p", 0.05, 1.0, 0.95, step=0.05, disabled=not use_top_p)

    st.subheader("Model")
    st.caption(
        f"{model.num_params() / 1e6:.1f}M params · {config.n_layers}L × {config.n_heads}H × "
        f"d{config.d_model}\nvocab {config.vocab_size} · ctx {config.context_length} · device `{device}`"
    )

prompt = st.text_area("Prompt", value="Once upon a time", height=100)
go = st.button("Generate", type="primary")

if go:
    stream = stream_generate(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens,
        temperature,
        top_k if use_top_k else None,
        top_p if use_top_p else None,
    )
    start = time.perf_counter()
    with st.container(border=True):
        text = st.write_stream(stream)
    elapsed = time.perf_counter() - start
    n_new = len(tokenizer.encode(text)) - len(tokenizer.encode(prompt))
    st.caption(f"{n_new} tokens in {elapsed:.1f}s ({n_new / max(elapsed, 1e-6):.1f} tok/s)")
