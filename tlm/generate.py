import torch
import torch.nn.functional as F

from tlm.model import TransformerLM


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _, kv_caches = model(idx)
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        next_token = sample_token(next_logits, temperature, top_k, top_p)
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        idx = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _, kv_caches = model(idx, kv_caches=kv_caches)

    return generated


def sample_token(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k is not None:
        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if top_p is not None:
        probs = apply_top_p(probs, top_p)

    return int(torch.multinomial(probs, num_samples=1).item())


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    renormalized = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    result = torch.zeros_like(probs)
    result.scatter_(-1, sorted_idx, renormalized)
    return result
