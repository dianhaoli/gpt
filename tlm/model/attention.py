import torch
import torch.nn.functional as F
from torch import nn

from tlm.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_length: int, rope_theta: float, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, context_length, rope_theta)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.qkv_proj(x).split(d_model, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        is_causal = kv_cache is None
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(attn_out), new_cache
