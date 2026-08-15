import torch
from torch import nn

from tlm.model.attention import CausalSelfAttention
from tlm.model.config import TransformerConfig
from tlm.model.feedforward import SwiGLU
from tlm.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(
            config.d_model, config.n_heads, config.context_length, config.rope_theta, config.dropout
        )
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        attn_out, new_cache = self.attn(self.attn_norm(x), positions, kv_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_cache
