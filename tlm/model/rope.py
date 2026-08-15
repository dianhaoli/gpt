import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, context_length: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(context_length).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions].repeat_interleave(2, dim=-1)
        sin = self.sin_cached[positions].repeat_interleave(2, dim=-1)
        rotated = x.float() * cos + rotate_half(x.float()) * sin
        return rotated.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
