from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1536
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
