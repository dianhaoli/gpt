import math

import torch
import torch.nn.functional as F
from torch import nn

from tlm.model.block import TransformerBlock
from tlm.model.config import TransformerConfig
from tlm.model.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple] | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple]]:
        _, seq_len = input_ids.shape
        if positions is None:
            offset = kv_caches[0][0].shape[2] if kv_caches is not None else 0
            positions = torch.arange(offset, offset + seq_len, device=input_ids.device)

        x = self.dropout(self.token_embedding(input_ids))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, positions, cache)
            new_caches.append(new_cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.token_embedding.weight.numel()
        return n
