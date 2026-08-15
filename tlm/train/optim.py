import math

from torch import nn
from torch.optim import AdamW


def cosine_with_warmup_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float], fused: bool) -> AdamW:
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=betas, fused=fused)
