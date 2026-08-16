import dataclasses
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from tlm.model.config import TransformerConfig


def save_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, step: int, config: TransformerConfig) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(config),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: Optimizer | None = None) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state"].items()}
    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["step"]


def load_config(path: str) -> TransformerConfig:
    checkpoint = torch.load(path, map_location="cpu")
    return TransformerConfig(**checkpoint["config"])
