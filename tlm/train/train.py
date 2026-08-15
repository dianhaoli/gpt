import argparse
import time

import torch

from tlm.model import TransformerConfig, TransformerLM
from tlm.train.checkpoint import save_checkpoint
from tlm.train.data import BatchPrefetcher, get_batch, load_dataset
from tlm.train.optim import configure_optimizer, cosine_with_warmup_lr


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def estimate_loss(model: TransformerLM, dataset, batch_size: int, context_length: int, device: str, eval_iters: int) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        _, loss, _ = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.bin")
    parser.add_argument("--valid-data", default="data/valid.bin")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-dir", default="workspace/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tinystories-lm")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"device={device} dtype={dtype}")

    config = TransformerConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        dropout=args.dropout,
    )
    model = TransformerLM(config).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M total, {model.num_params(exclude_embedding=True) / 1e6:.2f}M non-embedding")

    if device == "cuda" and not args.no_compile:
        model = torch.compile(model)

    optimizer = configure_optimizer(model, args.max_lr, args.weight_decay, (args.beta1, args.beta2), fused=(device == "cuda"))

    start_step = 0
    if args.resume:
        from tlm.train.checkpoint import load_checkpoint

        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at step {start_step}")

    train_data = load_dataset(args.train_data)
    valid_data = load_dataset(args.valid_data)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    autocast_enabled = dtype != torch.float32
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))
    prefetcher = BatchPrefetcher(train_data, args.batch_size, args.context_length, device)

    t0 = time.perf_counter()
    for step in range(start_step, args.max_steps):
        lr = cosine_with_warmup_lr(step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum_steps):
            x, y = prefetcher.next()
            with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
                _, loss, _ = model(x, y)
            scaler.scale(loss / args.grad_accum_steps).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            tokens_per_sec = args.batch_size * args.context_length * args.grad_accum_steps * max(1, step - start_step + 1) / elapsed
            print(f"step {step} | loss {loss.item():.4f} | lr {lr:.2e} | grad_norm {grad_norm:.2f} | tok/s {tokens_per_sec:,.0f}")
            if wandb_run:
                wandb_run.log({"train/loss": loss.item(), "lr": lr, "grad_norm": grad_norm}, step=step)

        if step % args.eval_interval == 0 and step > start_step:
            val_loss = estimate_loss(model, valid_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"step {step} | val_loss {val_loss:.4f} | val_ppl {torch.exp(torch.tensor(val_loss)).item():.2f}")
            if wandb_run:
                wandb_run.log({"val/loss": val_loss}, step=step)

        if step % args.checkpoint_interval == 0 and step > start_step:
            ckpt_path = f"{args.checkpoint_dir}/step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, config)
            print(f"saved checkpoint -> {ckpt_path}")

    prefetcher.close()
    save_checkpoint(f"{args.checkpoint_dir}/final.pt", model, optimizer, args.max_steps, config)
    print(f"saved final checkpoint -> {args.checkpoint_dir}/final.pt")


if __name__ == "__main__":
    main()
