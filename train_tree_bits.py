from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tree_nibble_nn import (
    TreeNibbleMLP,
    file_to_nibble_tokens,
    make_tree_context_batch,
    resolve_device,
    save_tree_checkpoint,
    select_tree_path_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional bit-tree nibble probability model.")
    parser.add_argument("--input", required=True, help="training file path")
    parser.add_argument("--model-out", default="tree_bit_model.pt", help="checkpoint output path")
    parser.add_argument("--context", type=int, required=True, help="context length M, in 4-bit tokens")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-bytes", type=int, default=None, help="read only the first N bytes for quick tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    input_path = Path(args.input)
    tokens = file_to_nibble_tokens(input_path, max_bytes=args.max_bytes)
    token_count = int(tokens.numel())
    if token_count <= args.context:
        raise ValueError(f"need more than context={args.context} tokens, got {token_count}")

    model = TreeNibbleMLP(context_length=args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    max_start = token_count - args.context

    measured_bytes = input_path.stat().st_size if args.max_bytes is None else min(args.max_bytes, input_path.stat().st_size)
    print(f"input={input_path}")
    print(f"bytes={measured_bytes}")
    print(f"tokens={token_count} context={args.context} device={device}")
    print(f"model: {args.context} -> 16 -> 16 -> 16 -> 15")

    global_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_steps = 0
        for step in range(1, args.steps_per_epoch + 1):
            starts = torch.randint(0, max_start, (args.batch_size,), dtype=torch.long)
            contexts, bit_targets = make_tree_context_batch(tokens, starts, args.context, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(contexts)
            path_logits = select_tree_path_logits(logits, bit_targets)
            loss = F.binary_cross_entropy_with_logits(path_logits, bit_targets, reduction="none").sum(dim=1).mean()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            running_steps += 1
            global_step += 1
            if step % args.log_every == 0 or step == args.steps_per_epoch:
                avg_loss = running_loss / running_steps
                bits_per_token = avg_loss / math.log(2.0)
                print(
                    f"epoch={epoch} step={step}/{args.steps_per_epoch} "
                    f"loss={avg_loss:.4f} bpt={bits_per_token:.4f}"
                )
                running_loss = 0.0
                running_steps = 0

    elapsed = time.perf_counter() - started
    save_tree_checkpoint(
        args.model_out,
        model,
        args.context,
        train_args=vars(args) | {"train_seconds": elapsed, "token_count": token_count},
    )
    print(f"saved={args.model_out}")
    print(f"train_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
