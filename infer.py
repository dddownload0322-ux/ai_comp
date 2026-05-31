from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nibble_nn import (
    VOCAB_SIZE,
    file_to_nibble_tokens,
    load_checkpoint,
    logits_to_frequencies,
    resolve_device,
    sequential_context_batch,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer per-token nibble probabilities and measure bandwidth.")
    parser.add_argument("--input", required=True, help="file to infer")
    parser.add_argument("--model", required=True, help="trained checkpoint path")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--prob-out", default=None, help="optional .npy output for distributions")
    parser.add_argument("--prob-format", choices=("freq-u16", "prob-f32"), default="freq-u16")
    parser.add_argument("--freq-total", type=int, default=4096, help="frequency total for freq-u16")
    parser.add_argument("--max-bytes", type=int, default=None, help="read only first N bytes for quick tests")
    return parser.parse_args()


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.prob_format == "freq-u16" and args.freq_total > np.iinfo(np.uint16).max:
        raise ValueError("freq-u16 requires --freq-total <= 65535")
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.model, device)
    context = int(checkpoint["context_length"])

    read_started = time.perf_counter()
    tokens = file_to_nibble_tokens(args.input, max_bytes=args.max_bytes)
    read_seconds = time.perf_counter() - read_started
    token_count = int(tokens.numel())
    pred_count = max(0, token_count - context)
    input_size = Path(args.input).stat().st_size
    measured_bytes = input_size if args.max_bytes is None else min(input_size, args.max_bytes)

    out_array = None
    if args.prob_out is not None:
        dtype = np.uint16 if args.prob_format == "freq-u16" else np.float32
        out_array = np.lib.format.open_memmap(
            args.prob_out,
            mode="w+",
            dtype=dtype,
            shape=(pred_count, VOCAB_SIZE),
        )

    total_loss = 0.0
    total_targets = 0
    synchronize_if_needed(device)
    infer_started = time.perf_counter()
    with torch.inference_mode():
        for first_target in range(context, token_count, args.batch_size):
            last_target = min(token_count, first_target + args.batch_size)
            contexts, targets = sequential_context_batch(tokens, first_target, last_target, context, device)
            logits = model(contexts)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_targets += int(targets.numel())

            if out_array is not None:
                row_start = first_target - context
                row_stop = last_target - context
                if args.prob_format == "freq-u16":
                    out_array[row_start:row_stop] = logits_to_frequencies(logits, args.freq_total)
                else:
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
                    out_array[row_start:row_stop] = probs
    if out_array is not None:
        out_array.flush()
    synchronize_if_needed(device)
    infer_seconds = time.perf_counter() - infer_started

    bandwidth_mib = (measured_bytes / (1024 * 1024)) / infer_seconds if infer_seconds > 0 else float("inf")
    bpt = total_loss / max(1, total_targets) / math.log(2.0)
    theoretical_ratio = bpt / 4.0

    print(f"input={args.input}")
    print(f"model={args.model}")
    print(f"tokens={token_count} predicted_tokens={pred_count} context={context} device={device}")
    print(f"read_seconds={read_seconds:.3f}")
    print(f"infer_seconds={infer_seconds:.3f}")
    print(f"infer_bandwidth={bandwidth_mib:.3f} MiB/s")
    print(f"cross_entropy={bpt:.4f} bits/token, ideal_compressed/original={theoretical_ratio:.4f}")
    if args.prob_out is not None:
        sidecar = str(Path(args.prob_out).with_suffix(Path(args.prob_out).suffix + ".json"))
        write_json(
            sidecar,
            {
                "input": str(args.input),
                "model": str(args.model),
                "context_length": context,
                "token_count": token_count,
                "predicted_tokens": pred_count,
                "format": args.prob_format,
                "freq_total": args.freq_total if args.prob_format == "freq-u16" else None,
                "shape": [pred_count, VOCAB_SIZE],
                "dtype": str(out_array.dtype),
            },
        )
        print(f"prob_out={args.prob_out}")
        print(f"prob_meta={sidecar}")


if __name__ == "__main__":
    main()
