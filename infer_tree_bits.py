from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tree_nibble_nn import (
    BIT_ORDER,
    TREE_LOGIT_COUNT,
    load_tree_checkpoint,
    file_to_nibble_tokens,
    logits_to_tree_frequencies,
    resolve_device,
    select_tree_path_logits,
    sequential_tree_context_batch_view,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer conditional bit-tree probabilities per nibble token.")
    parser.add_argument("--input", required=True, help="file to infer")
    parser.add_argument("--model", required=True, help="trained tree checkpoint path")
    parser.add_argument("--batch-size", type=int, default=1048576)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--prob-out", default=None, help="optional .npy output for tree probabilities")
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
    if args.freq_total < 2:
        raise ValueError("--freq-total must be at least 2")

    device = resolve_device(args.device)
    model, checkpoint = load_tree_checkpoint(args.model, device)
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
            shape=(pred_count, TREE_LOGIT_COUNT),
        )

    total_loss = 0.0
    total_tokens = 0
    synchronize_if_needed(device)
    infer_started = time.perf_counter()
    with torch.inference_mode():
        for first_target in range(context, token_count, args.batch_size):
            last_target = min(token_count, first_target + args.batch_size)
            contexts, bit_targets, _ = sequential_tree_context_batch_view(tokens, first_target, last_target, context, device)
            logits = model(contexts)
            path_logits = select_tree_path_logits(logits, bit_targets)
            loss = F.binary_cross_entropy_with_logits(path_logits, bit_targets, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_tokens += int(bit_targets.shape[0])

            if out_array is not None:
                row_start = first_target - context
                row_stop = last_target - context
                if args.prob_format == "freq-u16":
                    out_array[row_start:row_stop] = logits_to_tree_frequencies(logits, args.freq_total)
                else:
                    probs_one = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                    out_array[row_start:row_stop] = probs_one
    if out_array is not None:
        out_array.flush()
    synchronize_if_needed(device)
    infer_seconds = time.perf_counter() - infer_started

    bandwidth_mib = (measured_bytes / (1024 * 1024)) / infer_seconds if infer_seconds > 0 else float("inf")
    bpt = total_loss / max(1, total_tokens) / math.log(2.0)
    theoretical_ratio = bpt / 4.0

    print(f"input={args.input}")
    print(f"model={args.model}")
    print(f"tokens={token_count} predicted_tokens={pred_count} context={context} device={device}")
    print(f"read_seconds={read_seconds:.3f}")
    print(f"infer_seconds={infer_seconds:.3f}")
    print(f"infer_bandwidth={bandwidth_mib:.3f} MiB/s")
    print(f"conditional_bit_cross_entropy={bpt:.4f} bits/token, ideal_compressed/original={theoretical_ratio:.4f}")
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
                "shape": [pred_count, TREE_LOGIT_COUNT],
                "dtype": str(out_array.dtype),
                "value": "frequency_of_bit_one" if args.prob_format == "freq-u16" else "probability_of_bit_one",
                "bit_order": BIT_ORDER,
                "tree": "heap indices: root=0, level starts 0,1,3,7",
            },
        )
        print(f"prob_out={args.prob_out}")
        print(f"prob_meta={sidecar}")


if __name__ == "__main__":
    main()
