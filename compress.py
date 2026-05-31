from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nibble_nn import (
    VOCAB_SIZE,
    TOKEN_ORDER,
    cumulative_from_frequencies,
    file_to_nibble_tokens,
    load_checkpoint,
    logits_to_frequencies,
    pack_nibbles,
    probabilities_to_frequencies,
    resolve_device,
    sequential_context_batch,
    sha256_file,
)
from range_coder import ArithmeticEncoder, BitOutputStream


MAGIC = b"NNRC1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress nibble tokens with model probabilities and arithmetic coding.")
    parser.add_argument("--input", required=True, help="file to compress")
    parser.add_argument("--output", default=None, help="compressed output path")
    parser.add_argument("--model", default=None, help="checkpoint path; used for on-the-fly inference")
    parser.add_argument("--prob-in", default=None, help="optional .npy probability/frequency table from infer.py")
    parser.add_argument("--prob-format", choices=("freq-u16", "prob-f32"), default="freq-u16")
    parser.add_argument("--context", type=int, default=None, help="context M if --prob-in is used without --model")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--freq-total", type=int, default=4096)
    parser.add_argument("--max-bytes", type=int, default=None, help="read only first N bytes for quick tests")
    parser.add_argument("--estimate-only", action="store_true", help="do not write range-coded payload; only estimate entropy length")
    parser.add_argument("--progress-every", type=int, default=50, help="print progress every N inference batches; 0 disables it")
    return parser.parse_args()


def cumulative_rows_from_prob_file(prob_path: str, prob_format: str, freq_total: int):
    array = np.load(prob_path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != VOCAB_SIZE:
        raise ValueError(f"probability table must have shape [N, {VOCAB_SIZE}]")
    for row in array:
        if prob_format == "freq-u16":
            yield cumulative_from_frequencies(np.asarray(row, dtype=np.int64))
        else:
            freqs = probabilities_to_frequencies(np.asarray(row, dtype=np.float32)[None, :], freq_total)[0]
            yield cumulative_from_frequencies(freqs)


def write_header(out, metadata: dict, seed_tokens: torch.Tensor) -> int:
    header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    seed_bytes = pack_nibbles(seed_tokens)
    out.write(MAGIC)
    out.write(len(header).to_bytes(4, "little"))
    out.write(header)
    out.write(seed_bytes)
    return len(MAGIC) + 4 + len(header) + len(seed_bytes)


def maybe_print_progress(args: argparse.Namespace, batch_index: int, done_tokens: int, pred_count: int, started: float) -> None:
    if args.progress_every <= 0 or batch_index % args.progress_every != 0:
        return
    elapsed = time.perf_counter() - started
    rate = done_tokens / elapsed if elapsed > 0 else 0.0
    percent = 100.0 * done_tokens / pred_count if pred_count else 100.0
    print(
        f"progress={percent:.2f}% predicted_tokens={done_tokens}/{pred_count} "
        f"tokens_per_sec={rate:.0f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.model is None and args.prob_in is None:
        raise ValueError("provide --model for on-the-fly inference, or --prob-in from infer.py")
    if args.prob_format == "freq-u16" and args.freq_total > np.iinfo(np.uint16).max:
        raise ValueError("freq-u16 requires --freq-total <= 65535")

    output_path = Path(args.output) if args.output else Path(str(args.input) + ".nnrc")
    device = resolve_device(args.device)
    model = None
    checkpoint = None
    context = args.context
    model_sha256 = None
    if args.model is not None:
        model, checkpoint = load_checkpoint(args.model, device)
        context = int(checkpoint["context_length"])
        model_sha256 = sha256_file(args.model)

    tokens = file_to_nibble_tokens(args.input, max_bytes=args.max_bytes)
    token_count = int(tokens.numel())
    input_size = Path(args.input).stat().st_size
    measured_bytes = input_size if args.max_bytes is None else min(input_size, args.max_bytes)

    if args.prob_in is not None:
        prob_array = np.load(args.prob_in, mmap_mode="r")
        if prob_array.ndim != 2 or prob_array.shape[1] != VOCAB_SIZE:
            raise ValueError(f"probability table must have shape [N, {VOCAB_SIZE}]")
        if context is None:
            context = token_count - int(prob_array.shape[0])
        if int(prob_array.shape[0]) != token_count - int(context):
            raise ValueError("probability table row count must equal token_count - context")

    if context is None:
        raise ValueError("could not determine context length")
    if context < 0 or context > token_count:
        raise ValueError(f"invalid context={context} for token_count={token_count}")

    pred_count = token_count - context
    seed_tokens = tokens[:context]
    metadata = {
        "version": 1,
        "original_size": measured_bytes,
        "token_count": token_count,
        "context_length": context,
        "seed_token_count": int(seed_tokens.numel()),
        "token_order": TOKEN_ORDER,
        "coder": "arithmetic_v1",
        "freq_total": args.freq_total,
        "model_sha256": model_sha256,
        "prob_in": str(args.prob_in) if args.prob_in else None,
    }

    started = time.perf_counter()
    header_bytes = len(MAGIC) + 4 + len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    header_bytes += len(pack_nibbles(seed_tokens))
    payload_bytes = 0

    if args.estimate_only:
        bits = 0.0
        if args.prob_in is not None:
            prob_array = np.load(args.prob_in, mmap_mode="r")
            actual = tokens[context:].numpy()
            for row_start in range(0, pred_count, args.batch_size):
                row_stop = min(pred_count, row_start + args.batch_size)
                targets_np = actual[row_start:row_stop]
                rows = np.asarray(prob_array[row_start:row_stop])
                if args.prob_format == "freq-u16":
                    freqs = rows.astype(np.float64, copy=False)
                    chosen = freqs[np.arange(row_stop - row_start), targets_np]
                    totals = freqs.sum(axis=1)
                    bits += float(-np.log2(chosen / totals).sum())
                else:
                    probs = np.maximum(rows[np.arange(row_stop - row_start), targets_np], 1e-12)
                    bits += float(-np.log2(probs).sum())
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with torch.inference_mode():
                batch_index = 0
                for first_target in range(context, token_count, args.batch_size):
                    batch_index += 1
                    last_target = min(token_count, first_target + args.batch_size)
                    contexts, targets = sequential_context_batch(tokens, first_target, last_target, context, device)
                    logits = model(contexts)
                    freqs = logits_to_frequencies(logits, args.freq_total).astype(np.float64, copy=False)
                    targets_np = targets.detach().cpu().numpy()
                    chosen = freqs[np.arange(targets_np.size), targets_np]
                    totals = freqs.sum(axis=1)
                    bits += float(-np.log2(chosen / totals).sum())
                    maybe_print_progress(args, batch_index, last_target - context, pred_count, started)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        payload_bytes = math.ceil(float(bits) / 8.0)
        output_bytes = header_bytes + payload_bytes
    else:
        with output_path.open("wb") as raw_out:
            header_bytes = write_header(raw_out, metadata, seed_tokens)
            bitout = BitOutputStream(raw_out)
            encoder = ArithmeticEncoder(bitout)

            if args.prob_in is not None:
                rows = cumulative_rows_from_prob_file(args.prob_in, args.prob_format, args.freq_total)
                for target, cumulative in zip(tokens[context:].numpy(), rows, strict=True):
                    encoder.write(cumulative, int(target))
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                with torch.inference_mode():
                    batch_index = 0
                    for first_target in range(context, token_count, args.batch_size):
                        batch_index += 1
                        last_target = min(token_count, first_target + args.batch_size)
                        contexts, targets = sequential_context_batch(tokens, first_target, last_target, context, device)
                        logits = model(contexts)
                        freqs = logits_to_frequencies(logits, args.freq_total)
                        for row, target in zip(freqs, targets.detach().cpu().numpy(), strict=True):
                            encoder.write(cumulative_from_frequencies(row), int(target))
                        maybe_print_progress(args, batch_index, last_target - context, pred_count, started)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            encoder.finish()
            payload_bytes = bitout.bytes_written
        output_bytes = output_path.stat().st_size

    elapsed = time.perf_counter() - started
    compressed_over_original = output_bytes / measured_bytes if measured_bytes else 0.0
    original_over_compressed = measured_bytes / output_bytes if output_bytes else float("inf")

    print(f"input={args.input}")
    print(f"output={output_path if not args.estimate_only else '(estimate only)'}")
    print(f"context={context} tokens={token_count} predicted_tokens={pred_count}")
    print(f"header_seed_bytes={header_bytes}")
    print(f"range_payload_bytes={payload_bytes}")
    print(f"encoded_length_bytes={output_bytes}")
    print(f"original_bytes={measured_bytes}")
    print(f"compressed/original={compressed_over_original:.6f}")
    print(f"compression_ratio_original/compressed={original_over_compressed:.6f}")
    print(f"seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
