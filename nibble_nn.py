from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


VOCAB_SIZE = 16
TOKEN_ORDER = "high_nibble_first"
INPUT_SCALE = 15.0


class NibbleMLP(nn.Module):
    """Four fully connected layers: M -> 16 -> 16 -> 16 -> 16."""

    def __init__(self, context_length: int, hidden_dim: int = 16) -> None:
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self.context_length = int(context_length)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.context_length, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, VOCAB_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_to_nibble_tokens(path: str | Path, max_bytes: int | None = None) -> torch.Tensor:
    path = Path(path)
    count = -1 if max_bytes is None else int(max_bytes)
    byte_values = np.fromfile(path, dtype=np.uint8, count=count)
    tokens = np.empty(byte_values.size * 2, dtype=np.uint8)
    tokens[0::2] = byte_values >> 4
    tokens[1::2] = byte_values & 0x0F
    return torch.from_numpy(tokens)


def pack_nibbles(tokens: np.ndarray | torch.Tensor) -> bytes:
    if isinstance(tokens, torch.Tensor):
        values = tokens.detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        values = np.asarray(tokens, dtype=np.uint8)
    if values.size == 0:
        return b""
    padded = values
    if values.size % 2:
        padded = np.concatenate([values, np.zeros(1, dtype=np.uint8)])
    byte_values = ((padded[0::2] & 0x0F) << 4) | (padded[1::2] & 0x0F)
    return byte_values.astype(np.uint8, copy=False).tobytes()


def unpack_nibbles(data: bytes, token_count: int) -> np.ndarray:
    byte_values = np.frombuffer(data, dtype=np.uint8)
    tokens = np.empty(byte_values.size * 2, dtype=np.uint8)
    tokens[0::2] = byte_values >> 4
    tokens[1::2] = byte_values & 0x0F
    return tokens[:token_count].copy()


def make_context_batch(
    tokens: torch.Tensor,
    context_starts: torch.Tensor,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(context_length, dtype=torch.long)
    indices = context_starts.to(dtype=torch.long).unsqueeze(1) + offsets.unsqueeze(0)
    contexts = tokens[indices].to(dtype=torch.float32).div_(INPUT_SCALE).to(device)
    targets = tokens[context_starts.to(dtype=torch.long) + context_length].to(dtype=torch.long).to(device)
    return contexts, targets


def sequential_context_batch(
    tokens: torch.Tensor,
    first_target: int,
    last_target_exclusive: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.arange(
        first_target - context_length,
        last_target_exclusive - context_length,
        dtype=torch.long,
    )
    return make_context_batch(tokens, starts, context_length, device)


def probabilities_to_frequencies(probs: np.ndarray, total: int) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[1] != VOCAB_SIZE:
        raise ValueError(f"probs must have shape [N, {VOCAB_SIZE}]")
    if total < VOCAB_SIZE:
        raise ValueError("frequency total must be at least 16")
    probs64 = probs.astype(np.float64, copy=False)
    row_sums = probs64.sum(axis=1, keepdims=True)
    probs64 = np.divide(probs64, row_sums, out=np.full_like(probs64, 1.0 / VOCAB_SIZE), where=row_sums > 0)
    extra_total = total - VOCAB_SIZE
    raw_extra = probs64 * extra_total
    extra_floor = np.floor(raw_extra).astype(np.int64)
    freqs = extra_floor + 1
    remainders = total - freqs.sum(axis=1)
    fractions = raw_extra - extra_floor
    for row, remaining in enumerate(remainders):
        if remaining > 0:
            order = np.argpartition(-fractions[row], int(remaining) - 1)[: int(remaining)]
            freqs[row, order] += 1
    return freqs.astype(np.uint16 if total <= np.iinfo(np.uint16).max else np.uint32)


def logits_to_frequencies(logits: torch.Tensor, total: int) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probabilities_to_frequencies(probs, total)


def cumulative_from_frequencies(freqs: np.ndarray) -> list[int]:
    values = [int(x) for x in freqs]
    cumulative = [0]
    running = 0
    for value in values:
        if value <= 0:
            raise ValueError("all frequencies must be positive")
        running += value
        cumulative.append(running)
    return cumulative


def save_checkpoint(
    path: str | Path,
    model: NibbleMLP,
    context_length: int,
    train_args: dict[str, Any],
) -> None:
    checkpoint = {
        "version": 1,
        "context_length": int(context_length),
        "hidden_dim": int(model.hidden_dim),
        "vocab_size": VOCAB_SIZE,
        "token_order": TOKEN_ORDER,
        "input_scale": INPUT_SCALE,
        "model_state_dict": model.state_dict(),
        "train_args": train_args,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[NibbleMLP, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    context_length = int(checkpoint["context_length"])
    hidden_dim = int(checkpoint.get("hidden_dim", 16))
    model = NibbleMLP(context_length=context_length, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
