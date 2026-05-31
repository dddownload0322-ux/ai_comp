from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nibble_nn import (
    INPUT_SCALE,
    TOKEN_ORDER,
    file_to_nibble_tokens,
    make_context_batch,
    pack_nibbles,
    resolve_device,
    sha256_file,
    write_json,
)


BIT_COUNT = 4
TREE_LOGIT_COUNT = (1 << BIT_COUNT) - 1
BIT_ORDER = "msb_first"
OUTPUT_KIND = "conditional_nibble_bit_tree"


class TreeNibbleMLP(nn.Module):
    """Four fully connected layers: M -> 16 -> 16 -> 16 -> 15."""

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
            nn.Linear(self.hidden_dim, TREE_LOGIT_COUNT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nibble_tokens_to_bit_targets(tokens: torch.Tensor) -> torch.Tensor:
    shifts = torch.tensor([3, 2, 1, 0], dtype=torch.long, device=tokens.device)
    values = tokens.to(dtype=torch.long).unsqueeze(-1)
    return ((values >> shifts) & 1).to(dtype=torch.float32)


def tree_path_indices_from_bits(bit_targets: torch.Tensor) -> torch.Tensor:
    bits = bit_targets.to(dtype=torch.long)
    prefix0 = bits.new_zeros(bits.shape[0])
    prefix1 = bits[:, 0]
    prefix2 = (bits[:, 0] << 1) | bits[:, 1]
    prefix3 = (bits[:, 0] << 2) | (bits[:, 1] << 1) | bits[:, 2]
    return torch.stack(
        (
            prefix0,
            1 + prefix1,
            3 + prefix2,
            7 + prefix3,
        ),
        dim=1,
    )


def select_tree_path_logits(logits: torch.Tensor, bit_targets: torch.Tensor) -> torch.Tensor:
    path_indices = tree_path_indices_from_bits(bit_targets)
    return torch.gather(logits, 1, path_indices)


def nibble_tokens_to_tree_paths_numpy(tokens: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(tokens, torch.Tensor):
        values = tokens.detach().cpu().numpy().astype(np.uint8, copy=False)
    else:
        values = np.asarray(tokens, dtype=np.uint8)
    bits = np.empty((values.size, BIT_COUNT), dtype=np.uint8)
    bits[:, 0] = (values >> 3) & 1
    bits[:, 1] = (values >> 2) & 1
    bits[:, 2] = (values >> 1) & 1
    bits[:, 3] = values & 1

    path_indices = np.empty((values.size, BIT_COUNT), dtype=np.uint8)
    path_indices[:, 0] = 0
    path_indices[:, 1] = 1 + bits[:, 0]
    path_indices[:, 2] = 3 + ((bits[:, 0] << 1) | bits[:, 1])
    path_indices[:, 3] = 7 + ((bits[:, 0] << 2) | (bits[:, 1] << 1) | bits[:, 2])
    return bits, path_indices


def make_tree_context_batch(
    tokens: torch.Tensor,
    context_starts: torch.Tensor,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, nibble_targets = make_context_batch(tokens, context_starts, context_length, device)
    return contexts, nibble_tokens_to_bit_targets(nibble_targets)


def _sequential_context_batch_view(
    tokens: torch.Tensor,
    first_target: int,
    last_target_exclusive: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if first_target < context_length:
        raise ValueError("first_target must be at least context_length")
    if last_target_exclusive < first_target:
        raise ValueError("last_target_exclusive must be >= first_target")
    row_start = first_target - context_length
    row_stop = last_target_exclusive - context_length
    pred_count = int(tokens.numel()) - context_length
    if row_stop > pred_count:
        raise ValueError("target range exceeds token count")

    windows = torch.as_strided(tokens, size=(pred_count, context_length), stride=(1, 1))
    contexts = windows[row_start:row_stop].to(device=device, dtype=torch.float32).div_(INPUT_SCALE)
    targets = tokens[first_target:last_target_exclusive].to(device=device, dtype=torch.long)
    return contexts, targets


def sequential_tree_context_batch_view(
    tokens: torch.Tensor,
    first_target: int,
    last_target_exclusive: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts, nibble_targets = _sequential_context_batch_view(
        tokens,
        first_target,
        last_target_exclusive,
        context_length,
        device,
    )
    bit_targets = nibble_tokens_to_bit_targets(nibble_targets)
    return contexts, bit_targets, nibble_targets


def bernoulli_probs_to_frequencies(probs_one: np.ndarray, total: int) -> np.ndarray:
    if probs_one.ndim != 2 or probs_one.shape[1] != TREE_LOGIT_COUNT:
        raise ValueError(f"probs_one must have shape [N, {TREE_LOGIT_COUNT}]")
    if total < 2:
        raise ValueError("frequency total must be at least 2")

    probs64 = probs_one.astype(np.float64, copy=False)
    probs64 = np.nan_to_num(probs64, nan=0.5, posinf=1.0, neginf=0.0)
    freq_one = np.rint(probs64 * total).astype(np.int64)
    freq_one = np.clip(freq_one, 1, total - 1)
    dtype = np.uint16 if total <= np.iinfo(np.uint16).max else np.uint32
    return freq_one.astype(dtype, copy=False)


def logits_to_tree_frequencies(logits: torch.Tensor, total: int) -> np.ndarray:
    probs_one = torch.sigmoid(logits).detach().cpu().numpy()
    return bernoulli_probs_to_frequencies(probs_one, total)


def bit_frequency_to_cumulative(freq_one: int, total: int) -> list[int]:
    freq_one = int(freq_one)
    total = int(total)
    if freq_one <= 0 or freq_one >= total:
        raise ValueError("one-bit frequency must be in [1, total - 1]")
    return [0, total - freq_one, total]


def save_tree_checkpoint(
    path: str | Path,
    model: TreeNibbleMLP,
    context_length: int,
    train_args: dict[str, Any],
) -> None:
    checkpoint = {
        "version": 1,
        "output_kind": OUTPUT_KIND,
        "context_length": int(context_length),
        "hidden_dim": int(model.hidden_dim),
        "bit_count": BIT_COUNT,
        "tree_logit_count": TREE_LOGIT_COUNT,
        "bit_order": BIT_ORDER,
        "token_order": TOKEN_ORDER,
        "input_scale": INPUT_SCALE,
        "model_state_dict": model.state_dict(),
        "train_args": train_args,
    }
    torch.save(checkpoint, path)


def load_tree_checkpoint(path: str | Path, device: torch.device) -> tuple[TreeNibbleMLP, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    output_kind = checkpoint.get("output_kind")
    if output_kind not in (None, OUTPUT_KIND):
        raise ValueError(f"checkpoint output_kind={output_kind!r} is not {OUTPUT_KIND!r}")
    context_length = int(checkpoint["context_length"])
    hidden_dim = int(checkpoint.get("hidden_dim", 16))
    model = TreeNibbleMLP(context_length=context_length, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
