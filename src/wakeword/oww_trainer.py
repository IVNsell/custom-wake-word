"""Train a small DNN on precomputed features and export to ONNX (no openWakeWord train.py)."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class WakeWordDNN(nn.Module):
    """Same architecture as openWakeWord model_type=dnn (layer_dim=32 default)."""

    def __init__(self, input_shape: tuple[int, int] = (16, 96), layer_dim: int = 32, n_blocks: int = 1):
        super().__init__()
        self.input_shape = input_shape

        class FCNBlock(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.fcn = nn.Linear(dim, dim)
                self.relu = nn.ReLU()
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                return self.relu(self.norm(self.fcn(x)))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                flat = input_shape[0] * input_shape[1]
                self.flatten = nn.Flatten()
                self.layer1 = nn.Linear(flat, layer_dim)
                self.relu1 = nn.ReLU()
                self.norm1 = nn.LayerNorm(layer_dim)
                self.blocks = nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
                self.last = nn.Linear(layer_dim, 1)
                self.act = nn.Sigmoid()

            def forward(self, x):
                x = self.relu1(self.norm1(self.layer1(self.flatten(x))))
                for block in self.blocks:
                    x = block(x)
                return self.act(self.last(x))

        self.model = Net()

    def forward(self, x):
        return self.model(x)


def _sample_batch(
    pos: np.ndarray,
    neg: np.ndarray,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random mini-batch of positive and negative feature vectors."""
    pi = rng.integers(0, pos.shape[0], size=n_pos)
    ni = rng.integers(0, neg.shape[0], size=n_neg)
    x = np.vstack([pos[pi], neg[ni]]).astype(np.float32)
    y = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def _export_onnx(model: WakeWordDNN, input_shape: tuple[int, int], output_onnx: Path) -> None:
    """Export to ONNX; suppress stdout to avoid Windows cp1251 emoji crashes."""
    model.cpu().eval()
    dummy = torch.randn(1, *input_shape)
    buf = io.StringIO()
    kwargs = dict(input_names=["x"], output_names=["y"], opset_version=18)
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            torch.onnx.export(model, dummy, str(output_onnx), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(output_onnx), **kwargs)


def train_and_export_onnx(
    pos_features_path: Path,
    neg_features_path: Path,
    output_onnx: Path,
    *,
    layer_size: int = 32,
    steps: int = 35000,
    n_pos_batch: int = 64,
    n_neg_batch: int = 960,
    max_negative_weight: float = 1500.0,
    val_split: float = 0.15,
    device: str | None = None,
) -> Path:
    """Train classifier on .npy features and save best checkpoint as ONNX."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    pos = np.load(pos_features_path, mmap_mode="r")
    neg = np.load(neg_features_path, mmap_mode="r")
    if pos.ndim != 3 or neg.ndim != 3:
        raise ValueError(f"Expected shape (N, 16, 96), got pos={pos.shape}, neg={neg.shape}")

    input_shape = (int(pos.shape[1]), int(pos.shape[2]))
    n_val = max(1, int(pos.shape[0] * val_split))
    rng = np.random.default_rng(42)
    order = rng.permutation(pos.shape[0])
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    pos_train = np.asarray(pos[train_idx])
    pos_val = np.asarray(pos[val_idx])

    # Validation set: held-out positives + random negatives
    n_val_neg = min(2000, neg.shape[0])
    neg_val_idx = rng.integers(0, neg.shape[0], size=n_val_neg)
    neg_val = np.asarray(neg[neg_val_idx])
    x_val = torch.from_numpy(np.vstack([pos_val, neg_val]).astype(np.float32))
    y_val = torch.from_numpy(
        np.array([1.0] * len(pos_val) + [0.0] * len(neg_val), dtype=np.float32)
    )

    model = WakeWordDNN(input_shape=input_shape, layer_dim=layer_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCELoss(reduction="none")
    rng_train = np.random.default_rng(123)

    best_state = None
    best_score = -1.0

    pbar = tqdm(range(steps), desc="train")
    for step in pbar:
        # Gradually increase weight on negatives to reduce false activations
        neg_w = 1.0 + (max_negative_weight - 1.0) * (step / max(steps - 1, 1))
        x, y = _sample_batch(pos_train, neg, n_pos_batch, n_neg_batch, rng_train)
        x, y = x.to(device), y.to(device)

        pred = model(x).squeeze(-1)
        weights = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, neg_w))
        loss = (bce(pred, y) * weights).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vp = model(x_val.to(device)).squeeze(-1).cpu()
                acc = ((vp >= 0.5) == (y_val >= 0.5)).float().mean().item()
                recall = ((vp >= 0.5) & (y_val >= 0.5)).sum().item() / max((y_val >= 0.5).sum().item(), 1)
            model.train()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}", rec=f"{recall:.3f}")
            score = recall * 0.7 + acc * 0.3
            if score >= best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    weights_path = output_onnx.with_suffix(".pt")
    torch.save(model.state_dict(), weights_path)

    _export_onnx(model, input_shape, output_onnx)
    print(f"OK: ONNX saved: {output_onnx}")
    print(f"OK: PyTorch weights: {weights_path}")
    return output_onnx


def export_onnx_from_weights(
    weights_path: Path,
    output_onnx: Path,
    input_shape: tuple[int, int] = (16, 96),
    layer_size: int = 32,
) -> Path:
    """Re-export ONNX from saved .pt weights without retraining."""
    model = WakeWordDNN(input_shape=input_shape, layer_dim=layer_size)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    _export_onnx(model, input_shape, output_onnx)
    print(f"OK: ONNX from weights: {output_onnx}")
    return output_onnx
