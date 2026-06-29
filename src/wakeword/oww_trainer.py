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


def _mine_confusing_negatives(
    pos_train: np.ndarray,
    neg: np.ndarray,
    *,
    n_sample: int = 60000,
    top_k: int = 2500,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick rows from the shared corpus that look most like the wake phrase.

    This finds confusers in general speech/noise (short vowel/consonant patterns,
    similar energy) without listing specific words like alex or aizek.
    """
    if neg.shape[0] == 0 or pos_train.shape[0] == 0:
        return np.zeros((0, *neg.shape[1:]), dtype=np.float32)

    pos_flat = pos_train.reshape(len(pos_train), -1).astype(np.float32)
    proto = pos_flat.mean(axis=0)
    proto /= np.linalg.norm(proto) + 1e-8

    n_sample = min(n_sample, neg.shape[0])
    idx = rng.choice(neg.shape[0], size=n_sample, replace=False)
    sample = np.asarray(neg[idx], dtype=np.float32)
    flat = sample.reshape(len(sample), -1)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    sim = flat @ proto

    k = min(top_k, len(sample))
    pick = np.argpartition(sim, -k)[-k:]
    return sample[pick]


def _mine_model_confusers(
    model: WakeWordDNN,
    neg: np.ndarray,
    *,
    device: str,
    n_sample: int = 24000,
    top_k: int = 1500,
    rng: np.random.Generator,
    score_threshold: float = 0.12,
) -> np.ndarray:
    """Find shared-corpus rows the current model scores too high on."""
    if neg.shape[0] == 0:
        return np.zeros((0, *neg.shape[1:]), dtype=np.float32)

    n_sample = min(n_sample, neg.shape[0])
    idx = rng.choice(neg.shape[0], size=n_sample, replace=False)
    sample = np.asarray(neg[idx], dtype=np.float32)

    model.eval()
    scores: list[float] = []
    batch = 512
    with torch.no_grad():
        for start in range(0, len(sample), batch):
            x = torch.from_numpy(sample[start : start + batch]).to(device)
            pred = model(x).squeeze(-1).cpu().numpy()
            scores.extend(pred.tolist())

    scores_arr = np.asarray(scores, dtype=np.float32)
    mask = scores_arr >= score_threshold
    if not mask.any():
        pick = np.argsort(scores_arr)[-min(top_k, len(scores_arr)) :]
        return sample[pick]

    ranked = np.where(mask)[0]
    ranked = ranked[np.argsort(scores_arr[ranked])][::-1]
    pick = ranked[: min(top_k, len(ranked))]
    return sample[pick]


def _merge_confusers(*arrays: np.ndarray | None) -> np.ndarray | None:
    parts = [a for a in arrays if a is not None and a.shape[0] > 0]
    if not parts:
        return None
    return np.vstack(parts)


def _sample_batch(
    pos: np.ndarray,
    neg: np.ndarray,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    hard_neg: np.ndarray | None = None,
    n_hard_neg: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random mini-batch with per-sample loss weights (pos / platform neg / confuser)."""
    pi = rng.integers(0, pos.shape[0], size=n_pos)
    parts = [pos[pi]]
    labels = [1.0] * n_pos
    weights = [2.0] * n_pos

    n_platform_neg = n_neg
    if hard_neg is not None and hard_neg.shape[0] > 0 and n_hard_neg > 0:
        hi = rng.integers(0, hard_neg.shape[0], size=n_hard_neg)
        parts.append(hard_neg[hi])
        labels.extend([0.0] * n_hard_neg)
        weights.extend([0.0] * n_hard_neg)
        n_platform_neg = max(1, n_neg - n_hard_neg)

    ni = rng.integers(0, neg.shape[0], size=n_platform_neg)
    parts.append(neg[ni])
    labels.extend([0.0] * n_platform_neg)
    weights.extend([1.0] * n_platform_neg)

    x = np.vstack(parts).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)


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
    hard_neg_features_path: Path | None = None,
    layer_size: int = 32,
    steps: int = 35000,
    n_pos_batch: int = 64,
    n_neg_batch: int = 960,
    n_confuser_batch: int = 384,
    max_negative_weight: float = 1500.0,
    confuser_weight: float = 4000.0,
    mine_confusers: bool = True,
    confuser_top_k: int = 2500,
    val_split: float = 0.15,
    remine_every: int = 5000,
    device: str | None = None,
) -> Path:
    """Train classifier on .npy features and save best checkpoint as ONNX."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    pos = np.load(pos_features_path, mmap_mode="r")
    neg = np.load(neg_features_path, mmap_mode="r")
    if pos.ndim != 3 or neg.ndim != 3:
        raise ValueError(f"Expected shape (N, 16, 96), got pos={pos.shape}, neg={neg.shape}")
    hard_neg = None
    if hard_neg_features_path and hard_neg_features_path.exists():
        hard_neg = np.load(hard_neg_features_path, mmap_mode="r")
        if hard_neg.ndim != 3:
            raise ValueError(f"Expected optional hard negatives shape (N, 16, 96), got {hard_neg.shape}")
        print(f"Optional user hard negatives: {hard_neg.shape[0]} feature rows")

    input_shape = (int(pos.shape[1]), int(pos.shape[2]))
    n_val = max(1, int(pos.shape[0] * val_split))
    rng = np.random.default_rng(42)
    order = rng.permutation(pos.shape[0])
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    pos_train = np.asarray(pos[train_idx])
    pos_val = np.asarray(pos[val_idx])

    confusers = None
    if mine_confusers:
        confusers = _mine_confusing_negatives(
            pos_train,
            neg,
            top_k=confuser_top_k,
            rng=rng,
        )
        print(f"Auto-mined confusers from shared corpus: {confusers.shape[0]} rows")
    confuser_pool = _merge_confusers(
        confusers,
        np.asarray(hard_neg) if hard_neg is not None else None,
    )

    # Validation set: held-out positives + random negatives
    n_val_neg = min(2000, neg.shape[0])
    neg_val_idx = rng.integers(0, neg.shape[0], size=n_val_neg)
    neg_val = np.asarray(neg[neg_val_idx])
    val_parts = [pos_val, neg_val]
    val_labels = [1.0] * len(pos_val) + [0.0] * len(neg_val)
    confuser_val = None
    if confuser_pool is not None and confuser_pool.shape[0] > 0:
        n_val_conf = min(500, confuser_pool.shape[0])
        conf_val_idx = rng.integers(0, confuser_pool.shape[0], size=n_val_conf)
        confuser_val = confuser_pool[conf_val_idx]
        val_parts.append(confuser_val)
        val_labels.extend([0.0] * n_val_conf)
    x_val = torch.from_numpy(np.vstack(val_parts).astype(np.float32))
    y_val = torch.from_numpy(np.array(val_labels, dtype=np.float32))
    confuser_val_t = (
        torch.from_numpy(confuser_val.astype(np.float32)) if confuser_val is not None else None
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
        hard_w = 1.0 + (confuser_weight - 1.0) * (step / max(steps - 1, 1))
        x, y, sample_kind = _sample_batch(
            pos_train,
            neg,
            n_pos_batch,
            n_neg_batch,
            rng_train,
            hard_neg=confuser_pool,
            n_hard_neg=n_confuser_batch if confuser_pool is not None else 0,
        )
        x, y, sample_kind = x.to(device), y.to(device), sample_kind.to(device)

        pred = model(x).squeeze(-1)
        weights = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, neg_w))
        if confuser_pool is not None and n_confuser_batch > 0:
            weights = torch.where(sample_kind <= 0.0, torch.full_like(y, hard_w), weights)

        loss = (bce(pred, y) * weights).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if mine_confusers and remine_every > 0 and step > 0 and step % remine_every == 0:
            remined = _mine_model_confusers(
                model,
                neg,
                device=device,
                rng=rng_train,
            )
            confusers = _merge_confusers(confusers, remined)
            confuser_pool = _merge_confusers(
                confusers,
                np.asarray(hard_neg) if hard_neg is not None else None,
            )
            if confuser_pool is not None:
                n_val_conf = min(500, confuser_pool.shape[0])
                conf_val_idx = rng.integers(0, confuser_pool.shape[0], size=n_val_conf)
                confuser_val = confuser_pool[conf_val_idx]
                confuser_val_t = torch.from_numpy(confuser_val.astype(np.float32))

        if step % 500 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vp = model(x_val.to(device)).squeeze(-1).cpu()
                acc = ((vp >= 0.5) == (y_val >= 0.5)).float().mean().item()
                recall = ((vp >= 0.5) & (y_val >= 0.5)).sum().item() / max((y_val >= 0.5).sum().item(), 1)
                confuser_reject = 1.0
                if confuser_val_t is not None:
                    cp = model(confuser_val_t.to(device)).squeeze(-1).cpu()
                    confuser_fp = (cp >= 0.5).float().mean().item()
                    confuser_reject = 1.0 - confuser_fp
            model.train()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.3f}",
                rec=f"{recall:.3f}",
                conf=f"{confuser_reject:.3f}",
            )
            score = recall * 0.45 + acc * 0.15 + confuser_reject * 0.40
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
