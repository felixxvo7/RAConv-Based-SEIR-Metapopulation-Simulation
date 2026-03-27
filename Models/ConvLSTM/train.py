#!/usr/bin/env python
"""
Train RAConv on SEIR grid data (predicting I / infectious).

Paper-aligned settings:
  - P = 8 past frames
  - Q = 4 future frames
  - 16×16 spatial grid (one cluster of base stations)

Normalization:
  The paper (Eq. 1) normalises per base-station using the min and max
  of the FULL time series at that station (not train-split only).
  This matches "d^(m,n)_s = (d_hat - min) / (max - min)" with no
  mention of a train-only window.

Optimiser / LR:
  Adam with default settings (lr=1e-3) and ReduceLROnPlateau scheduler
  for stable convergence.

Loss:
  MSE, as the paper minimises the mean-squared error between
  predictions and ground truth (Section III-B-3).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from fullModel import RAConv


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class SplitRanges:
    train: Tuple[int, int]
    val:   Tuple[int, int]
    test:  Tuple[int, int]


class SEIRWindowDataset(Dataset):
    """
    Sliding-window dataset over a normalised [T, H, W] array.
    Returns:
        x : [P, 1, H, W]  — P past frames
        y : [Q, 1, H, W]  — Q future frames (ground truth)
    """

    def __init__(self, series: np.ndarray, P: int, Q: int,
                 start: int, end: int):
        self.series = series
        self.P = P
        self.Q = Q
        self.start = start
        self.length = (end - start) - (P + Q) + 1
        if self.length <= 0:
            raise ValueError(
                f"Split [{start},{end}) is too small for P={P}, Q={Q}.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        t0 = self.start + idx
        x  = self.series[t0        : t0 + self.P]           # [P, H, W]
        y  = self.series[t0 + self.P : t0 + self.P + self.Q] # [Q, H, W]
        x  = x[:, None, :, :].astype(np.float32)            # [P, 1, H, W]
        y  = y[:, None, :, :].astype(np.float32)            # [Q, 1, H, W]
        return torch.from_numpy(x), torch.from_numpy(y)


def compute_splits(T: int, train_ratio: float,
                   val_ratio: float) -> SplitRanges:
    train_end = int(T * train_ratio)
    val_end   = train_end + int(T * val_ratio)
    val_end   = min(val_end, T - 1)
    return SplitRanges(
        train=(0, train_end),
        val  =(train_end, val_end),
        test =(val_end, T),
    )


def per_station_normalize(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-cell (base-station) min-max normalisation to [0, 1].

    Paper Eq. (1):
        d^(m,n) = (d_hat^(m,n) - min(d_hat^(m,n))) /
                  (max(d_hat^(m,n)) - min(d_hat^(m,n)))

    min / max are computed over the FULL time axis of each cell
    (no train-only restriction) as the paper's formula uses all Ts steps.
    """
    # series: [T, H, W]
    min_v = series.min(axis=0, keepdims=True)          # [1, H, W]
    max_v = series.max(axis=0, keepdims=True)          # [1, H, W]
    denom = np.maximum(max_v - min_v, 1e-8)
    normed = (series - min_v) / denom
    return normed, min_v, max_v


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    model.eval()
    mse_sum, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            pred  = model(x)
            mse_sum += nn.functional.mse_loss(pred, y,
                                              reduction='sum').item()
            count   += y.numel()
    return float(np.sqrt(mse_sum / max(count, 1)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- load data ----
    data_path = Path(args.data_path).expanduser().resolve()
    raw = np.load(data_path)          # [T, H, W, 4]
    if raw.ndim != 4 or raw.shape[-1] < 3:
        raise ValueError(f"Unexpected shape {raw.shape}; expected [T,H,W,4].")

    series = raw[:, :, :, 2]          # I channel → [T, H, W]
    T, H, W = series.shape
    print(f"Data shape: T={T}, H={H}, W={W}")

    # ---- normalise using full time-series per cell (paper Eq. 1) ----
    series_norm, min_v, max_v = per_station_normalize(series)

    # ---- build train / val / test splits ----
    splits = compute_splits(T, args.train_ratio, args.val_ratio)
    print(f"Splits  train={splits.train}  val={splits.val}  test={splits.test}")

    train_ds = SEIRWindowDataset(series_norm, args.P, args.Q, *splits.train)
    val_ds   = SEIRWindowDataset(series_norm, args.P, args.Q, *splits.val)
    test_ds  = SEIRWindowDataset(series_norm, args.P, args.Q, *splits.test)
    print(f"Windows  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # ---- build model ----
    model = RAConv(
        in_channels=1,
        hidden_3d=[64, 96, 128],
        hidden_lstm=256,
        P=args.P,
        Q=args.Q,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ---- optimiser + scheduler ----
    # Paper uses Adam (default settings inferred from standard practice).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    # Halve LR if val RMSE does not improve for 10 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    criterion = nn.MSELoss()

    # ---- checkpoint directory ----
    ckpt_dir  = Path(args.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "raconv_best.pt"
    best_val  = float("inf")

    # ---- training loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, batches = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # Gradient clipping for stable training with LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item()
            batches += 1

        train_rmse = float(np.sqrt(running / max(batches, 1)))
        val_rmse   = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        print(f"Epoch {epoch:03d} | "
              f"train RMSE {train_rmse:.6f} | val RMSE {val_rmse:.6f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "P":                args.P,
                "Q":                args.Q,
                "min_vals":         torch.from_numpy(min_v),
                "max_vals":         torch.from_numpy(max_v),
            }, best_path)
            print(f"  ✓ saved best checkpoint (val RMSE {best_val:.6f})")

    # ---- final test evaluation ----
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded best checkpoint from epoch {ckpt['epoch']}")

    test_rmse = evaluate(model, test_loader, device)
    print(f"Test RMSE: {test_rmse:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train RAConv to predict the infectious (I) channel.")
    p.add_argument("--data-path",       type=str,   default="../../src_data/seir_grid_data.npy")
    p.add_argument("--checkpoint-dir",  type=str,   default="./checkpoints")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--P",               type=int,   default=8)
    p.add_argument("--Q",               type=int,   default=4)
    p.add_argument("--train-ratio",     type=float, default=0.7)
    p.add_argument("--val-ratio",       type=float, default=0.15)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--cpu",             action="store_true")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
