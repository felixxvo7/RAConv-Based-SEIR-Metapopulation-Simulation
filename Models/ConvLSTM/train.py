#!/usr/bin/env python
"""
Train RAConv on SEIR grid data (predicting I / infectious).

Expects input from seir_preprocessed.npz which already contains:
  - Pre-split (train / val / test)
  - Per-cell min-max normalised
  - Sliding-window sampled
arrays with keys:
  X_train, Y_train, X_val, Y_val, X_test, Y_test   — shape (N, T, 16, 16)
  norm_min, norm_max                                  — for inverse transform

Default preprocessing settings (from seir_preprocessing.py):
  - P = 14 past frames (seq_len)
  - Q = 7  future frames (pred_len)
  - 16×16 spatial grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from fullModel import RAConv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(data_path: Path):
    """
    Load pre-processed seir_preprocessed.npz and return
    TensorDatasets ready for DataLoader.

    The .npz contains arrays of shape (N, seq_len, 16, 16).
    RAConv expects input [B, P, 1, H, W], so we insert a channel dim.
    """
    data = np.load(data_path)

    required_keys = {'X_train', 'Y_train', 'X_val', 'Y_val',
                     'X_test', 'Y_test', 'norm_min', 'norm_max'}
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"Missing keys in .npz: {missing}")

    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        # arr: (N, T, H, W) → (N, T, 1, H, W)
        return torch.from_numpy(arr[:, :, None, :, :].astype(np.float32))

    X_train = _to_tensor(data['X_train'])
    Y_train = _to_tensor(data['Y_train'])
    X_val   = _to_tensor(data['X_val'])
    Y_val   = _to_tensor(data['Y_val'])
    X_test  = _to_tensor(data['X_test'])
    Y_test  = _to_tensor(data['Y_test'])

    norm_min = data['norm_min']
    norm_max = data['norm_max']

    print(f"Loaded {data_path.name}")
    print(f"  X_train: {list(X_train.shape)}  Y_train: {list(Y_train.shape)}")
    print(f"  X_val:   {list(X_val.shape)}    Y_val:   {list(Y_val.shape)}")
    print(f"  X_test:  {list(X_test.shape)}   Y_test:  {list(Y_test.shape)}")

    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val,   Y_val)
    test_ds  = TensorDataset(X_test,  Y_test)

    return train_ds, val_ds, test_ds, norm_min, norm_max


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

    # ---- load pre-processed data ----
    data_path = Path(args.data_path).expanduser().resolve()
    train_ds, val_ds, test_ds, norm_min, norm_max = load_npz(data_path)

    # Infer P and Q from the loaded data
    P = train_ds[0][0].shape[0]   # seq_len  dimension
    Q = train_ds[0][1].shape[0]   # pred_len dimension
    print(f"Inferred P={P}, Q={Q} from data")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # ---- build model (P and Q come from the data) ----
    model = RAConv(
        in_channels=1,
        hidden_3d=[64, 96, 128],
        hidden_lstm=256,
        P=P,
        Q=Q,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ---- optimiser + scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
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
                "P":                P,
                "Q":                Q,
                "min_vals":         torch.from_numpy(norm_min),
                "max_vals":         torch.from_numpy(norm_max),
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
    p.add_argument("--data-path",       type=str,   default="../Preprocessing/preprocessed_output/seir_preprocessed.npz")
    p.add_argument("--checkpoint-dir",  type=str,   default="./checkpoints")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--cpu",             action="store_true")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
