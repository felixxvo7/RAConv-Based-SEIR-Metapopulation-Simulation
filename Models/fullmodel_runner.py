"""
RAConv Experiment Runner
========================
End-to-end pipeline: load preprocessed SEIR data, train the full RAConv model
(ResBlock3D + AConvLSTM), evaluate on test set, and produce diagnostic plots.

Trains separately for each lookback window P (P4, P6, P8, P14 by default).
Results are saved to results_raconv/P<N>/ subdirectories.

Usage:
    python fullmodel_runner.py                          # trains P4, P6, P8, P14
    python fullmodel_runner.py --p 4 8                  # trains only P4 and P8
    python fullmodel_runner.py --epochs 200 --lr 5e-4
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from RAConvLSTM import RAConv  # noqa: E402

NPZ_DIR = SCRIPT_DIR / "Preprocessing" / "preprocessed_output"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    epochs=50,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,

    results_dir=str(SCRIPT_DIR / "results_raconv"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _make_loader(X: np.ndarray, Y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    """
    X: (N, T, H, W) → (N, 1, T, H, W)   Conv3D layout for RAConv
    Y: (N, Q, H, W) → (N, Q, 1, H, W)
    """
    Xt = torch.from_numpy(X).float().unsqueeze(1)
    Yt = torch.from_numpy(Y).float().unsqueeze(2)
    return DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True)


def build_loaders(data: Dict[str, np.ndarray],
                  batch_size: int) -> Dict[str, DataLoader]:
    return {
        "train": _make_loader(data["X_train"], data["Y_train"], batch_size, shuffle=True),
        "val":   _make_loader(data["X_val"],   data["Y_val"],   batch_size, shuffle=False),
        "test":  _make_loader(data["X_test"],  data["Y_test"],  batch_size, shuffle=False),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(Q: int, device: torch.device) -> RAConv:
    model = RAConv(in_channels=1, out_steps=Q)
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _run_epoch(model: RAConv, loader: DataLoader,
               criterion: nn.Module, optimizer=None,
               device: torch.device = torch.device("cpu")) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)

            loss = criterion(preds, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

    return total_loss / max(n, 1)


def train_model(model: RAConv, loaders: Dict[str, DataLoader],
                device: torch.device,
                ckpt_path: str) -> Tuple[List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                                 weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    train_hist: List[float] = []
    val_hist:   List[float] = []
    best_val = float("inf")

    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CFG["epochs"] + 1):
        t_loss = _run_epoch(model, loaders["train"], criterion,
                            optimizer, device)
        v_loss = _run_epoch(model, loaders["val"], criterion,
                            device=device)
        scheduler.step(v_loss)
        train_hist.append(t_loss)
        val_hist.append(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{CFG['epochs']}  "
              f"train={t_loss:.6f}  val={v_loss:.6f}  lr={lr_now:.1e}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ↳ best val={best_val:.6f} — checkpoint saved")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return train_hist, val_hist


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _inverse_transform(tensor: torch.Tensor,
                       norm_min: np.ndarray,
                       norm_max: np.ndarray) -> torch.Tensor:
    """(B, Q, 1, H, W) normalised → real scale using global min/max scalars."""
    mn = torch.from_numpy(norm_min).float()   # scalar (0-D tensor)
    mx = torch.from_numpy(norm_max).float()   # scalar (0-D tensor)
    return tensor * (mx - mn + 1e-8) + mn


@torch.no_grad()
def evaluate(model: RAConv, loader: DataLoader,
             device: torch.device,
             norm_min: np.ndarray = None,
             norm_max: np.ndarray = None) -> Dict:
    model.eval()
    all_preds, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # --- normalised-space metrics ---
    mse  = ((preds - targets) ** 2).mean().item()
    mae  = (preds - targets).abs().mean().item()
    rmse = mse ** 0.5

    per_step_mse = ((preds - targets) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
    per_step_mae = (preds - targets).abs().mean(dim=(0, 2, 3, 4)).tolist()
    spatial_mse  = ((preds - targets) ** 2).mean(dim=(0, 1, 2)).numpy()

    result = {
        "MSE": mse, "MAE": mae, "RMSE": rmse,
        "per_step_mse": per_step_mse,
        "per_step_mae": per_step_mae,
        "spatial_mse":  spatial_mse,
        "preds":   preds,
        "targets": targets,
    }

    # --- real-space metrics (inverse-transformed) ---
    if norm_min is not None and norm_max is not None:
        preds_real   = _inverse_transform(preds, norm_min, norm_max).clamp(min=0)
        targets_real = _inverse_transform(targets, norm_min, norm_max)

        real_mse  = ((preds_real - targets_real) ** 2).mean().item()
        real_mae  = (preds_real - targets_real).abs().mean().item()
        real_rmse = real_mse ** 0.5

        real_per_step_mse = ((preds_real - targets_real) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
        real_per_step_mae = (preds_real - targets_real).abs().mean(dim=(0, 2, 3, 4)).tolist()

        result.update({
            "real_MSE": real_mse, "real_MAE": real_mae, "real_RMSE": real_rmse,
            "real_per_step_mse": real_per_step_mse,
            "real_per_step_mae": real_per_step_mae,
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_hist, val_hist, save_path, p_label):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_hist) + 1)
    ax.plot(epochs, train_hist, label="Train")
    ax.plot(epochs, val_hist,   label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"RAConv Training & Validation Loss  [{p_label}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved loss curves -> {save_path}")


def plot_per_step_metrics(per_step_mse, per_step_mae, save_path, p_label):
    Q = len(per_step_mse)
    steps = range(1, Q + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(steps, per_step_mse, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Forecast Step")
    ax1.set_ylabel("MSE")
    ax1.set_title(f"MSE by Forecast Step  [{p_label}]")
    ax1.set_xticks(list(steps))
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(steps, per_step_mae, color="coral", alpha=0.8)
    ax2.set_xlabel("Forecast Step")
    ax2.set_ylabel("MAE")
    ax2.set_title(f"MAE by Forecast Step  [{p_label}]")
    ax2.set_xticks(list(steps))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-step metrics -> {save_path}")


def plot_spatial_error(spatial_mse, save_path, p_label):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(spatial_mse, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(f"Spatial MSE  [{p_label}]")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    plt.colorbar(im, ax=ax, label="MSE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved spatial error map -> {save_path}")


def plot_predictions(preds, targets, sample_indices, cells, save_path, p_label):
    Q = preds.shape[1]
    steps = range(1, Q + 1)
    n_samples = len(sample_indices)
    n_cells   = len(cells)
    fig, axes = plt.subplots(n_samples, n_cells,
                             figsize=(6 * n_cells, 4 * n_samples),
                             squeeze=False)
    for i, idx in enumerate(sample_indices):
        for j, (r, c) in enumerate(cells):
            ax = axes[i][j]
            ax.plot(steps, targets[idx, :, 0, r, c].numpy(),
                    "k-", marker="s", linewidth=2, label="Actual")
            ax.plot(steps, preds[idx, :, 0, r, c].numpy(),
                    "b-", marker="o", label="Predicted")
            ax.set_xlabel("Forecast Step")
            ax.set_ylabel("Normalised I")
            ax.set_title(f"[{p_label}] Sample {idx}, Cell ({r},{c})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction plots -> {save_path}")


def _print_metrics(results: Dict, p_label: str):
    w = 55
    print(f"\n{'=' * w}  [{p_label}]")

    # normalised space
    print(f"\n  {'Metric (normalised)':<30s} {'Value':>12s}")
    print(f"  {'-' * 44}")
    print(f"  {'Test MSE':<30s} {results['MSE']:12.6f}")
    print(f"  {'Test MAE':<30s} {results['MAE']:12.6f}")
    print(f"  {'Test RMSE':<30s} {results['RMSE']:12.6f}")

    # real space
    if "real_MSE" in results:
        print(f"\n  {'Metric (real scale)':<30s} {'Value':>14s}")
        print(f"  {'-' * 46}")
        print(f"  {'Test MSE':<30s} {results['real_MSE']:14.2f}")
        print(f"  {'Test MAE':<30s} {results['real_MAE']:14.2f}")
        print(f"  {'Test RMSE':<30s} {results['real_RMSE']:14.2f}")

    print(f"\n{'=' * w}")

    # per-step breakdown
    has_real = "real_per_step_mse" in results
    print("\nPer-step breakdown:")
    if has_real:
        print(f"  {'Step':<6s} {'MSE(norm)':>10s} {'MAE(norm)':>10s}  "
              f"{'MSE(real)':>12s} {'MAE(real)':>12s}")
        print(f"  {'-' * 56}")
        for step, (mse, mae, rmse, rmae) in enumerate(
            zip(results["per_step_mse"], results["per_step_mae"],
                results["real_per_step_mse"], results["real_per_step_mae"]), 1
        ):
            print(f"  {step:<6d} {mse:10.6f} {mae:10.6f}  {rmse:12.2f} {rmae:12.2f}")
    else:
        print(f"  {'Step':<6s} {'MSE':>10s} {'MAE':>10s}")
        print(f"  {'-' * 28}")
        for step, (mse, mae) in enumerate(
            zip(results["per_step_mse"], results["per_step_mae"]), 1
        ):
            print(f"  {step:<6d} {mse:10.6f} {mae:10.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PER-P EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(p: int, device: torch.device):
    p_label  = f"P{p}"
    npz_path = NPZ_DIR / f"seir_preprocessed_{p_label}.npz"
    out_dir  = Path(CFG["results_dir"]) / p_label
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / f"best_raconv_{p_label}.pth")

    print(f"\n{'#' * 65}")
    print(f"  Experiment: {p_label}   data: {npz_path}")
    print(f"{'#' * 65}")

    # data
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Re-run preprocessing:\n"
            f"  cd Models/Preprocessing && python seir_preprocessing.py"
        )
    data = load_npz(str(npz_path))
    Q = data["Y_train"].shape[1]   # always 7 (fixed pred_len)
    P_seq = data["X_train"].shape[1]  # lookback window = P

    assert P_seq == p, (
        f"DATA MISMATCH: {p_label} expects lookback P={p}, but the NPZ "
        f"contains X with time dim={P_seq}. "
        f"Re-run preprocessing:  cd Models/Preprocessing && python seir_preprocessing.py"
    )

    print(f"  Lookback window  P = {P_seq}")
    print(f"  Forecast horizon Q = {Q}")
    print(f"  X_train {data['X_train'].shape}  Y_train {data['Y_train'].shape}")
    print(f"  X_val   {data['X_val'].shape}    Y_val   {data['Y_val'].shape}")
    print(f"  X_test  {data['X_test'].shape}   Y_test  {data['Y_test'].shape}")

    loaders = build_loaders(data, CFG["batch_size"])

    # model
    print(f"\nBuilding RAConv for {p_label} ...")
    model = build_model(Q, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {total_params:,}")
    print(f"  Forecast steps (Q)   : {Q}")

    # train
    print(f"\n{'=' * 60}")
    print(f"Training RAConv  [{p_label}]")
    print("=" * 60)

    t0 = time.time()
    train_hist, val_hist = train_model(model, loaders, device, ckpt_path)
    elapsed = time.time() - t0

    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Epochs run    : {len(train_hist)}")
    print(f"  Best val loss : {min(val_hist):.6f}")
    print(f"  Checkpoint    : {ckpt_path}")

    # evaluate
    print(f"\nEvaluating on test set  [{p_label}] ...")
    norm_min = data.get("norm_min", None)
    norm_max = data.get("norm_max", None)
    results = evaluate(model, loaders["test"], device, norm_min, norm_max)
    _print_metrics(results, p_label)

    # save metrics
    metrics = {
        "P": p,
        "MSE":  results["MSE"],
        "MAE":  results["MAE"],
        "RMSE": results["RMSE"],
        "per_step_mse": results["per_step_mse"],
        "per_step_mae": results["per_step_mae"],
        "config": {k: v for k, v in CFG.items() if k != "results_dir"},
        "training_time_s": round(elapsed, 1),
        "total_params": total_params,
        "epochs_run": len(train_hist),
    }
    if "real_MSE" in results:
        metrics.update({
            "real_MSE":  results["real_MSE"],
            "real_MAE":  results["real_MAE"],
            "real_RMSE": results["real_RMSE"],
            "real_per_step_mse": results["real_per_step_mse"],
            "real_per_step_mae": results["real_per_step_mae"],
        })
    metrics_path = str(out_dir / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved metrics -> {metrics_path}")

    # plots
    plot_loss_curves(train_hist, val_hist,
                     str(out_dir / "loss_curves.png"), p_label)

    plot_per_step_metrics(results["per_step_mse"], results["per_step_mae"],
                          str(out_dir / "per_step_metrics.png"), p_label)

    plot_spatial_error(results["spatial_mse"],
                       str(out_dir / "spatial_error.png"), p_label)

    n_test = results["preds"].shape[0]
    sample_indices = sorted({0, n_test // 2, n_test - 1})
    cells = [(4, 4), (8, 8), (12, 12)]
    plot_predictions(results["preds"], results["targets"],
                     sample_indices, cells,
                     str(out_dir / "pred_vs_actual.png"), p_label)

    print(f"\n  All {p_label} results saved to {out_dir}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAConv Experiment Runner")
    parser.add_argument("--p", type=int, nargs="+", default=[4, 6, 8, 14],
                        choices=[4, 6, 8, 14],
                        help="Lookback window(s) P to train (default: 4 6 8 14).")
    parser.add_argument("--epochs",     type=int,   default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--seed",       type=int,   default=CFG["seed"])
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even when CUDA is available")
    args = parser.parse_args()

    CFG["epochs"]          = args.epochs
    CFG["batch_size"]      = args.batch_size
    CFG["lr"]              = args.lr

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    all_metrics = {}
    for p in args.p:
        all_metrics[f"P{p}"] = run_experiment(p, device)

    # ── summary across all P ──────────────────────────────────────────────
    has_real = any("real_RMSE" in m for m in all_metrics.values())
    print(f"\n{'=' * 80}")
    print("Summary")
    if has_real:
        print(f"{'P':<6} {'MSE(norm)':>12} {'MAE(norm)':>12} {'RMSE(norm)':>12}  "
              f"{'RMSE(real)':>12}")
        print("-" * 60)
        for label, m in all_metrics.items():
            print(f"{label:<6} {m['MSE']:12.6f} {m['MAE']:12.6f} {m['RMSE']:12.6f}  "
                  f"{m.get('real_RMSE', float('nan')):12.2f}")
    else:
        print(f"{'P':<6} {'MSE':>12} {'MAE':>12} {'RMSE':>12}")
        print("-" * 45)
        for label, m in all_metrics.items():
            print(f"{label:<6} {m['MSE']:12.6f} {m['MAE']:12.6f} {m['RMSE']:12.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
