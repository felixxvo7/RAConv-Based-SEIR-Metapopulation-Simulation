"""
AConvLSTM Experiment Runner
============================
End-to-end pipeline: load preprocessed SEIR data, train the AConvLSTM model,
evaluate on test set, and produce diagnostic plots.

Usage:
    python experiment_runner.py
    python experiment_runner.py --data path/to/seir_preprocessed.npz
    python experiment_runner.py --epochs 200 --batch_size 8 --lr 5e-4
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

# ── locate project modules ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from AConvLSTM import AConvLSTMLayers  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_NPZ = str(
    SCRIPT_DIR.parent / "Preprocessing" / "preprocessed_output" / "seir_preprocessed.npz"
)

CFG = dict(
    epochs=150,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    seed=42,

    hidden_channels=[256, 256],
    kernel_sizes=[3, 3],
    num_layers=2,

    results_dir=str(SCRIPT_DIR / "results"),
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
    X: (N, P, H, W) → (N, P, 1, H, W)
    Y: (N, Q, H, W) → (N, Q, 1, H, W)
    """
    Xt = torch.from_numpy(X).float().unsqueeze(2)
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

def build_model(device: torch.device) -> AConvLSTMLayers:
    model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=CFG["hidden_channels"],
        kernel_size=CFG["kernel_sizes"],
        num_layers=CFG["num_layers"],
        bias=True,
        use_attention=True,
    )
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _run_epoch(model: AConvLSTMLayers, loader: DataLoader, Q: int,
               criterion: nn.Module, optimizer=None,
               device: torch.device = torch.device("cpu")) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            _, last_states = model(xb)
            first_input = xb[:, -1, :, :, :]
            preds = model.predict_future(last_states, Q, first_input)

            loss = criterion(preds, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

    return total_loss / max(n, 1)


def train_model(model: AConvLSTMLayers, loaders: Dict[str, DataLoader],
                Q: int, device: torch.device
                ) -> Tuple[List[float], List[float], str]:
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                                 weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    train_hist: List[float] = []
    val_hist: List[float] = []
    best_val, wait = float("inf"), 0

    ckpt_dir = Path(CFG["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "best_aconvlstm.pth")

    for epoch in range(1, CFG["epochs"] + 1):
        t_loss = _run_epoch(model, loaders["train"], Q, criterion,
                            optimizer, device)
        v_loss = _run_epoch(model, loaders["val"], Q, criterion,
                            device=device)
        scheduler.step(v_loss)
        train_hist.append(t_loss)
        val_hist.append(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{CFG['epochs']}  "
              f"train={t_loss:.6f}  val={v_loss:.6f}  lr={lr_now:.1e}")

        if v_loss < best_val:
            best_val = v_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= CFG["patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return train_hist, val_hist, ckpt_path


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: AConvLSTMLayers, loader: DataLoader,
             Q: int, device: torch.device) -> Dict:
    model.eval()
    all_preds, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, last_states = model(xb)
        first_input = xb[:, -1, :, :, :]
        preds = model.predict_future(last_states, Q, first_input)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    rmse = mse ** 0.5

    per_step_mse = ((preds - targets) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
    per_step_mae = (preds - targets).abs().mean(dim=(0, 2, 3, 4)).tolist()

    spatial_mse = ((preds - targets) ** 2).mean(dim=(0, 1, 2)).numpy()

    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse,
        "per_step_mse": per_step_mse,
        "per_step_mae": per_step_mae,
        "spatial_mse": spatial_mse,
        "preds": preds,
        "targets": targets,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_hist: List[float], val_hist: List[float],
                     save_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_hist) + 1)
    ax.plot(epochs, train_hist, label="Train")
    ax.plot(epochs, val_hist, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("AConvLSTM Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved loss curves -> {save_path}")


def plot_per_step_metrics(per_step_mse: List[float],
                          per_step_mae: List[float],
                          save_path: str):
    Q = len(per_step_mse)
    steps = range(1, Q + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(steps, per_step_mse, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Forecast Step")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE by Forecast Step")
    ax1.set_xticks(list(steps))
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(steps, per_step_mae, color="coral", alpha=0.8)
    ax2.set_xlabel("Forecast Step")
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE by Forecast Step")
    ax2.set_xticks(list(steps))
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-step metrics -> {save_path}")


def plot_spatial_error(spatial_mse: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(spatial_mse, cmap="YlOrRd", interpolation="nearest")
    ax.set_title("Spatial MSE (averaged over samples & time steps)")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    plt.colorbar(im, ax=ax, label="MSE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved spatial error map -> {save_path}")


def plot_predictions(preds: torch.Tensor, targets: torch.Tensor,
                     sample_indices: List[int],
                     cells: List[Tuple[int, int]],
                     save_path: str):
    n_samples = len(sample_indices)
    n_cells = len(cells)
    fig, axes = plt.subplots(n_samples, n_cells,
                             figsize=(6 * n_cells, 4 * n_samples),
                             squeeze=False)

    Q = preds.shape[1]
    steps = range(1, Q + 1)

    for i, idx in enumerate(sample_indices):
        for j, (r, c) in enumerate(cells):
            ax = axes[i][j]
            pred_vals = preds[idx, :, 0, r, c].numpy()
            true_vals = targets[idx, :, 0, r, c].numpy()
            ax.plot(steps, true_vals, "k-", marker="s", linewidth=2,
                    label="Actual")
            ax.plot(steps, pred_vals, "b-", marker="o", label="Predicted")
            ax.set_xlabel("Forecast Step")
            ax.set_ylabel("Normalised I")
            ax.set_title(f"Sample {idx}, Cell ({r},{c})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction plots -> {save_path}")


def _print_metrics(results: Dict):
    w = 50
    print("\n" + "=" * w)
    print(f"{'Metric':<25s} {'Value':>12s}")
    print("-" * w)
    print(f"{'Test MSE':<25s} {results['MSE']:12.6f}")
    print(f"{'Test MAE':<25s} {results['MAE']:12.6f}")
    print(f"{'Test RMSE':<25s} {results['RMSE']:12.6f}")
    print("=" * w)

    print("\nPer-step breakdown:")
    print(f"  {'Step':<6s} {'MSE':>10s} {'MAE':>10s}")
    print(f"  {'-'*28}")
    for step, (mse, mae) in enumerate(
        zip(results["per_step_mse"], results["per_step_mae"]), 1
    ):
        print(f"  {step:<6d} {mse:10.6f} {mae:10.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AConvLSTM Experiment Runner")
    parser.add_argument("--data", type=str, default=DEFAULT_NPZ,
                        help="Path to seir_preprocessed.npz")
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    parser.add_argument("--patience", type=int, default=CFG["patience"])
    parser.add_argument("--seed", type=int, default=CFG["seed"])
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even when CUDA is available")
    args = parser.parse_args()

    CFG["epochs"] = args.epochs
    CFG["batch_size"] = args.batch_size
    CFG["lr"] = args.lr
    CFG["patience"] = args.patience

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data}")
    data = load_npz(args.data)
    Q = data["Y_train"].shape[1]
    print(f"  X_train {data['X_train'].shape}  Y_train {data['Y_train'].shape}")
    print(f"  X_val   {data['X_val'].shape}    Y_val   {data['Y_val'].shape}")
    print(f"  X_test  {data['X_test'].shape}   Y_test  {data['Y_test'].shape}")
    print(f"  Forecast horizon Q = {Q}")

    loaders = build_loaders(data, CFG["batch_size"])

    # ── model ─────────────────────────────────────────────────────────────
    print("\nBuilding AConvLSTM model ...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {total_params:,}")
    print(f"  Hidden channels      : {CFG['hidden_channels']}")
    print(f"  Kernel sizes         : {CFG['kernel_sizes']}")
    print(f"  Layers               : {CFG['num_layers']}")

    # ── train ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training AConvLSTM")
    print("=" * 60)

    t0 = time.time()
    train_hist, val_hist, ckpt_path = train_model(model, loaders, Q, device)
    elapsed = time.time() - t0

    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Epochs run       : {len(train_hist)}")
    print(f"  Best val loss    : {min(val_hist):.6f}")
    print(f"  Checkpoint saved : {ckpt_path}")

    # ── evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    results = evaluate(model, loaders["test"], Q, device)
    _print_metrics(results)

    # ── save metrics ──────────────────────────────────────────────────────
    out_dir = Path(CFG["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "MSE": results["MSE"],
        "MAE": results["MAE"],
        "RMSE": results["RMSE"],
        "per_step_mse": results["per_step_mse"],
        "per_step_mae": results["per_step_mae"],
        "config": {k: v for k, v in CFG.items() if k != "results_dir"},
        "training_time_s": round(elapsed, 1),
        "total_params": total_params,
        "epochs_run": len(train_hist),
    }
    metrics_path = str(out_dir / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved metrics -> {metrics_path}")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_loss_curves(train_hist, val_hist,
                     str(out_dir / "loss_curves.png"))

    plot_per_step_metrics(results["per_step_mse"], results["per_step_mae"],
                          str(out_dir / "per_step_metrics.png"))

    plot_spatial_error(results["spatial_mse"],
                       str(out_dir / "spatial_error.png"))

    n_test = results["preds"].shape[0]
    sample_indices = sorted({0, n_test // 2, n_test - 1})
    cells = [(4, 4), (8, 8), (12, 12)]
    plot_predictions(results["preds"], results["targets"],
                     sample_indices, cells,
                     str(out_dir / "pred_vs_actual.png"))

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
