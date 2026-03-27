re """
AConvLSTM Ablation Experiment Runner
=====================================
End-to-end pipeline: load preprocessed data, train two model variants,
evaluate, and produce comparison plots.

Variant 1 — ConvLSTM  (use_attention=False)
Variant 2 — AConvLSTM (use_attention=True)

Usage:
    python experiment_runner.py
    python experiment_runner.py --data path/to/seir_preprocessed.npz
"""

import os
import sys
import argparse
import json
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
sys.path.insert(0, str(SCRIPT_DIR.parent / "Preprocessing"))

from AConvLSTM import AConvLSTMLayers  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_NPZ = str(
    SCRIPT_DIR.parent / "Preprocessing" / "preprocessed_output" / "seir_preprocessed.npz"
)

CFG = dict(
    # training
    epochs=150,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-5,
    patience=8,

    # model
    hidden_channels=(256, 256, 256),
    kernel_sizes=(3, 3),
    num_layers=2,

    # output
    results_dir=str(SCRIPT_DIR / "results"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int,
                shuffle: bool) -> DataLoader:
    """
    X arrives as (N, P, 16, 16).  Unsqueeze to (N, P, 1, 16, 16) so the
    model receives the 5-D tensor it expects: (B, T, C, H, W).
    Y is treated the same way: (N, Q, 16, 16) → (N, Q, 1, 16, 16).
    """
    Xt = torch.from_numpy(X).float().unsqueeze(2)   # (N,P,1,H,W)
    Yt = torch.from_numpy(Y).float().unsqueeze(2)   # (N,Q,1,H,W)
    return DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True)


def build_loaders(data: Dict[str, np.ndarray],
                  batch_size: int) -> Dict[str, DataLoader]:
    return {
        "train": make_loader(data["X_train"], data["Y_train"], batch_size, shuffle=True),
        "val":   make_loader(data["X_val"],   data["Y_val"],   batch_size, shuffle=False),
        "test":  make_loader(data["X_test"],  data["Y_test"],  batch_size, shuffle=False),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(use_attention: bool, device: torch.device) -> AConvLSTMLayers:
    model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=list(CFG["hidden_channels"]),
        kernel_size=list(CFG["kernel_sizes"]),
        num_layers=CFG["num_layers"],
        bias=True,
        use_attention=use_attention,
    )
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model: AConvLSTMLayers, loader: DataLoader, Q: int,
              criterion: nn.Module, optimizer=None,
              device: torch.device = torch.device("cpu")) -> float:
    """Run one epoch.  If optimizer is None → eval mode (no grad)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            _, last_states = model(xb)
            first_input = xb[:, -1, :, :, :]   # (B, C, H, W) — last frame
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
                Q: int, device: torch.device,
                tag: str = "") -> Tuple[List[float], List[float], str]:
    """Full training loop with early stopping.  Returns histories + ckpt path."""
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                                weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    train_hist, val_hist = [], []
    best_val, wait = float("inf"), 0

    ckpt_dir = Path(CFG["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / f"best_{tag}.pth")

    for epoch in range(1, CFG["epochs"] + 1):
        t_loss = run_epoch(model, loaders["train"], Q, criterion,
                           optimizer, device)
        v_loss = run_epoch(model, loaders["val"], Q, criterion,
                           device=device)
        scheduler.step(v_loss)
        train_hist.append(t_loss)
        val_hist.append(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  [{tag}] Epoch {epoch:3d}/{CFG['epochs']}  "
              f"train={t_loss:.6f}  val={v_loss:.6f}  lr={lr_now:.1e}")

        if v_loss < best_val:
            best_val = v_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= CFG["patience"]:
                print(f"  [{tag}] Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return train_hist, val_hist, ckpt_path


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: AConvLSTMLayers, loader: DataLoader, Q: int,
             device: torch.device) -> Dict[str, float]:
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
    return {"MSE": mse, "MAE": mae, "preds": preds, "targets": targets}


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(histories: Dict[str, Tuple[List[float], List[float]]],
                     save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split in zip(axes, ("Train", "Val")):
        idx = 0 if split == "Train" else 1
        for name, (th, vh) in histories.items():
            vals = th if split == "Train" else vh
            ax.plot(range(1, len(vals) + 1), vals, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{split} Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved loss curves → {save_path}")


def plot_prediction_comparison(
    results: Dict[str, Dict],
    sample_idx: int,
    cell: Tuple[int, int],
    save_path: str,
):
    """Overlay predicted vs actual for one grid cell across Q steps."""
    r, c = cell
    fig, ax = plt.subplots(figsize=(10, 5))
    Q = None

    for name, res in results.items():
        preds = res["preds"][sample_idx, :, 0, r, c].numpy()   # (Q,)
        Q = len(preds)
        ax.plot(range(1, Q + 1), preds, marker="o", label=f"{name} (pred)")

    actual = list(results.values())[0]["targets"][sample_idx, :, 0, r, c].numpy()
    ax.plot(range(1, Q + 1), actual, "k--", marker="s", linewidth=2,
            label="Actual")

    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("Normalised Infected (I)")
    ax.set_title(f"Prediction vs Actual — grid cell ({r},{c}), sample {sample_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction plot → {save_path}")


def print_metrics_table(results: Dict[str, Dict]):
    header = f"{'Variant':<20s} {'Test MSE':>12s} {'Test MAE':>12s}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(f"{name:<20s} {res['MSE']:12.6f} {res['MAE']:12.6f}")
    print("=" * len(header))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AConvLSTM Ablation Runner")
    parser.add_argument("--data", type=str, default=DEFAULT_NPZ,
                        help="Path to seir_preprocessed.npz")
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    args = parser.parse_args()

    CFG["epochs"] = args.epochs
    CFG["batch_size"] = args.batch_size
    CFG["lr"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data}")
    data = load_npz(args.data)
    Q = data["Y_train"].shape[1]
    print(f"  X_train {data['X_train'].shape}  Y_train {data['Y_train'].shape}  Q={Q}")

    loaders = build_loaders(data, CFG["batch_size"])

    # ── ablation variants ─────────────────────────────────────────────────
    variants = {
        "ConvLSTM":  False,   # use_attention = False
        "AConvLSTM": True,    # use_attention = True
    }

    histories: Dict[str, Tuple[List[float], List[float]]] = {}
    test_results: Dict[str, Dict] = {}
    out_dir = Path(CFG["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, use_att in variants.items():
        print(f"\n{'=' * 60}")
        print(f"Training  {name}  (use_attention={use_att})")
        print("=" * 60)

        model = build_model(use_attention=use_att, device=device)
        th, vh, ckpt = train_model(model, loaders, Q, device, tag=name)
        histories[name] = (th, vh)

        print(f"\n  Evaluating {name} on test set ...")
        res = evaluate(model, loaders["test"], Q, device)
        test_results[name] = res
        print(f"  {name} test MSE={res['MSE']:.6f}  MAE={res['MAE']:.6f}")

    # ── metrics table ─────────────────────────────────────────────────────
    print_metrics_table(test_results)

    metrics_path = str(out_dir / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {k: {"MSE": v["MSE"], "MAE": v["MAE"]} for k, v in test_results.items()},
            f, indent=2,
        )
    print(f"  Saved metrics → {metrics_path}")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_loss_curves(histories, str(out_dir / "loss_curves.png"))

    sample_idx = 0
    cell = (8, 8)
    plot_prediction_comparison(test_results, sample_idx, cell,
                               str(out_dir / "pred_vs_actual.png"))

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
