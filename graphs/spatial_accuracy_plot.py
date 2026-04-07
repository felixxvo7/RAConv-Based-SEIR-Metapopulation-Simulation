"""
Spatial Accuracy Comparison — Day-level Heatmap (Fig. 8 style)
==============================================================
Produces a side-by-side 16x16 heatmap comparing:
    Ground Truth  |  RAConv (Full Model)  |  AConvLSTM (Ablation)
for a single target day across all 256 cities.

Pixel brightness indicates the infected count of the corresponding city.
All three panels share a common colour scale for direct comparison.

Usage:
    python spatial_accuracy_plot.py
    python spatial_accuracy_plot.py --day 255 --p 14
    python spatial_accuracy_plot.py --cpu
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / "Models"
sys.path.insert(0, str(MODELS_DIR))

from RAConvLSTM import RAConv                    # noqa: E402
from AConvLSTM.AConvLSTM import AConvLSTMLayers  # noqa: E402

GRID_H, GRID_W = 16, 16
TRAIN_DAYS = 240


def load_npz(npz_path: Path):
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def find_all_windows_for_day(target_day: int, P: int, Q: int, n_windows: int):
    """Return all (window_idx, step_idx) pairs that produce *target_day*.
    Averaging across these gives a holistic view over all forecast horizons."""
    test_start = TRAIN_DAYS
    pairs = []
    for step in range(Q):
        win = target_day - test_start - P - step
        if 0 <= win < n_windows:
            pairs.append((win, step))
    if not pairs:
        raise ValueError(
            f"Day {target_day} is not reachable in the test set "
            f"(test starts at day {test_start}, P={P}, Q={Q}, "
            f"n_windows={n_windows})"
        )
    return pairs


@torch.no_grad()
def predict_single(model, x_tensor, model_type, Q):
    """Run inference on a single sample and return (1, Q, 1, H, W) tensor."""
    model.eval()
    if model_type == "raconv":
        return model(x_tensor)
    _, last_states = model(x_tensor)
    first_input = x_tensor[:, -1, :, :, :]
    return model.predict_future(last_states, Q, first_input)


def inverse_transform(arr, norm_min, norm_max):
    return arr * (norm_max - norm_min + 1e-8) + norm_min


def main():
    parser = argparse.ArgumentParser(
        description="Spatial accuracy heatmap comparison on a specific day."
    )
    parser.add_argument("--day", type=int, default=275,
                        help="Target simulation day to visualise (default 275: all Q steps available).")
    parser.add_argument("--p", type=int, default=14, choices=[4, 6, 8, 10, 14],
                        help="Lookback window P (determines which NPZ / checkpoints to load).")
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--raconv-ckpt", type=Path, default=None)
    parser.add_argument("--aconvlstm-ckpt", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    p_label = f"P{args.p}"
    preproc_dir = ROOT_DIR / "Models" / "Preprocessing" / "preprocessed_output"

    if args.npz is None:
        args.npz = preproc_dir / f"seir_preprocessed_{p_label}.npz"
    if args.raconv_ckpt is None:
        args.raconv_ckpt = (
            ROOT_DIR / "Models" / "results_fullmodel" / p_label / f"best_raconv_{p_label}.pth"
        )
    if args.aconvlstm_ckpt is None:
        args.aconvlstm_ckpt = (
            ROOT_DIR / "Models" / "results_ablation" / p_label / f"best_aconvlstm_{p_label}.pth"
        )

    for tag, path in [
        ("NPZ", args.npz),
        ("RAConv checkpoint", args.raconv_ckpt),
        ("AConvLSTM checkpoint", args.aconvlstm_ckpt),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{tag} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    data = load_npz(args.npz)
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    P = X_test.shape[1]
    Q = Y_test.shape[1]
    norm_min = float(data["norm_min"])
    norm_max = float(data["norm_max"])

    n_windows = X_test.shape[0]
    pairs = find_all_windows_for_day(args.day, P, Q, n_windows)
    print(f"Day {args.day} -> {len(pairs)} (window, step) pairs: {pairs}")

    raconv_model = RAConv(in_channels=1, out_steps=Q).to(device)
    aconv_model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=[256, 256],
        kernel_size=[3, 3],
        num_layers=2,
        bias=True,
        use_attention=True,
        dropout=0.2,
    ).to(device)

    with torch.no_grad():
        _ = raconv_model(torch.zeros(1, 1, P, GRID_H, GRID_W, device=device))
        _ = aconv_model(torch.zeros(1, P, 1, GRID_H, GRID_W, device=device))

    raconv_model.load_state_dict(
        torch.load(args.raconv_ckpt, map_location=device, weights_only=True)
    )
    aconv_model.load_state_dict(
        torch.load(args.aconvlstm_ckpt, map_location=device, weights_only=True)
    )

    gt_acc     = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    raconv_acc = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    aconv_acc  = np.zeros((GRID_H, GRID_W), dtype=np.float64)

    for win_idx, step_idx in pairs:
        x_sample = X_test[win_idx : win_idx + 1]
        x_raconv = torch.from_numpy(x_sample).float().unsqueeze(1).to(device)
        x_aconv  = torch.from_numpy(x_sample).float().unsqueeze(2).to(device)

        raconv_pred = predict_single(raconv_model, x_raconv, "raconv", Q)
        aconv_pred  = predict_single(aconv_model, x_aconv, "aconvlstm", Q)

        gt_acc     += Y_test[win_idx, step_idx].astype(np.float64)
        raconv_acc += raconv_pred[0, step_idx, 0].cpu().numpy().astype(np.float64)
        aconv_acc  += aconv_pred[0, step_idx, 0].cpu().numpy().astype(np.float64)

    n = len(pairs)
    gt_norm     = (gt_acc / n).astype(np.float32)
    raconv_norm = (raconv_acc / n).astype(np.float32)
    aconv_norm  = (aconv_acc / n).astype(np.float32)

    gt_real     = inverse_transform(gt_norm, norm_min, norm_max)
    raconv_real = np.clip(inverse_transform(raconv_norm, norm_min, norm_max), 0, None)
    aconv_real  = np.clip(inverse_transform(aconv_norm, norm_min, norm_max), 0, None)

    all_vals = np.concatenate([gt_real.ravel(), raconv_real.ravel(), aconv_real.ravel()])
    positive_vals = all_vals[all_vals > 0]
    if positive_vals.size > 0:
        vmin = positive_vals.min()
    else:
        vmin = 1.0
    vmax = max(gt_real.max(), raconv_real.max(), aconv_real.max())
    if vmax <= vmin:
        vmax = vmin * 10

    gt_plot     = np.clip(gt_real, vmin, None)
    raconv_plot = np.clip(raconv_real, vmin, None)
    aconv_plot  = np.clip(aconv_real, vmin, None)

    log_norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(19, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.30)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    panels = [
        ("Ground Truth",         gt_plot),
        ("RAConv (Full Model)",  raconv_plot),
        ("AConvLSTM (Ablation)", aconv_plot),
    ]

    tick_positions = np.arange(0, GRID_W, 2)
    for ax, (title, grid) in zip(axes, panels):
        im = ax.imshow(grid, cmap="viridis", norm=log_norm,
                       interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Grid Column", fontsize=11)
        ax.set_ylabel("Grid Row", fontsize=11)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(labelsize=9)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Infected Count (log scale)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Daily Infected Count Prediction - Day {args.day}  [{p_label}]\n"
        f"(log scale, real range [{vmin:.1f}, {vmax:.1f}])",
        fontsize=15, fontweight="bold",
    )

    fig.subplots_adjust(top=0.84)
    save_path = args.outdir / f"spatial_accuracy_day{args.day}_{p_label}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {save_path}")

    raconv_mse  = np.mean((raconv_real - gt_real) ** 2)
    aconv_mse   = np.mean((aconv_real  - gt_real) ** 2)
    raconv_mae  = np.mean(np.abs(raconv_real - gt_real))
    aconv_mae   = np.mean(np.abs(aconv_real  - gt_real))

    print(f"\nSpatial metrics on day {args.day} (real scale):")
    print(f"  {'Model':<24s} {'MSE':>12s} {'MAE':>12s}")
    print(f"  {'-' * 50}")
    print(f"  {'RAConv (Full)':<24s} {raconv_mse:12.4f} {raconv_mae:12.4f}")
    print(f"  {'AConvLSTM (Ablation)':<24s} {aconv_mse:12.4f} {aconv_mae:12.4f}")


if __name__ == "__main__":
    main()
