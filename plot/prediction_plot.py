"""
Plot RAConv + AConvLSTM predictions vs. ground truth on test data.

Default targets:
- NPZ:      Models/Preprocessing/preprocessed_output/seir_preprocessed_P14.npz
- RAConv:   Models/results_fullmodel/P14/best_raconv_P14.pth
- AConvLSTM Models/results_ablation/P14/best_aconvlstm_P14.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / "Models"
sys.path.insert(0, str(MODELS_DIR))

from RAConvLSTM import RAConv  # noqa: E402
from AConvLSTM.AConvLSTM import AConvLSTMLayers  # noqa: E402


def load_npz(npz_path: Path):
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def build_test_loader(x_test: np.ndarray, y_test: np.ndarray, batch_size: int, model_type: str):
    # RAConv expects (B, C, T, H, W); AConvLSTM expects (B, T, C, H, W)
    if model_type == "raconv":
        x_tensor = torch.from_numpy(x_test).float().unsqueeze(1)
    else:
        x_tensor = torch.from_numpy(x_test).float().unsqueeze(2)
    y_tensor = torch.from_numpy(y_test).float().unsqueeze(2)  # (B, Q, 1, H, W)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )


@torch.no_grad()
def predict_test(model, loader: DataLoader, device: torch.device, model_type: str):
    model.eval()
    all_preds = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        if model_type == "raconv":
            preds = model(xb)
        else:
            _, last_states = model(xb)
            first_input = xb[:, -1, :, :, :]
            preds = model.predict_future(last_states, yb.shape[1], first_input)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())
    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


def inverse_transform(tensor: torch.Tensor, norm_min: np.ndarray, norm_max: np.ndarray):
    mn = torch.as_tensor(norm_min, dtype=torch.float32)
    mx = torch.as_tensor(norm_max, dtype=torch.float32)
    return tensor * (mx - mn + 1e-8) + mn


def plot_average_predictions_comparison(
    raconv_preds: torch.Tensor,
    aconvlstm_preds: torch.Tensor,
    targets: torch.Tensor,
    save_path: Path,
    title: str,
    y_label: str,
):
    # Average over test samples and spatial map: (N, Q, 1, H, W) -> (Q,)
    raconv_mean = raconv_preds.mean(dim=(0, 2, 3, 4)).numpy()
    aconvlstm_mean = aconvlstm_preds.mean(dim=(0, 2, 3, 4)).numpy()
    target_mean = targets.mean(dim=(0, 2, 3, 4)).numpy()

    q_steps = raconv_preds.shape[1]
    x_axis = np.arange(1, q_steps + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_axis, target_mean, "k-", marker="s", linewidth=2, label="Ground Truth (Average)")
    ax.plot(x_axis, raconv_mean, "b-", marker="o", linewidth=2, label="RAConv (Average)")
    ax.plot(x_axis, aconvlstm_mean, "r-", marker="^", linewidth=2, label="AConvLSTM (Average)")
    ax.set_title(title)
    ax.set_xlabel("Forecast Step")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_axis)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_full_test_timeline(
    preds: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Convert windowed Q-step outputs into a continuous test-period timeline.
    Overlapping days from different windows are averaged.

    preds/targets: (N, Q, 1, H, W)
    Returns:
        pred_timeline:   (T_test - P,)
        target_timeline: (T_test - P,)
    """
    n_windows, q_steps = preds.shape[0], preds.shape[1]
    timeline_len = n_windows + q_steps - 1

    # Average each window over spatial dimensions first -> (N, Q)
    pred_win_mean = preds.mean(dim=(2, 3, 4)).numpy()
    target_win_mean = targets.mean(dim=(2, 3, 4)).numpy()

    pred_sum = np.zeros(timeline_len, dtype=np.float64)
    target_sum = np.zeros(timeline_len, dtype=np.float64)
    counts = np.zeros(timeline_len, dtype=np.float64)

    for i in range(n_windows):
        pred_sum[i:i + q_steps] += pred_win_mean[i]
        target_sum[i:i + q_steps] += target_win_mean[i]
        counts[i:i + q_steps] += 1.0

    pred_timeline = pred_sum / np.maximum(counts, 1.0)
    target_timeline = target_sum / np.maximum(counts, 1.0)
    return pred_timeline, target_timeline


def plot_full_timeline(
    raconv_timeline: np.ndarray,
    aconvlstm_timeline: np.ndarray,
    target_timeline: np.ndarray,
    save_path: Path,
    title: str,
    y_label: str,
    start_day_offset: int,
):
    x_axis = np.arange(start_day_offset, start_day_offset + len(target_timeline))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_axis, target_timeline, "k-", linewidth=2, label="Ground Truth")
    ax.plot(x_axis, raconv_timeline, "b-", linewidth=2, label="RAConv")
    ax.plot(x_axis, aconvlstm_timeline, "r-", linewidth=2, label="AConvLSTM")
    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot test predictions for both RAConv and AConvLSTM.")
    parser.add_argument(
        "--npz",
        type=Path,
        default=ROOT_DIR / "Models" / "Preprocessing" / "preprocessed_output" / "seir_preprocessed_P14.npz",
        help="Path to preprocessed .npz file.",
    )
    parser.add_argument(
        "--raconv-ckpt",
        type=Path,
        default=ROOT_DIR / "Models" / "results_fullmodel" / "P14" / "best_raconv_P14.pth",
        help="Path to trained RAConv checkpoint (.pth).",
    )
    parser.add_argument(
        "--aconvlstm-ckpt",
        type=Path,
        default=ROOT_DIR / "Models" / "results_ablation" / "P14" / "best_aconvlstm_P14.pth",
        help="Path to trained AConvLSTM checkpoint (.pth).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory where plots are saved.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ not found: {args.npz}")
    if not args.raconv_ckpt.exists():
        raise FileNotFoundError(f"RAConv checkpoint not found: {args.raconv_ckpt}")
    if not args.aconvlstm_ckpt.exists():
        raise FileNotFoundError(f"AConvLSTM checkpoint not found: {args.aconvlstm_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.npz)
    x_test = data["X_test"]
    y_test = data["Y_test"]
    q_steps = y_test.shape[1]
    loader_raconv = build_test_loader(x_test, y_test, args.batch_size, "raconv")
    loader_aconvlstm = build_test_loader(x_test, y_test, args.batch_size, "aconvlstm")

    raconv_model = RAConv(in_channels=1, out_steps=q_steps).to(device)
    aconvlstm_model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=[256, 256],
        kernel_size=[3, 3],
        num_layers=2,
        bias=True,
        use_attention=True,
        dropout=0.2,
    ).to(device)

    # Warm-up pass creates lazy peephole params before loading weights.
    with torch.no_grad():
        raconv_warmup = torch.zeros(
            1,
            1,
            x_test.shape[1],
            x_test.shape[2],
            x_test.shape[3],
            device=device,
            dtype=torch.float32,
        )
        _ = raconv_model(raconv_warmup)

        aconvlstm_warmup = torch.zeros(
            1,
            x_test.shape[1],
            1,
            x_test.shape[2],
            x_test.shape[3],
            device=device,
            dtype=torch.float32,
        )
        _ = aconvlstm_model(aconvlstm_warmup)

    raconv_model.load_state_dict(torch.load(args.raconv_ckpt, map_location=device, weights_only=True))
    aconvlstm_model.load_state_dict(torch.load(args.aconvlstm_ckpt, map_location=device, weights_only=True))

    raconv_preds, targets = predict_test(raconv_model, loader_raconv, device, "raconv")
    aconvlstm_preds, targets_aconv = predict_test(aconvlstm_model, loader_aconvlstm, device, "aconvlstm")
    if not torch.allclose(targets, targets_aconv):
        raise RuntimeError("Target mismatch between RAConv and AConvLSTM loaders.")
    lookback_steps = x_test.shape[1]

    if "norm_min" not in data or "norm_max" not in data:
        raise ValueError("Real-scale plotting requires 'norm_min' and 'norm_max' in the NPZ file.")

    raconv_real = inverse_transform(raconv_preds, data["norm_min"], data["norm_max"]).clamp(min=0)
    aconvlstm_real = inverse_transform(aconvlstm_preds, data["norm_min"], data["norm_max"]).clamp(min=0)
    targets_real = inverse_transform(targets, data["norm_min"], data["norm_max"])

    raconv_timeline_real, target_timeline_real = build_full_test_timeline(
        preds=raconv_real,
        targets=targets_real,
    )
    aconvlstm_timeline_real, _ = build_full_test_timeline(
        preds=aconvlstm_real,
        targets=targets_real,
    )
    timeline_real_path = args.outdir / "full_timeline_pred_comparison_vs_ground_truth_test_real_scale.png"
    plot_full_timeline(
        raconv_timeline=raconv_timeline_real,
        aconvlstm_timeline=aconvlstm_timeline_real,
        target_timeline=target_timeline_real,
        save_path=timeline_real_path,
        title="RAConv vs AConvLSTM: Average Daily Infected Count Per City Timeline Prediction",
        y_label="Infected Count",
        start_day_offset=lookback_steps,
    )
    print(f"Saved real-scale full-timeline plot: {timeline_real_path}")


if __name__ == "__main__":
    main()
