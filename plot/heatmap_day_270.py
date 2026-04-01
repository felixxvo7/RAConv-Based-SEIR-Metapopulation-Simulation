"""
Generate heatmap-style plots (actual vs. RAConv vs. AConvLSTM) for a specific day.

Default targets:
- SEIR CSV:  Models/Preprocessing/seir_baseline_300days_256cities.csv
- GEO CSV:   Models/Preprocessing/tx_pd.csv
- NPZ:       Models/Preprocessing/preprocessed_output/seir_preprocessed_P14.npz
- RAConv:    Models/results_fullmodel/P14/best_raconv_P14.pth
- AConvLSTM: Models/results_ablation/P14/best_aconvlstm_P14.pth
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
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / "Models"
PREPROC_DIR = MODELS_DIR / "Preprocessing"

sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(PREPROC_DIR))

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


def derive_split_lengths(x_split: np.ndarray, y_split: np.ndarray) -> int:
    seq_len = x_split.shape[1]
    pred_len = y_split.shape[1]
    return x_split.shape[0] + seq_len + pred_len - 1


def build_day_prediction_map(preds: torch.Tensor, seq_len: int, target_day: int):
    """
    preds: (N, Q, 1, H, W) in real scale.
    target_day: day index within the test split (0-based).
    Returns:
        (H, W) averaged prediction map for the target day.
    """
    n_windows, q_steps, _, h, w = preds.shape
    total = np.zeros((h, w), dtype=np.float64)
    count = 0.0

    preds_np = preds.numpy()
    for i in range(n_windows):
        for q in range(q_steps):
            day_idx = i + seq_len + q
            if day_idx == target_day:
                total += preds_np[i, q, 0]
                count += 1.0

    if count == 0:
        raise RuntimeError(
            f"No prediction coverage for test day {target_day}. "
            f"Predictions start at day {seq_len} within the test split."
        )

    return total / count


def build_day_target_map(targets: torch.Tensor, seq_len: int, target_day: int):
    """
    targets: (N, Q, 1, H, W) in real scale.
    target_day: day index within the test split (0-based).
    Returns:
        (H, W) averaged ground-truth map for the target day.
    """
    n_windows, q_steps, _, h, w = targets.shape
    total = np.zeros((h, w), dtype=np.float64)
    count = 0.0

    targets_np = targets.numpy()
    for i in range(n_windows):
        for q in range(q_steps):
            day_idx = i + seq_len + q
            if day_idx == target_day:
                total += targets_np[i, q, 0]
                count += 1.0

    if count == 0:
        raise RuntimeError(
            f"No target coverage for test day {target_day}. "
            f"Targets start at day {seq_len} within the test split."
        )

    return total / count


def plot_heatmap(matrix: np.ndarray, save_path: Path, title: str, vmin: float, vmax: float, log_scale: bool):
    plot_matrix = matrix
    norm = None
    if log_scale:
        vmin = max(vmin, 1e-6)
        plot_matrix = np.maximum(matrix, vmin)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(
        plot_matrix,
        cmap="viridis",
        interpolation="nearest",
        vmin=None if log_scale else vmin,
        vmax=None if log_scale else vmax,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate heatmap plots for a specific day.")
    parser.add_argument("--day", type=int, default=270, help="Absolute day index in the raw SEIR data.")
    parser.add_argument(
        "--test-day",
        type=int,
        default=None,
        help="Day index within the test split (0-based). Overrides --day if provided.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=PREPROC_DIR / "preprocessed_output" / "seir_preprocessed_P14.npz",
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
    parser.add_argument("--log-scale", action="store_true", help="Use log color scale for heatmaps.")
    args = parser.parse_args()

    for path in (args.npz, args.raconv_ckpt, args.aconvlstm_ckpt):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed windows and infer split sizes.
    data = load_npz(args.npz)
    x_train, y_train = data["X_train"], data["Y_train"]
    x_val, y_val = data["X_val"], data["Y_val"]
    x_test, y_test = data["X_test"], data["Y_test"]

    seq_len = x_test.shape[1]
    pred_len = y_test.shape[1]
    train_len = derive_split_lengths(x_train, y_train)
    val_len = derive_split_lengths(x_val, y_val)
    test_len = derive_split_lengths(x_test, y_test)
    train_days = train_len + val_len

    # Interpret --day as absolute day index; map to test-split day unless --test-day provided.
    if args.test_day is not None:
        day_in_test = args.test_day
        if day_in_test < 0 or day_in_test >= test_len:
            raise ValueError(
                f"--test-day {day_in_test} out of range 0..{test_len - 1}."
            )
        absolute_day = train_days + day_in_test
    else:
        day_in_test = args.day - train_days
        if day_in_test < 0 or day_in_test >= test_len:
            raise ValueError(
                f"Day {args.day} is outside the test split. "
                f"Test days cover [{train_days}, {train_days + test_len - 1}]. "
                f"Use a day in that range to plot test-set heatmaps."
            )
        absolute_day = args.day

    # Build loaders and models.
    loader_raconv = build_test_loader(x_test, y_test, args.batch_size, "raconv")
    loader_aconvlstm = build_test_loader(x_test, y_test, args.batch_size, "aconvlstm")

    raconv_model = RAConv(in_channels=1, out_steps=pred_len).to(device)
    aconvlstm_model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=[256, 256],
        kernel_size=[3, 3],
        num_layers=2,
        bias=True,
        use_attention=True,
        dropout=0.2,
    ).to(device)

    # Warm-up to initialize lazy parameters.
    with torch.no_grad():
        raconv_warmup = torch.zeros(1, 1, seq_len, 16, 16, device=device, dtype=torch.float32)
        _ = raconv_model(raconv_warmup)

        aconvlstm_warmup = torch.zeros(1, seq_len, 1, 16, 16, device=device, dtype=torch.float32)
        _ = aconvlstm_model(aconvlstm_warmup)

    raconv_model.load_state_dict(torch.load(args.raconv_ckpt, map_location=device, weights_only=True))
    aconvlstm_model.load_state_dict(torch.load(args.aconvlstm_ckpt, map_location=device, weights_only=True))

    raconv_preds, targets = predict_test(raconv_model, loader_raconv, device, "raconv")
    aconvlstm_preds, targets_aconv = predict_test(aconvlstm_model, loader_aconvlstm, device, "aconvlstm")
    if not torch.allclose(targets, targets_aconv):
        raise RuntimeError("Target mismatch between RAConv and AConvLSTM loaders.")

    if "norm_min" not in data or "norm_max" not in data:
        raise ValueError("Real-scale plotting requires 'norm_min' and 'norm_max' in the NPZ file.")

    raconv_real = inverse_transform(raconv_preds, data["norm_min"], data["norm_max"]).clamp(min=0)
    aconvlstm_real = inverse_transform(aconvlstm_preds, data["norm_min"], data["norm_max"]).clamp(min=0)
    targets_real = inverse_transform(targets, data["norm_min"], data["norm_max"]).clamp(min=0)

    actual_map = build_day_target_map(targets_real, seq_len, day_in_test)
    raconv_map = build_day_prediction_map(raconv_real, seq_len, day_in_test)
    aconvlstm_map = build_day_prediction_map(aconvlstm_real, seq_len, day_in_test)

    vmin = float(min(actual_map.min(), raconv_map.min(), aconvlstm_map.min()))
    vmax = float(max(actual_map.max(), raconv_map.max(), aconvlstm_map.max()))

    if args.log_scale:
        # Use minimum positive value across maps for stable LogNorm
        positive_min = min(
            np.min(actual_map[actual_map > 0]) if np.any(actual_map > 0) else vmin,
            np.min(raconv_map[raconv_map > 0]) if np.any(raconv_map > 0) else vmin,
            np.min(aconvlstm_map[aconvlstm_map > 0]) if np.any(aconvlstm_map > 0) else vmin,
        )
        vmin = float(max(positive_min, 1e-6))

    actual_path = args.outdir / f"heatmap_day{absolute_day}_actual.png"
    raconv_path = args.outdir / f"heatmap_day{absolute_day}_raconv.png"
    aconvlstm_path = args.outdir / f"heatmap_day{absolute_day}_aconvlstm.png"

    plot_heatmap(actual_map, actual_path, f"Actual (Day {absolute_day})", vmin, vmax, args.log_scale)
    plot_heatmap(raconv_map, raconv_path, f"RAConv Prediction (Day {absolute_day})", vmin, vmax, args.log_scale)
    plot_heatmap(aconvlstm_map, aconvlstm_path, f"AConvLSTM Prediction (Day {absolute_day})", vmin, vmax, args.log_scale)

    print(f"Saved: {actual_path}")
    print(f"Saved: {raconv_path}")
    print(f"Saved: {aconvlstm_path}")


if __name__ == "__main__":
    main()
