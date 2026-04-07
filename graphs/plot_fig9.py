"""
Plot Fig. 9-style metric comparison for our project:
RAConv vs AConvLSTM across different lookback windows P.

Expected directory structure:
- Models/results_fullmodel/P7/metrics.json
- Models/results_fullmodel/P10/metrics.json
- Models/results_fullmodel/P14/metrics.json
- Models/results_ablation/P7/metrics.json
- Models/results_ablation/P10/metrics.json
- Models/results_ablation/P14/metrics.json

Each metrics.json should contain at least:
{
  "P": 14,
  "MSE": ...,
  "MAE": ...,
  "RMSE": ...
}
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MODELS_DIR = ROOT_DIR / "Models"


def load_metrics(metrics_path: Path):
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def collect_model_metrics(base_dir: Path, p_values):
    """
    Returns:
        {
            "MSE":  [..],
            "MAE":  [..],
            "RMSE": [..],
        }
    """
    results = {"MSE": [], "MAE": [], "RMSE": []}

    for p in p_values:
        metrics_path = base_dir / f"P{p}" / "metrics.json"
        metrics = load_metrics(metrics_path)

        metrics_upper = {k.upper(): v for k, v in metrics.items()}

        for key in results.keys():
            if key not in metrics_upper:
                raise KeyError(f"Missing '{key}' in {metrics_path}")
            results[key].append(float(metrics_upper[key]))

    return results


def plot_single_metric_comparison(
    p_values,
    raconv_values,
    aconvlstm_values,
    metric_name,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        p_values,
        raconv_values,
        "b-o",
        linewidth=2,
        markersize=7,
        label="RAConv",
    )
    ax.plot(
        p_values,
        aconvlstm_values,
        "r-s",
        linewidth=2,
        markersize=7,
        label="AConvLSTM",
    )

    ax.set_xlabel("Lookback Window P")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Lookback Window P")
    ax.set_xticks(p_values)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric_name} plot -> {save_path}")


def plot_combined_metrics(
    p_values,
    raconv_metrics,
    aconvlstm_metrics,
    save_path: Path,
):
    metric_names = ["RMSE", "MAE", "MSE"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, metric_names):
        ax.plot(
            p_values,
            raconv_metrics[metric],
            "b-o",
            linewidth=2,
            markersize=6,
            label="RAConv",
        )
        ax.plot(
            p_values,
            aconvlstm_metrics[metric],
            "r-s",
            linewidth=2,
            markersize=6,
            label="AConvLSTM",
        )

        ax.set_title(metric)
        ax.set_xlabel("Lookback Window P")
        ax.set_ylabel(metric)
        ax.set_xticks(p_values)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("RAConv vs AConvLSTM Across Lookback Windows", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined metrics plot -> {save_path}")


def print_summary_table(p_values, raconv_metrics, aconvlstm_metrics):
    print("\nSummary")
    print("=" * 72)
    print(f"{'P':<6} {'Model':<14} {'RMSE':>12} {'MAE':>12} {'MSE':>12}")
    print("-" * 72)

    for i, p in enumerate(p_values):
        print(
            f"{p:<6} {'RAConv':<14} "
            f"{raconv_metrics['RMSE'][i]:12.6f} "
            f"{raconv_metrics['MAE'][i]:12.6f} "
            f"{raconv_metrics['MSE'][i]:12.6f}"
        )
        print(
            f"{'':<6} {'AConvLSTM':<14} "
            f"{aconvlstm_metrics['RMSE'][i]:12.6f} "
            f"{aconvlstm_metrics['MAE'][i]:12.6f} "
            f"{aconvlstm_metrics['MSE'][i]:12.6f}"
        )
        print("-" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Fig. 9-style metric comparison: RAConv vs AConvLSTM across P."
    )
    parser.add_argument(
        "--p",
        type=int,
        nargs="+",
        default=[4, 6, 8, 14],
        help="Lookback windows P to include (default: 4 6 8 14)",
    )
    parser.add_argument(
        "--raconv-dir",
        type=Path,
        default=MODELS_DIR / "results_fullmodel",
        help="Directory containing RAConv result folders P4/P6/P8/P14...",
    )
    parser.add_argument(
        "--aconvlstm-dir",
        type=Path,
        default=MODELS_DIR / "results_ablation",
        help="Directory containing AConvLSTM result folders P4/P6/P8/P14...",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory to save plots",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    requested_p = sorted(args.p)
    valid_p = []

    for p in requested_p:
        r_path = args.raconv_dir / f"P{p}" / "metrics.json"
        a_path = args.aconvlstm_dir / f"P{p}" / "metrics.json"
        if r_path.exists() and a_path.exists():
            valid_p.append(p)
        else:
            missing = []
            if not r_path.exists(): missing.append("RAConv")
            if not a_path.exists(): missing.append("AConvLSTM")
            print(f"Skipping P={p}: metrics.json missing for {', '.join(missing)}")

    if not valid_p:
        print(f"Error: No common metrics found in {args.raconv_dir} and {args.aconvlstm_dir}")
        return

    raconv_metrics = collect_model_metrics(args.raconv_dir, valid_p)
    aconvlstm_metrics = collect_model_metrics(args.aconvlstm_dir, valid_p)

    print_summary_table(valid_p, raconv_metrics, aconvlstm_metrics)

    # Combined 3-panel plot
    combined_path = args.outdir / "fig9_style_metrics_comparison.png"
    plot_combined_metrics(
        valid_p,
        raconv_metrics,
        aconvlstm_metrics,
        combined_path,
    )

    # Individual plots
    plot_single_metric_comparison(
        valid_p,
        raconv_metrics["RMSE"],
        aconvlstm_metrics["RMSE"],
        "RMSE",
        args.outdir / "fig9_style_rmse.png",
    )

    plot_single_metric_comparison(
        valid_p,
        raconv_metrics["MAE"],
        aconvlstm_metrics["MAE"],
        "MAE",
        args.outdir / "fig9_style_mae.png",
    )

    plot_single_metric_comparison(
        valid_p,
        raconv_metrics["MSE"],
        aconvlstm_metrics["MSE"],
        "MSE",
        args.outdir / "fig9_style_mse.png",
    )


if __name__ == "__main__":
    main()