"""
Statewide SEIR / reported-case plots from Task 4 simulation output.

Default input: seir_baseline_300days_256cities.npy (shape: days x cities x 4 [S,E,I,R]).
Pass a .csv path to use the legacy long-format file instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Match Task4_SEIR_Simulation.py (observables)
GAMMA = 0.100
REPORTING_FRACTION = 0.10


def load_statewide_from_npy(path: Path) -> tuple[pd.DataFrame, int]:
    """Aggregate SEIR arrays to statewide daily totals."""
    results = np.load(path)
    if results.ndim != 3 or results.shape[2] != 4:
        raise ValueError(
            f"Expected .npy shape (n_days, n_cities, 4) [S,E,I,R]; got {results.shape}"
        )

    n_cities = results.shape[1]
    s = results[:, :, 0].sum(axis=1)
    e = results[:, :, 1].sum(axis=1)
    i_tot = results[:, :, 2].sum(axis=1)
    r = results[:, :, 3].sum(axis=1)

    i_rep = i_tot * REPORTING_FRACTION
    new_rep = GAMMA * i_tot * REPORTING_FRACTION

    n_days = results.shape[0]
    day = np.arange(n_days, dtype=int)

    df = pd.DataFrame(
        {
            "day": day,
            "S": s,
            "E": e,
            "I": i_tot,
            "R": r,
            "I_rep": i_rep,
            "new_rep": new_rep,
        }
    )
    return df, n_cities


def load_statewide_from_csv(path: Path) -> tuple[pd.DataFrame, int]:
    """Legacy long-format CSV: day, city, S, E, I, R, I_rep, new_rep."""
    df = pd.read_csv(path)
    required_cols = {"day", "city", "S", "E", "I", "R", "I_rep", "new_rep"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n_cities = df["city"].nunique()
    statewide = (
        df.groupby("day", as_index=False)[["S", "E", "I", "R", "I_rep", "new_rep"]]
        .sum()
        .sort_values("day")
    )
    return statewide, n_cities


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_npy = script_dir / "seir_baseline_300days_256cities.npy"

    parser = argparse.ArgumentParser(
        description="Plot statewide SEIR from Task 4 .npy or legacy .csv output."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=f"Path to .npy or .csv (default: {default_npy.name} next to this script).",
    )
    args = parser.parse_args()

    if args.input:
        file_path = Path(args.input).expanduser().resolve()
    else:
        file_path = default_npy

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        statewide, n_cities = load_statewide_from_npy(file_path)
    elif suffix == ".csv":
        statewide, n_cities = load_statewide_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type {suffix!r}; use .npy or .csv")

    n_days = statewide["day"].nunique()
    print(f"Loaded {file_path.name}: {n_days} days, {n_cities} cities (statewide sums).")

    # Plot 1: Statewide SEIR totals
    plt.figure(figsize=(12, 6))
    plt.plot(statewide["day"], statewide["S"], label="S")
    plt.plot(statewide["day"], statewide["E"], label="E")
    plt.plot(statewide["day"], statewide["I"], label="I")
    plt.plot(statewide["day"], statewide["R"], label="R")
    plt.title("Statewide SEIR Time Series (All Cities Summed)")
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Statewide reported totals
    plt.figure(figsize=(12, 6))
    plt.plot(statewide["day"], statewide["I_rep"], label="Reported active cases (I_rep)")
    plt.plot(statewide["day"], statewide["new_rep"], label="Reported new cases/day (new_rep)")
    plt.title("Statewide Reported Time Series (All Cities Summed)")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    peak_I_row = statewide.loc[statewide["I"].idxmax()]
    peak_R_row = statewide.loc[statewide["R"].idxmax()]
    peak_Irep_row = statewide.loc[statewide["I_rep"].idxmax()]
    peak_new_row = statewide.loc[statewide["new_rep"].idxmax()]

    print("\nSummary")
    print("-" * 40)
    print(f"Peak active infected I: day {int(peak_I_row['day'])}, value {peak_I_row['I']:,.0f}")
    print(f"Peak recovered R: day {int(peak_R_row['day'])}, value {peak_R_row['R']:,.0f}")
    print(f"Peak reported active I_rep: day {int(peak_Irep_row['day'])}, value {peak_Irep_row['I_rep']:,.0f}")
    print(f"Peak reported new cases/day new_rep: day {int(peak_new_row['day'])}, value {peak_new_row['new_rep']:,.0f}")

    out_csv = script_dir / "statewide_seir_summary.csv"
    statewide.to_csv(out_csv, index=False)
    print(f"\nSaved statewide daily totals to {out_csv}")


if __name__ == "__main__":
    main()
