"""
SEIR Preprocessing for ConvLSTM-based Models
=============================================
Transforms raw SEIR CSV into normalised sliding-window tensors.

Pipeline:
  1. Load SEIR CSV  +  geographic CSV (lat/lng)
  2. Map 256 cities → 16×16 grid (geographic snake-sort)
  3. Reshape to (days, 16, 16) using infected column I
  4. Split chronologically into train / val / test
  5. Per-cell min-max normalisation (fit on train only)
  6. Sliding-window sampling → X (N, P, 16, 16), Y (N, Q, 16, 16)
  7. Save everything in a single .npz file
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict

# ── constants ────────────────────────────────────────────────────────────────
GRID_H, GRID_W = 16, 16
NUM_CITIES = GRID_H * GRID_W          # 256
EPS = 1e-8                             # avoids division by zero in normalisation


# ── data loading ─────────────────────────────────────────────────────────────

def load_data(seir_path: str, geo_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (seir_df, geo_df) after basic validation."""
    seir_df = pd.read_csv(seir_path)
    required = {'day', 'city', 'I'}
    if not required.issubset(seir_df.columns):
        raise ValueError(f"SEIR CSV must contain {required}; found {set(seir_df.columns)}")

    geo_df = pd.read_csv(geo_path)
    required_geo = {'city', 'lat', 'lng'}
    if not required_geo.issubset(geo_df.columns):
        raise ValueError(f"Geo CSV must contain {required_geo}; found {set(geo_df.columns)}")

    print(f"SEIR : {len(seir_df)} rows | {seir_df['day'].nunique()} days | "
          f"{seir_df['city'].nunique()} cities")
    print(f"GEO  : {len(geo_df)} cities with lat/lng")
    return seir_df, geo_df


# ── geographic grid mapping ──────────────────────────────────────────────────

def build_geographic_grid(
    seir_df: pd.DataFrame,
    geo_df: pd.DataFrame
) -> Dict[str, Tuple[int, int]]:
    """
    Map cities onto a 16×16 grid using a latitude-sorted snake pattern
    so that geographic neighbours occupy adjacent grid cells.
    """
    day0_cities = set(seir_df.loc[seir_df['day'] == 0, 'city'])
    geo = geo_df[geo_df['city'].isin(day0_cities)].copy()

    if len(geo) != NUM_CITIES:
        raise ValueError(f"Expected {NUM_CITIES} cities in geo_df, got {len(geo)}")

    geo = geo.sort_values('lat').reset_index(drop=True)

    ordered_cities = []
    for row in range(GRID_H):
        chunk = geo.iloc[row * GRID_W:(row + 1) * GRID_W].copy()
        chunk = chunk.sort_values('lng', ascending=(row % 2 == 0))
        ordered_cities.extend(chunk['city'].tolist())

    return {city: (i // GRID_W, i % GRID_W) for i, city in enumerate(ordered_cities)}


# ── reshape ──────────────────────────────────────────────────────────────────

def reshape_to_grid(
    seir_df: pd.DataFrame,
    city_to_grid: Dict[str, Tuple[int, int]]
) -> np.ndarray:
    """Pivot the long-format SEIR table into shape (days, 16, 16)."""
    pivot = seir_df.pivot(index='day', columns='city', values='I')
    num_days = len(pivot)
    grid = np.zeros((num_days, GRID_H, GRID_W), dtype=np.float32)
    for city, (r, c) in city_to_grid.items():
        if city in pivot.columns:
            grid[:, r, c] = pivot[city].values
    return grid


# ── normalisation (global min-max, fit on full dataset) ──────────────────────

def minmax_fit(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global min/max over the entire grid (all days, all cells)."""
    mn = grid.min()   # scalar
    mx = grid.max()   # scalar
    return mn, mx


def minmax_transform(data: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return (data - mn) / (mx - mn + EPS)


def minmax_inverse(data: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return data * (mx - mn + EPS) + mn


# ── sliding window ───────────────────────────────────────────────────────────

def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sampling.

    Returns
        X : (N, seq_len, 16, 16)
        Y : (N, pred_len, 16, 16)
    """
    T = data.shape[0]
    win = seq_len + pred_len
    if T < win:
        raise ValueError(f"Data length {T} < window {win}")

    N = (T - win) // stride + 1
    X = np.zeros((N, seq_len, GRID_H, GRID_W), dtype=np.float32)
    Y = np.zeros((N, pred_len, GRID_H, GRID_W), dtype=np.float32)
    for i in range(N):
        s = i * stride
        X[i] = data[s:s + seq_len]
        Y[i] = data[s + seq_len:s + win]
    return X, Y


# ── full pipeline ────────────────────────────────────────────────────────────

def preprocess(
    seir_path: str,
    geo_path: str,
    save_path: str,
    seq_len: int = 14,
    pred_len: int = 7,
    train_days: int = 200,
    val_ratio: float = 0.20,
    stride: int = 1,
) -> str:
    """
    Run the full preprocessing pipeline and save a single .npz file.

    Saved arrays
    -------------
    X_train, Y_train, X_val, Y_val, X_test, Y_test
    norm_min, norm_max          (for inverse-transform)
    """
    seir_df, geo_df = load_data(seir_path, geo_path)

    # 1  geographic grid
    city_grid = build_geographic_grid(seir_df, geo_df)
    print(f"[1/5] Mapped {len(city_grid)} cities → {GRID_H}×{GRID_W} grid")

    # 2  reshape
    grid = reshape_to_grid(seir_df, city_grid)
    print(f"[2/5] Spatial tensor : {grid.shape}")

    # 3  global min-max normalisation (fit on full 300-day grid)
    mn, mx = minmax_fit(grid)
    grid_norm = minmax_transform(grid, mn, mx)
    print(f"[3/5] Normalised   : global [{mn:.4f}, {mx:.4f}] → grid [{grid_norm.min():.4f}, {grid_norm.max():.4f}]")

    # 4  chronological split (on already-normalised data)
    val_days   = int(train_days * val_ratio)
    train_norm = grid_norm[:train_days - val_days]
    val_norm   = grid_norm[train_days - val_days:train_days]
    test_norm  = grid_norm[train_days:]
    print(f"[4/5] Split : train {len(train_norm)} | val {len(val_norm)} | test {len(test_norm)} days")

    # 5  sliding windows
    X_train, Y_train = create_sequences(train_norm, seq_len, pred_len, stride)
    X_val,   Y_val   = create_sequences(val_norm,   seq_len, pred_len, stride)
    X_test,  Y_test  = create_sequences(test_norm,  seq_len, pred_len, stride)
    print(f"[5/5] Sequences    : train {X_train.shape[0]} | val {X_val.shape[0]} | test {X_test.shape[0]}")
    print(f"      X shape      : {X_train.shape}")
    print(f"      Y shape      : {Y_train.shape}")

    # save
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(
        save_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val,     Y_val=Y_val,
        X_test=X_test,   Y_test=Y_test,
        norm_min=mn,     norm_max=mx,
    )
    print(f"\nSaved → {save_path}")
    return save_path


def load_preprocessed(path: str) -> Dict[str, np.ndarray]:
    """Load the .npz produced by *preprocess()*."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    seq_lens = [7, 10, 14]          # P values: lookback window length
    Q = 7                           # fixed forecast horizon

    for P in seq_lens:
        print(f"\n{'=' * 60}")
        print(f"  Running preprocessing with P (seq_len) = {P}, Q (pred_len) = {Q}")
        print(f"{'=' * 60}")
        preprocess(
            seir_path=os.path.join(base, 'seir_baseline_300days_256cities.csv'),
            geo_path=os.path.join(base, 'tx_pd.csv'),
            save_path=os.path.join(base, 'preprocessed_output', f'seir_preprocessed_P{P}.npz'),
            seq_len=P,
            pred_len=Q,
            train_days=200,
            val_ratio=0.20,
            stride=1,
        )


if __name__ == '__main__':
    main()
