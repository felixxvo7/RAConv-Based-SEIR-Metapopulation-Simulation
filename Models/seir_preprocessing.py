"""
SEIR Data Preprocessing for ConvLSTM-based Models
==================================================

This script preprocesses SEIR epidemic simulation data for:
- ConvLSTM (baseline)
- AConvLSTM (Attention ConvLSTM)
- ResAConvLSTM (Residual Attention ConvLSTM / RAConv)

Based on the paper's approach (WWICC2022):
- Input: Infected (I) compartment values only
- Spatial arrangement: 256 cities → 16x16 grid
- Training period: 170 days
- Input format: 5D tensor (batch, time, channels, height, width)

Author: Generated based on WWICC2022 paper methodology
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict, Optional, List
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================

class SEIRConfig:
    """Configuration parameters for SEIR data preprocessing."""
    
    # Spatial configuration (paper: 16x16 grid of base stations per cluster)
    GRID_HEIGHT = 16
    GRID_WIDTH = 16
    NUM_CITIES = 256  # 16 * 16
    
    # Temporal configuration
    TOTAL_DAYS = 301  # Day 0 to Day 300
    TRAIN_DAYS = 170  # Training period
    
    # Sequence parameters (paper: P observations, Q predictions)
    DEFAULT_SEQ_LEN = 8    # P: number of observation steps
    DEFAULT_PRED_LEN = 4   # Q: number of prediction steps
    
    # Data parameters
    TARGET_COLUMN = 'I'    # Infected compartment (as specified)
    
    # Normalization
    NORM_METHOD = 'minmax'  # 'minmax', 'standard', or 'log'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_seir_data(filepath: str) -> pd.DataFrame:
    """
    Load SEIR simulation data from CSV file.
    
    Args:
        filepath: Path to the SEIR CSV file
        
    Returns:
        DataFrame with columns: day, city, S, E, I, R, I_rep, new_rep
    """
    df = pd.read_csv(filepath)
    
    # Validate expected columns
    expected_cols = {'day', 'city', 'S', 'E', 'I', 'R'}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing expected columns. Found: {df.columns.tolist()}")
    
    print(f"Loaded data: {len(df)} rows, {df['day'].nunique()} days, {df['city'].nunique()} cities")
    return df


def load_statewide_summary(filepath: str) -> pd.DataFrame:
    """
    Load statewide aggregated SEIR summary data.
    
    Args:
        filepath: Path to the statewide summary CSV
        
    Returns:
        DataFrame with daily statewide SEIR values
    """
    df = pd.read_csv(filepath)
    print(f"Loaded statewide summary: {len(df)} days")
    return df


# =============================================================================
# SPATIAL ARRANGEMENT
# =============================================================================

def create_city_to_grid_mapping(
    df: pd.DataFrame,
    method: str = 'population_order'
) -> Dict[str, Tuple[int, int]]:
    """
    Create mapping from city names to 2D grid positions.
    
    The paper uses spectral clustering based on traffic similarity.
    For SEIR data, we use population-based ordering (larger cities first)
    which maintains a natural hierarchical structure.
    
    Args:
        df: DataFrame containing city data
        method: Mapping method ('population_order', 'alphabetical', 'random')
        
    Returns:
        Dictionary mapping city name to (row, col) grid position
    """
    # Get unique cities in order of first appearance (typically sorted by population)
    day0_data = df[df['day'] == 0].copy()
    
    if method == 'population_order':
        # Use S (susceptible) as proxy for population, sort descending
        day0_data = day0_data.sort_values('S', ascending=False)
    elif method == 'alphabetical':
        day0_data = day0_data.sort_values('city')
    elif method == 'random':
        day0_data = day0_data.sample(frac=1, random_state=42)
    
    cities = day0_data['city'].tolist()
    
    if len(cities) != SEIRConfig.NUM_CITIES:
        warnings.warn(f"Expected {SEIRConfig.NUM_CITIES} cities, got {len(cities)}")
    
    # Map cities to grid positions (row-major order)
    city_to_grid = {}
    for idx, city in enumerate(cities):
        row = idx // SEIRConfig.GRID_WIDTH
        col = idx % SEIRConfig.GRID_WIDTH
        city_to_grid[city] = (row, col)
    
    return city_to_grid


def reshape_to_spatial_grid(
    df: pd.DataFrame,
    city_to_grid: Dict[str, Tuple[int, int]],
    column: str = 'I'
) -> np.ndarray:
    """
    Reshape time-series data into spatial grid format.
    
    Args:
        df: DataFrame with columns [day, city, I, ...]
        city_to_grid: Mapping from city names to grid positions
        column: Column to extract ('I' for infected)
        
    Returns:
        3D array of shape (num_days, height, width)
    """
    num_days = df['day'].nunique()
    height, width = SEIRConfig.GRID_HEIGHT, SEIRConfig.GRID_WIDTH
    
    # Initialize spatial grid: (days, height, width)
    spatial_data = np.zeros((num_days, height, width), dtype=np.float32)
    
    for day in range(num_days):
        day_data = df[df['day'] == day]
        for _, row in day_data.iterrows():
            city = row['city']
            if city in city_to_grid:
                r, c = city_to_grid[city]
                spatial_data[day, r, c] = row[column]
    
    return spatial_data


# =============================================================================
# NORMALIZATION
# =============================================================================

class DataNormalizer:
    """
    Handles normalization and denormalization of SEIR data.
    
    Supports multiple normalization methods suitable for epidemic data:
    - MinMax: Scale to [0, 1] range
    - Standard: Z-score normalization
    - Log: Log transformation (handles exponential growth)
    """
    
    def __init__(self, method: str = 'minmax', epsilon: float = 1e-8):
        self.method = method
        self.epsilon = epsilon
        self.scaler = None
        self.fitted = False
        
        # Store parameters for inverse transform
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
        
    def fit(self, data: np.ndarray) -> 'DataNormalizer':
        """Fit normalizer to training data."""
        if self.method == 'minmax':
            self.min_val = np.min(data)
            self.max_val = np.max(data)
        elif self.method == 'standard':
            self.mean_val = np.mean(data)
            self.std_val = np.std(data)
        elif self.method == 'log':
            self.min_val = np.min(data)
            
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
            
        if self.method == 'minmax':
            range_val = self.max_val - self.min_val + self.epsilon
            return (data - self.min_val) / range_val
        elif self.method == 'standard':
            return (data - self.mean_val) / (self.std_val + self.epsilon)
        elif self.method == 'log':
            return np.log1p(data - self.min_val + self.epsilon)
        
        return data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform to original scale."""
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
            
        if self.method == 'minmax':
            range_val = self.max_val - self.min_val + self.epsilon
            return data * range_val + self.min_val
        elif self.method == 'standard':
            return data * (self.std_val + self.epsilon) + self.mean_val
        elif self.method == 'log':
            return np.expm1(data) + self.min_val - self.epsilon
        
        return data


# =============================================================================
# SEQUENCE GENERATION
# =============================================================================

def create_sequences(
    data: np.ndarray,
    seq_len: int = 8,
    pred_len: int = 4,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequence pairs using sliding window.
    
    Based on paper's formulation:
    - Input: P consecutive observations (seq_len)
    - Output: Q future predictions (pred_len)
    
    Args:
        data: 3D array of shape (time, height, width)
        seq_len: Number of input time steps (P in paper)
        pred_len: Number of output time steps (Q in paper)
        stride: Sliding window stride
        
    Returns:
        X: Input sequences of shape (num_samples, seq_len, 1, height, width)
        Y: Target sequences of shape (num_samples, pred_len, 1, height, width)
    """
    total_time = data.shape[0]
    window_size = seq_len + pred_len
    
    if total_time < window_size:
        raise ValueError(f"Data length {total_time} < required window {window_size}")
    
    num_samples = (total_time - window_size) // stride + 1
    
    height, width = data.shape[1], data.shape[2]
    
    # Add channel dimension: (time, height, width) -> (time, 1, height, width)
    data = data[:, np.newaxis, :, :]
    
    X = np.zeros((num_samples, seq_len, 1, height, width), dtype=np.float32)
    Y = np.zeros((num_samples, pred_len, 1, height, width), dtype=np.float32)
    
    for i in range(num_samples):
        start_idx = i * stride
        X[i] = data[start_idx:start_idx + seq_len]
        Y[i] = data[start_idx + seq_len:start_idx + seq_len + pred_len]
    
    return X, Y


def create_sequences_single_step(
    data: np.ndarray,
    seq_len: int = 8,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for single-step prediction (predict next frame only).
    
    Args:
        data: 3D array of shape (time, height, width)
        seq_len: Number of input time steps
        stride: Sliding window stride
        
    Returns:
        X: Input sequences of shape (num_samples, seq_len, 1, height, width)
        Y: Target frames of shape (num_samples, 1, height, width)
    """
    total_time = data.shape[0]
    
    if total_time <= seq_len:
        raise ValueError(f"Data length {total_time} <= seq_len {seq_len}")
    
    num_samples = (total_time - seq_len - 1) // stride + 1
    height, width = data.shape[1], data.shape[2]
    
    # Add channel dimension
    data = data[:, np.newaxis, :, :]
    
    X = np.zeros((num_samples, seq_len, 1, height, width), dtype=np.float32)
    Y = np.zeros((num_samples, 1, height, width), dtype=np.float32)
    
    for i in range(num_samples):
        start_idx = i * stride
        X[i] = data[start_idx:start_idx + seq_len]
        Y[i] = data[start_idx + seq_len]
    
    return X, Y


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class SEIRDataset(Dataset):
    """
    PyTorch Dataset for SEIR epidemic data.
    
    Provides data in the format required by ConvLSTM models:
    - Input: (batch, time, channels, height, width)
    - Target: (batch, pred_len, channels, height, width) or (batch, channels, height, width)
    """
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Args:
            X: Input sequences
            Y: Target sequences/frames
            transform: Optional transform to apply
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.Y[idx]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            
        return x, y


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

class SEIRPreprocessor:
    """
    Complete preprocessing pipeline for SEIR data.
    
    Handles the full workflow:
    1. Load raw CSV data
    2. Create spatial grid mapping
    3. Reshape to 3D spatial-temporal format
    4. Normalize data
    5. Split train/val/test
    6. Create sequences
    7. Build PyTorch DataLoaders
    """
    
    def __init__(
        self,
        seq_len: int = 8,
        pred_len: int = 4,
        train_days: int = 170,
        norm_method: str = 'minmax',
        target_column: str = 'I',
        val_ratio: float = 0.15,
        stride: int = 1,
        single_step: bool = False
    ):
        """
        Args:
            seq_len: Number of input time steps (P)
            pred_len: Number of prediction steps (Q)
            train_days: Number of days for training
            norm_method: Normalization method ('minmax', 'standard', 'log')
            target_column: SEIR column to use ('I' for infected)
            val_ratio: Proportion of training data for validation
            stride: Sliding window stride
            single_step: If True, predict only next frame
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_days = train_days
        self.norm_method = norm_method
        self.target_column = target_column
        self.val_ratio = val_ratio
        self.stride = stride
        self.single_step = single_step
        
        # Will be set during preprocessing
        self.normalizer = None
        self.city_to_grid = None
        self.grid_to_city = None
        self.spatial_data = None
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        mapping_method: str = 'population_order'
    ) -> Dict[str, np.ndarray]:
        """
        Execute full preprocessing pipeline.
        
        Args:
            df: Raw SEIR DataFrame
            mapping_method: Method for city-to-grid mapping
            
        Returns:
            Dictionary containing train/val/test data splits
        """
        print("=" * 60)
        print("SEIR DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Create spatial mapping
        print("\n[1/5] Creating spatial grid mapping...")
        self.city_to_grid = create_city_to_grid_mapping(df, method=mapping_method)
        self.grid_to_city = {v: k for k, v in self.city_to_grid.items()}
        print(f"      Mapped {len(self.city_to_grid)} cities to {SEIRConfig.GRID_HEIGHT}x{SEIRConfig.GRID_WIDTH} grid")
        
        # Step 2: Reshape to spatial grid
        print(f"\n[2/5] Reshaping to spatial grid (column: {self.target_column})...")
        self.spatial_data = reshape_to_spatial_grid(
            df, self.city_to_grid, column=self.target_column
        )
        print(f"      Spatial data shape: {self.spatial_data.shape}")
        
        # Step 3: Split data temporally
        print(f"\n[3/5] Splitting data (train: {self.train_days} days)...")
        train_data = self.spatial_data[:self.train_days]
        test_data = self.spatial_data[self.train_days:]
        
        val_days = int(self.train_days * self.val_ratio)
        train_data_final = train_data[:-val_days] if val_days > 0 else train_data
        val_data = train_data[-val_days:] if val_days > 0 else None
        
        print(f"      Train: {len(train_data_final)} days")
        print(f"      Val: {len(val_data) if val_data is not None else 0} days")
        print(f"      Test: {len(test_data)} days")
        
        # Step 4: Normalize (fit on training data only)
        print(f"\n[4/5] Normalizing data (method: {self.norm_method})...")
        self.normalizer = DataNormalizer(method=self.norm_method)
        train_data_norm = self.normalizer.fit_transform(train_data_final)
        val_data_norm = self.normalizer.transform(val_data) if val_data is not None else None
        test_data_norm = self.normalizer.transform(test_data)
        
        print(f"      Train range: [{train_data_norm.min():.4f}, {train_data_norm.max():.4f}]")
        
        # Step 5: Create sequences
        print(f"\n[5/5] Creating sequences (seq_len={self.seq_len}, pred_len={self.pred_len})...")
        
        seq_func = create_sequences_single_step if self.single_step else create_sequences
        seq_kwargs = {'seq_len': self.seq_len, 'stride': self.stride}
        if not self.single_step:
            seq_kwargs['pred_len'] = self.pred_len
        
        X_train, Y_train = seq_func(train_data_norm, **seq_kwargs)
        X_val, Y_val = seq_func(val_data_norm, **seq_kwargs) if val_data_norm is not None else (None, None)
        X_test, Y_test = seq_func(test_data_norm, **seq_kwargs)
        
        print(f"      Train samples: {len(X_train)}")
        print(f"      Val samples: {len(X_val) if X_val is not None else 0}")
        print(f"      Test samples: {len(X_test)}")
        print(f"      Input shape: {X_train.shape}")
        print(f"      Target shape: {Y_train.shape}")
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return {
            'X_train': X_train, 'Y_train': Y_train,
            'X_val': X_val, 'Y_val': Y_val,
            'X_test': X_test, 'Y_test': Y_test,
            'spatial_data': self.spatial_data,
            'normalizer': self.normalizer,
            'city_to_grid': self.city_to_grid
        }
    
    def create_dataloaders(
        self,
        data_dict: Dict[str, np.ndarray],
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from preprocessed data.
        
        Args:
            data_dict: Output from fit_transform()
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            
        Returns:
            Dictionary containing train/val/test DataLoaders
        """
        train_dataset = SEIRDataset(data_dict['X_train'], data_dict['Y_train'])
        test_dataset = SEIRDataset(data_dict['X_test'], data_dict['Y_test'])
        
        loaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        }
        
        if data_dict['X_val'] is not None:
            val_dataset = SEIRDataset(data_dict['X_val'], data_dict['Y_val'])
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        
        return loaders


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_data_summary(data_dict: Dict[str, np.ndarray]) -> None:
    """Print summary of preprocessed data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"{key:15s}: shape={value.shape}, dtype={value.dtype}")
            print(f"                 min={value.min():.6f}, max={value.max():.6f}, mean={value.mean():.6f}")
        elif value is not None:
            print(f"{key:15s}: {type(value).__name__}")


def verify_model_compatibility(data_dict: Dict[str, np.ndarray]) -> None:
    """
    Verify data format is compatible with ConvLSTM models.
    
    Expected format: (batch, time, channels, height, width)
    """
    X = data_dict['X_train']
    print("\n" + "=" * 60)
    print("MODEL COMPATIBILITY CHECK")
    print("=" * 60)
    
    assert len(X.shape) == 5, f"Expected 5D tensor, got {len(X.shape)}D"
    batch, time, channels, height, width = X.shape
    
    print(f"✓ Tensor is 5D: (batch, time, channels, height, width)")
    print(f"  Batch size: {batch}")
    print(f"  Sequence length: {time}")
    print(f"  Channels: {channels}")
    print(f"  Spatial: {height}x{width}")
    
    # Check compatibility with paper's architecture
    assert height == SEIRConfig.GRID_HEIGHT, f"Height mismatch: {height} != {SEIRConfig.GRID_HEIGHT}"
    assert width == SEIRConfig.GRID_WIDTH, f"Width mismatch: {width} != {SEIRConfig.GRID_WIDTH}"
    
    print(f"\n✓ Compatible with ConvLSTM (input_channels={channels})")
    print(f"✓ Compatible with AConvLSTM (spatial={height}x{width})")
    print(f"✓ Compatible with ResAConvLSTM/RAConv (16x16 grid)")


def save_preprocessed_data(
    data_dict: Dict[str, np.ndarray],
    preprocessor: 'SEIRPreprocessor',
    save_dir: str = '.'
) -> None:
    """
    Save all preprocessed data, normalizer state, and metadata so that
    downstream models can load everything without re-running preprocessing.

    Saves:
      - seir_preprocessed.npz  : X_train, Y_train, X_val, Y_val, X_test,
                                  Y_test, spatial_data arrays
      - seir_normalizer.npz    : normalizer parameters for inverse-transform
      - seir_city_grid.npz     : city ↔ grid mapping
      - seir_config.npz        : preprocessing hyper-parameters
    """
    import os, json

    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Arrays (train / val / test splits) ---
    arrays_to_save = {k: v for k, v in data_dict.items() if isinstance(v, np.ndarray)}
    arr_path = os.path.join(save_dir, 'seir_preprocessed.npz')
    np.savez_compressed(arr_path, **arrays_to_save)
    print(f"  Saved arrays         → {arr_path}")

    # --- 2. Normalizer state ---
    norm = preprocessor.normalizer
    norm_dict = {
        'method': norm.method,
        'epsilon': norm.epsilon,
        'min_val': norm.min_val if norm.min_val is not None else np.nan,
        'max_val': norm.max_val if norm.max_val is not None else np.nan,
        'mean_val': norm.mean_val if norm.mean_val is not None else np.nan,
        'std_val': norm.std_val if norm.std_val is not None else np.nan,
    }
    norm_path = os.path.join(save_dir, 'seir_normalizer.npz')
    np.savez(norm_path, **{k: np.array(v) for k, v in norm_dict.items()})
    print(f"  Saved normalizer     → {norm_path}")

    # --- 3. City ↔ grid mapping (JSON for easy use in any language) ---
    grid_path = os.path.join(save_dir, 'seir_city_grid.json')
    mapping = {city: list(pos) for city, pos in preprocessor.city_to_grid.items()}
    with open(grid_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"  Saved city grid map  → {grid_path}")

    # --- 4. Config / hyper-params ---
    config = {
        'seq_len': preprocessor.seq_len,
        'pred_len': preprocessor.pred_len,
        'train_days': preprocessor.train_days,
        'norm_method': preprocessor.norm_method,
        'target_column': preprocessor.target_column,
        'val_ratio': preprocessor.val_ratio,
        'stride': preprocessor.stride,
        'single_step': preprocessor.single_step,
        'grid_height': SEIRConfig.GRID_HEIGHT,
        'grid_width': SEIRConfig.GRID_WIDTH,
        'num_cities': SEIRConfig.NUM_CITIES,
    }
    config_path = os.path.join(save_dir, 'seir_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config         → {config_path}")


def load_preprocessed_data(load_dir: str = '.') -> Dict:
    """
    Load everything saved by save_preprocessed_data().

    Returns a dictionary with keys:
      'arrays'     – dict of numpy arrays (X_train, Y_train, …)
      'normalizer' – a fitted DataNormalizer instance
      'city_to_grid' – dict mapping city name → (row, col)
      'config'     – dict of preprocessing hyper-parameters
    """
    import os, json

    # Arrays
    arr_path = os.path.join(load_dir, 'seir_preprocessed.npz')
    with np.load(arr_path) as data:
        arrays = {k: data[k] for k in data.files}

    # Normalizer
    norm_path = os.path.join(load_dir, 'seir_normalizer.npz')
    with np.load(norm_path, allow_pickle=True) as nf:
        norm = DataNormalizer(method=str(nf['method']), epsilon=float(nf['epsilon']))
        norm.min_val  = None if np.isnan(float(nf['min_val']))  else float(nf['min_val'])
        norm.max_val  = None if np.isnan(float(nf['max_val']))  else float(nf['max_val'])
        norm.mean_val = None if np.isnan(float(nf['mean_val'])) else float(nf['mean_val'])
        norm.std_val  = None if np.isnan(float(nf['std_val']))  else float(nf['std_val'])
        norm.fitted = True

    # City grid mapping
    grid_path = os.path.join(load_dir, 'seir_city_grid.json')
    with open(grid_path, 'r') as f:
        city_to_grid = {k: tuple(v) for k, v in json.load(f).items()}

    # Config
    config_path = os.path.join(load_dir, 'seir_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    return {
        'arrays': arrays,
        'normalizer': norm,
        'city_to_grid': city_to_grid,
        'config': config,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution demonstrating the preprocessing pipeline.
    """
    import os
    
    # Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(base_dir, 'seir_baseline_300days_256cities.csv')
    SAVE_DIR  = os.path.join(base_dir, 'preprocessed_output')
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        print("Please provide the path to seir_baseline_300days_256cities.csv")
        return
    
    # Load data
    print("Loading SEIR data...")
    df = load_seir_data(DATA_PATH)
    
    # Initialize preprocessor with paper-aligned parameters
    preprocessor = SEIRPreprocessor(
        seq_len=8,           # P = 8 observations (paper)
        pred_len=4,          # Q = 4 predictions (paper)
        train_days=170,      # Training period as specified
        norm_method='minmax',
        target_column='I',   # Infected compartment only
        val_ratio=0.15,
        stride=1,
        single_step=False    # Multi-step prediction
    )
    
    # Run preprocessing
    data_dict = preprocessor.fit_transform(df)
    
    # Print summary
    print_data_summary(data_dict)
    
    # Verify compatibility
    verify_model_compatibility(data_dict)
    
    # Create DataLoaders
    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)
    
    dataloaders = preprocessor.create_dataloaders(
        data_dict,
        batch_size=16,
        num_workers=0
    )
    
    for name, loader in dataloaders.items():
        print(f"{name:10s} DataLoader: {len(loader)} batches")
    
    # Example: Get a batch
    print("\n" + "=" * 60)
    print("EXAMPLE BATCH")
    print("=" * 60)
    
    batch_x, batch_y = next(iter(dataloaders['train']))
    print(f"Input batch shape: {batch_x.shape}")
    print(f"Target batch shape: {batch_y.shape}")
    print(f"Input dtype: {batch_x.dtype}")
    print(f"Target dtype: {batch_y.dtype}")
    
    # Save preprocessed data for downstream models
    print("\n" + "=" * 60)
    print("SAVING PREPROCESSED OUTPUT")
    print("=" * 60)
    save_preprocessed_data(data_dict, preprocessor, save_dir=SAVE_DIR)
    print(f"\n✓ All outputs saved to: {SAVE_DIR}")
    print("  Load in another script with:")
    print("    from seir_preprocessing import load_preprocessed_data")
    print(f"    data = load_preprocessed_data('{SAVE_DIR}')")
    print("    X_train = data['arrays']['X_train']")
    print("    normalizer = data['normalizer']")
    
    return data_dict, dataloaders, preprocessor


if __name__ == "__main__":
    data_dict, dataloaders, preprocessor = main()
