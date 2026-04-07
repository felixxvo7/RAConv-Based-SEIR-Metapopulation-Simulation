# RAConv-Based SEIR Metapopulation Simulation

Spatiotemporal epidemic forecasting across 256 Texas cities using a deep learning architecture adapted from cellular traffic prediction. Built for **COMP 4360 -- Machine Learning** at the University of Manitoba.

![Geographic wave spread](plot/geographic_wave_spread.gif)

## What This Project Does

This project transfers the **RAConv** (Residual + Attention + ConvLSTM) architecture from its original domain of cellular traffic prediction to **epidemiological forecasting**. A synthetic SEIR metapopulation model generates ground-truth epidemic data across 256 Texas cities over 300 days, and the deep learning models learn to predict future infection patterns from past spatial snapshots.

The key question: *can a model designed for cell tower traffic also learn the spatiotemporal dynamics of disease spread?*

## Results

RAConv consistently outperforms the AConvLSTM ablation across all lookback windows, with the gap widening at shorter horizons where residual 3D convolutions capture more useful short-range spatial patterns.

| Lookback (P) | RAConv RMSE | AConvLSTM RMSE | RAConv MAE | AConvLSTM MAE | Improvement |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4  | 4,558 | 7,746 | 624  | 739  | 41% |
| 6  | 3,775 | 7,724 | 484  | 1,261 | 51% |
| 8  | 3,482 | 6,136 | 477  | 717  | 43% |
| 14 | 3,229 | 4,453 | 469  | 538  | 27% |

*All metrics are in real scale (infected count). Forecast horizon Q = 7 days.*

## Architecture

```
Input: (B, 1, P, 16, 16)           # P past days on a 16x16 city grid
        │
        ▼
┌─────────────────────┐
│   Conv3D + BN + ReLU│             # Initial feature extraction
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  ResBlock3D  (×3)   │             # 64 → 64 → 96 → 128 channels
│  (5 Conv3D layers   │             # Two residual shortcuts per block
│   per block)        │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  AConvLSTM  (×2)    │             # 2-layer attention ConvLSTM
│  256 hidden channels│             # Attention on input + output gates
│  3×3 kernels        │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Autoregressive     │             # Rolls forward Q steps using
│  Future Prediction  │             # feedback from previous prediction
└─────────┬───────────┘
          ▼
Output: (B, Q, 1, 16, 16)          # Q future days predicted
```

**RAConv**: 12.3M parameters | **AConvLSTM** (ablation, no ResBlock3D): 7.4M parameters

## Project Structure

```
├── data_generation/                 # Synthetic SEIR data pipeline
│   ├── Task1_Data_Collection.py     # Load 256 Texas cities, build distance matrix
│   ├── Task2_Data_Preprocessing.py  # Validate population and coordinate data
│   ├── Task3_Mobility_Matrix.py     # Gravity-based inter-city mobility matrix
│   ├── Task4_SEIR_Simulation.py     # Coupled SEIR ODE with time-varying β and mobility
│   ├── Task5_SEIR_Preprocess.py     # Map cities to 16×16 grid for model input
│   ├── SEIR calibrated parameters.py# Peer-reviewed β(t) and mobility schedules
│   └── plot.py                      # Statewide SEIR summary plots
│
├── Models/
│   ├── RAConvLSTM.py                # Full RAConv model (ResBlock3D + AConvLSTM)
│   ├── fullmodel_runner.py          # Train/eval RAConv across lookback windows
│   ├── AConvLSTM/
│   │   ├── AConvLSTM.py             # Attention ConvLSTM cell and stacked layers
│   │   └── experiment_runner.py     # Train/eval AConvLSTM (ablation baseline)
│   ├── ResBlock/
│   │   └── ResBlock.py              # 3D residual blocks (Fig. 4 from paper)
│   ├── Preprocessing/
│   │   └── seir_preprocessing.py    # CSV → normalized sliding-window .npz tensors
│   ├── results_fullmodel/           # RAConv checkpoints, metrics, plots per P
│   └── results_ablation/            # AConvLSTM checkpoints, metrics, plots per P
│
├── plot/                            # Visualization scripts
│   ├── geographic_wave_gif.py       # Animated epidemic spread across Texas
│   ├── heatmap_day_270.py           # Actual vs predicted 16×16 heatmaps
│   ├── prediction_plot.py           # Timeline comparison (RAConv vs AConvLSTM)
│   ├── plot_fig9.py                 # Metric comparison across lookback windows
│   └── spatial_accuracy_plot.py     # Side-by-side spatial accuracy heatmaps
│
├── src_data/                        # Source data (city coordinates, populations)
├── graphs/                          # Generated HTML maps and assignment plots
└── plan/                            # Project proposal and methodology documents
```

## Setup

### Requirements

- Python 3.9+
- PyTorch 1.12+ (CUDA recommended)
- NumPy, Pandas, SciPy, Matplotlib, scikit-learn, Pillow

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn pillow
```

### Data Generation (from scratch)

Run the data generation pipeline in order from the `data_generation/` directory:

```bash
cd data_generation

python Task1_Data_Collection.py      # Produces tx_pd.csv and distance_df.csv
python Task2_Data_Preprocessing.py   # Validates the generated data
python Task3_Mobility_Matrix.py      # Produces mobility_matrix.csv/.npy
python Task4_SEIR_Simulation.py      # Runs 300-day SEIR simulation (takes ~30s)
python Task5_SEIR_Preprocess.py      # Maps cities to 16×16 grid
```

Task 1 requires `uscities.csv` in `src_data/`. Tasks 3-4 run from the `data_generation/` directory and produce outputs there.

### Preprocessing for Models

Convert raw SEIR CSV into model-ready tensors for all lookback windows:

```bash
cd Models/Preprocessing
python seir_preprocessing.py
```

This produces `seir_preprocessed_P{4,6,8,14}.npz` files containing normalized train/val/test splits with sliding-window sequences.

### Training

**Full model (RAConv):**
```bash
cd Models
python fullmodel_runner.py                  # Train all P values (P4, P6, P8, P14)
python fullmodel_runner.py --p 14           # Train only P14
python fullmodel_runner.py --epochs 100 --lr 5e-4 --cpu
```

**Ablation (AConvLSTM only, no ResBlock3D):**
```bash
cd Models/AConvLSTM
python experiment_runner.py                 # Train all P values
python experiment_runner.py --p 8 14        # Train only P8 and P14
```

Both runners save checkpoints, metrics JSON, loss curves, per-step metrics, spatial error maps, and prediction plots to their respective `results_*/P{N}/` directories.

### Visualization

All plot scripts run from the `plot/` directory and auto-locate data files relative to the project root:

```bash
python plot/geographic_wave_gif.py          # Animated GIF of epidemic spread
python plot/prediction_plot.py              # RAConv vs AConvLSTM timeline
python plot/heatmap_day_270.py --day 270    # Spatial heatmap for a specific day
python plot/spatial_accuracy_plot.py --day 275 --p 14
python plot/plot_fig9.py                    # Metric comparison across P values
```

## SEIR Simulation Details

The epidemic simulation uses a coupled SEIR metapopulation model with:

- **256 cities** connected by a gravity-based mobility matrix
- **Time-varying transmission** β(t) calibrated to Texas COVID-19 policy events (stay-at-home orders, mask mandates, phased reopenings)
- **Time-varying mobility** from Google Community Mobility Reports
- **Nonlinear feedback**: state-dependent behavioral response suppresses transmission at high prevalence
- **Local mobility suppression**: cities with active outbreaks reduce outbound travel

The simulation produces a two-wave epidemic (Days 0-300) with a visible trough between waves, seeded from Houston with 1,000 initial infections.

## References

- Wang, Z., & Wong, V. W. (2022). *Cellular Traffic Prediction Using Deep Convolutional Neural Network with Attention Mechanism*. IEEE ICC.
- Shi, X., et al. (2015). *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*. NeurIPS.
- Yu, D., et al. (2021). *Assessing effects of reopening policies on COVID-19 pandemic in Texas*. Infectious Disease Modelling, 6, 461-473.
- Keeling, M. J., & Rohani, P. (2008). *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press.
- Balcan, D., et al. (2009). *Multiscale mobility networks and the spatial spreading of infectious diseases*. PNAS, 106(51), 21484-21489.
