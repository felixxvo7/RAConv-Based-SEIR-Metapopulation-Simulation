# Spatiotemporal Forecasting under Domain Shift using RAConv

## Overview
This project studies **spatiotemporal forecasting under domain shift** by adapting the **RAConv (Residual + Attention + Convolutional Network)** architecture from its original application in **cellular traffic prediction** to **epidemiological forecasting**.

The work is conducted as part of **COMP 4360 – Machine Learning (University of Manitoba)** and focuses on understanding, refactoring, and evaluating a state-of-the-art deep learning model when applied to a fundamentally different problem domain.

Rather than reproducing results, the project emphasizes **model adaptation, synthetic data generation, and systematic evaluation**.

---

## Problem Domain
The original RAConv model was designed to predict future cellular traffic loads at each spatial location by learning:
- Local short-term spatiotemporal patterns (neighbor interactions)
- Long-term dependencies (daily/weekly cycles and trends)

In this project, we transfer the same modeling assumptions to **epidemiology**, where:
- Each spatial cell represents a city
- Temporal dynamics reflect disease progression
- Spatial coupling reflects human mobility and disease spread

This constitutes a clear **domain shift** from network traffic to infectious disease modeling.

---

## Dataset and Data Generation

### Synthetic Epidemiological Data
Instead of using real epidemiological data, we generate **synthetic spatiotemporal data** using a **mechanistic metapopulation SEIR model**.

Key properties:
- 256 cities arranged into a 16 × 16 spatial grid
- Each city follows SEIR ordinary differential equations
- Cities are connected through a mobility matrix controlling inter-city transmission
- Disease parameters fall within ranges reported by CDC and peer-reviewed COVID-19 literature

Simulation details:
- Duration: 300 days
- State variables: S, E, I, R
- Grid size: 16 × 16
- Total state dimension: 1024 variables
- Output stored as a 3D tensor (time × space × space)

This approach allows full control over regime changes, mobility strength, and transmission intensity while avoiding privacy or reporting biases.

---

## Model Architecture

### RAConv
The adapted RAConv architecture consists of:

1. **Conv3D + ResConv3D blocks**
   - Three residual blocks with 3D convolutions
   - Capture local short-term spatiotemporal patterns
   - Residual connections stabilize deep training

2. **Attention-aided ConvLSTM (AConvLSTM)**
   - Two ConvLSTM layers with spatial attention
   - Capture long-term spatiotemporal dependencies
   - Attention highlights influential regions and time steps

3. **Multi-step forecasting head**
   - Input: past P time steps
   - Output: next Q future frames

The input is treated as a **video-like tensor**, where each frame represents the spatial disease state at a given time.

---

## Adaptation Strategy
In the original paper, 4,096 base stations are clustered into 16 groups, each modeled separately.  
In this project:

- All 256 cities are treated as **one spatial cluster**
- A single shared RAConv model is trained
- This reduces computational cost while preserving spatial structure
- The setup remains comparable since the original paper reports results from a single local model

Each city maps to one grid cell, forming a 16 × 16 spatial representation of disease incidence at time *t*.

---

## Experimental Plan

### Baseline
- ConvLSTM without residual blocks or attention

### Ablation Studies
1. **Baseline ConvLSTM**
   - No ResConv3D
   - No attention mechanism

2. **Attention ConvLSTM**
   - Attention enabled
   - ResConv3D removed

3. **Full RAConv**
   - ResConv3D + Attention ConvLSTM

Performance is compared using RMSE and MAPE to quantify the contribution of each architectural component.

---

## Evaluation Goals
- Measure forecasting accuracy under domain shift
- Evaluate robustness to regime changes in transmission and mobility
- Understand the contribution of residual learning and attention
- Identify failure modes and limitations of RAConv in epidemiological settings

Results are not intended for real-world policy use.

---

## Implementation
- Python
- PyTorch
- NumPy / Pandas
- SciPy (ODE simulation)
- Matplotlib / Seaborn for visualization

Training is performed on university computing resources, with local machines used for preprocessing.

---

## References
- Wang, Z., & Wong, V. W. (2022). *Cellular Traffic Prediction Using Deep Convolutional Neural Network with Attention Mechanism*. IEEE ICC.
- Shi, X., et al. (2015). *Convolutional LSTM Network*. NeurIPS.
- Keeling, M. J., & Rohani, P. (2008). *Modeling Infectious Diseases in Humans and Animals*.
- Balcan, D., et al. (2009). *Multiscale mobility networks and disease spread*. PNAS.

---

## Notes
This repository represents an **academic model adaptation and analysis project**.  
Synthetic data is used for transparency, reproducibility, and controlled experimentation.
