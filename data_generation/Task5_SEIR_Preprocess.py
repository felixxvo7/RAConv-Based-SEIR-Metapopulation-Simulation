"""
Task 5: Preprocess generated SEIR data
==========================================
Reshape the SEIR dataset into a 16×16 spatial grid tensor for model input.
Each grid cell represents one station, and the assignment preserves geographical neighborhood relationships between stations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("Task 5: Data Collection")
    print("=" * 60)
    
    # Step 0: Extract coordinates and station names
    tx_pd = pd.read_csv('../src_data/tx_pd.csv')
    seir_df = pd.read_csv('../data_generation/seir_baseline_300days_256cities.csv')
    coords = tx_pd[['lat','lng']].to_numpy()  # shape = (256,2)
    station_names = tx_pd['city'].unique().tolist()  # unique stations

    # Step 1: Normalize coordinates to [0,1]
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

    # Step 2: Create 16x16 grid coordinates
    grid_size = 16
    grid_coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    grid_coords_norm = grid_coords / (grid_size - 1)

    # Step 3: Compute assignment cost
    cost_matrix = cdist(coords_norm, grid_coords_norm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    station_to_grid = {station_names[i]: tuple(grid_coords[col_ind[i]]) for i in range(len(station_names))}

    # Step 4: Build seir_grid_data
    features = ['S', 'E', 'I', 'R']
    num_features = len(features)

    num_days = seir_df['day'].max() + 1  # since day starts at 0
    seir_grid_data = np.zeros((num_days, grid_size, grid_size, num_features))

    for _, row in seir_df.iterrows():
        day_idx = row['day'] # directly use day as index
        city = row['city']
        r, c = station_to_grid[city]
        seir_grid_data[day_idx, r, c, :] = row[features].to_numpy()

    save_dir = 'seir_grid_data.npy'
    np.save(save_dir, seir_grid_data)
    
    print("\n" + "=" * 60)
    print("Task 5 Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()