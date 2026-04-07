"""
Task 5: Preprocess generated SEIR data
==========================================
Reshape the SEIR dataset into a 16x16 spatial grid tensor for model input.
Each grid cell represents one city, and the assignment preserves geographical
neighborhood relationships between cities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def main():
    print("=" * 60)
    print("Task 5: Data Collection")
    print("=" * 60)

    tx_pd = pd.read_csv('../src_data/tx_pd.csv')
    seir_df = pd.read_csv('../src_data/seir_baseline_300days_256cities.csv')
    coords = tx_pd[['lat', 'lng']].to_numpy()
    cities_name = tx_pd['city'].unique().tolist()

    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

    grid_size = 16
    grid_coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    grid_coords_norm = grid_coords / (grid_size - 1)

    cost_matrix = cdist(coords_norm, grid_coords_norm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    city_to_grid_coords = {cities_name[i]: tuple(grid_coords[col_ind[i]]) for i in range(len(cities_name))}

    features = ['S', 'E', 'I', 'R']
    num_features = len(features)

    num_days = seir_df['day'].max() + 1  # since day starts at 0
    seir_grid_data = np.zeros((num_days, grid_size, grid_size, num_features))

    for _, row in seir_df.iterrows():
        day_idx = row['day']
        city = row['city']
        r, c = city_to_grid_coords[city]
        seir_grid_data[day_idx, r, c, :] = row[features].to_numpy()

    save_dir = '../src_data/seir_grid_data.npy'
    np.save(save_dir, seir_grid_data)

    plt.figure(figsize=(6, 6))
    plt.scatter(coords_norm[:, 0], coords_norm[:, 1], c='blue', label='Original cities')
    plt.scatter(grid_coords_norm[:, 0], grid_coords_norm[:, 1], c='red', marker='s', label='Grid cells')
    for i in range(len(coords)):
        plt.plot([coords_norm[i, 0], grid_coords_norm[col_ind[i], 0]],
                [coords_norm[i, 1], grid_coords_norm[col_ind[i], 1]],
                c='gray', linewidth=0.5)
    plt.legend()
    plt.title("City coordinates → 16x16 grid assignment")
    plt.savefig('../graphs/city_to_grid_mapping.png', dpi=150)
    print("Saved city_to_grid_mapping.png")

    print("\n" + "=" * 60)
    print("Task 5 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
