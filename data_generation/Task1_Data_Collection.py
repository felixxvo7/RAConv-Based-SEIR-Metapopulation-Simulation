"""
Task 1: Data Collection
==========================================
This script load the most populated 256 Texas cities data and calculate their distance matrix.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np
import os

def main():
    print("=" * 60)
    print("Task 1: Data Collection")
    print("=" * 60)

    # Get uscities.csv
    root_dir = os.getcwd()
    csv_path = '../src_data/uscities.csv'
    df = pd.read_csv(csv_path)

    # Select 256 of the most populated cites in Texas
    # Create a data of those Texas cities with thier population and coordinates 
    tx_city_population_coor = df[(df['state_id'] == 'TX')]

    tx_city_sorted = tx_city_population_coor.sort_values(by=['population'], ascending=False)
    tx_city_sample = tx_city_sorted.iloc[:256]

    total_population = tx_city_sample.agg({'population': 'sum'})

    tx_pd = tx_city_sample[['city', 'population', 'lat', 'lng']].reset_index(drop=True)

    print("\ntx_pd (first 5 rows):")
    print(tx_pd.head())
    save_dir = '../src_data/tx_pd.csv'
    tx_pd.to_csv(save_dir, index=False)
    print("Saved tx_pd.csv")

    # Create distance matrix of the cities
    # Extract coordinates and convert to radians
    coords = tx_pd[['lat', 'lng']].map(radians).values

    # Calculate distance matrix using haversine_distances
    # Multiply by Earth's radius (6371 km) to get distances in kilometers
    distance_matrix = haversine_distances(coords, coords) * 6371

    # Convert the numpy array to a pandas DataFrame for better readability
    # Use city names as index and columns for clarity
    distance_df = pd.DataFrame(distance_matrix, index=tx_pd['city'], columns=tx_pd['city'])

    print("\nDistance matrix (first 5 rows):")
    print(distance_df.head())

    # Save distance matrix
    save_dir = '../src_data/distance_df.csv'
    distance_df.to_csv(save_dir, index=True)
    print("Saved distance_df.csv")
    
    print("\n" + "=" * 60)
    print("Task 1 Complete!")
    print("=" * 60)
    
if __name__ == "__main__":
    main()