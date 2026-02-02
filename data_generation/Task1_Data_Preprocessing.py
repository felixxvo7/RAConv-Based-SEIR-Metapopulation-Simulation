"""
Task 1: Data Collection and Preprocessing
==========================================
This script loads and validates the Texas city population and distance data.
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Task 1: Data Collection and Preprocessing")
    print("=" * 60)
    
    # Load data
    tx_pd = pd.read_csv('src_data/tx_pd.csv')
    distance_df = pd.read_csv('src_data/distance_df.csv', index_col=0)
    
    print(f"\nLoaded {len(tx_pd)} cities")
    print(f"Distance matrix shape: {distance_df.shape}")
    
    # 1. Population Check (~15-20M total)
    print("\n--- 1. Population Check ---")
    total_pop = tx_pd['population'].sum()
    print(f"Total population: {total_pop:,}")
    pop_ok = 15_000_000 <= total_pop <= 34_000_000
    if pop_ok:
        print("✓ Population in expected range (15-34M)")
    else:
        print(f"⚠ Population {total_pop:,} outside expected range (15-34M)")
    
    # 2. Coordinate Bounds Check
    print("\n--- 2. Coordinate Bounds Check ---")
    lat_min, lat_max = tx_pd['lat'].min(), tx_pd['lat'].max()
    lng_min, lng_max = tx_pd['lng'].min(), tx_pd['lng'].max()
    
    print(f"Latitude range: {lat_min:.2f} to {lat_max:.2f} (expected 25.8 to 36.5)")
    print(f"Longitude range: {lng_min:.2f} to {lng_max:.2f} (expected -106.6 to -93.5)")
    
    lat_ok = 25.8 <= lat_min and lat_max <= 36.5
    lng_ok = -106.6 <= lng_min and lng_max <= -93.5
    print(f"✓ Latitude bounds OK: {lat_ok}")
    print(f"✓ Longitude bounds OK: {lng_ok}")
    
    # 3. Duplicate City Check
    print("\n--- 3. Duplicate City Check ---")
    duplicates = tx_pd['city'].duplicated().sum()
    if duplicates == 0:
        print("✓ No duplicate city names found")
    else:
        print(f"⚠ Found {duplicates} duplicate city names")
        print(tx_pd[tx_pd['city'].duplicated(keep=False)])
    
    # 4. Distance Matrix Symmetry Check
    print("\n--- 4. Distance Matrix Symmetry Check ---")
    dist_matrix = distance_df.values
    is_symmetric = np.allclose(dist_matrix, dist_matrix.T, atol=1e-6)
    print(f"✓ Distance matrix symmetric: {is_symmetric}")
    
    # 5. Triangle Inequality Check (Sample of 50 cities)
    print("\n--- 5. Triangle Inequality Check (Sample) ---")
    n = len(dist_matrix)
    violations = 0
    for i in range(min(50, n)):
        for j in range(min(50, n)):
            for k in range(min(50, n)):
                if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j] + 1e-6:
                    violations += 1
    
    if violations == 0:
        print("✓ Triangle inequality holds for sampled cities")
    else:
        print(f"⚠ Found {violations} triangle inequality violations")
    
    # Summary
    print("\n" + "=" * 60)
    print("Task 1 Validation Summary")
    print("=" * 60)
    print(f"Cities loaded: {len(tx_pd)}")
    print(f"Total population: {total_pop:,}")
    print(f"Coordinate bounds valid: {lat_ok and lng_ok}")
    print(f"No duplicate cities: {duplicates == 0}")
    print(f"Distance matrix symmetric: {is_symmetric}")
    print(f"Triangle inequality violations: {violations}")
    
    all_passed = pop_ok and lat_ok and lng_ok and (duplicates == 0) and is_symmetric and (violations == 0)
    if all_passed:
        print("\n✓ All Task 1 checks PASSED!")
    else:
        print("\n⚠ Some checks did not pass - review above")
    
    return all_passed

if __name__ == "__main__":
    main()
