"""
Task 2: Mobility Matrix Design
==============================
Implements a gravity-based mobility matrix Θ = [θ_ij] representing movement rates between cities.

Formula: θ_ij = c * N_j / D_ij^α  for i ≠ j, θ_ii = 0

Parameters:
- α = 2.0 (distance decay exponent)
- c = scaling constant (normalized so each row sums to 1-3% of source city population per day)
"""

import pandas as pd
import numpy as np

def create_mobility_matrix(tx_pd, distance_df, alpha=2.0, daily_outflow_rate=0.02):
    """
    Create gravity-based mobility matrix.
    
    Args:
        tx_pd: DataFrame with city, population columns
        distance_df: DataFrame with pairwise distances (km)
        alpha: Distance decay exponent (default 2.0)
        daily_outflow_rate: Target fraction of population moving out daily (default 2%)
    
    Returns:
        theta: Mobility matrix (256 x 256)
    """
    n = len(tx_pd)
    populations = tx_pd['population'].values
    distances = distance_df.values
    
    # Initialize mobility matrix
    theta = np.zeros((n, n))
    
    # Compute raw gravity values
    for i in range(n):
        for j in range(n):
            if i != j and distances[i, j] > 0:
                # θ_ij = N_j / D_ij^α
                theta[i, j] = populations[j] / (distances[i, j] ** alpha)
    
    # Normalize each row so outflow sums to target rate * population
    for i in range(n):
        row_sum = theta[i, :].sum()
        if row_sum > 0:
            # Scale so total outflow = daily_outflow_rate * N_i
            target_outflow = daily_outflow_rate * populations[i]
            theta[i, :] = theta[i, :] * (target_outflow / row_sum)
    
    return theta


def validate_mobility_matrix(theta, tx_pd, alpha=2.0):
    """Validate the mobility matrix."""
    n = len(tx_pd)
    populations = tx_pd['population'].values
    
    print("\n--- Mobility Matrix Validation ---")
    
    # 1. Check row sums (1-3% daily outflow)
    print("\n1. Row Sums (Daily Outflow Rate):")
    row_sums = theta.sum(axis=1)
    outflow_rates = row_sums / populations
    
    print(f"   Min outflow rate: {outflow_rates.min():.4f} ({outflow_rates.min()*100:.2f}%)")
    print(f"   Max outflow rate: {outflow_rates.max():.4f} ({outflow_rates.max()*100:.2f}%)")
    print(f"   Mean outflow rate: {outflow_rates.mean():.4f} ({outflow_rates.mean()*100:.2f}%)")
    
    in_range = ((outflow_rates >= 0.01) & (outflow_rates <= 0.03)).sum()
    print(f"   Cities with 1-3% outflow: {in_range}/{n}")
    
    # 2. Large-city dominance check
    print("\n2. Large-City Dominance Check:")
    city_inflow = theta.sum(axis=0)  # Column sums = total inflow to each city
    city_outflow = theta.sum(axis=1)  # Row sums = total outflow from each city
    
    # Find top 5 cities by population
    top5_idx = np.argsort(populations)[-5:][::-1]
    print("   Top 5 cities by population:")
    for idx in top5_idx:
        print(f"     {tx_pd.iloc[idx]['city']}: pop={populations[idx]:,}, "
              f"inflow={city_inflow[idx]:.1f}, outflow={city_outflow[idx]:.1f}")
    
    # 3. Distance decay verification (log-log plot data)
    print("\n3. Distance Decay Check:")
    print(f"   Expected slope (α): -{alpha}")
    
    # Sample some city pairs for verification
    sample_i = 0  # First city
    nonzero_j = np.where(theta[sample_i, :] > 0)[0][:10]
    if len(nonzero_j) > 0:
        print(f"   Sample flows from {tx_pd.iloc[sample_i]['city']}:")
        for j in nonzero_j[:5]:
            dist = np.sqrt((tx_pd.iloc[sample_i]['lat'] - tx_pd.iloc[j]['lat'])**2 + 
                          (tx_pd.iloc[sample_i]['lng'] - tx_pd.iloc[j]['lng'])**2) * 111  # approx km
            print(f"     -> {tx_pd.iloc[j]['city']}: flow={theta[sample_i,j]:.2f}")
    
    return True


def main():
    print("=" * 60)
    print("Task 2: Mobility Matrix Design")
    print("=" * 60)
    
    # Load data
    tx_pd = pd.read_csv('src_data/tx_pd.csv')
    distance_df = pd.read_csv('src_data/distance_df.csv', index_col=0)
    
    print(f"\nLoaded {len(tx_pd)} cities")
    
    # Parameters from plan
    alpha = 2.0  # Distance decay exponent
    daily_outflow_rate = 0.02  # 2% daily outflow (middle of 1-3% range)
    
    print(f"\nParameters:")
    print(f"  α (distance decay): {alpha}")
    print(f"  Daily outflow rate: {daily_outflow_rate*100}%")
    
    # Create mobility matrix
    print("\nCreating mobility matrix...")
    theta = create_mobility_matrix(tx_pd, distance_df, alpha, daily_outflow_rate)
    
    print(f"Mobility matrix shape: {theta.shape}")
    print(f"Non-zero entries: {(theta > 0).sum()}")
    print(f"Total daily mobility: {theta.sum():,.0f} person-trips")
    
    # Validate
    validate_mobility_matrix(theta, tx_pd, alpha)
    
    # Save mobility matrix
    output_file = 'mobility_matrix.npy'
    np.save(output_file, theta)
    print(f"\n✓ Saved mobility matrix to {output_file}")
    
    # Also save as CSV for inspection
    mobility_df = pd.DataFrame(theta, index=tx_pd['city'], columns=tx_pd['city'])
    mobility_df.to_csv('mobility_matrix.csv')
    print(f"✓ Saved mobility matrix to mobility_matrix.csv")
    
    print("\n" + "=" * 60)
    print("Task 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
