"""
Task 3: SEIR ODE Integration (Optimized with Time-Varying Interventions)
=========================================================================
Implements a coupled SEIR metapopulation model across 256 Texas cities.

SEIR Equations (per city i):
    dS_i/dt = -β(t) * (S_i * I_i / N_i) + Σ_j (θ_ji(t) * S_j - θ_ij(t) * S_i)
    dE_i/dt =  β(t) * (S_i * I_i / N_i) - σ * E_i + Σ_j (θ_ji(t) * E_j - θ_ij(t) * E_i)
    dI_i/dt =  σ * E_i - γ * I_i + Σ_j (θ_ji(t) * I_j - θ_ij(t) * I_i)
    dR_i/dt =  γ * I_i + Σ_j (θ_ji(t) * R_j - θ_ij(t) * R_i)

Parameters (COVID-19-like baseline with interventions):
    β(t) = time-varying transmission rate (0.35 baseline -> 0.20 lockdown -> 0.25 reopening)
    σ = 0.192 /day (incubation rate, 1/σ = 5.2 days)
    γ = 0.1 /day (recovery rate, 1/γ = 10 days)
    mobility_scale(t) = time-varying mobility (5% -> 0.5% lockdown -> 2% reopening)

Initial Conditions:
    Houston seeded with 10 infected; all others fully susceptible.

Output:
    seir_baseline_300days_256cities.npy  (shape: [301, 256, 4])
    seir_baseline_300days_256cities.csv  (long format for inspection)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import time

# =============================================================================
# SEIR Parameters (Table 1 from plan)
# =============================================================================
SIGMA = 0.2    # Incubation rate (/day), 1/σ = 5 days
GAMMA = 0.1      # Recovery rate (/day), 1/γ = 10 days

# Simulation parameters
T_MAX = 300      # Simulation period (days)
DT = 1.0         # Output resolution (days)

# =============================================================================
# Time-Varying Parameters (Regime Shifts)
# =============================================================================

def beta_t(t):
    """
    Literature-based transmission regimes.
    
    With γ=0.1:
        β=0.35 → R₀=3.5 (baseline COVID-19)
        β=0.30 → R₀=3.0 (early intervention)
        β=0.18 → R₀=1.8 (lockdown)
        β=0.25 → R₀=2.5 (reopening)
    """
    if t < 45:
        return 0.35   # Baseline (R₀≈3.5) [Liu et al. 2020]
    elif t < 65:
        return 0.30   # Early intervention (R₀≈3.0)
    elif t < 150:
        return 0.18   # Lockdown (R₀≈1.8) [Gatto et al. 2020]
    else:
        return 0.25   # Reopening (R₀≈2.5)


def mobility_scale_t(t):
    """
    Literature-based mobility regimes (absolute rates).
    
    Range: 0.001-0.05 (0.1%-5% daily) [Balcan et al. 2009]
    Lockdown reductions: 70-90% [Google Mobility 2020]
    """
    if t < 45:
        return 0.030   # 3.0% normal mobility [Apolloni et al. 2014]
    elif t < 65:
        return 0.020   # 2.0% early intervention (33% reduction)
    elif t < 110:
        return 0.006   # 0.6% lockdown (80% reduction) [Kraemer et al. 2020]
    else:
        return 0.020   # 2.0% reopening


def load_data():
    """Load population and mobility data."""
    tx_pd = pd.read_csv('src_data/tx_pd.csv')
    theta = np.load('mobility_matrix.npy')
    return tx_pd, theta


def rescale_mobility_matrix(theta, N, target_outflow_rate=1.0):
    """
    Rescale mobility matrix to base rate = 1.0.
    
    Assumes theta is already a per-capita rate matrix (row sums ≈ 1.0).
    """
    row_sums = theta.sum(axis=1)
    
    # ============== DIAGNOSTIC CHECK ==============
    # Check if theta is already normalized (row sums ≈ 1) or absolute flows (row sums >> 1)
    max_row_sum = row_sums.max()
    mean_row_sum = row_sums.mean()
    
    print(f"\n  Mobility Matrix Diagnostic:")
    print(f"    Max row sum: {max_row_sum:.6f}")
    print(f"    Mean row sum: {mean_row_sum:.6f}")
    print(f"    Min row sum: {row_sums.min():.6f}")
    
    if max_row_sum < 2.0:
        print(f"    → Matrix appears to be ALREADY NORMALIZED (row sums ≈ 1)")
        print(f"    → Using matrix AS-IS (no per-capita conversion needed)")
        theta_rate = theta.copy()  # Already per-capita rates
    else:
        print(f"    → Matrix appears to be ABSOLUTE FLOWS (row sums >> 1)")
        
        # Check for flow balance (Crucial for population conservation)
        col_sums = theta.sum(axis=0)
        total_flow = theta.sum()
        imbalance = np.abs(row_sums - col_sums).sum() / total_flow
        print(f"    → Flow Imbalance: {imbalance*100:.2f}% (Target < 1%)")
        
        if imbalance > 0.01:
            print(f"    → ⚠ HIGH IMBALANCE DETECTED! Enforcing symmetry to conserve population.")
            theta = (theta + theta.T) / 2.0
            print(f"    → Symmetrized flows (T_ij = T_ji)")
            
        print(f"    → Converting to per-capita rates")
        theta_rate = theta / N.reshape(-1, 1)
    # ==============================================
    
    # Rescale to base rate = 1.0
    current_mean_rate = theta_rate.sum(axis=1).mean()
    scale_factor = target_outflow_rate / current_mean_rate
    theta_scaled = theta_rate * scale_factor
    
    # Verify
    final_rates = theta_scaled.sum(axis=1)
    print(f"\n  Mobility Rescaling:")
    print(f"    Input row sum range: [{row_sums.min():.4f}, {row_sums.max():.4f}]")
    print(f"    Normalized to base rate: {final_rates.mean():.6f}")
    print(f"    Will be scaled by mobility_scale_t() values (0.006-0.03)")
    
    return theta_scaled


def create_seir_ode(N, theta_base, n_cities):
    """
    Factory function to create an optimized SEIR ODE with time-varying parameters.
    
    Returns a function suitable for solve_ivp that uses closure for efficiency.
    
    Args:
        N: Population vector
        theta_base: Base mobility matrix (already rescaled to reasonable rates)
        n_cities: Number of cities
    """
    # Precompute base transpose for efficiency
    theta_base_T = theta_base.T.copy()
    outflow_rate_base = theta_base.sum(axis=1)
    
    def seir_ode(t, y):
        """
        Optimized SEIR ODE system with time-varying β(t) and mobility coupling.
        """
        # Reshape state into compartments
        S = y[0:n_cities]
        E = y[n_cities:2*n_cities]
        I = y[2*n_cities:3*n_cities]
        R = y[3*n_cities:4*n_cities]
        
        # Get time-varying parameters
        beta = beta_t(t)
        mobility_scale = mobility_scale_t(t)
        
        # Compute infection term (vectorized) with time-varying β
        infection = beta * S * I / N
        
        # Compute mobility terms with time-varying scale
        # Scale the base mobility by the current mobility factor
        S_net = mobility_scale * (theta_base_T @ S - outflow_rate_base * S)
        E_net = mobility_scale * (theta_base_T @ E - outflow_rate_base * E)
        I_net = mobility_scale * (theta_base_T @ I - outflow_rate_base * I)
        R_net = mobility_scale * (theta_base_T @ R - outflow_rate_base * R)
        
        # SEIR derivatives
        dSdt = -infection + S_net
        dEdt = infection - SIGMA * E + E_net
        dIdt = SIGMA * E - GAMMA * I + I_net
        dRdt = GAMMA * I + R_net
        
        return np.concatenate([dSdt, dEdt, dIdt, dRdt])
    
    return seir_ode


def run_simulation(tx_pd, theta):
    """Run the SEIR simulation using RK45 with time-varying interventions."""
    n_cities = len(tx_pd)
    N = tx_pd['population'].values.astype(float)
    
    # Rescale mobility matrix to base rate (normalized to 1.0)
    theta_scaled = rescale_mobility_matrix(theta, N, target_outflow_rate=1.0)
    
    print(f"\nSimulation Parameters:")
    print(f"  σ (incubation):   {SIGMA} /day (period = {1/SIGMA:.1f} days)")
    print(f"  γ (recovery):     {GAMMA} /day (period = {1/GAMMA:.1f} days)")
    print(f"  β(t):             Time-varying (0.35 -> 0.30 -> 0.18 -> 0.25)")
    print(f"  Mobility(t):      Time-varying (3.0% -> 2.0% -> 0.6% -> 2.0%)")
    print(f"  Cities:           {n_cities}")
    print(f"  Duration:         {T_MAX} days")
    print(f"  State dimension:  {n_cities * 4} variables")
    
    print(f"\n  Regime Schedule:")
    print(f"    Day 0-44:   β=0.35 (R₀≈3.5), mobility=3.0% (baseline)")
    print(f"    Day 45-64:  β=0.30 (R₀≈3.0), mobility=2.0% (early intervention)")
    print(f"    Day 65-109: β=0.18 (R₀≈1.8), mobility=0.6% (lockdown)")
    print(f"    Day 110-149: β=0.18 (R₀≈1.8), mobility=2.0% (reopening)")
    print(f"    Day 150+:   β=0.25 (R₀≈2.5), mobility=2.0% (partial reopening)")
    
    # Initial conditions: Seed Houston with 10 infected
    S0 = N.copy()
    E0 = np.zeros(n_cities)
    I0 = np.zeros(n_cities)
    R0_init = np.zeros(n_cities)
    
    # Find Houston (largest city)
    houston_idx = tx_pd['population'].idxmax()
    houston_name = tx_pd.loc[houston_idx, 'city']
    print(f"\n  Seeding {houston_name} (index {houston_idx}) with 10 initial infected")
    
    S0[houston_idx] -= 10
    I0[houston_idx] = 10
    
    # Flatten initial state
    y0 = np.concatenate([S0, E0, I0, R0_init])
    
    # Time points for output
    t_eval = np.arange(0, T_MAX + DT, DT)
    
    # Create optimized ODE function with rescaled mobility
    seir_ode = create_seir_ode(N, theta_scaled, n_cities)
    
    print(f"\nRunning ODE solver (RK45 - explicit Runge-Kutta method)...")
    start_time = time.time()
    
    # Solve ODE using RK45 (explicit 4th/5th order Runge-Kutta)
    solution = solve_ivp(
        fun=seir_ode,
        t_span=(0, T_MAX),
        y0=y0,
        method='RK45',       # Explicit Runge-Kutta 4(5)
        t_eval=t_eval,
        rtol=1e-6,           # Relative tolerance (high accuracy)
        atol=1e-9            # Absolute tolerance (high accuracy)
    )
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    print(f"  Solver status: {solution.message}")
    print(f"  Number of function evaluations: {solution.nfev}")
    
    if not solution.success:
        print("  WARNING: Solver did not converge!")
        return None, None
    
    # Reshape results: (n_times, n_cities, 4 compartments)
    n_times = len(solution.t)
    results = np.zeros((n_times, n_cities, 4))
    
    for i in range(n_times):
        results[i, :, 0] = solution.y[0:n_cities, i]              # S
        results[i, :, 1] = solution.y[n_cities:2*n_cities, i]     # E
        results[i, :, 2] = solution.y[2*n_cities:3*n_cities, i]   # I
        results[i, :, 3] = solution.y[3*n_cities:4*n_cities, i]   # R
    
    return results, solution.t


def validate_results(results, tx_pd):
    """Validate simulation results thoroughly."""
    print("\n" + "=" * 60)
    print("Validation Checks")
    print("=" * 60)
    
    n_cities = len(tx_pd)
    N = tx_pd['population'].values
    total_N = N.sum()
    
    # 1. Population conservation (per city and global)
    print("\n1. Population Conservation:")
    total_pop_per_time = results.sum(axis=2)  # Sum S+E+I+R for each city at each time
    
    # Per-city conservation
    city_drift = np.abs(total_pop_per_time - N)
    max_city_drift = city_drift.max()
    max_drift_pct = (max_city_drift / N.min()) * 100
    print(f"   Max per-city drift: {max_city_drift:.4f} ({max_drift_pct:.4f}%)")
    
    # Global conservation
    global_pop_per_time = total_pop_per_time.sum(axis=1)
    global_drift = np.abs(global_pop_per_time - total_N)
    max_global_drift = global_drift.max()
    print(f"   Max global drift: {max_global_drift:.4f} ({max_global_drift/total_N*100:.6f}%)")
    
    if max_drift_pct < 0.1:
        print("   ✓ Population conserved (drift < 0.1%)")
    else:
        print(f"   ⚠ Population drift detected ({max_drift_pct:.2f}%)")
    
    # 2. Non-negativity
    print("\n2. Non-Negativity:")
    min_S = results[:, :, 0].min()
    min_E = results[:, :, 1].min()
    min_I = results[:, :, 2].min()
    min_R = results[:, :, 3].min()
    print(f"   Min S: {min_S:.6f}, Min E: {min_E:.6f}, Min I: {min_I:.6f}, Min R: {min_R:.6f}")
    
    if min(min_S, min_E, min_I, min_R) >= -1e-6:
        print("   ✓ All compartments non-negative")
    else:
        print("   ⚠ Negative values detected!")
    
    # 3. Houston Epidemic Curve (seed city)
    print("\n3. Houston Epidemic Curve:")
    houston_idx = tx_pd['population'].idxmax()
    houston_name = tx_pd.loc[houston_idx, 'city']
    houston_I = results[:, houston_idx, 2]
    houston_R = results[:, houston_idx, 3]
    houston_S = results[:, houston_idx, 0]
    
    peak_day = np.argmax(houston_I)
    peak_infected = houston_I[peak_day]
    final_recovered = houston_R[-1]
    final_susceptible = houston_S[-1]
    houston_pop = N[houston_idx]
    
    print(f"   City: {houston_name} (pop: {houston_pop:,})")
    print(f"   Peak day: {peak_day}")
    print(f"   Peak infected: {peak_infected:,.0f} ({peak_infected/houston_pop*100:.1f}%)")
    print(f"   Final recovered: {final_recovered:,.0f} ({final_recovered/houston_pop*100:.1f}%)")
    print(f"   Final susceptible: {final_susceptible:,.0f} ({final_susceptible/houston_pop*100:.1f}%)")
    
    # 4. State-wide summary
    print("\n4. State-Wide Summary:")
    total_I = results[:, :, 2].sum(axis=1)  # Total infected over time
    total_R = results[:, :, 3].sum(axis=1)  # Total recovered over time
    total_S = results[:, :, 0].sum(axis=1)  # Total susceptible over time
    
    peak_day_state = np.argmax(total_I)
    peak_infected_state = total_I[peak_day_state]
    final_R_state = total_R[-1]
    final_S_state = total_S[-1]
    attack_rate = final_R_state / total_N
    
    print(f"   Peak day (state): {peak_day_state}")
    print(f"   Peak infected (state): {peak_infected_state:,.0f} ({peak_infected_state/total_N*100:.1f}%)")
    print(f"   Final recovered (state): {final_R_state:,.0f}")
    print(f"   Final susceptible (state): {final_S_state:,.0f} ({final_S_state/total_N*100:.1f}%)")
    print(f"   Attack rate: {attack_rate*100:.1f}%")
    
    # 5. Spread verification (did epidemic reach other cities?)
    print("\n5. Spatial Spread Verification:")
    cities_with_cases = (results[-1, :, 3] > 10).sum()  # Cities with >10 recovered
    print(f"   Cities with >10 recovered at day 300: {cities_with_cases}/{n_cities}")
    
    # Find 5 cities with highest attack rates
    city_attack_rates = results[-1, :, 3] / N
    top5_idx = np.argsort(city_attack_rates)[-5:][::-1]
    print("   Top 5 cities by attack rate:")
    for idx in top5_idx:
        print(f"     {tx_pd.iloc[idx]['city']}: {city_attack_rates[idx]*100:.1f}%")
    
    # 6. Regime shift verification
    print("\n6. Regime Shift Verification:")
    # Check if infections slowed down after interventions
    I_day60 = total_I[60]
    I_day80 = total_I[80]
    I_day120 = total_I[120]
    I_day150 = total_I[150]
    I_day200 = total_I[200]
    
    print(f"   Infected at day 60 (pre-intervention): {I_day60:,.0f}")
    print(f"   Infected at day 80 (lockdown): {I_day80:,.0f}")
    print(f"   Infected at day 120 (post-lockdown): {I_day120:,.0f}")
    print(f"   Infected at day 150 (reopening): {I_day150:,.0f}")
    print(f"   Infected at day 200: {I_day200:,.0f}")
    
    return True


def save_results(results, t, tx_pd):
    """Save results to .npy and .csv files."""
    
    # Save as NumPy array (primary format for ML)
    npy_file = 'seir_baseline_300days_256cities.npy'
    np.save(npy_file, results)
    print(f"\n✓ Saved to {npy_file}")
    print(f"  Shape: {results.shape} (days, cities, compartments)")
    print(f"  Compartment order: [S, E, I, R]")
    
    # Save as CSV (long format for inspection/analysis)
    print("\nGenerating CSV (long format)...")
    
    # Efficient CSV generation using list comprehension
    cities = tx_pd['city'].values
    n_cities = len(cities)
    n_times = len(t)
    
    # Pre-allocate arrays
    days = np.repeat(t.astype(int), n_cities)
    city_names = np.tile(cities, n_times)
    S_vals = results[:, :, 0].flatten()
    E_vals = results[:, :, 1].flatten()
    I_vals = results[:, :, 2].flatten()
    R_vals = results[:, :, 3].flatten()
    
    df = pd.DataFrame({
        'day': days,
        'city': city_names,
        'S': S_vals,
        'E': E_vals,
        'I': I_vals,
        'R': R_vals
    })
    
    csv_file = 'seir_baseline_300days_256cities.csv'
    df.to_csv(csv_file, index=False, float_format='%.2f')
    print(f"✓ Saved to {csv_file} ({len(df):,} rows)")


def main():
    print("=" * 60)
    print("Task 3: SEIR ODE Integration (Time-Varying Interventions)")
    print("=" * 60)
    
    # Load data
    tx_pd, theta = load_data()
    print(f"\nLoaded {len(tx_pd)} cities")
    print(f"Mobility matrix shape: {theta.shape}")
    print(f"Total mobility (person-trips/day): {theta.sum():,.0f}")
    
    # Run simulation
    results, t = run_simulation(tx_pd, theta)
    
    if results is None:
        print("\nSimulation failed!")
        return
    
    # Validate
    validate_results(results, tx_pd)
    
    # Save results
    save_results(results, t, tx_pd)
    
    print("\n" + "=" * 60)
    print("Task 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
