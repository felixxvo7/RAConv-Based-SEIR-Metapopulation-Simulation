"""
Task 3: SEIR ODE Integration (Optimized)
=========================================
Implements a coupled SEIR metapopulation model across 256 Texas cities.

SEIR Equations (per city i):
    dS_i/dt = -β * (S_i * I_i / N_i) + Σ_j (θ_ji * S_j - θ_ij * S_i)
    dE_i/dt =  β * (S_i * I_i / N_i) - σ * E_i + Σ_j (θ_ji * E_j - θ_ij * E_i)
    dI_i/dt =  σ * E_i - γ * I_i + Σ_j (θ_ji * I_j - θ_ij * I_i)
    dR_i/dt =  γ * I_i + Σ_j (θ_ji * R_j - θ_ij * R_i)

Parameters (COVID-19-like baseline):
    β = 0.35 /day (transmission rate, R0=3.5, γ=0.1 => β=R0*γ)
    σ = 0.192 /day (incubation rate, 1/σ = 5.2 days)
    γ = 0.1 /day (recovery rate, 1/γ = 10 days)

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
BETA = 0.35      # Transmission rate (/day)
SIGMA = 0.192    # Incubation rate (/day), 1/σ = 5.2 days
GAMMA = 0.1      # Recovery rate (/day), 1/γ = 10 days
R0_CALC = BETA / GAMMA  # Basic reproduction number = 3.5

# Simulation parameters
T_MAX = 300      # Simulation period (days)
DT = 1.0         # Output resolution (days)


def load_data():
    """Load population and mobility data."""
    tx_pd = pd.read_csv('src_data/tx_pd.csv')
    theta = np.load('mobility_matrix.npy')
    return tx_pd, theta


def create_seir_ode(N, theta, n_cities):
    """
    Factory function to create an optimized SEIR ODE with precomputed values.
    
    Returns a function suitable for solve_ivp that uses closure for efficiency.
    """
    # Precompute transpose and outflow rates for efficiency
    theta_T = theta.T.copy()
    outflow_rate = theta.sum(axis=1)
    
    def seir_ode(t, y):
        """
        Optimized SEIR ODE system with mobility coupling.
        Uses precomputed values and fully vectorized operations.
        """
        # Reshape state into compartments
        S = y[0:n_cities]
        E = y[n_cities:2*n_cities]
        I = y[2*n_cities:3*n_cities]
        R = y[3*n_cities:4*n_cities]
        
        # Compute infection term (vectorized)
        infection = BETA * S * I / N
        
        # Compute mobility terms using matrix multiplication (vectorized)
        # Inflow: θ^T @ X (sum over sources j)
        # Outflow: outflow_rate * X
        S_net = theta_T @ S - outflow_rate * S
        E_net = theta_T @ E - outflow_rate * E
        I_net = theta_T @ I - outflow_rate * I
        R_net = theta_T @ R - outflow_rate * R
        
        # SEIR derivatives
        dSdt = -infection + S_net
        dEdt = infection - SIGMA * E + E_net
        dIdt = SIGMA * E - GAMMA * I + I_net
        dRdt = GAMMA * I + R_net
        
        return np.concatenate([dSdt, dEdt, dIdt, dRdt])
    
    return seir_ode


def run_simulation(tx_pd, theta):
    """Run the SEIR simulation using LSODA (adaptive for stiff/non-stiff)."""
    n_cities = len(tx_pd)
    N = tx_pd['population'].values.astype(float)
    
    print(f"\nSimulation Parameters:")
    print(f"  β (transmission): {BETA} /day")
    print(f"  σ (incubation):   {SIGMA} /day (period = {1/SIGMA:.1f} days)")
    print(f"  γ (recovery):     {GAMMA} /day (period = {1/GAMMA:.1f} days)")
    print(f"  R0:               {R0_CALC:.1f}")
    print(f"  Cities:           {n_cities}")
    print(f"  Duration:         {T_MAX} days")
    print(f"  State dimension:  {n_cities * 4} variables")
    
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
    
    # Create optimized ODE function
    seir_ode = create_seir_ode(N, theta, n_cities)
    
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
    
    peak_day = np.argmax(houston_I)
    peak_infected = houston_I[peak_day]
    final_recovered = houston_R[-1]
    houston_pop = N[houston_idx]
    
    print(f"   City: {houston_name} (pop: {houston_pop:,})")
    print(f"   Peak day: {peak_day}")
    print(f"   Peak infected: {peak_infected:,.0f} ({peak_infected/houston_pop*100:.1f}%)")
    print(f"   Final recovered: {final_recovered:,.0f} ({final_recovered/houston_pop*100:.1f}%)")
    
    # 4. State-wide summary
    print("\n4. State-Wide Summary:")
    total_I = results[:, :, 2].sum(axis=1)  # Total infected over time
    total_R = results[:, :, 3].sum(axis=1)  # Total recovered over time
    
    peak_day_state = np.argmax(total_I)
    peak_infected_state = total_I[peak_day_state]
    final_R_state = total_R[-1]
    attack_rate = final_R_state / total_N
    
    print(f"   Peak day (state): {peak_day_state}")
    print(f"   Peak infected (state): {peak_infected_state:,.0f} ({peak_infected_state/total_N*100:.1f}%)")
    print(f"   Final recovered (state): {final_R_state:,.0f}")
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
    print("Task 3: SEIR ODE Integration")
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
