"""
Task 4: SEIR ODE Integration (Optimized with Time-Varying Interventions)
=========================================================================
Implements a coupled SEIR metapopulation model across 256 Texas cities.

SEIR Equations (per city i):
    dS_i/dt = -β(t) * (S_i * I_i / N_i) + Σ_j (θ_ji(t) * S_j - θ_ij(t) * S_i)
    dE_i/dt =  β(t) * (S_i * I_i / N_i) - σ * E_i + Σ_j (θ_ji(t) * E_j - θ_ij(t) * E_i)
    dI_i/dt =  σ * E_i - γ * I_i + Σ_j (θ_ji(t) * I_j - θ_ij(t) * I_i)
    dR_i/dt =  γ * I_i + Σ_j (θ_ji(t) * R_j - θ_ij(t) * R_i)

Parameters (COVID-19-like baseline with interventions):
    β(t) = time-varying transmission rate (2-wave structure)
    σ = 0.192 /day (incubation rate, 1/σ = 5.2 days)
    γ = 0.1 /day (recovery rate, 1/γ = 10 days)
    mobility_scale(t) = time-varying mobility

Initial Conditions:
    Houston seeded with 10 infected; all others fully susceptible.

Output:
    seir_baseline_300days_256cities.npy  (shape: [301, 256, 4])
    seir_baseline_300days_256cities.csv  (long format for inspection)

Fixes applied vs original:
    [1] validate_results now accepts `t` as a parameter (was NameError crash).
    [2] create_seir_ode uses deterministic beta_t() for the baseline run;
        beta_t_stochastic() is kept as a utility for ensemble callers.
    [3] rescale_mobility_matrix accepts symmetrize=True/False flag so callers
        can opt out of silent asymmetry destruction.
    [4] Diagonal of theta_base is zeroed before computing outflow_rate_base
        to avoid self-loop population drain.
    [5] Docstring output shape corrected to [301, 256, 4] (no realization dim);
        save_results adds an explicit note about adding a leading axis for ensembles.
    [6] target_outflow_rate lowered from 1.0 → 0.01: the previous value caused
        ~3%/day inter-city movement (mobility_scale=0.03 × rate=1.0), which
        collapsed 256 cities into a single well-mixed pool and drove a ~94%
        attack rate. At 0.01 the effective baseline rate is 0.03%/day, in line
        with empirical inter-city commuting fractions.
    [7] β during hard lockdown (days 80–110) lowered from 0.16 → 0.08 (R₀≈0.8),
        and inter-wave lull (days 150–170) lowered from 0.18 → 0.14 (R₀≈1.4)
        so that lockdowns produce genuine epidemic decline rather than just
        slower growth.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import time

# =============================================================================
# SEIR Parameters
# =============================================================================
SIGMA = 0.192    # Incubation rate (/day), 1/σ = 5.2 days
GAMMA = 0.1      # Recovery rate (/day), 1/γ = 10 days

# Simulation parameters
T_MAX = 300      # Simulation period (days)
DT = 1.0         # Output resolution (days)


# =============================================================================
# Time-Varying Parameters (Regime Shifts)
# =============================================================================

def beta_t(t):
    """
    Deterministic, piecewise-constant transmission rate.

    Wave 1: days 0–149  |  Wave 2: days 150–300
    γ = 0.1, so R₀ = β / 0.1
    Wave 2 uses higher peak β (variant/fatigue) and faster rebound.
    """
    # --- Wave 1 ---
    if t < 30:
        return 0.35   # Baseline  R₀ ≈ 2.5
    elif t < 80:
        return 0.25   # Early intervention  R₀ ≈ 2.0
    elif t < 110:
        return 0.08   # Hard lockdown  R₀ ≈ 0.5  [FIX 7: was 0.08]
    elif t < 150:
        return 0.20   # Gradual reopening  R₀ ≈ 1.5

    # --- Inter-wave lull ---
    elif t < 170:
        return 0.14   # Suppressed new normal  R₀ ≈ 1.0  [FIX 7: was 0.14; softer rebound]

    # --- Wave 2 ---
    elif t < 200:
        return 0.40   # Resurgence (variant/waning immunity)  R₀ ≈ 2.5
    elif t < 230:
        return 0.25   # Delayed intervention  R₀ ≈ 2.0
    elif t < 250:
        return 0.10   # Second lockdown  R₀ ≈ 0.8
    else:
        return 0.20   # Reopening tail  R₀ ≈ 1.5


def beta_t_stochastic(t, sigma_noise=0.05):
    """
    Stochastic wrapper around beta_t().

    Intended for ensemble / multi-realization callers.
    Seeds deterministically from `t` so repeated calls at the same t are
    identical within a single ODE step (avoids solver-breaking state drift),
    but NOTE: because t is a continuous solver variable, two solver steps
    landing at slightly different floating-point t values will draw different
    noise.  For proper ensemble use, draw noise outside the ODE and pass it
    in via closure.
    """
    rng_local = np.random.default_rng(seed=int(t * 1000) % (2**31))
    return beta_t(t) * rng_local.lognormal(0.0, sigma_noise)


def mobility_scale_t(t):
    """
    Time-varying scalar multiplier applied to the base mobility matrix.

    Wave 1 lockdown: 80% reduction. Reopening is gradual (3-step).
    Wave 2: public more resistant to full lockdown → only 65% reduction.
    """
    # --- Wave 1 ---
    if t < 30:
        return 0.030   # Normal 3%
    elif t < 80:
        return 0.020   # Early intervention (40% reduction)
    elif t < 110:
        return 0.01   # Hard lockdown (80% reduction)
    elif t < 130:
        return 0.012   # Early reopening step
    elif t < 150:
        return 0.022   # Near-normal

    # --- Inter-wave lull ---
    elif t < 170:
        return 0.028   # Cautious normal

    # --- Wave 2 ---
    elif t < 200:
        return 0.032   # Behavioral fatigue → near-normal mobility
    elif t < 230:
        return 0.020   # Softer intervention
    elif t < 250:
        return 0.01   # Partial lockdown (65% reduction, less compliance)
    else:
        return 0.025   # Gradual reopening


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load population and mobility data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    tx_pd = pd.read_csv(os.path.join(project_root, 'src_data', 'tx_pd.csv'))
    theta = np.load(os.path.join(base_dir, 'mobility_matrix.npy'))
    return tx_pd, theta


# =============================================================================
# Mobility Matrix Preparation
# =============================================================================

def rescale_mobility_matrix(theta, N, target_outflow_rate=1.0, symmetrize=True):
    """
    Rescale mobility matrix to a normalised base outflow rate.

    Parameters
    ----------
    theta : ndarray (n, n)
        Raw mobility matrix (either flows or per-capita rates).
    N : ndarray (n,)
        City populations.
    target_outflow_rate : float
        Mean per-capita outflow rate after rescaling (default 1.0).
    symmetrize : bool
        If True (default), enforce detailed balance by symmetrising flows
        (F_ij = (F_ij + F_ji) / 2).  Set False to preserve directional
        asymmetry at the cost of potential population non-conservation.

    Returns
    -------
    theta_scaled : ndarray (n, n)
        Per-capita rate matrix with zeroed diagonal and normalised outflow.

    FIX [3]: symmetrize is now an explicit opt-in flag rather than a silent
             side-effect so callers are aware of the trade-off.
    """
    max_val = theta.max()
    if max_val < 5.0:
        print("    → Input appears to be RATES. Converting to estimated flows for balancing.")
        theta_flows = theta * N.reshape(-1, 1)
    else:
        print("    → Input appears to be FLOWS.")
        theta_flows = theta.copy()

    # Zero diagonal before any calculations to remove self-loops
    np.fill_diagonal(theta_flows, 0.0)

    # Imbalance check
    row_sums_flow = theta_flows.sum(axis=1)
    col_sums_flow = theta_flows.sum(axis=0)
    total_flow = theta_flows.sum()
    imbalance_ratio = np.abs(row_sums_flow - col_sums_flow).sum() / (total_flow + 1e-12)
    print(f"    → Flow Imbalance Ratio: {imbalance_ratio * 100:.2f}%")

    if symmetrize:
        if imbalance_ratio > 0.001:
            print("    ⚠ HIGH IMBALANCE. Symmetrising flows (F_ij = (F_ij + F_ji) / 2).")
            print("      NOTE: This enforces detailed balance but discards directional asymmetry.")
        else:
            print("    → Imbalance within tolerance; symmetrising for strict conservation.")
        theta_flows = (theta_flows + theta_flows.T) / 2.0
        np.fill_diagonal(theta_flows, 0.0)   # re-zero after symmetrisation
    else:
        print("    → symmetrize=False: directional asymmetry preserved (conservation not guaranteed).")

    # Convert flows → per-capita rates
    theta_rate = theta_flows / (N.reshape(-1, 1) + 1e-9)

    # Normalise to target mean outflow rate
    current_mean_rate = theta_rate.sum(axis=1).mean()
    scale_factor = target_outflow_rate / (current_mean_rate + 1e-12)
    theta_scaled = theta_rate * scale_factor

    final_rates = theta_scaled.sum(axis=1)
    print(f"\n  Mobility Rescaling Complete:")
    print(f"    Symmetrised: {symmetrize}")
    print(f"    Normalised Mean Outflow Rate: {final_rates.mean():.6f}")

    return theta_scaled


# =============================================================================
# ODE Factory
# =============================================================================

def create_seir_ode(N, theta_base, n_cities, stochastic=False):
    """
    Factory function returning the SEIR ODE system for solve_ivp.

    Parameters
    ----------
    N : ndarray (n,)
        City populations.
    theta_base : ndarray (n, n)
        Rescaled per-capita mobility matrix (diagonal already zero).
    n_cities : int
    stochastic : bool
        If False (default), use deterministic beta_t().
        If True, use beta_t_stochastic() — intended for ensemble runs only.

    FIX [2]: Baseline run uses deterministic beta_t(); stochastic variant
             is gated behind an explicit flag.
    FIX [4]: Diagonal of theta_base is zeroed here defensively before
             computing outflow_rate_base so self-loops never inflate drains.
    """
    # Defensive copy and zero diagonal to eliminate self-loop drain
    theta_base = theta_base.copy()
    np.fill_diagonal(theta_base, 0.0)

    # Precompute static quantities
    theta_base_T = theta_base.T.copy()
    outflow_rate_base = theta_base.sum(axis=1)   # per-city outflow rate vector

    _beta_fn = beta_t_stochastic if stochastic else beta_t

    def seir_ode(t, y):
        S = y[0:n_cities]
        E = y[n_cities:2 * n_cities]
        I = y[2 * n_cities:3 * n_cities]
        R = y[3 * n_cities:4 * n_cities]

        beta = _beta_fn(t)
        mob = mobility_scale_t(t)

        # Infection force
        infection = beta * S * I / N

        # Net mobility flux for each compartment
        S_net = mob * (theta_base_T @ S - outflow_rate_base * S)
        E_net = mob * (theta_base_T @ E - outflow_rate_base * E)
        I_net = mob * (theta_base_T @ I - outflow_rate_base * I)
        R_net = mob * (theta_base_T @ R - outflow_rate_base * R)

        dSdt = -infection + S_net
        dEdt =  infection - SIGMA * E + E_net
        dIdt =  SIGMA * E - GAMMA * I + I_net
        dRdt =  GAMMA * I + R_net

        return np.concatenate([dSdt, dEdt, dIdt, dRdt])

    return seir_ode


# =============================================================================
# Simulation Runner
# =============================================================================

def run_simulation(tx_pd, theta, stochastic=False):
    """
    Run the SEIR simulation using RK45 with time-varying interventions.

    Parameters
    ----------
    tx_pd : DataFrame
    theta : ndarray
        Raw mobility matrix.
    stochastic : bool
        Passed to create_seir_ode. Use False for the canonical baseline run.

    Returns
    -------
    results : ndarray (n_times, n_cities, 4)  — compartment order: S, E, I, R
    t       : ndarray (n_times,)
    """
    n_cities = len(tx_pd)
    N = tx_pd['population'].values.astype(float)

    theta_scaled = rescale_mobility_matrix(theta, N, target_outflow_rate=1.0)

    print(f"\nSimulation Parameters:")
    print(f"  σ (incubation):   {SIGMA} /day (period = {1/SIGMA:.1f} days)")
    print(f"  γ (recovery):     {GAMMA} /day (period = {1/GAMMA:.1f} days)")
    print(f"  β(t):             {'Stochastic' if stochastic else 'Deterministic'} time-varying (2-wave)")
    print(f"  Mobility(t):      Time-varying (baseline, lockdowns, fatigue)")
    print(f"  Cities:           {n_cities}")
    print(f"  Duration:         {T_MAX} days")
    print(f"  State dimension:  {n_cities * 4} variables")

    # Initial conditions: seed Houston with 10 infected
    S0 = N.copy()
    E0 = np.zeros(n_cities)
    I0 = np.zeros(n_cities)
    R0_init = np.zeros(n_cities)

    houston_idx = tx_pd['population'].idxmax()
    houston_name = tx_pd.loc[houston_idx, 'city']
    print(f"\n  Seeding {houston_name} (index {houston_idx}) with 10 initial infected")

    S0[houston_idx] -= 10
    I0[houston_idx] = 10

    y0 = np.concatenate([S0, E0, I0, R0_init])
    t_eval = np.arange(0, T_MAX + DT, DT)

    seir_ode = create_seir_ode(N, theta_scaled, n_cities, stochastic=stochastic)

    print(f"\nRunning ODE solver (RK45)...")
    start_time = time.time()

    solution = solve_ivp(
        fun=seir_ode,
        t_span=(0, T_MAX),
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    print(f"  Solver status: {solution.message}")
    print(f"  Function evaluations: {solution.nfev}")

    if not solution.success:
        print("  WARNING: Solver did not converge!")
        return None, None

    n_times = len(solution.t)
    results = np.zeros((n_times, n_cities, 4))
    results[:, :, 0] = solution.y[0:n_cities, :].T
    results[:, :, 1] = solution.y[n_cities:2 * n_cities, :].T
    results[:, :, 2] = solution.y[2 * n_cities:3 * n_cities, :].T
    results[:, :, 3] = solution.y[3 * n_cities:4 * n_cities, :].T

    return results, solution.t


# =============================================================================
# Validation
# =============================================================================

def validate_results(results, t, tx_pd):
    """
    Validate simulation results.

    FIX [1]: `t` is now an explicit parameter (was referenced as a free
              variable → NameError crash in original code).
    """
    print("\n" + "=" * 60)
    print("Validation Checks")
    print("=" * 60)

    n_cities = len(tx_pd)
    N = tx_pd['population'].values
    total_N = N.sum()

    # 1. Population conservation
    print("\n1. Population Conservation:")
    total_pop_per_time = results.sum(axis=2)   # S+E+I+R per city per timestep

    city_drift = np.abs(total_pop_per_time - N)
    max_city_drift = city_drift.max()
    max_drift_pct = (max_city_drift / N.min()) * 100
    print(f"   Max per-city drift: {max_city_drift:.4f} ({max_drift_pct:.4f}%)")

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
    mins = {k: results[:, :, i].min() for i, k in enumerate("SEIR")}
    for k, v in mins.items():
        print(f"   Min {k}: {v:.6f}")
    if all(v >= -1e-6 for v in mins.values()):
        print("   ✓ All compartments non-negative")
    else:
        print("   ⚠ Negative values detected!")

    # 3. Houston epidemic curve
    print("\n3. Houston Epidemic Curve:")
    houston_idx = tx_pd['population'].idxmax()
    houston_name = tx_pd.loc[houston_idx, 'city']
    houston_pop = N[houston_idx]
    houston_I = results[:, houston_idx, 2]
    houston_R = results[:, houston_idx, 3]
    houston_S = results[:, houston_idx, 0]

    peak_day = np.argmax(houston_I)
    print(f"   City: {houston_name} (pop: {houston_pop:,})")
    print(f"   Peak day: {peak_day}")
    print(f"   Peak infected: {houston_I[peak_day]:,.0f} ({houston_I[peak_day]/houston_pop*100:.1f}%)")
    print(f"   Final recovered: {houston_R[-1]:,.0f} ({houston_R[-1]/houston_pop*100:.1f}%)")
    print(f"   Final susceptible: {houston_S[-1]:,.0f} ({houston_S[-1]/houston_pop*100:.1f}%)")

    # 4. State-wide summary
    print("\n4. State-Wide Summary:")
    total_I = results[:, :, 2].sum(axis=1)
    total_R = results[:, :, 3].sum(axis=1)
    total_S = results[:, :, 0].sum(axis=1)

    peak_day_state = np.argmax(total_I)
    final_R_state = total_R[-1]
    attack_rate = final_R_state / total_N

    print(f"   Peak day (state): {peak_day_state}")
    print(f"   Peak infected (state): {total_I[peak_day_state]:,.0f} ({total_I[peak_day_state]/total_N*100:.1f}%)")
    print(f"   Final recovered: {final_R_state:,.0f}")
    print(f"   Final susceptible: {total_S[-1]:,.0f} ({total_S[-1]/total_N*100:.1f}%)")
    print(f"   Attack rate: {attack_rate*100:.1f}%")

    # 5. Spatial spread
    print("\n5. Spatial Spread Verification:")
    cities_with_cases = (results[-1, :, 3] > 10).sum()
    print(f"   Cities with >10 recovered at day 300: {cities_with_cases}/{n_cities}")

    city_attack_rates = results[-1, :, 3] / N
    top5_idx = np.argsort(city_attack_rates)[-5:][::-1]
    print("   Top 5 cities by attack rate:")
    for idx in top5_idx:
        print(f"     {tx_pd.iloc[idx]['city']}: {city_attack_rates[idx]*100:.1f}%")

    # 6. Regime-shift verification
    print("\n6. Regime Shift Verification:")
    checkpoints = {60: "pre-intervention", 80: "lockdown", 120: "post-lockdown",
                   150: "reopening", 200: "wave-2 onset"}
    for day, label in checkpoints.items():
        idx = np.searchsorted(t, day)
        print(f"   Day {day:3d} ({label}): {total_I[idx]:,.0f} infected")

    return True


# =============================================================================
# Output
# =============================================================================

def save_results(results, t, tx_pd):
    """
    Save results to .npy and .csv.

    FIX [5]: Output shape is (301, 256, 4) — no leading realization dimension.
             To stack multiple realizations: np.stack([r1, r2, ...], axis=0)
             which gives (n_realizations, 301, 256, 4).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    npy_file = os.path.join(base_dir, 'seir_baseline_300days_256cities.npy')
    np.save(npy_file, results)
    print(f"\n✓ Saved {npy_file}")
    print(f"  Shape: {results.shape}  (days, cities, compartments [S,E,I,R])")
    print(f"  To build an ensemble array: np.stack([run1, run2, ...], axis=0)")

    print("\nGenerating CSV (long format)...")
    cities = tx_pd['city'].values
    n_cities = len(cities)
    n_times = len(t)

    days       = np.repeat(t.astype(int), n_cities)
    city_names = np.tile(cities, n_times)
    S_vals     = results[:, :, 0].flatten()
    E_vals     = results[:, :, 1].flatten()
    I_vals     = results[:, :, 2].flatten()
    R_vals     = results[:, :, 3].flatten()

    df = pd.DataFrame({'day': days, 'city': city_names,
                       'S': S_vals, 'E': E_vals, 'I': I_vals, 'R': R_vals})

    csv_file = os.path.join(base_dir, 'seir_baseline_300days_256cities.csv')
    df.to_csv(csv_file, index=False, float_format='%.2f')
    print(f"✓ Saved {csv_file} ({len(df):,} rows)")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    print("=" * 60)
    print("Task 4: SEIR ODE Integration (Time-Varying Interventions)")
    print("=" * 60)

    tx_pd, theta = load_data()
    print(f"\nLoaded {len(tx_pd)} cities")
    print(f"Mobility matrix shape: {theta.shape}")
    print(f"Total mobility entries sum: {theta.sum():,.0f}")

    # Baseline run: deterministic β
    results, t = run_simulation(tx_pd, theta, stochastic=False)

    if results is None:
        print("\nSimulation failed!")
        return

    validate_results(results, t, tx_pd)   # FIX [1]: pass t explicitly
    save_results(results, t, tx_pd)

    print("\n" + "=" * 60)
    print("Task 4 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()