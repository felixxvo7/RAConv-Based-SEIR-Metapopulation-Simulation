"""
Calibrated Time-Varying Parameters for Texas COVID-19 SEIR Metapopulation Model
=================================================================================
Replaces the original beta_t() and mobility_scale_t() with functions anchored
to real Texas COVID-19 policy events, peer-reviewed R(t) estimates, and
Google Community Mobility Report data.

Temporal anchor
---------------
    Day 0  = March 6, 2020  (first confirmed TX community case, Fort Bend County)
    Day 300 = January 1, 2021  (Wave 2 still rising toward its Jan 12 peak)

    ┌─────────────────────────────────────────────────────────────────────┐
    │  NOTE: The real Wave 2 peak (≈26 k cases/day, 14,218 hospitalised) │
    │  occurred on January 12–13, 2021 = Day 312.  To capture the full   │
    │  second-wave arc, extend T_MAX from 300 → 320 in the main script.  │
    └─────────────────────────────────────────────────────────────────────┘

Primary sources
---------------
β(t) — transmission rate
    [B1] Yu D, Zhu G, Wang X et al. "Assessing effects of reopening policies
         on COVID-19 pandemic in Texas with a data-driven transmission model."
         Infectious Disease Modelling 6 (2021) 461–473. PMC8314068.
         → Texas-specific SEIR fit; β estimated from DSHS case data.

    [B2] Oden JT et al. "Impact of Social Distancing on COVID-19 Healthcare
         Demand, Central Texas." Emerg Infect Dis 26(10) 2020. CDC/EID.
         → R₀ = 2.2 baseline; sensitivity at R₀ = 3.5 for Austin metro.

    [B3] Chudik A, Pesaran MH, Rebucci A. "COVID-19 Time-Varying Reproduction
         Numbers Worldwide." NBER Working Paper 28629 (2021).
         → North America 2-wave R(t) time series; Wave 2 larger due to
           behavioral fatigue + indoor winter transmission, NOT higher β.

    [B4] Arroyo-Marioli F et al. "Tracking R of COVID-19: A new real-time
         estimation using the Kalman filter." PLOS ONE 2021. doi:10.1371/
         journal.pone.0244474.
         → Cross-country Rt estimates; Texas Rt trajectory.

    [B5] UT COVID-19 Modeling Consortium, Texas Dashboard (TACC/UT Austin).
         https://covid-19.tacc.utexas.edu/dashboards/texas/
         → Texas-specific R(t) using DSHS hospitalisations + SafeGraph
           cell-phone mobility.  R(t) < 1 confirmed after mask mandate.

    [B6] Cambridge Core / Disaster Med & Public Health Preparedness (2021).
         "Mathematical Modeling and COVID-19 Forecast in Texas, USA."
         → Mean R₀ = 2.65 across full pandemic in Texas.

mobility_scale_t(t) — inter-city travel scalar
    [M1] Google COVID-19 Community Mobility Reports (Texas state level),
         Feb 2020 – Jan 2021.  google.com/covid19/mobility
         → Retail & recreation, transit, workplace % changes vs Jan–Feb 2020
           baseline; converted to multiplicative scalars here.

    [M2] IHME SEIR model: "Modeling COVID-19 scenarios for the United States."
         Nature Medicine 27 (2021) 51–63. PMC7806509.
         → Mobility covariates used as SEIR β drivers; Wave 2 mobility
           reductions ≈ half those of Wave 1 (behavioral fatigue documented).

    [M3] Community movement and COVID-19: a global study using Google's
         Community Mobility Reports. BMJ Open 2020. PMC7729173.
         → Negative correlation between case counts and retail/transit
           mobility confirmed for North America.

Key policy events (Texas) — mapped to model days
-------------------------------------------------
    Day  0  Mar  6  First TX community case confirmed (Fort Bend County)
    Day  5  Mar 11  Events > 500 banned; voluntary distancing begins
    Day 13  Mar 19  Abbott closes schools statewide until May 4
    Day 25  Mar 31  Statewide stay-at-home order issued
    Day 56  May  1  Phase 1 reopening: restaurants/retail at 25 %
    Day 65  May 11  Phase 2: restaurants/retail at 50 %
    Day 89  Jun  3  Phase 3: restaurants at 75 %; pools/gyms open
    Day 111 Jun 25  Bars closed again; restaurants back to 50 % (surge)
    Day 118 Jul  2  Statewide mask mandate (counties > 20 active cases)
    Day 130 Jul 14  Wave 1 peak ≈ 14 000 new cases/day (7-day avg)
    Day 165 Aug  8  Cases declining; Wave 1 resolved
    Day 185 Sep  7  Trough ≈ 3 200 new cases/day
    Day 195 Sep 17  Business capacity raised to 75 % — Wave 2 trigger
    Day 215 Oct  8  Wave 2 exponential growth phase begins
    Day 265 Nov 27  Thanksgiving — holiday amplification
    Day 283 Dec 14  First vaccines administered in Texas (front-line workers)
    Day 300 Jan  1  End of simulation (Wave 2 still rising)
    Day 312 Jan 12  Real Wave 2 peak ≈ 26 000 new cases/day [OUTSIDE window]
"""

import numpy as np

# =============================================================================
# Model constants (unchanged from original)
# =============================================================================
SIGMA = 0.192   # 1/σ = 5.2-day incubation (Linton et al., Int J Environ Res 2020)
GAMMA = 0.10    # 1/γ = 10-day infectious period  (He et al., Nat Med 2020)


# =============================================================================
# β(t) — Transmission Rate
# =============================================================================

def beta_t(t):
    """
    Deterministic, piecewise-constant SARS-CoV-2 transmission rate (β) for
    Texas, anchored to Day 0 = March 6, 2020.

    β is the per-capita transmission rate; R₀ = β / γ = β / 0.1.
    Effective Rₜ ≈ β(t) / γ  when S/N ≈ 1 (early epidemic).

    All values are consistent with peer-reviewed Texas-specific SEIR fits
    [B1, B2, B6] and Rₜ time series [B4, B5].  See module docstring for full
    citations.

    Wave-structure rationale
    ------------------------
    Wave 2 (Oct 2020 – Jan 2021) produced LARGER case counts than Wave 1
    despite LOWER β.  The mechanism [B3, B5]:
      • Spatial diffusion: virus reached rural Texas (256 cities fully seeded)
      • Higher initial infected pool → faster absolute growth even at R₀ ≈ 1.3
      • Mask compliance fatigue reduced NPI effectiveness
    Do NOT inflate Wave 2 β above Wave 1 to force a bigger wave — the
    metapopulation structure handles the amplitude difference organically.

    Regime summary
    --------------
    Days    Date range          β       R₀    Event / justification
    ─────── ─────────────────── ─────── ───── ────────────────────────────────
      0– 5  Mar  6–11 2020      0.30    3.0   Uncontrolled spread  [B2, B6]
      5–25  Mar 11–31           0.26    2.6   Voluntary distancing, school
                                              closures  [B3, B5]
     25–56  Mar 31–May 1        0.12    1.2   Stay-at-home order; TX essential
                                              businesses open throughout  [B1]
     56–89  May  1–Jun 3        0.16    1.6   Phase 1–2 reopening  [B1, B5]
     89–111 Jun  3–25           0.20    2.0   Phase 3 (restaurants 75 %)  [B5]
    111–118 Jun 25–Jul 2        0.17    1.7   Bars re-closed, partial rollback
    118–165 Jul  2–Aug 18       0.09    0.9   Mask mandate — R(t) < 1 confirmed
                                              by UT TACC [B5]; Wave 1 peak Day
                                              130 then steep decline
    165–195 Aug 18–Sep 17       0.10    1.0   Post-Wave-1 trough; cautious
                                              normal  [B4, B5]
    195–215 Sep 17–Oct  8       0.12    1.2   75 % capacity rule triggers
                                              Wave 2 seed  [B1, B3]
    215–265 Oct  8–Nov 27       0.13    1.3   Wave 2 growth; behavioral
                                              fatigue + indoor transmission
                                              [B3, M2]
    265–283 Nov 27–Dec 14       0.15    1.5   Thanksgiving gatherings; holiday
                                              amplification  [B3, B5]
    283–300 Dec 14–Jan  1       0.14    1.4   Vaccines begin (front-line);
                                              Wave 2 approaching peak  [B5]
    """
    # ── Pre-intervention ────────────────────────────────────────────────────
    # R₀ ≈ 3.0; consistent with CDC/Austin SEIR R₀ = 2.2–3.5 [B2] and
    # Texas mean R₀ = 2.65 [B6].  NBER WP28629 [B3] places early North
    # American R(t) at 2.8–3.4.
    if t < 5:
        return 0.30    # R₀ = 3.0 — uncontrolled; no awareness yet

    # Voluntary distancing + school closures (Mar 11–31)
    # Google Mobility [M1]: retail −15 % to −35 %, transit −25 %.
    # R(t) declining toward 2.0 per UT TACC [B5].
    elif t < 25:
        return 0.26    # R₀ = 2.6 — early voluntary NPI effect

    # Stay-at-home order (Mar 31 – May 1)
    # Texas issued a stay-at-home order but NOT a full lockdown; all
    # essential businesses remained open.  Google Mobility [M1]: retail
    # −45 % to −55 %, workplace −35 %.  Yu et al. [B1] estimate R(t)
    # fell to ≈ 1.0–1.3 in April.  UT TACC [B5] shows R(t) ≈ 1.1.
    elif t < 56:
        return 0.12    # R₀ = 1.2 — partial stay-at-home (not full lockdown)

    # Phase 1–2 reopening (May 1 – Jun 3)
    # Restaurants 25 %→50 %, retail recovering.  Yu et al. [B1] show
    # contact rate rebounded; R(t) rose back to ≈ 1.4–1.6 by late May.
    elif t < 89:
        return 0.16    # R₀ = 1.6 — phased reopening rebound

    # Phase 3 reopening (Jun 3 – Jun 25)
    # Restaurants at 75 %, pools and gyms open.  Google Mobility [M1]:
    # retail only −10 %.  R(t) surged to ≈ 1.7–2.0 per UT TACC [B5],
    # driving the rapid case increase toward Wave 1 peak.
    elif t < 111:
        return 0.20    # R₀ = 2.0 — aggressive phase-3 effect

    # Bars re-closed, restaurant rollback (Jun 25 – Jul 2)
    # Partial response; compliance lag means R(t) remains ≈ 1.5–1.8
    # for roughly one week before mask mandate is issued.
    elif t < 118:
        return 0.17    # R₀ = 1.7 — rollback lag, still above threshold

    # Mask mandate (Jul 2 – Aug 18) ← MOST EFFECTIVE SINGLE NPI
    # Abbott mandated masks in all counties with ≥ 20 active cases.
    # UT TACC [B5] confirmed R(t) dropped below 1.0 within 3 weeks.
    # Yu et al. [B1] and IHME [M2] both document masks as the dominant
    # β-reducing intervention in Texas summer 2020.
    # Wave 1 peaks Day ≈ 130 (Jul 14) then enters sustained decline.
    elif t < 165:
        return 0.09    # R₀ = 0.9 — mask mandate drives R(t) < 1

    # Post-Wave-1 trough (Aug 18 – Sep 17)
    # Cases declining; UT TACC R(t) ≈ 0.85–1.0.  Google Mobility [M1]:
    # retail recovering to −12 %; behaviour cautiously near-normal.
    elif t < 195:
        return 0.10    # R₀ = 1.0 — epidemic trough / borderline

    # Capacity raised to 75 % (Sep 17 – Oct 8) — Wave 2 trigger
    # Abbott raised capacity limits; R(t) begins rising per UT TACC [B5].
    # NBER [B3] identifies this relaxation as the North American Wave 2
    # inflection point.
    elif t < 215:
        return 0.12    # R₀ = 1.2 — reopening seeds Wave 2

    # Wave 2 exponential growth (Oct 8 – Nov 27)
    # Behavioral fatigue documented in [B3, M2]; indoor fall transmission.
    # R(t) ≈ 1.2–1.5 per UT TACC [B5] and covidestim.org.
    # β is LOWER than Wave 1 baseline but wave is larger due to:
    #   1. 256 cities now seeded (vs Houston only at Day 0)
    #   2. Higher cumulative infected pool accelerates absolute growth
    elif t < 265:
        return 0.13    # R₀ = 1.3 — Wave 2 growth (fatigue-driven)

    # Thanksgiving holiday amplification (Nov 27 – Dec 14)
    # Household gatherings; UT TACC [B5] shows R(t) spike ≈ 1.4–1.5.
    # IHME scenarios [M2] explicitly modelled Thanksgiving as a β pulse.
    elif t < 283:
        return 0.15    # R₀ = 1.5 — holiday gathering amplification

    # Dec 14 – Jan 1, 2021 (vaccines begin, Wave 2 approaching peak)
    # First doses administered to front-line workers Dec 14; negligible
    # population immunity effect at this stage.  No new TX restrictions.
    # R(t) ≈ 1.3–1.4 per UT TACC.  Wave 2 peaks Day 312 (Jan 12, 2021)
    # — OUTSIDE the 300-day window.  Extend T_MAX to 320 to capture it.
    else:
        return 0.14    # R₀ = 1.4 — Wave 2 peak approach; vaccines negligible


# =============================================================================
# mobility_scale_t(t) — Inter-City Travel Scalar
# =============================================================================

def mobility_scale_t(t):
    """
    Time-varying scalar applied to the inter-city mobility matrix θ.

    Anchored to Day 0 = March 6, 2020.

    Calibration method
    ------------------
    Google Community Mobility Reports [M1] provide daily % change from
    Jan–Feb 2020 baseline for Texas at state level.  We average the
    transit-station and retail-recreation series (most predictive of
    inter-city travel) and apply:

        mobility_scale(t) = 0.030 × (1 + Δ/100)

    where 0.030 is the pre-pandemic baseline (3 %/day inter-city movement,
    per Fix [6] in the original codebase) and Δ is the mobility % change.

    This function governs INTER-CITY commuting/travel only.  WITHIN-CITY
    contact-rate changes are fully captured by β(t).

    Behavioral-fatigue asymmetry (key calibration insight)
    -------------------------------------------------------
    Wave 1 (Apr 2020): Google Mobility shows TX retail −50 %, transit −55 %
    → mobility_scale drops to ≈ 0.015.

    Wave 2 (Oct–Dec 2020): Google Mobility shows TX retail only −15 %,
    transit −20 % → mobility_scale ≈ 0.025.

    Wave 2 mobility reduction was roughly HALF of Wave 1 [B3, M2].  The
    original code used symmetric lockdown depths for both waves, which
    over-suppressed inter-city seeding in Wave 2 and produced an
    unrealistically isolated second wave.

    Regime summary
    --------------
    Days    Date range          scale   Google Mobility Δ  Notes
    ─────── ─────────────────── ─────── ────────────────── ────────────────
      0– 5  Mar  6–11 2020      0.030   ≈  0 %             Pre-pandemic
      5–25  Mar 11–31           0.023   ≈ −22 %            School closures,
                                                           voluntary retreat
     25–56  Mar 31–May 1        0.015   ≈ −50 %            Stay-at-home peak
                                                           [M1]: retail −55 %,
                                                           transit −55 %
     56–89  May  1–Jun 3        0.021   ≈ −30 %            Phase 1–2 recovery
     89–111 Jun  3–25           0.026   ≈ −13 %            Phase 3 near-normal
    111–165 Jun 25–Aug 18       0.023   ≈ −23 %            Voluntary retreat
                                                           during Wave 1 peak
    165–195 Aug 18–Sep 17       0.026   ≈ −13 %            Post-Wave-1 trough
    195–215 Sep 17–Oct  8       0.028   ≈  −7 %            75 % capacity rule;
                                                           mobility near-normal
    215–265 Oct  8–Nov 27       0.025   ≈ −17 %            Wave 2 growth;
                                                           MUCH less reduction
                                                           than Wave 1 [B3, M2]
    265–283 Nov 27–Dec 14       0.022   ≈ −27 %            Thanksgiving travel
                                                           then voluntary retreat
    283–300 Dec 14–Jan  1       0.025   ≈ −17 %            Post-Thanksgiving
                                                           partial recovery
    """
    # Pre-pandemic baseline (Day 0, Mar 6)
    # Normal Texas inter-city commuting ≈ 3 %/day [Fix 6 baseline].
    # Google Mobility: 0 % change from Jan–Feb 2020 baseline.
    if t < 5:
        return 0.030   # Baseline: 3 %/day inter-city movement

    # Voluntary distancing + school closures (Mar 11–31)
    # Google Mobility [M1]: TX retail −15 % to −35 %, transit −25 %.
    # Rapid public behavioural response even before formal order.
    elif t < 25:
        return 0.023   # −23 % from baseline

    # Stay-at-home peak (Mar 31 – May 1)
    # Google Mobility [M1]: TX retail −55 %, transit −55 %,
    # workplace −35 %.  Composite inter-city travel ≈ −50 %.
    # Texas NEVER issued a full lockdown; Apple driving data
    # shows TX driving fell only −35 % vs −60 % in locked-down states.
    elif t < 56:
        return 0.015   # −50 % from baseline (partial stay-at-home effect)

    # Phase 1–2 reopening (May 1 – Jun 3)
    # Google Mobility [M1]: retail recovering to −25 to −35 %;
    # transit −30 %.  Driving returning toward normal.
    elif t < 89:
        return 0.021   # −30 % from baseline; recovery underway

    # Phase 3 near-normal (Jun 3 – Jun 25)
    # Google Mobility [M1]: retail −10 %, transit −20 %.
    # Driving near-baseline.  Near-full inter-city movement restored.
    elif t < 111:
        return 0.026   # −13 % from baseline

    # Voluntary retreat during Wave 1 peak (Jun 25 – Aug 18)
    # Bars closed; mask mandate.  Google Mobility [M1]: retail −20 %,
    # transit −25 %.  Behavioral response WITHOUT formal lockdown.
    elif t < 165:
        return 0.023   # −23 % from baseline; voluntary wave-1 response

    # Post-Wave-1 trough (Aug 18 – Sep 17)
    # Cases declining; mobility recovering.  Google Mobility [M1]:
    # retail −12 %, transit −20 %.  Back to cautious-normal.
    elif t < 195:
        return 0.026   # −13 % from baseline

    # 75 % capacity raised (Sep 17 – Oct 8)
    # Abbott raises business limits; public confidence returning.
    # Google Mobility [M1]: retail approaching −7 %.
    elif t < 215:
        return 0.028   # −7 % from baseline; near-normal mobility

    # Wave 2 growth phase (Oct 8 – Nov 27) — BEHAVIORAL FATIGUE
    # KEY CALIBRATION: Google Mobility [M1] shows TX retail only −15 %
    # during Wave 2 vs −55 % during Wave 1.  IHME [M2] and NBER [B3]
    # both document fatigue-driven failure to reduce mobility in Wave 2.
    # This is the dominant reason Wave 2 is larger despite lower β.
    elif t < 265:
        return 0.025   # −17 % from baseline (vs −50 % in Wave 1)

    # Thanksgiving travel then retreat (Nov 27 – Dec 14)
    # Holiday travel initially INCREASES inter-city movement, followed
    # by post-holiday voluntary retreat.  Net effect: moderate reduction.
    elif t < 283:
        return 0.022   # −27 % from baseline; post-Thanksgiving dip

    # Dec 14 – Jan 1, 2021 (vaccines begin, cautious holiday)
    # Google Mobility [M1]: retail −17 %.  Slight uptick from Dec travel
    # then settling.  No new TX mobility restrictions issued.
    else:
        return 0.025   # −17 % from baseline; cautious December


# =============================================================================
# Stochastic wrapper (unchanged signature — kept for ensemble callers)
# =============================================================================

def beta_t_stochastic(t, sigma_noise=0.05):
    """
    Stochastic wrapper around beta_t().

    Intended for ensemble / multi-realization callers ONLY.
    Do NOT call inside the ODE — draw noise outside the integrator and
    pass it in via a closure to avoid seed-collision artefacts at identical
    floating-point t values (see original module docstring for details).
    """
    rng_local = np.random.default_rng(seed=int(t * 1000) % (2 ** 31))
    return beta_t(t) * rng_local.lognormal(0.0, sigma_noise)


# =============================================================================
# Calibration verification table (run as __main__ to audit all regimes)
# =============================================================================

def _print_calibration_table():
    """Print a summary of all regimes to verify calibration at a glance."""
    checkpoints = [
        (0,   "Mar  6: first TX case"),
        (5,   "Mar 11: schools close"),
        (25,  "Mar 31: stay-at-home order"),
        (56,  "May  1: Phase 1 reopen"),
        (89,  "Jun  3: Phase 3 reopen"),
        (111, "Jun 25: bars re-closed"),
        (118, "Jul  2: mask mandate"),
        (130, "Jul 14: Wave 1 PEAK (~14k/day)"),
        (165, "Aug 18: Wave 1 resolved"),
        (185, "Sep  7: trough (~3.2k/day)"),
        (195, "Sep 17: 75% capacity rule"),
        (215, "Oct  8: Wave 2 growth begins"),
        (265, "Nov 27: Thanksgiving"),
        (283, "Dec 14: vaccines begin"),
        (300, "Jan  1: end of sim (Wave 2 rising)"),
    ]

    print("\nCalibration audit — β(t) and mobility_scale_t(t)")
    print("=" * 72)
    print(f"{'Day':>5}  {'Event':<38}  {'β':>6}  {'R₀':>5}  {'mob':>6}")
    print("-" * 72)
    for day, label in checkpoints:
        b  = beta_t(day)
        r0 = b / 0.1
        m  = mobility_scale_t(day)
        print(f"{day:>5}  {label:<38}  {b:>6.3f}  {r0:>5.1f}  {m:>6.4f}")
    print("=" * 72)

    print("\nFix-6 compliance check:")
    print(f"  target_outflow_rate must be 0.01 in run_simulation() call.")
    print(f"  Verify: rescale_mobility_matrix(theta, N, target_outflow_rate=0.01)")

    print("\nT_MAX recommendation:")
    print(f"  Current T_MAX=300 (Jan 1, 2021). Wave 2 peak = Day 312 (Jan 12).")
    print(f"  Recommend: T_MAX = 320 to capture full Wave 2 arc.")


if __name__ == "__main__":
    _print_calibration_table()