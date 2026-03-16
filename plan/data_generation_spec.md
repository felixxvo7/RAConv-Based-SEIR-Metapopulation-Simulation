# Data Generation Specification: RAConv-Based SEIR Metapopulation Simulation

## 1. Simulation Workflow

The data generation pipeline consists of three sequential tasks:

1\. **Data Preprocessing**: Loading and cleaning census population data and geographic coordinates.

2\. **Mobility Network Construction**: Generating a sparse mobility matrix (flow between cities) using a Gravity Model based on population and distance.

3\. **SEIR Simulation**: Integrating the coupled differential equations over time with regime shifts (interventions).

------------------------------------------------------------------------

## 2. Mobility Matrix Calculation

The mobility network is constructed using a **doubly-constrained Gravity Model** approximation to estimate travel flows between cities.

### Gravity Formula

The raw attraction force from city $i$ to city $j$ is calculated as: $$
\text{Attraction}_{ij} = \frac{P_j}{D_{ij}^\alpha}
$$ Where: \* $P_j$ is the population of the *destination* city $j$. \* $D_{ij}$ is the distance between city $i$ and city $j$ (typically in km). \* $\alpha$ is the distance decay exponent (set to $2.0$).

### Normalization (Outflow Constraint)

To ensure realistic travel volume, the raw attraction values are normalized such that the total outflow from any city $i$ equals a specific fraction of its population (the **daily outflow rate**, e.g., 2%).

The probability of moving from $i$ to $j$ is: $$
P(j|i) = \frac{\text{Attraction}_{ij}}{\sum_{k \neq i} \text{Attraction}_{ik}}
$$

The final flow rate (people per day) is: $$
\theta_{ij} = (\text{Daily Outflow Rate} \times P_i) \times P(j|i)
$$

This results in a sparse matrix $\Theta$ where $\theta_{ij}$ represents the number of people moving from city $i$ to city $j$ per day.

------------------------------------------------------------------------

## 3. Mathematical Model

We use a **deterministic Metapopulation SEIR model**. The population of each city $i$ is divided into four compartments: \* $S_i$: Susceptible \* $E_i$: Exposed (infected but not yet infectious) \* $I_i$: Infectious \* $R_i$: Recovered

### Differential Equations (per city $i$)

$$
\begin{aligned}
\frac{dS_i}{dt} &= -\beta(t) \frac{S_i I_i}{N_i} - \sum_{j} (\theta_{ij} S_i - \theta_{ji} S_j) \\
\frac{dE_i}{dt} &= \beta(t) \frac{S_i I_i}{N_i} - \sigma E_i - \sum_{j} (\theta_{ij} E_i - \theta_{ji} E_j) \\
\frac{dI_i}{dt} &= \sigma E_i - \gamma I_i - \sum_{j} (\theta_{ij} I_i - \theta_{ji} I_j) \\
\frac{dR_i}{dt} &= \gamma I_i - \sum_{j} (\theta_{ij} R_i - \theta_{ji} R_j)
\end{aligned}
$$

**Note regarding coupling:** The implementation uses a simplified net flow calculation: `mobility_scale * (theta_T @ X - outflow_rate * X)`. \* $\theta_{ji} X_j$: Inflow from $j$ to $i$. \* $\theta_{ij} X_i$: Outflow from $i$ to $j$.

### Terms

-   $\beta(t)$: The effectively contacted transmission rate. It varies over time to simulate policy changes.
-   **Coupling (**$\sum$): Represents the net flow of individuals between cities.
-   $N_i$: Population of city $i$.

------------------------------------------------------------------------

## 3. Parameters

### Fixed Epidemiological Parameters

| Parameter | Symbol | Value | Description |
|:---|:--:|:--:|:---|
| **Incubation Rate** | $\sigma$ | $0.2 \text{ day}^{-1}$ | Average latent period of 5 days ($1/\sigma$). |
| **Recovery Rate** | $\gamma$ | $0.1 \text{ day}^{-1}$ | Average infectious period of 10 days ($1/\gamma$). |

### Mobility Parameters

-   **Gravity Model Exponent (**$\alpha$): $2.0$ (Inverse square law for distance decay).
-   **Base Mobility Rate**: Scaled such that approximately **2%** of a city's population travels daily in the baseline scenario.

------------------------------------------------------------------------

## 4. Switch Regimes (Time-Variant Interventions)

To simulate a realistic pandemic timeline (e.g., initial outbreak, lockdown, reopening), the simulation employs **regime switching**. Both the transmission rate $\beta(t)$ and mobility magnitude are piecewise constant functions of time.

### Regime Schedule

The simulation runs for **300 days**.

| Regime Name | Time Period (Days) | Transmission Rate ($\beta$) | Implied $R_0$ ($\beta/\gamma$) | Mobility Scale |
|:---|:--:|:--:|:--:|:--:|
| **1. Baseline** | Day 0 - 45 | **0.35** | 3.5 | **100% (3.0%)** |
| **2. Early Action** | Day 45 - 75 | **0.30** | 3.0 | **66% (2.0%)** |
| **3. Lockdown** | Day 75 - 105 | **0.18** | 1.8 | **20% (0.6%)** |
| **4. Reopening** | Day 105 - 135 | **0.18** | 1.8 | **66% (2.0%)** |
| **5. New Normal** | Day 135 - 150 | **0.25** | 2.5 | **66% (2.0%)** |

*Note: Mobility Scale is applied as a multiplier to the base mobility matrix* $\theta_{ij}$. A 20% scale means travel is reduced by 80%.

US mean: 58\~60 days before lockdown

Fact: Texas lockdown time is 28 days -\> 30 days

------------------------------------------------------------------------

## 5. Assume

-   Single epidemic wave

<!-- -->

-   No waning immunity

-   No reinfection

-   Deterministic dynamics

-   No random noise

------------------------------------------------------------------------

## 6. Validation Checks

The simulation performs the following detailed diagnostics to ensure physical realism and correctness:

1.  **Population Conservation**:
    -   **Per-City Drift**: Max deviation of total population ($S+E+I+R$) from census data.
    -   **Global Drift**: Total state population deviation.
    -   *Target*: $< 0.1\%$ drift.
2.  **Non-Negativity**:
    -   Verifies that all compartments ($S, E, I, R$) remain $\ge 0$ throughout the simulation.
