import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Input file
# --------------------------------------------------
file_path = Path("seir_baseline_300days_256cities.csv")

if not file_path.exists():
    raise FileNotFoundError(f"Could not find: {file_path.resolve()}")

# --------------------------------------------------
# Load data
# Expected columns:
# day, city, S, E, I, R, I_rep, new_rep
# --------------------------------------------------
df = pd.read_csv(file_path)

required_cols = {"day", "city", "S", "E", "I", "R", "I_rep", "new_rep"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# --------------------------------------------------
# Sum all cities by day
# --------------------------------------------------
statewide = (
    df.groupby("day", as_index=False)[["S", "E", "I", "R", "I_rep", "new_rep"]]
    .sum()
    .sort_values("day")
)

n_days = statewide["day"].nunique()
n_cities = df["city"].nunique()

print(f"Loaded {len(df):,} rows across {n_cities} cities and {n_days} days.")

# --------------------------------------------------
# Plot 1: Statewide SEIR totals
# --------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(statewide["day"], statewide["S"], label="S")
plt.plot(statewide["day"], statewide["E"], label="E")
plt.plot(statewide["day"], statewide["I"], label="I")
plt.plot(statewide["day"], statewide["R"], label="R")
plt.title("Statewide SEIR Time Series (All Cities Summed)")
plt.xlabel("Day")
plt.ylabel("Population")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 2: Statewide reported totals
# --------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(statewide["day"], statewide["I_rep"], label="Reported active cases (I_rep)")
plt.plot(statewide["day"], statewide["new_rep"], label="Reported new cases/day (new_rep)")
plt.title("Statewide Reported Time Series (All Cities Summed)")
plt.xlabel("Day")
plt.ylabel("Cases")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Summary stats
# --------------------------------------------------
peak_I_row = statewide.loc[statewide["I"].idxmax()]
peak_R_row = statewide.loc[statewide["R"].idxmax()]
peak_Irep_row = statewide.loc[statewide["I_rep"].idxmax()]
peak_new_row = statewide.loc[statewide["new_rep"].idxmax()]

print("\nSummary")
print("-" * 40)
print(f"Peak active infected I: day {int(peak_I_row['day'])}, value {peak_I_row['I']:,.0f}")
print(f"Peak recovered R: day {int(peak_R_row['day'])}, value {peak_R_row['R']:,.0f}")
print(f"Peak reported active I_rep: day {int(peak_Irep_row['day'])}, value {peak_Irep_row['I_rep']:,.0f}")
print(f"Peak reported new cases/day new_rep: day {int(peak_new_row['day'])}, value {peak_new_row['new_rep']:,.0f}")

# --------------------------------------------------
# Optional: save statewide daily totals
# --------------------------------------------------
statewide.to_csv("statewide_seir_summary.csv", index=False)
print("\nSaved statewide daily totals to statewide_seir_summary.csv")