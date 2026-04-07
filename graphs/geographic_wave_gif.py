"""
Geographic Wave Propagation GIF — RAConv P14 SEIR Metapopulation
================================================================
Visualises the two-wave epidemic spread across 256 Texas cities
over 300 days, with Houston as the Wave-1 origin hotspot.

Usage
-----
    python plot/geographic_wave_gif.py

Outputs
-------
    plot/geographic_wave_spread.gif
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEIR_CSV = os.path.join(ROOT, "Models", "Preprocessing",
                        "seir_baseline_300days_256cities.csv")
GEO_CSV  = os.path.join(ROOT, "Models", "Preprocessing", "tx_pd.csv")
OUT_GIF  = os.path.join(ROOT, "plot", "geographic_wave_spread.gif")

WAVE_PHASES = [
    (0,   99,  "Wave 1 — Exponential Growth",  "#e74c3c"),
    (100, 169, "NPI Trough — Decline",          "#3498db"),
    (170, 254, "Wave 2 — Reopening Surge",      "#e67e22"),
    (255, 284, "NPI-2 — Second Decline",        "#3498db"),
    (285, 300, "Reopening Tail",                "#7f8c8d"),
]

LABEL_CITIES = [
    "Houston", "Dallas", "San Antonio", "Austin",
    "Fort Worth", "El Paso", "McAllen", "Corpus Christi",
]


def phase_for_day(day: int):
    for start, end, label, colour in WAVE_PHASES:
        if start <= day <= end:
            return label, colour
    return "", "#7f8c8d"


def load_data():
    seir = pd.read_csv(SEIR_CSV)
    geo  = pd.read_csv(GEO_CSV)
    geo_map = geo.set_index("city")[["lat", "lng", "population"]].to_dict("index")
    return seir, geo, geo_map


def build_frame(day_df, geo_map, day, vmax_global, fig, ax):
    ax.clear()

    lngs, lats, intensities, pops = [], [], [], []
    for _, row in day_df.iterrows():
        city = row["city"]
        if city not in geo_map:
            continue
        g = geo_map[city]
        lngs.append(g["lng"])
        lats.append(g["lat"])
        intensities.append(row["I"])
        pops.append(g["population"])

    lngs = np.array(lngs)
    lats = np.array(lats)
    intensities = np.array(intensities)
    pops = np.array(pops)

    norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax_global)
    normed = norm(intensities)

    hot_cmap = plt.cm.get_cmap("hot_r")
    colors = hot_cmap(normed)
    colors[:, 3] = np.clip(0.20 + 0.80 * normed, 0.20, 1.0)

    base_size = 6 + 55 * (pops / pops.max()) ** 0.5
    size_boost = 1.0 + 2.5 * normed
    sizes = base_size * size_boost

    ax.set_facecolor("#0d1117")
    fig.set_facecolor("#0d1117")

    ax.scatter(lngs, lats, s=sizes, c=colors, edgecolors="none",
               linewidths=0, zorder=2)

    glow_mask = normed > 0.15
    if glow_mask.any():
        ax.scatter(lngs[glow_mask], lats[glow_mask],
                   s=sizes[glow_mask] * 2.8,
                   c=colors[glow_mask] * np.array([1, 1, 1, 0.18]),
                   edgecolors="none", zorder=1)

    for lbl_city in LABEL_CITIES:
        if lbl_city not in geo_map:
            continue
        g = geo_map[lbl_city]
        city_I = day_df.loc[day_df["city"] == lbl_city, "I"].values
        if len(city_I) == 0:
            continue
        val = city_I[0]
        brightness = norm(val)
        txt_color = "#ffeedd" if brightness > 0.3 else "#8899aa"
        ax.annotate(
            lbl_city,
            (g["lng"], g["lat"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=6.5,
            color=txt_color,
            fontweight="bold" if brightness > 0.3 else "normal",
            zorder=5,
        )

    houston = geo_map.get("Houston")
    if houston:
        h_I = day_df.loc[day_df["city"] == "Houston", "I"].values
        if len(h_I) and h_I[0] > vmax_global * 0.05:
            ax.scatter([houston["lng"]], [houston["lat"]],
                       s=350, facecolors="none",
                       edgecolors="#ff6b6b", linewidths=1.5,
                       linestyle="--", zorder=4, alpha=0.7)

    phase_label, phase_color = phase_for_day(day)

    ax.set_xlim(-107.5, -93.0)
    ax.set_ylim(25.5, 37.0)
    ax.set_aspect(1.15)

    ax.tick_params(colors="#556677", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a3a4a")
        spine.set_linewidth(0.8)
    ax.set_xlabel("Longitude", color="#8899aa", fontsize=8, labelpad=4)
    ax.set_ylabel("Latitude",  color="#8899aa", fontsize=8, labelpad=4)

    ax.text(0.02, 0.97, f"Day {day:>3d} / 300",
            transform=ax.transAxes, fontsize=16, fontweight="bold",
            color="white", va="top", ha="left", zorder=10,
            fontfamily="monospace")

    ax.text(0.02, 0.91, phase_label,
            transform=ax.transAxes, fontsize=10,
            color=phase_color, va="top", ha="left", zorder=10,
            fontweight="bold")

    state_I = day_df["I"].sum()
    ax.text(0.98, 0.97,
            f"Statewide Active I: {state_I:,.0f}",
            transform=ax.transAxes, fontsize=9, color="#ccddee",
            va="top", ha="right", zorder=10, fontfamily="monospace")

    peak_city_idx = day_df["I"].idxmax()
    peak_city = day_df.loc[peak_city_idx, "city"]
    peak_I = day_df.loc[peak_city_idx, "I"]
    ax.text(0.98, 0.92,
            f"Hotspot: {peak_city} ({peak_I:,.0f})",
            transform=ax.transAxes, fontsize=8, color="#ff9966",
            va="top", ha="right", zorder=10)

    n_infected = (day_df["I"] > 1.0).sum()
    ax.text(0.98, 0.87,
            f"Cities w/ I>1: {n_infected}/256",
            transform=ax.transAxes, fontsize=8, color="#88aacc",
            va="top", ha="right", zorder=10)

    ax.text(0.50, 0.015,
            "RAConv-P14 · SEIR Metapopulation · 256 Texas Cities",
            transform=ax.transAxes, fontsize=7, color="#556677",
            ha="center", va="bottom", zorder=10)


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                pad_inches=0.15, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def main():
    print("Loading data...")
    seir, geo, geo_map = load_data()

    days = sorted(seir["day"].unique())
    n_days = len(days)
    print(f"  {n_days} days,  {seir['city'].nunique()} cities")

    vmax_global = seir["I"].quantile(0.997)
    print(f"  Colour ceiling (99.7th percentile): {vmax_global:,.0f}")

    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    frames = []
    frame_days = list(range(0, 301, 1))

    for i, day in enumerate(frame_days):
        if day not in seir["day"].values:
            continue
        day_df = seir[seir["day"] == day].copy()
        build_frame(day_df, geo_map, day, vmax_global, fig, ax)
        img = fig_to_pil(fig)
        frames.append(img)
        if i % 30 == 0:
            print(f"  Frame {i+1}/{len(frame_days)}  (day {day})")

    plt.close(fig)

    print(f"\nAssembling GIF ({len(frames)} frames)...")
    durations = []
    for day in frame_days:
        if day in [0, 1, 2]:
            durations.append(200)
        elif 95 <= day <= 105:
            durations.append(120)
        elif 165 <= day <= 175:
            durations.append(120)
        elif 250 <= day <= 260:
            durations.append(120)
        elif day >= 295:
            durations.append(300)
        else:
            durations.append(70)
    durations = durations[:len(frames)]

    os.makedirs(os.path.dirname(OUT_GIF), exist_ok=True)

    frames_rgb = [f.convert("RGB") for f in frames]
    frames_rgb[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(OUT_GIF) / (1024 * 1024)
    print(f"\nSaved → {OUT_GIF}")
    print(f"  Size : {size_mb:.1f} MB")
    print(f"  Frames: {len(frames)}")
    print(f"  Total duration: {sum(durations)/1000:.1f}s")


if __name__ == "__main__":
    main()
