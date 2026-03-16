import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

def load_data():
    """Load city data and simulation results from likely paths."""
    
    # helper to check paths
    def get_path(filename, search_paths):
        for path in search_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path
        return None

    # Current dir and parent dir
    base_dirs = ['.', '..', 'data_generation', 'src_data']
    
    # 1. Load City Data (tx_pd.csv)
    # usually in src_data/tx_pd.csv
    city_file = get_path('tx_pd.csv', [os.path.join(d, 'src_data') for d in ['.', '..']] + ['src_data'])
    if not city_file:
         # Fallback try direct search
         city_file = get_path('tx_pd.csv', ['.', '..', 'src_data'])
         
    if not city_file:
        print("Error: Could not find tx_pd.csv")
        return None, None

    print(f"Loading city data from {city_file}...")
    df_cities = pd.read_csv(city_file)
    
    # 2. Load Simulation Results (.npy)
    # usually in data_generation/seir_baseline... or just in current dir
    npy_filename = 'seir_baseline_300days_256cities.npy'
    npy_file = get_path(npy_filename, ['.', 'data_generation', '..'])
    
    if not npy_file:
        print(f"Error: Could not find {npy_filename}")
        return None, None
        
    print(f"Loading simulation results from {npy_file}...")
    results = np.load(npy_file) # Shape: (days, cities, 4)
    
    return df_cities, results

def create_infection_gif():
    # Load data
    df_cities, results = load_data()
    if df_cities is None or results is None:
        return

    # Extract Infected compartment (Index 2)
    # results shape: (T, N, 4) -> S, E, I, R
    Infected = results[:, :, 2]
    
    n_days = Infected.shape[0]
    n_cities = len(df_cities)
    
    print(f"Simulation data: {n_days} days, {n_cities} cities")
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Map bounds (Texas roughly)
    ax.set_xlim(-107, -93)
    ax.set_ylim(25.5, 37)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Background: All cities (static, gray circles)
    # Size proportional to population
    pop_sizes = df_cities['population'] / 5000  # Scaling factor
    ax.scatter(
        df_cities['lng'], 
        df_cities['lat'], 
        s=pop_sizes, 
        c='lightgray', 
        alpha=0.5, 
        edgecolors='gray', 
        linewidth=0.5,
        label='Cities'
    )
    
    # Foreground: Infected cases (dynamic, red bubbles)
    # Initial state (Day 0)
    scat = ax.scatter(
        df_cities['lng'], 
        df_cities['lat'], 
        s=0, 
        c='red', 
        alpha=0.6, 
        edgecolors='darkred',
        linewidth=0.5
    )
    
    # Text annotation for Day
    day_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Total Infected validation
    total_infected_text = ax.text(0.02, 0.91, '', transform=ax.transAxes, fontsize=10)

    # Add legend
    # Create a dummy handle for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Population (fixed)',
               markerfacecolor='lightgray', markeredgecolor='gray', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Infected (dynamic)',
               markerfacecolor='red', markeredgecolor='darkred', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    def update(frame):
        # Current infections for all cities
        I_current = Infected[frame]
        
        # Update Bubble Sizes
        # Size proportional to sqrt(Infected) to make it visible but not overwhelming
        # Filter out 0 to avoid warnings or clutter, though scatter handles 0 fine usually
        
        # Dynamic scaling: 
        # Visualization trick: Sqrt scale usually looks better for magnitude diffs
        sizes = np.sqrt(I_current) * 5 
        
        scat.set_sizes(sizes)
        
        # Optional: Color intensity based on prevalence? 
        # For now, keep solid red, size indicates magnitude.
        
        # Update Text
        day_text.set_text(f"Day: {frame}")
        total_I = I_current.sum()
        total_infected_text.set_text(f"Total Infected: {total_I:,.0f}")
        
        return scat, day_text, total_infected_text

    print("Generating animation (this may take a minute)...")
    
    # Create Animation
    # Interval: ms between frames. 300 frames. 
    # If we want 15 seconds duration: 15*1000 / 300 = 50ms
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=range(0, n_days, 2), # Skip every other frame to speed up rendering/reduce size
        interval=50, 
        blit=True
    )
    
    # Save GIF
    output_file = 'infection_spread.gif'
    # Try saving in current dir or data_generation if accessible
    if os.path.basename(os.getcwd()) == 'RAConv-Based-SEIR-Metapopulation-Simulation':
         # if in root, verify if we should save to data_generation
         if os.path.exists('data_generation'):
             output_file = os.path.join('data_generation', output_file)

    try:
        print(f"Saving to {output_file}...")
        anim.save(output_file, writer='pillow', fps=15)
        print(f"✓ Animation saved successfully to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure 'pillow' is installed (pip install pillow)")

if __name__ == "__main__":
    create_infection_gif()
