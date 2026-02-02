import pandas as pd
import matplotlib.pyplot as plt
import os

def create_population_map():
    # Load data
    try:
        df = pd.read_csv('src_data/tx_pd.csv')
    except FileNotFoundError:
        print("Error: src_data/tx_pd.csv not found.")
        return

    print(f"Loaded {len(df)} cities.")

    # 1. Static Scatter Plot (Fallback/Quick view)
    print("\nGenerating static scatter plot...")
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        df['lng'], 
        df['lat'], 
        c=df['population'], 
        s=df['population'] / 1000, 
        cmap='hot_r',
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Population')
    plt.title('Texas Cities Population Distribution')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Try to add simple background if geopandas is missing
    plt.xlim(-107, -93)
    plt.ylim(25.5, 37)
    
    output_file = 'population_map.png'
    plt.savefig(output_file, dpi=150)
    print(f"✓ Saved static map to {output_file}")

    # 2. Interactive Map with Folium (Best for "on the map" view)
    print("\nGenerating interactive map with Folium...")
    try:
        import folium
        from folium.plugins import HeatMap
        
        # Center on mean coordinates
        center_lat = df['lat'].mean()
        center_lng = df['lng'].mean()
        
        # Create map (OpenStreetMap tiles by default)
        m = folium.Map(location=[center_lat, center_lng], zoom_start=6, tiles='OpenStreetMap')
        
        # Add Heatmap layer
        # Data format: [lat, lng, weight]
        # Normalize weights for better visualization
        max_pop = df['population'].max()
        heat_data = [[row['lat'], row['lng'], row['population']/max_pop] for _, row in df.iterrows()]
        
        HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)
        
        # Add Circles for cities
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=max(3, row['population'] / 50000), # Dynamic radius
                popup=f"<b>{row['city']}</b><br>Pop: {row['population']:,}",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6
            ).add_to(m)

        html_file = 'population_map.html'
        m.save(html_file)
        print(f"✓ Saved interactive map to {html_file}")
        print(f"  -> Open {os.path.abspath(html_file)} in your browser to see the cities on the map!")
        
    except ImportError:
        print("⚠ 'folium' library not found.")
        print("  To see the map background, please run: pip install folium")

if __name__ == "__main__":
    create_population_map()
