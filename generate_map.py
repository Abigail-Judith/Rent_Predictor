import folium
import pandas as pd

def generate_rent_map(csv_path="data/rent_listings_sample.csv", output_path="static/rent_map.html"):
    """Generates an interactive rent map using Folium."""

    df = pd.read_csv(csv_path)

    # Ensure dataset has lat/lon info
    if not {"latitude", "longitude", "locality", "rent"}.issubset(df.columns):
        raise ValueError("Dataset must include latitude, longitude, locality, and rent columns.")

    # Center on average Bengaluru coordinates
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    rent_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    # Add listings
    for _, row in df.iterrows():
        popup_html = f"""
        <b>🏠 Locality:</b> {row['locality']}<br>
        <b>BHK:</b> {row['bhk']} | <b>Size:</b> {row['size_sqft']} sqft<br>
        <b>Rent:</b> ₹{row['rent']}
        """
        color = "blue" if row["rent"] < 25000 else "orange" if row["rent"] < 45000 else "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=popup_html,
            color=color,
            fill=True,
            fill_opacity=0.7,
        ).add_to(rent_map)

    rent_map.save(output_path)
    print(f"✅ Rent map updated → {output_path}")
