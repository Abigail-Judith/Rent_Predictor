import folium
from folium.plugins import MarkerCluster
import pandas as pd

def generate_rent_map_obj(csv_path="data/rent_listings_sample.csv"):
    df = pd.read_csv(csv_path)
    center_lat = df["latitude"].median()
    center_lon = df["longitude"].median()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster()
    p25, p75 = df["rent"].quantile(0.25), df["rent"].quantile(0.75)
    for _, r in df.iterrows():
        rent = r["rent"]
        color = "blue" if rent < p25 else "red" if rent > p75 else "orange"
        popup = f"<b>Locality:</b> {r['locality']}<br><b>BHK:</b> {r['bhk']} | <b>Size:</b> {r['size_sqft']} sqft<br><b>Rent:</b> ₹{int(r['rent']):,}"
        folium.CircleMarker(location=[r["latitude"], r["longitude"]], radius=6, color=color, fill=True, fill_opacity=0.7, popup=popup).add_to(mc)
    mc.add_to(m)
    return m
