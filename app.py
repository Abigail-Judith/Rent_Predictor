from flask import Flask, render_template, request
from markupsafe import Markup
import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from difflib import get_close_matches
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__)

# -----------------------------
# Config / Paths
# -----------------------------
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_PATH = MODELS_DIR / "smartrent_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
RAW_CSV = DATA_DIR / "rent_listings_sample.csv"

# -----------------------------
# Load model + encoders + data
# -----------------------------
if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
    raise FileNotFoundError("Model or encoder missing — run train_model.py")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

le_city = encoders["city"]
le_locality = encoders["locality"]
le_furnish = encoders["furnishing"]

raw = pd.read_csv(RAW_CSV).dropna(subset=["latitude", "longitude"])

# -----------------------------
# Fallbacks
# -----------------------------
most_common_city = raw["city"].mode().iloc[0]
most_common_locality = raw["locality"].mode().iloc[0]
most_common_furn = raw["furnishing"].mode().iloc[0]


# -----------------------------
# Encoding Helpers
# -----------------------------
def encode_column_series(series, encoder):
    """Encode Series safely with fuzzy matching."""
    out = []
    classes = list(map(str, encoder.classes_))

    for v in series.astype(str).str.strip():
        vstr = v.lower()

        # 1. exact match
        match = next((c for c in classes if c.lower() == vstr), None)
        if match:
            out.append(int(encoder.transform([match])[0]))
            continue

        # 2. fuzzy
        close = get_close_matches(v, classes, n=1, cutoff=0.6)
        if close:
            out.append(int(encoder.transform([close[0]])[0]))
            continue

        # 3. fallback
        fb = series.mode().iloc[0]
        fb_match = next((c for c in classes if c.lower() == str(fb).lower()), classes[0])
        out.append(int(encoder.transform([fb_match])[0]))

    return np.array(out, dtype=int)


def safe_encode_input(value, encoder, raw_series):
    classes = list(map(str, encoder.classes_))
    v = str(value).strip().lower()

    # exact match
    match = next((c for c in classes if c.lower() == v), None)
    if match:
        return int(encoder.transform([match])[0])

    # fuzzy
    close = get_close_matches(value, classes, n=1, cutoff=0.6)
    if close:
        return int(encoder.transform([close[0]])[0])

    # fallback
    fb = raw_series.mode().iloc[0]
    fb_match = next((c for c in classes if c.lower() == str(fb).lower()), classes[0])
    return int(encoder.transform([fb_match])[0])


# -----------------------------
# Build Encoded DataFrame
# -----------------------------
enc_df = raw.copy()
enc_df["city_enc"] = encode_column_series(enc_df["city"], le_city)
enc_df["locality_enc"] = encode_column_series(enc_df["locality"], le_locality)
enc_df["furnishing_enc"] = encode_column_series(enc_df["furnishing"], le_furnish)

# Predict full dataset (for suggestions)
FEATURES = ["city", "locality", "bhk", "size_sqft", "bathrooms", "furnishing", "parking", "age_yrs"]

X_all = pd.DataFrame({
    "city": enc_df["city_enc"],
    "locality": enc_df["locality_enc"],
    "bhk": enc_df["bhk"],
    "size_sqft": enc_df["size_sqft"],
    "bathrooms": enc_df["bathrooms"],
    "furnishing": enc_df["furnishing_enc"],
    "parking": enc_df["parking"],
    "age_yrs": enc_df["age_yrs"]
})

enc_df["pred_rent"] = model.predict(X_all)
enc_df["value_score"] = (enc_df["pred_rent"] - enc_df["rent"]) / enc_df["pred_rent"]


# -----------------------------
# Geolocation + Math Helpers
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1) / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) *
         math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_latlon_for_locality(locality, city):
    li = locality.strip().lower()

    # 1. exact locality
    m = raw[raw["locality"].str.strip().str.lower() == li]
    if not m.empty:
        r = m.iloc[0]
        return float(r["latitude"]), float(r["longitude"])

    # 2. fuzzy
    all_loc = raw["locality"].str.strip().unique().tolist()
    close = get_close_matches(locality, all_loc, n=1, cutoff=0.6)
    if close:
        r = raw[raw["locality"].str.strip() == close[0]].iloc[0]
        return float(r["latitude"]), float(r["longitude"])

    # 3. city centroid
    cm = raw[raw["city"].str.lower() == city.lower()]
    if not cm.empty:
        return float(cm["latitude"].median()), float(cm["longitude"].median())

    # 4. fallback
    return float(raw["latitude"].median()), float(raw["longitude"].median())


def format_currency(x):
    return f"₹{int(x):,}"


# -----------------------------
# Home + Map Routes
# -----------------------------
@app.route("/")
def home():
    center = [raw["latitude"].median(), raw["longitude"].median()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster().add_to(m)

    p25, p75 = raw["rent"].quantile(0.25), raw["rent"].quantile(0.75)

    for _, r in raw.iterrows():
        rent = r["rent"]
        color = "blue" if rent < p25 else "red" if rent > p75 else "orange"

        popup = f"<b>{r['locality']}</b><br>{r['bhk']} BHK · {r['size_sqft']} sqft<br>Rent: {format_currency(rent)}"
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6, color=color, fill=True, fill_opacity=0.8,
            popup=popup
        ).add_to(mc)

    return render_template("index.html", map_html=Markup(m._repr_html_()))


# -----------------------------
# PREDICT + MINI-MAP ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # collect input
    city = request.form["city"].strip()
    locality = request.form["locality"].strip()
    bhk = int(request.form["bhk"])
    size = float(request.form["size"])
    bath = int(request.form["bath"])
    furnishing = request.form["furnishing"].strip()
    parking = int(request.form["parking"])
    age = int(request.form["age"])
    listed = float(request.form["listed_rent"])

    # encode input
    city_enc = safe_encode_input(city, le_city, raw["city"])
    locality_enc = safe_encode_input(locality, le_locality, raw["locality"])
    furnish_enc = safe_encode_input(furnishing, le_furnish, raw["furnishing"])

    # model predict
    X = np.array([[city_enc, locality_enc, bhk, size, bath, furnish_enc, parking, age]])
    pred = float(model.predict(X)[0])

    diff = listed - pred
    pct = (diff / pred) * 100

    if pct > 10:
        verdict = f"❌ Overpriced by {pct:.1f}%"
        color_class = "red"
    elif pct < -10:
        verdict = f"🟢 Underpriced! (~{abs(pct):.1f}% below market)"
        color_class = "green"
    else:
        verdict = "✅ Fairly Priced"
        color_class = "blue"

    insight = f"For a {bhk}BHK ~{int(size)} sqft in {locality}, fair rent is around {format_currency(pred)}."

    # suggestions
    df = enc_df.copy()

    lat_user, lon_user = find_latlon_for_locality(locality, city)
    df["dist"] = df.apply(lambda r: haversine(lat_user, lon_user, r["latitude"], r["longitude"]), axis=1)

    cand = df[df["city_enc"] == city_enc].copy()

    # budget logic
    max_budget = pred * (1.05 if listed > pred else 1.3)

    def pick(df, min_s=0.8, max_s=1.2, max_d=6):
        m = (
            (df["bhk"] == bhk)
            & df["size_sqft"].between(size * min_s, size * max_s)
            & (df["dist"] <= max_d)
            & (df["rent"] <= max_budget)
        )
        return df[m].sort_values("value_score", ascending=False).head(5)

    sug = pick(cand)

    if sug.empty:
        sug = pick(cand, 0.7, 1.3, 8)

    fallback = ""
    if sug.empty:
        sug = cand.sort_values("value_score", ascending=False).head(5)
        fallback = "⚠️ No nearby matches. Showing best deals in the city."
    else:
        fallback = f"🏙️ Best value rentals near {locality}"

    # build suggestion list + map markers
    suggestions = []
    map_markers = []

    # user pin
    map_markers.append({
        "type": "user",
        "lat": lat_user,
        "lon": lon_user,
        "color": "black",
        "popup": f"You: {locality}<br>{bhk} BHK • {int(size)} sqft<br>Rent: {format_currency(listed)}"
    })

    for _, r in sug.iterrows():
        suggestions.append({
            "locality": r["locality"],
            "bhk": int(r["bhk"]),
            "size_sqft": int(r["size_sqft"]),
            "bathrooms": int(r["bathrooms"]),
            "furnishing": r["furnishing"],
            "rent": int(r["rent"]),
            "pred_rent": int(r["pred_rent"]),
            "value_score": round(float(r["value_score"]), 2),
            "distance_km": round(float(r["dist"]), 2)
        })

        score = r["value_score"]
        color = "green" if score >= 0.12 else ("orange" if score >= 0.05 else "red")

        map_markers.append({
            "type": "suggestion",
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "color": color,
            "popup": f"<b>{r['locality']}</b><br>{r['bhk']} BHK • {int(r['size_sqft'])} sqft<br>"
                     f"Rent: {format_currency(r['rent'])}"
        })

    return render_template(
        "result.html",
        predicted=int(pred),
        listed=int(listed),
        verdict=verdict,
        color_class=color_class,
        insight=insight,
        suggestions=suggestions,
        fallback_message=fallback,
        map_markers=map_markers
    )


if __name__ == "__main__":
    app.run(debug=True)
