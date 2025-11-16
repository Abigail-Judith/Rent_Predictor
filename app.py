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
# Encoding Helpers
# -----------------------------
def encode_column_series(series, encoder):
    out = []
    classes = list(map(str, encoder.classes_))

    for v in series.astype(str).str.strip():
        vstr = v.lower()

        # exact
        match = next((c for c in classes if c.lower() == vstr), None)
        if match:
            out.append(int(encoder.transform([match])[0]))
            continue

        # fuzzy
        close = get_close_matches(v, classes, n=1, cutoff=0.6)
        if close:
            out.append(int(encoder.transform([close[0]])[0]))
            continue

        # fallback
        fb = series.mode().iloc[0]
        fb_match = next((c for c in classes if c.lower() == str(fb).lower()), classes[0])
        out.append(int(encoder.transform([fb_match])[0]))

    return np.array(out, dtype=int)


def safe_encode_input(value, encoder, raw_series):
    classes = list(map(str, encoder.classes_))
    v = str(value).strip().lower()

    # exact
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

# Predict full dataset (fair rent)
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

enc_df["pred_rent"] = model.predict(X_all)        # fair rent
enc_df["value_score"] = (enc_df["pred_rent"] - enc_df["rent"]) / enc_df["pred_rent"]


# -----------------------------
# Utility Functions
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1)/2)**2 +
         math.cos(lat1)*math.cos(lat2)*math.sin((lon2 - lon1)/2)**2)
    return 2*R*math.asin(math.sqrt(a))


def format_currency(x):
    return f"₹{int(x):,}"


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", active_page="predict")


# -----------------------------
# SUGGESTIONS PAGE (TABLE FIRST + MAP BELOW + DIST + FAIR RENT)
# -----------------------------
@app.route("/suggestions")
def suggestions_page():

    # Calculate city center for distance reference
    center_lat = raw["latitude"].median()
    center_lon = raw["longitude"].median()

    enc_df["dist"] = enc_df.apply(
        lambda r: haversine(center_lat, center_lon, r["latitude"], r["longitude"]),
        axis=1
    )

    # Pick top 5 best deals
    best = enc_df.sort_values("value_score", ascending=False).head(5)

    suggestions = []
    for _, r in best.iterrows():
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

    # Generate map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster().add_to(m)

    for _, r in enc_df.iterrows():
        popup = (
            f"<b>{r['locality']}</b><br>"
            f"{r['bhk']} BHK • {r['size_sqft']} sqft<br>"
            f"Rent: ₹{int(r['rent']):,}<br>"
            f"Fair Rent: ₹{int(r['pred_rent']):,}"
        )

        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(mc)

    # auto-fit
    m.fit_bounds(m.get_bounds(), padding=(20, 20))

    return render_template(
        "suggestions.html",
        active_page="suggestions",
        suggestions=suggestions,
        map_html=Markup(m._repr_html_())
    )


# -----------------------------
# RESULT PAGE (same as before)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    city = request.form["city"].strip()
    locality = request.form["locality"].strip()
    bhk = int(request.form["bhk"])
    size = float(request.form["size"])
    bath = int(request.form["bath"])
    furnishing = request.form["furnishing"].strip()
    parking = int(request.form["parking"])
    age = int(request.form["age"])
    listed = float(request.form["listed_rent"])

    city_enc = safe_encode_input(city, le_city, raw["city"])
    locality_enc = safe_encode_input(locality, le_locality, raw["locality"])
    furnish_enc = safe_encode_input(furnishing, le_furnish, raw["furnishing"])

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

    return render_template(
        "result.html",
        predicted=int(pred),
        listed=int(listed),
        verdict=verdict,
        color_class=color_class,
        insight=insight,
        active_page="predict"
    )


if __name__ == "__main__":
    app.run(debug=True)
