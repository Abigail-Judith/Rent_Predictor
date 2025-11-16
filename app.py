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
# Load model, encoders, data
# -----------------------------
if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
    raise FileNotFoundError("models/smartrent_model.pkl or models/encoders.pkl missing. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)  # dict with LabelEncoders: city, locality, furnishing
le_city = encoders["city"]
le_locality = encoders["locality"]
le_furnish = encoders["furnishing"]

# Raw dataframe with lat/lon (source of truth)
if not RAW_CSV.exists():
    raise FileNotFoundError(f"{RAW_CSV} not found.")
raw = pd.read_csv(RAW_CSV).dropna(subset=["latitude", "longitude"])

# Precompute most common for fallbacks
most_common_city = raw["city"].mode().iloc[0]
most_common_locality = raw["locality"].mode().iloc[0]
most_common_furn = raw["furnishing"].mode().iloc[0]

# Pre-encode the dataset for model predictions (safe, no encoder mutation)
def encode_column_series(series, encoder):
    """Encode a pandas Series using encoder; if unseen values appear, use fuzzy matching or fallback."""
    out = []
    classes = list(map(str, encoder.classes_))
    for v in series.astype(str).str.strip():
        v_str = v.strip()
        # 1) exact case-insensitive match to any class
        match = next((c for c in classes if c.lower() == v_str.lower()), None)
        if match is not None:
            out.append(int(encoder.transform([match])[0]))
            continue
        # 2) fuzzy match
        close = get_close_matches(v_str, classes, n=1, cutoff=0.6)
        if close:
            out.append(int(encoder.transform([close[0]])[0]))
            continue
        # 3) fallback to most common (use original series' mode if available)
        fallback = series.mode().iloc[0] if not series.mode().empty else classes[0]
        # if fallback not in encoder classes, use encoder.classes_[0]
        fb_match = next((c for c in classes if c.lower() == str(fallback).strip().lower()), None)
        if fb_match is None:
            fb_match = classes[0]
        out.append(int(encoder.transform([fb_match])[0]))
    return np.array(out, dtype=int)

# Create a copy with encoded columns for predictions + suggestions
enc_df = raw.copy()
enc_df["city_enc"] = encode_column_series(enc_df["city"], le_city)
enc_df["locality_enc"] = encode_column_series(enc_df["locality"], le_locality)
enc_df["furnishing_enc"] = encode_column_series(enc_df["furnishing"], le_furnish)

# Features used by the model (ensure same order as training)
FEATURE_COLS = ["city", "locality", "bhk", "size_sqft", "bathrooms", "furnishing", "parking", "age_yrs"]
# We'll create a numeric feature frame using encoded columns for the categorical fields
def df_features_from_enc_df(df_enc):
    X = pd.DataFrame({
        "city": df_enc["city_enc"],
        "locality": df_enc["locality_enc"],
        "bhk": df_enc["bhk"].astype(int),
        "size_sqft": df_enc["size_sqft"].astype(float),
        "bathrooms": df_enc["bathrooms"].astype(int),
        "furnishing": df_enc["furnishing_enc"],
        "parking": df_enc["parking"].astype(int),
        "age_yrs": df_enc["age_yrs"].astype(int),
    })
    return X

# Precompute model predictions for the dataset (used for suggestions & map coloring)
features_all = df_features_from_enc_df(enc_df)
enc_df["pred_rent"] = model.predict(features_all)
enc_df["value_score"] = (enc_df["pred_rent"] - enc_df["rent"]) / enc_df["pred_rent"].replace(0, np.nan)

# -----------------------------
# Helpers
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def safe_encode_input(value, encoder, raw_series):
    """Encode a single user input value using encoder with fuzzy fallback. Returns int code."""
    classes = list(map(str, encoder.classes_))
    v = str(value).strip()
    # exact case-insensitive
    match = next((c for c in classes if c.lower() == v.lower()), None)
    if match:
        return int(encoder.transform([match])[0])
    # fuzzy
    close = get_close_matches(v, classes, n=1, cutoff=0.6)
    if close:
        return int(encoder.transform([close[0]])[0])
    # fallback to raw data mode for that column
    fallback = raw_series.mode().iloc[0] if not raw_series.mode().empty else classes[0]
    fb_match = next((c for c in classes if c.lower() == str(fallback).strip().lower()), classes[0])
    return int(encoder.transform([fb_match])[0])

def find_latlon_for_locality(locality_input, city_input=None):
    """Find best lat/lon for a user locality string (case-insensitive exact, then city centroid, then dataset first row)."""
    li = str(locality_input).strip().lower()
    # exact case-insensitive match
    mask = raw["locality"].astype(str).str.strip().str.lower() == li
    if not raw[mask].empty:
        r = raw[mask].iloc[0]
        return float(r["latitude"]), float(r["longitude"])
    # fuzzy match across all localities
    candidates = raw["locality"].astype(str).str.strip().unique().tolist()
    close = get_close_matches(locality_input, candidates, n=1, cutoff=0.6)
    if close:
        r = raw[raw["locality"].astype(str).str.strip() == close[0]].iloc[0]
        return float(r["latitude"]), float(r["longitude"])
    # try city centroid
    if city_input is not None:
        cmask = raw["city"].astype(str).str.strip().str.lower() == str(city_input).strip().lower()
        if not raw[cmask].empty:
            # return median lat/lon for city
            lat = float(raw[cmask]["latitude"].median())
            lon = float(raw[cmask]["longitude"].median())
            return lat, lon
    # fallback to dataset mean coordinates
    return float(raw["latitude"].median()), float(raw["longitude"].median())

def format_currency(x):
    try:
        return f"₹{int(x):,}"
    except Exception:
        return x

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    # generate a compact folium map and pass HTML fragment into template (no static file)
    m = folium.Map(location=[raw["latitude"].median(), raw["longitude"].median()], zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster()
    # color by rent percentiles
    p25, p75 = raw["rent"].quantile(0.25), raw["rent"].quantile(0.75)
    for _, r in raw.iterrows():
        rent = r["rent"]
        if rent < p25:
            color = "blue"
        elif rent > p75:
            color = "red"
        else:
            color = "orange"
        popup = f"<b>Locality:</b> {r['locality']}<br><b>BHK:</b> {r['bhk']} | <b>Size:</b> {r['size_sqft']} sqft<br><b>Rent:</b> {format_currency(rent)}"
        folium.CircleMarker(location=[r["latitude"], r["longitude"]], radius=6, color=color, fill=True, fill_opacity=0.7, popup=popup).add_to(mc)
    mc.add_to(m)
    map_html = m._repr_html_()
    return render_template("index.html", map_html=Markup(map_html))

@app.route("/map")
def rent_map():
    # same as above but renders full page for map route
    m = folium.Map(location=[raw["latitude"].median(), raw["longitude"].median()], zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster()
    p25, p75 = raw["rent"].quantile(0.25), raw["rent"].quantile(0.75)
    for _, r in raw.iterrows():
        rent = r["rent"]
        if rent < p25:
            color = "blue"
        elif rent > p75:
            color = "red"
        else:
            color = "orange"
        popup = f"<b>Locality:</b> {r['locality']}<br><b>BHK:</b> {r['bhk']} | <b>Size:</b> {r['size_sqft']} sqft<br><b>Rent:</b> {format_currency(rent)}"
        folium.CircleMarker(location=[r["latitude"], r["longitude"]], radius=6, color=color, fill=True, fill_opacity=0.7, popup=popup).add_to(mc)
    mc.add_to(m)
    map_html = m._repr_html_()
    return render_template("map.html", map_html=Markup(map_html))

@app.route("/predict", methods=["POST"])
def predict():
    # 1. collect & validate input
    try:
        city = request.form["city"].strip()
        locality = request.form["locality"].strip()
        bhk = int(request.form["bhk"])
        size = float(request.form["size"])
        bath = int(request.form["bath"])
        furnishing = request.form["furnishing"].strip()
        parking = int(request.form["parking"])
        age = int(request.form["age"])
        listed = float(request.form["listed_rent"])
    except Exception as e:
        return render_template("result.html", error="Invalid input. Please check your form values."), 400

    # 2. encode safely using fuzzy fallback (no encoder mutation)
    city_enc = safe_encode_input(city, le_city, raw["city"])
    locality_enc = safe_encode_input(locality, le_locality, raw["locality"])
    furnish_enc = safe_encode_input(furnishing, le_furnish, raw["furnishing"])

    # 3. predict
    X = np.array([[city_enc, locality_enc, bhk, size, bath, furnish_enc, parking, age]])
    pred = float(model.predict(X)[0])

    diff = listed - pred
    pct = (diff / pred) * 100 if pred else 0
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

    # 4. prepare suggestions dataset (use enc_df which already has model preds)
    df = enc_df.copy()

    # compute distances using lat/lon from raw dataset (find user's lat/lon)
    lat_user, lon_user = find_latlon_for_locality(locality, city)
    df["distance_km"] = df.apply(lambda r: haversine(lat_user, lon_user, float(r["latitude"]), float(r["longitude"])), axis=1)

    # candidates: same city (use encoded city match)
    candidates = df[df["city_enc"] == city_enc].copy()

    # value score already computed; recompute if needed for safety
    candidates["value_score"] = (candidates["pred_rent"] - candidates["rent"]) / candidates["pred_rent"].replace(0, np.nan)

    # budget logic
    if listed > pred:
        max_budget = pred * 1.05
    else:
        max_budget = pred * 1.3

    def top5_from_df(dff):
        cols = ["locality", "bhk", "size_sqft", "bathrooms", "furnishing", "rent", "pred_rent", "value_score", "distance_km"]
        out = dff.sort_values("value_score", ascending=False)[cols].head(5).copy()
        if not out.empty:
            out["pred_rent"] = out["pred_rent"].round(0).astype(int)
            out["value_score"] = out["value_score"].round(2)
            out["distance_km"] = out["distance_km"].round(2)
        return out

    # Step 1: nearby similar size ±20%, same bhk, distance <= 6, rent <= max_budget
    mask = (
        (candidates["bhk"] == bhk)
        & (candidates["size_sqft"].between(size * 0.8, size * 1.2))
        & (candidates["distance_km"] <= 6)
        & (candidates["rent"] <= max_budget)
    )
    sug = top5_from_df(candidates[mask])

    # Step 2: expand to 8 km and widen size tolerance
    if sug.empty:
        mask = (
            (candidates["bhk"] == bhk)
            & (candidates["size_sqft"].between(size * 0.7, size * 1.3))
            & (candidates["distance_km"] <= 8)
            & (candidates["rent"] <= max_budget)
        )
        sug = top5_from_df(candidates[mask])

    # Step 3: city-wide best deals (no budget)
    fallback_message = ""
    if sug.empty:
        sug = top5_from_df(candidates)
        fallback_message = "⚠️ No exact in-budget matches nearby. Showing best value rentals in the city."

    if not sug.empty and not fallback_message:
        fallback_message = f"🏙️ Showing best value rentals near {locality}."

    # format suggestions for template
    suggestions = []
    if not sug.empty:
        for _, r in sug.iterrows():
            suggestions.append({
                "locality": r["locality"],
                "bhk": int(r["bhk"]),
                "size_sqft": int(r["size_sqft"]),
                "bathrooms": int(r["bathrooms"]),
                "furnishing": r["furnishing"],
                "rent": int(r["rent"]),
                "pred_rent": int(r["pred_rent"]),
                "value_score": float(r["value_score"]),
                "distance_km": float(r["distance_km"]),
            })

    # return result
    return render_template(
        "result.html",
        predicted=int(round(pred)),
        listed=int(round(listed)),
        verdict=verdict,
        color_class=color_class,
        insight=insight,
        suggestions=suggestions,
        fallback_message=fallback_message,
        map_html=None,  # keep map out of result page to avoid heavy rendering here
    )

if __name__ == "__main__":
    app.run(debug=True)
