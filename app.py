from flask import Flask, render_template, request, redirect, url_for
from markupsafe import Markup
import pandas as pd
import numpy as np
import joblib
import math
from pathlib import Path
from difflib import get_close_matches
import folium
from folium.plugins import MarkerCluster
import urllib.parse

app = Flask(__name__)

# ----------------------------- PATHS -----------------------------
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_PATH = MODELS_DIR / "smartrent_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
RAW_CSV = DATA_DIR / "rent_listings_sample.csv"

# ----------------------------- LOAD MODEL + DATA -----------------------------
if not MODEL_PATH.exists() or not ENCODERS_PATH.exists():
    raise FileNotFoundError("Model or encoder missing — run train_model.py")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
le_city = encoders["city"]
le_locality = encoders["locality"]
le_furnish = encoders["furnishing"]

raw = pd.read_csv(RAW_CSV).dropna(subset=["latitude", "longitude"])

# ----------------------------- HELPERS -----------------------------
def safe_encode_input(value, encoder, series):
    """
    Try exact case-insensitive match, then fuzzy, else fallback to mode of series.
    Returns encoder integer code.
    """
    classes = list(map(str, encoder.classes_))
    v = str(value).strip()
    if not v:
        mode = str(series.mode().iloc[0])
        return int(encoder.transform([mode])[0])

    # exact case-insensitive
    match = next((c for c in classes if c.lower() == v.lower()), None)
    if match:
        return int(encoder.transform([match])[0])

    # fuzzy
    close = get_close_matches(v, classes, n=1, cutoff=0.6)
    if close:
        return int(encoder.transform([close[0]])[0])

    # fallback to series mode
    fallback = str(series.mode().iloc[0])
    fb_match = next((c for c in classes if c.lower() == fallback.lower()), classes[0])
    return int(encoder.transform([fb_match])[0])


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def find_latlon_for_locality(locality_input, city_input=None):
    """Return latitude and longitude for a locality string (exact -> fuzzy -> city centroid -> dataset median)."""
    li = str(locality_input).strip().lower()
    if li:
        # exact match
        m = raw[raw["locality"].astype(str).str.strip().str.lower() == li]
        if not m.empty:
            r = m.iloc[0]
            return float(r["latitude"]), float(r["longitude"])
        # fuzzy match among unique localities
        all_loc = raw["locality"].astype(str).str.strip().unique().tolist()
        close = get_close_matches(locality_input, all_loc, n=1, cutoff=0.6)
        if close:
            r = raw[raw["locality"].astype(str).str.strip() == close[0]].iloc[0]
            return float(r["latitude"]), float(r["longitude"])

    # try city centroid (median lat/lon)
    if city_input:
        cm = raw[raw["city"].astype(str).str.strip().str.lower() == str(city_input).strip().lower()]
        if not cm.empty:
            return float(cm["latitude"].median()), float(cm["longitude"].median())

    # fallback to dataset median
    return float(raw["latitude"].median()), float(raw["longitude"].median())


def format_currency(x):
    try:
        return f"₹{int(x):,}"
    except Exception:
        return x

# ----------------------------- PREPARE ENCODED DF & PREDICTIONS -----------------------------
enc_df = raw.copy()

# safe transform: if encoder classes don't contain some values (shouldn't happen if trained), handle errors
def safe_transform(encoder, series):
    vals = series.astype(str).str.strip().tolist()
    out = []
    classes = list(map(str, encoder.classes_))
    for v in vals:
        m = next((c for c in classes if c.lower() == v.lower()), None)
        if m is not None:
            out.append(int(encoder.transform([m])[0]))
        else:
            # fuzzy or mode fallback
            close = get_close_matches(v, classes, n=1, cutoff=0.6)
            if close:
                out.append(int(encoder.transform([close[0]])[0]))
            else:
                mode = str(series.mode().iloc[0])
                fm = next((c for c in classes if c.lower() == mode.lower()), classes[0])
                out.append(int(encoder.transform([fm])[0]))
    return np.array(out, dtype=int)

enc_df["city_enc"] = safe_transform(le_city, enc_df["city"])
enc_df["locality_enc"] = safe_transform(le_locality, enc_df["locality"])
enc_df["furnishing_enc"] = safe_transform(le_furnish, enc_df["furnishing"])

X_all = pd.DataFrame({
    "city": enc_df["city_enc"],
    "locality": enc_df["locality_enc"],
    "bhk": enc_df["bhk"],
    "size_sqft": enc_df["size_sqft"],
    "bathrooms": enc_df["bathrooms"],
    "furnishing": enc_df["furnishing_enc"],
    "parking": enc_df["parking"],
    "age_yrs": enc_df["age_yrs"],
})

enc_df["pred_rent"] = model.predict(X_all)
enc_df["value_score"] = (enc_df["pred_rent"] - enc_df["rent"]) / enc_df["pred_rent"].replace(0, np.nan)

# ----------------------------- HOME (index) - keep existing full-city clustered map -----------------------------
@app.route("/")
def home():
    center = [raw["latitude"].median(), raw["longitude"].median()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster().add_to(m)

    p25, p75 = raw["rent"].quantile(0.25), raw["rent"].quantile(0.75)
    for _, r in raw.iterrows():
        rent = r["rent"]
        color = "blue" if rent < p25 else ("red" if rent > p75 else "orange")
        popup = (
            f"<b>{r['locality']}</b><br>"
            f"{int(r['bhk'])} BHK · {int(r['size_sqft'])} sqft<br>"
            f"Rent: {format_currency(r['rent'])}"
        )
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(mc)

    map_html = m._repr_html_()
    return render_template("index.html", map_html=Markup(map_html), active_page="predict")

# ----------------------------- PREDICT (returns result with link to suggestions) -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # collect input
    city = request.form.get("city", "").strip()
    locality = request.form.get("locality", "").strip()
    bhk = int(request.form.get("bhk", 1))
    size = float(request.form.get("size", 500))
    bath = int(request.form.get("bath", 1))
    furnishing = request.form.get("furnishing", "").strip()
    parking = int(request.form.get("parking", 0))
    age = int(request.form.get("age", 1))
    listed = float(request.form.get("listed_rent", 0))

    # safe encode
    city_enc = safe_encode_input(city, le_city, raw["city"])
    locality_enc = safe_encode_input(locality, le_locality, raw["locality"])
    furnishing_enc = safe_encode_input(furnishing, le_furnish, raw["furnishing"])

    X = np.array([[city_enc, locality_enc, bhk, size, bath, furnishing_enc, parking, age]])
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

    # Build suggestions link: pass parameters via query string
    params = {
        "city": city,
        "locality": locality,
        "bhk": bhk,
        "size": size,
        "pred": int(round(pred)),
        "listed": int(round(listed))
    }
    qs = urllib.parse.urlencode(params, doseq=True)

    return render_template(
        "result.html",
        predicted=int(round(pred)),
        listed=int(round(listed)),
        verdict=verdict,
        color_class=color_class,
        insight=insight,
        suggestions_qs=qs,
        active_page="predict"
    )

# ----------------------------- SUGGESTIONS (table first, map of suggestions below) -----------------------------
@app.route("/suggestions")
def suggestions_page():
    """
    If query params (city, locality, bhk, size, pred) are provided — filter results:
      - same city
      - same bhk
      - distance <= 6 km from the provided locality
      - size between 70% and 130% of provided size
      - top 10 by value_score
    If no params — show top 10 city-wide best deals (value_score).
    """
    # read query params
    city_q = request.args.get("city", "").strip()
    locality_q = request.args.get("locality", "").strip()
    bhk_q = request.args.get("bhk", "").strip()
    size_q = request.args.get("size", "").strip()
    pred_q = request.args.get("pred", "").strip()
    listed_q = request.args.get("listed", "").strip()

    # default: top 10 best deals overall
    suggestions_df = enc_df.copy()

    if city_q and bhk_q and size_q:
        try:
            bhk_q_i = int(bhk_q)
            size_q_f = float(size_q)
        except Exception:
            bhk_q_i = None
            size_q_f = None

        # filter same city (case-insensitive)
        suggestions_df = suggestions_df[suggestions_df["city"].astype(str).str.strip().str.lower() == city_q.lower()]

        if bhk_q_i is not None:
            suggestions_df = suggestions_df[suggestions_df["bhk"] == bhk_q_i]

        # compute lat/lon reference from user locality (fallback to city centroid)
        lat_user, lon_user = find_latlon_for_locality(locality_q, city_q)

        # compute distance for every row
        suggestions_df = suggestions_df.copy()
        suggestions_df["distance_km"] = suggestions_df.apply(
            lambda r: haversine(lat_user, lon_user, float(r["latitude"]), float(r["longitude"])),
            axis=1
        )

        # distance filter <= 6 km
        suggestions_df = suggestions_df[suggestions_df["distance_km"] <= 6.0]

        # size tolerance 70% - 130%
        if size_q_f is not None:
            min_size = size_q_f * 0.7
            max_size = size_q_f * 1.3
            suggestions_df = suggestions_df[suggestions_df["size_sqft"].between(min_size, max_size)]

        # sort by value_score desc and keep top 10
        suggestions_df = suggestions_df.sort_values("value_score", ascending=False).head(10)

    else:
        # no filters provided: top 10 by value_score across dataset
        suggestions_df = suggestions_df.sort_values("value_score", ascending=False).head(10)
        suggestions_df["distance_km"] = np.nan

    # Build suggestions list for template
    suggestions = []
    for _, r in suggestions_df.iterrows():
        suggestions.append({
            "locality": r["locality"],
            "bhk": int(r["bhk"]),
            "size_sqft": int(r["size_sqft"]),
            "bathrooms": int(r["bathrooms"]),
            "furnishing": r["furnishing"],
            "rent": int(r["rent"]),
            "pred_rent": int(round(r["pred_rent"])),
            "value_score": round(float(r["value_score"]), 3),
            "distance_km": round(float(r["distance_km"]), 2) if not pd.isna(r["distance_km"]) else None,
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"])
        })

    # Prepare map for the suggestions only (top suggestions_df)
    # If suggestions_df empty, create an empty centered map
    if not suggestions_df.empty:
        center_lat = suggestions_df["latitude"].median()
        center_lon = suggestions_df["longitude"].median()
    else:
        center_lat = raw["latitude"].median()
        center_lon = raw["longitude"].median()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster().add_to(m)

    # add user marker if we had params
    if city_q and locality_q:
        lat_user, lon_user = find_latlon_for_locality(locality_q, city_q)
        folium.Marker(
            location=[lat_user, lon_user],
            popup=f"<b>You: {locality_q}</b>",
            icon=folium.Icon(color="black", icon="home")
        ).add_to(m)

    # add suggestion markers
    for s in suggestions:
        popup_html = (
            f"<b>{s['locality']}</b><br>"
            f"{s['bhk']} BHK · {s['size_sqft']} sqft<br>"
            f"Rent: {format_currency(s['rent'])}<br>"
            f"Fair Rent: {format_currency(s['pred_rent'])}<br>"
            + (f"Distance: {s['distance_km']} km" if s['distance_km'] is not None else "")
        )
        # color by value_score
        color = "green" if s["value_score"] >= 0.12 else ("orange" if s["value_score"] >= 0.05 else "red")
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup_html
        ).add_to(mc)

    m.fit_bounds(m.get_bounds(), padding=(20, 20)) if suggestions else None
    map_html = m._repr_html_()

    # pass predicted/listed from query if present so template can show them
    predicted_from_q = request.args.get("pred")
    listed_from_q = request.args.get("listed")

    return render_template(
        "suggestions.html",
        active_page="suggestions",
        suggestions=suggestions,
        map_html=Markup(map_html),
        predicted_from_q=predicted_from_q,
        listed_from_q=listed_from_q
    )

# ----------------------------- ANALYTICS (placeholder) -----------------------------
@app.route("/analytics")
def analytics():
    # simple analytics skeleton (you can expand later)
    avg_rent_locality = enc_df.groupby("locality")["rent"].mean().sort_values(ascending=False).head(10)
    bhk_rent = enc_df.groupby("bhk")["rent"].mean()
    furnish_count = enc_df["furnishing"].value_counts()
    size_vs_rent = enc_df[["size_sqft", "rent"]].to_dict(orient="records")

    return render_template(
        "analytics.html",
        active_page="analytics",
        avg_rent_locality=avg_rent_locality,
        bhk_rent=bhk_rent,
        furnish_count=furnish_count,
        size_vs_rent=size_vs_rent
    )

# ----------------------------- COMPARE (keeps current behaviour) -----------------------------
@app.route("/compare", methods=["GET", "POST"])
def compare_page():
    localities = sorted(raw["locality"].astype(str).str.strip().unique().tolist())
    if request.method == "GET":
        return render_template("compare.html", active_page="compare", localities=localities)

    loc_a = request.form.get("loc_a")
    loc_b = request.form.get("loc_b")
    if not loc_a or not loc_b:
        return render_template("compare.html", active_page="compare", localities=localities, error="Please select both localities.")
    def get_stats(loc):
        df = enc_df[enc_df["locality"].astype(str).str.strip().str.lower() == loc.strip().lower()]
        if df.empty:
            return None
        return {
            "count": int(df.shape[0]),
            "avg_rent": float(df["rent"].mean()),
            "avg_pred_rent": float(df["pred_rent"].mean()),
            "avg_size": float(df["size_sqft"].mean()),
            "avg_bhk": float(df["bhk"].mean()),
            "value_score": float(df["value_score"].mean()),
            "furnish_dist": df["furnishing"].value_counts(normalize=True).to_dict()
        }
    stats_a = get_stats(loc_a)
    stats_b = get_stats(loc_b)
    if not stats_a or not stats_b:
        return render_template("compare.html", active_page="compare", localities=localities, error="One or both localities have no data.")

    chart_labels = [loc_a, loc_b]
    listed_vals = [stats_a["avg_rent"], stats_b["avg_rent"]]
    fair_vals = [stats_a["avg_pred_rent"], stats_b["avg_pred_rent"]]

    return render_template(
        "compare.html",
        active_page="compare",
        localities=localities,
        loc_a=loc_a,
        loc_b=loc_b,
        stats_a=stats_a,
        stats_b=stats_b,
        chart_labels=chart_labels,
        listed_values=listed_vals,
        fair_values=fair_vals
    )

# ----------------------------- RUN -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
