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
import json

app = Flask(__name__)

# ----------------------------- PATHS -----------------------------
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_PATH = MODELS_DIR / "smartrent_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
RAW_CSV = DATA_DIR / "rent_listings_sample.csv"

# ----------------------------- MODEL LOAD -----------------------------
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

le_city = encoders["city"]
le_locality = encoders["locality"]
le_furnish = encoders["furnishing"]

raw = pd.read_csv(RAW_CSV).dropna(subset=["latitude", "longitude"])

# ----------------------------- HELPERS -----------------------------
def safe_encode_input(value, encoder, series):
    classes = list(map(str, encoder.classes_))
    value = value.strip().lower()

    # exact
    for c in classes:
        if c.lower() == value:
            return int(encoder.transform([c])[0])

    # fuzzy
    close = get_close_matches(value, classes, n=1, cutoff=0.6)
    if close:
        return int(encoder.transform([close[0]])[0])

    # fallback → use mode
    mode = str(series.mode().iloc[0])
    return int(encoder.transform([mode])[0])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = (math.sin((lat2 - lat1)/2)**2 +
         math.cos(lat1)*math.cos(lat2)*math.sin((lon2 - lon1)/2)**2)
    return 2*R*math.asin(math.sqrt(a))

def format_currency(x):
    return f"₹{int(x):,}"


# ----------------------------- ENCODE DATAFRAME -----------------------------
enc_df = raw.copy()
enc_df["city_enc"] = le_city.transform(enc_df["city"])
enc_df["locality_enc"] = le_locality.transform(enc_df["locality"])
enc_df["furnishing_enc"] = le_furnish.transform(enc_df["furnishing"])

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
enc_df["value_score"] = (enc_df["pred_rent"] - enc_df["rent"]) / enc_df["pred_rent"]


# ----------------------------- HOME -----------------------------
@app.route("/")
def home():
    return render_template("index.html", active_page="predict")


# ----------------------------- SUGGESTIONS (TABLE + MAP) -----------------------------
@app.route("/suggestions")
def suggestions_page():

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
            "value_score": float(r["value_score"]),
            "distance_km": 0
        })

    # Map
    center_lat = raw["latitude"].median()
    center_lon = raw["longitude"].median()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    mc = MarkerCluster().add_to(m)

    for _, r in enc_df.iterrows():
        popup = f"""
        <b>{r['locality']}</b><br>
        {r['bhk']} BHK • {r['size_sqft']} sqft<br>
        Rent: {format_currency(r['rent'])}<br>
        Fair Rent: {format_currency(r['pred_rent'])}
        """

        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=6,
            color="blue",
            fill=True,
            popup=popup
        ).add_to(mc)

    return render_template(
        "suggestions.html",
        active_page="suggestions",
        suggestions=suggestions,
        map_html=Markup(m._repr_html_())
    )


# ----------------------------- PREDICTION -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    city = request.form["city"]
    locality = request.form["locality"]
    bhk = int(request.form["bhk"])
    size = float(request.form["size"])
    bath = int(request.form["bath"])
    furn = request.form["furnishing"]
    park = int(request.form["parking"])
    age = int(request.form["age"])
    listed = float(request.form["listed_rent"])

    ce = safe_encode_input(city, le_city, raw["city"])
    le = safe_encode_input(locality, le_locality, raw["locality"])
    fe = safe_encode_input(furn, le_furnish, raw["furnishing"])

    X = np.array([[ce, le, bhk, size, bath, fe, park, age]])
    pred = float(model.predict(X)[0])

    diff = listed - pred
    pct = (diff / pred) * 100

    if pct > 10:
        verdict = f"❌ Overpriced by {pct:.1f}%"
        color = "red"
    elif pct < -10:
        verdict = f"🟢 Underpriced by {abs(pct):.1f}%"
        color = "green"
    else:
        verdict = "⚪ Fairly Priced"
        color = "blue"

    return render_template(
        "result.html",
        predicted=int(pred),
        listed=int(listed),
        verdict=verdict,
        color_class=color,
        insight=f"Fair rent for {bhk} BHK in {locality} is ₹{int(pred):,}.",
        active_page="predict"
    )


# ----------------------------- ANALYTICS -----------------------------
@app.route("/analytics")
def analytics():

    avg_rent_locality = (
        enc_df.groupby("locality")["rent"].mean().sort_values(ascending=False).head(10)
    )

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


# ----------------------------- COMPARE (LOCALITY vs LOCALITY) -----------------------------
@app.route("/compare", methods=["GET", "POST"])
def compare_page():
    localities = sorted(raw["locality"].unique().tolist())

    if request.method == "GET":
        return render_template("compare.html", active_page="compare", localities=localities)

    # POST: compare logic
    loc_a = request.form.get("loc_a")
    loc_b = request.form.get("loc_b")

    if not loc_a or not loc_b:
        return render_template(
            "compare.html",
            active_page="compare",
            localities=localities,
            error="Please select both localities."
        )

    def get_stats(loc):
        df = enc_df[enc_df["locality"].str.lower() == loc.lower()]
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
        return render_template(
            "compare.html",
            active_page="compare",
            localities=localities,
            error="One or both localities have no data."
        )

    # Prepare chart data
    chart_labels = [loc_a, loc_b]
    listed_vals = [stats_a["avg_rent"], stats_b["avg_rent"]]
    fair_vals = [stats_a["avg_pred_rent"], stats_b["avg_pred_rent"]]

    furn_types = sorted(set(list(stats_a["furnish_dist"].keys()) + list(stats_b["furnish_dist"].keys())))
    furn_a = [round(stats_a["furnish_dist"].get(f, 0) * 100, 1) for f in furn_types]
    furn_b = [round(stats_b["furnish_dist"].get(f, 0) * 100, 1) for f in furn_types]

    return render_template(
        "compare.html",
        active_page="compare",
        localities=localities,
        loc_a=loc_a,
        loc_b=loc_b,
        stats_a=stats_a,
        stats_b=stats_b,
        chart_labels=json.dumps(chart_labels),
        listed_values=json.dumps(listed_vals),
        fair_values=json.dumps(fair_vals),
        furn_types=json.dumps(furn_types),
        furn_a=json.dumps(furn_a),
        furn_b=json.dumps(furn_b)
    )


# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
