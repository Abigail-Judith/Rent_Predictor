from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import math
from generate_map import generate_rent_map  # ✅ We'll use this to generate the map

app = Flask(__name__)

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("models/smartrent_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# Extract individual encoders
le_city = encoders["city"]
le_locality = encoders["locality"]
le_furnish = encoders["furnishing"]

# Load dataset (includes latitude, longitude)
data = pd.read_csv("data/rent_listings_encoded.csv")


# -----------------------------
# Helper Functions
# -----------------------------
def safe_encode(label, encoder):
    """Encode user input safely."""
    if label not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, label)
    return encoder.transform([label])[0]


def ensure_encoded(series, encoder):
    """Ensure text column is encoded numerically."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    vals = series.astype(str).str.strip()
    unseen = np.setdiff1d(vals.unique(), encoder.classes_)
    if unseen.size:
        encoder.classes_ = np.concatenate([encoder.classes_, unseen])
    return encoder.transform(vals)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance (in km) between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    """Homepage with prediction form + interactive rent map"""
    # Automatically generate the latest rent map from data
    generate_rent_map("data/rent_listings_sample.csv", "static/rent_map.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --------- 1️⃣ Collect Input ---------
    city = request.form["city"]
    locality = request.form["locality"]
    bhk = int(request.form["bhk"])
    size = int(request.form["size"])
    bath = int(request.form["bath"])
    furnishing = request.form["furnishing"]
    parking = int(request.form["parking"])
    age = int(request.form["age"])
    listed = int(request.form["listed_rent"])

    # --------- 2️⃣ Encode Input ---------
    city_enc = safe_encode(city, le_city)
    locality_enc = safe_encode(locality, le_locality)
    furnish_enc = safe_encode(furnishing, le_furnish)

    # --------- 3️⃣ Predict Fair Rent ---------
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

    insight = f"For a {bhk}BHK ~{size} sqft in {locality}, fair rent is around ₹{int(pred):,}."

    # --------- 4️⃣ Prepare Dataset ---------
    df = data.copy()
    df["city"] = ensure_encoded(df["city"], le_city)
    df["locality"] = ensure_encoded(df["locality"], le_locality)
    df["furnishing"] = ensure_encoded(df["furnishing"], le_furnish)

    features = df[["city", "locality", "bhk", "size_sqft", "bathrooms",
                   "furnishing", "parking", "age_yrs"]]
    df["pred_rent"] = model.predict(features)
    df["value_score"] = (df["pred_rent"] - df["rent"]) / df["pred_rent"].replace(0, np.nan)

    # --------- 5️⃣ Compute Distances ---------
    if "latitude" in df.columns and "longitude" in df.columns:
        try:
            loc_ref = data[data["locality"] == locality].iloc[0]
            lat_user, lon_user = float(loc_ref["latitude"]), float(loc_ref["longitude"])
            df["distance_km"] = df.apply(
                lambda r: haversine(lat_user, lon_user, float(r["latitude"]), float(r["longitude"])),
                axis=1,
            )
        except Exception:
            df["distance_km"] = 9999.0
    else:
        df["distance_km"] = 9999.0

    # --------- 6️⃣ Generate Suggestions ---------
    candidates = df[df["city"] == city_enc].copy()

    def top5(df_):
        cols = ["locality", "bhk", "size_sqft", "bathrooms", "furnishing",
                "rent", "pred_rent", "value_score", "distance_km"]
        out = df_.sort_values("value_score", ascending=False)[cols].head(5).copy()
        if not out.empty:
            out["pred_rent"] = out["pred_rent"].round(0)
            out["value_score"] = out["value_score"].round(2)
            out["distance_km"] = out["distance_km"].round(2)
        return out

    fallback_message = ""

    # 🧠 Budget adjustment logic
    if listed > pred:
        # overpriced → show cheaper options
        max_budget = pred * 1.05
    else:
        # underpriced → show slightly better quality
        max_budget = pred * 1.3

    # Step 1: Nearby, similar size ±20%
    mask = (
        (candidates["bhk"] == bhk)
        & (candidates["size_sqft"].between(size * 0.8, size * 1.2))
        & (candidates["distance_km"] <= 6)
        & (candidates["rent"] <= max_budget)
    )
    sug = top5(candidates[mask])

    # Step 2: Expand to 8 km
    if sug.empty:
        mask = (
            (candidates["bhk"] == bhk)
            & (candidates["size_sqft"].between(size * 0.7, size * 1.3))
            & (candidates["distance_km"] <= 8)
            & (candidates["rent"] <= max_budget)
        )
        sug = top5(candidates[mask])

    # Step 3: City-wide top deals (no budget restriction)
    if sug.empty:
        sug = top5(candidates)
        fallback_message = "⚠️ No exact in-budget matches nearby. Showing best value rentals in Bengaluru."

    if not sug.empty and not fallback_message:
        fallback_message = f"🏙️ Showing best value rentals near {locality}."

    # --------- 7️⃣ Render Result Page ---------
    return render_template(
        "result.html",
        predicted=int(pred),
        listed=int(listed),
        verdict=verdict,
        color_class=color_class,
        insight=insight,
        suggestions=sug.to_dict(orient="records") if not sug.empty else [],
        fallback_message=fallback_message,
    )


@app.route("/map")
def rent_map():
    generate_rent_map("data/rent_listings_sample.csv", "static/rent_map.html")
    return render_template("map.html")


if __name__ == "__main__":
    app.run(debug=True)
