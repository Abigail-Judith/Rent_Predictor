from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import math

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

# Load dataset for suggestions (make sure it includes latitude & longitude)
data = pd.read_csv("data/rent_listings_encoded.csv")

# -----------------------------
# Helper Functions
# -----------------------------
def safe_encode(label, encoder):
    """Encode a single value safely (for user input)."""
    if label not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, label)
    return encoder.transform([label])[0]

def encode_column_safe(col, encoder):
    """Encode a pandas column safely (for suggestion dataset)."""
    encoded = []
    for val in col.astype(str).str.strip():
        if val not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, val)
        encoded.append(encoder.transform([val])[0])
    return encoded

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance (in km) between two lat/lon points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --------- 1️⃣ Collect User Input ---------
    city = request.form["city"]
    locality = request.form["locality"]
    bhk = int(request.form["bhk"])
    size = int(request.form["size"])
    bath = int(request.form["bath"])
    furnishing = request.form["furnishing"]
    parking = int(request.form["parking"])
    age = int(request.form["age"])
    listed = int(request.form["listed_rent"])

    # --------- 2️⃣ Encode Input Safely ---------
    city_enc = safe_encode(city, le_city)
    locality_enc = safe_encode(locality, le_locality)
    furnish_enc = safe_encode(furnishing, le_furnish)

    # --------- 3️⃣ Predict Fair Rent ---------
    X = np.array([[city_enc, locality_enc, bhk, size, bath, furnish_enc, parking, age]])
    pred = float(model.predict(X)[0])

    diff = listed - pred
    pct = (diff / pred) * 100 if pred else 0

    # --------- 4️⃣ Verdict ---------
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

    # --------- 5️⃣ Geo-Smart Suggestions ---------
    df = data.copy()

    # Ensure all encodings are numeric
    df["city"] = encode_column_safe(df["city"], le_city)
    df["locality"] = encode_column_safe(df["locality"], le_locality)
    df["furnishing"] = encode_column_safe(df["furnishing"], le_furnish)

    features = df[["city", "locality", "bhk", "size_sqft", "bathrooms",
                   "furnishing", "parking", "age_yrs"]]
    df["pred_rent"] = model.predict(features)
    df["value_score"] = df["pred_rent"] / df["rent"].clip(lower=1)

    # Get coordinates of the user's chosen locality (first match)
    if "latitude" in df.columns and "longitude" in df.columns:
        try:
            loc_ref = data[data["locality"] == locality].iloc[0]
            lat_user, lon_user = float(loc_ref["latitude"]), float(loc_ref["longitude"])
            df["distance_km"] = df.apply(
                lambda r: haversine(lat_user, lon_user, float(r["latitude"]), float(r["longitude"])), axis=1
            )
        except Exception:
            df["distance_km"] = 9999
    else:
        df["distance_km"] = 9999  # fallback if no lat/lon

    # Filter: same city, within 3 km, similar size ±20%
    df_filtered = df[
        (df["city"] == city_enc)
        & (df["bhk"] == bhk)
        & (df["size_sqft"].between(size * 0.8, size * 1.2))
        & (df["distance_km"] <= 3)
    ]

    # Fallback if empty
    if df_filtered.empty:
        df_filtered = df[df["city"] == city_enc]

    suggestions = (
        df_filtered.sort_values("value_score", ascending=False)
        .head(5)
        [["locality", "bhk", "size_sqft", "bathrooms", "furnishing",
          "rent", "pred_rent", "value_score", "distance_km"]]
    )

    if not suggestions.empty:
        suggestions["pred_rent"] = suggestions["pred_rent"].round(0)
        suggestions["value_score"] = suggestions["value_score"].round(2)
        suggestions["distance_km"] = suggestions["distance_km"].round(2)

    # --------- 6️⃣ Render Result Page ---------
    return render_template(
        "result.html",
        predicted=int(pred),
        listed=int(listed),
        verdict=verdict,
        color_class=color_class,
        insight=insight,
        suggestions=suggestions.to_dict(orient="records") if not suggestions.empty else []
    )


if __name__ == "__main__":
    app.run(debug=True)
