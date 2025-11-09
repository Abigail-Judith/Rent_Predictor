import pandas as pd
import numpy as np
import joblib
import os

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

# Load original dataset
df = pd.read_csv("data/rent_listings_sample.csv")

# Drop unnecessary columns if they exist
df.drop(columns=["latitude", "longitude"], inplace=True, errors="ignore")

# Load your saved encoders
encoders = joblib.load("models/encoders.pkl")
le_city = encoders["city"]
le_locality = encoders["locality"]
le_furnish = encoders["furnishing"]

# ---------- Safe Encode Function ----------
def safe_encode_column(col, encoder):
    """Safely encode categorical column with unseen label handling"""
    encoded = []
    for val in col.astype(str).str.strip():
        if val not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, val)
        encoded.append(encoder.transform([val])[0])
    return encoded

# ---------- Apply Encoding ----------
df["city"] = safe_encode_column(df["city"], le_city)
df["locality"] = safe_encode_column(df["locality"], le_locality)
df["furnishing"] = safe_encode_column(df["furnishing"], le_furnish)

# ---------- Convert all to numeric ----------
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# ---------- Verify ----------
print("🔍 Unique furnishing values before save:", df["furnishing"].unique())

# ---------- Save encoded file ----------
df.to_csv("data/rent_listings_encoded.csv", index=False)
print("\n✅ All categorical columns encoded and saved as data/rent_listings_encoded.csv")

# ---------- Double-check ----------
encoded_df = pd.read_csv("data/rent_listings_encoded.csv")
print("\n🧩 First few rows of encoded file:\n", encoded_df.head(3))
