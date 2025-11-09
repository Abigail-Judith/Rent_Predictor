import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path

# ✅ Use your new merged dataset (real + expanded)
DATA = Path("data/rent_listings_sample.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Load the dataset
df = pd.read_csv(DATA)

# Drop missing values just in case
df.dropna(inplace=True)

# Initialize encoders
le_city = LabelEncoder()
le_locality = LabelEncoder()
le_furnish = LabelEncoder()

# Encode text columns
df["city"] = le_city.fit_transform(df["city"])
df["locality"] = le_locality.fit_transform(df["locality"])
df["furnishing"] = le_furnish.fit_transform(df["furnishing"])

# ✅ Save encoders so Flask app can decode predictions
encoders = {
    "city": le_city,
    "locality": le_locality,
    "furnishing": le_furnish
}

# Define features and target
X = df[["city", "locality", "bhk", "size_sqft", "bathrooms", "furnishing", "parking", "age_yrs"]]
y = df["rent"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("✅ Training Complete")
print("R² Score:", round(r2_score(y_test, pred), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, pred), 2))

# Save model and encoders
joblib.dump(model, MODELS_DIR / "smartrent_model.pkl")
joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

print("\n✅ Saved models/smartrent_model.pkl and models/encoders.pkl")
print("You can now run: python app.py")
