import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path

DATA = Path("data/rent_listings_sample.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df.dropna(inplace=True)

# Label encoders for categorical fields
le_city = LabelEncoder()
le_locality = LabelEncoder()
le_furnish = LabelEncoder()

df["city"] = le_city.fit_transform(df["city"].astype(str).str.strip())
df["locality"] = le_locality.fit_transform(df["locality"].astype(str).str.strip())
df["furnishing"] = le_furnish.fit_transform(df["furnishing"].astype(str).str.strip())

# Features & target (order MUST match app.py)
X = df[["city", "locality", "bhk", "size_sqft", "bathrooms", "furnishing", "parking", "age_yrs"]]
y = df["rent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("✅ Training Complete")
print("R² Score:", round(r2_score(y_test, pred), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, pred), 2))

# Save model and encoders (encoders as dict)
encoders = {"city": le_city, "locality": le_locality, "furnishing": le_furnish}
joblib.dump(model, MODELS_DIR / "smartrent_model.pkl")
joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

print("\n✅ Saved models/smartrent_model.pkl and models/encoders.pkl")
