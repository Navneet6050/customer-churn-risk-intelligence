import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "../data/raw/telco_churn.csv"
MODEL_PATH = "../models/churn_model.pkl"
FEATURE_PATH = "../models/feature_names.pkl"
OUTPUT_PATH = "../data/processed/model_predictions.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID"])

df = pd.get_dummies(df, drop_first=True)

feature_names = joblib.load(FEATURE_PATH)

for col in feature_names:
    if col not in df.columns:
        df[col] = 0

df = df[feature_names + ["Churn"]]

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)

probs = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, probs)

# Save probabilities for dynamic thresholding
pred_df = pd.DataFrame({
    "actual": y_test.values,
    "probability": probs
})

pred_df.to_csv(OUTPUT_PATH, index=False)

print("Predictions saved for dynamic evaluation.")
print("ROC-AUC:", round(roc, 3))
