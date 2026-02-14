import pandas as pd
import numpy as np
import joblib
import shap
import os

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "../data/raw/telco_churn.csv"
MODEL_PATH = "../models/churn_model.pkl"
FEATURE_PATH = "../models/feature_names.pkl"

RISK_OUTPUT_PATH = "../data/processed/customer_risk_scores.csv"
ENCODED_OUTPUT_PATH = "../data/processed/customer_encoded_features.csv"

os.makedirs("../data/processed", exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

customer_ids = df["customerID"].copy()

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID column for encoding
df_model = df.drop(columns=["customerID"])

# One-hot encode
df_encoded = pd.get_dummies(df_model, drop_first=True)

# -----------------------------
# ALIGN FEATURES WITH TRAINING
# -----------------------------
feature_names = joblib.load(FEATURE_PATH)

for col in feature_names:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[feature_names]

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# RISK SCORES (SMOOTHED)
# -----------------------------
raw_probs = model.predict_proba(df_encoded)[:, 1]

# Smooth to avoid extreme clustering
risk_scores = 100 / (1 + np.exp(-6 * (raw_probs - 0.5)))
risk_scores = np.round(risk_scores, 2)

# -----------------------------
# SHAP EXPLANATIONS
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_encoded)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

feature_names = df_encoded.columns.tolist()

explanations = []

for i in range(len(df_encoded)):
    shap_row = shap_values[i]
    shap_pairs = list(zip(feature_names, shap_row))
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = shap_pairs[:3]
    readable = [feat.replace("_", " ") for feat, val in top_features]

    explanations.append("; ".join(readable))

# -----------------------------
# SAVE RISK OUTPUT
# -----------------------------
output_df = pd.DataFrame({
    "customerID": customer_ids,
    "risk_score": risk_scores,
    "top_risk_drivers": explanations
})

output_df.to_csv(RISK_OUTPUT_PATH, index=False)

# -----------------------------
# SAVE ENCODED MATRIX FOR SIMULATION
# -----------------------------
df_encoded_with_id = df_encoded.copy()
df_encoded_with_id["customerID"] = customer_ids.values
df_encoded_with_id.to_csv(ENCODED_OUTPUT_PATH, index=False)

print("Risk scoring complete.")
print(output_df.head())
