import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "../data/raw/telco_churn.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop non-useful columns
df = df.drop(columns=["customerID"])

# One-hot encode categoricals
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# SPLIT
# -----------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print("ROC-AUC:", round(auc, 3))

# -----------------------------
# SAVE
# -----------------------------
joblib.dump(model, f"{MODEL_DIR}/churn_model.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/feature_names.pkl")

print("Model training complete.")
