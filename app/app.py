import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# BASE PATH CONFIG (PRODUCTION SAFE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RISK_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_risk_scores.csv")
ENCODED_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_encoded_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
PRED_PATH = os.path.join(BASE_DIR, "data", "processed", "model_predictions.csv")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Risk Intelligence System",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(RISK_PATH)
df_encoded = pd.read_csv(ENCODED_PATH)
model = joblib.load(MODEL_PATH)

# -----------------------------
# HEADER
# -----------------------------
st.title("🚀 Customer Churn Risk Intelligence Dashboard")
st.markdown(
    "Predicts churn risk (0–100), enables retention simulation, and supports threshold tuning."
)

# ============================================================
# DYNAMIC MODEL EVALUATION SECTION
# ============================================================
st.markdown("---")
st.header("📈 Model Performance & Threshold Tuning")

try:
    pred_df = pd.read_csv(PRED_PATH)

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    preds = (pred_df["probability"] >= threshold).astype(int)

    tp = ((preds == 1) & (pred_df["actual"] == 1)).sum()
    tn = ((preds == 0) & (pred_df["actual"] == 0)).sum()
    fp = ((preds == 1) & (pred_df["actual"] == 0)).sum()
    fn = ((preds == 0) & (pred_df["actual"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    colA, colB = st.columns(2)
    colA.metric("Precision", round(precision, 3))
    colB.metric("Recall", round(recall, 3))

    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame({
        "": ["Actual Negative", "Actual Positive"],
        "Predicted Negative": [tn, fn],
        "Predicted Positive": [fp, tp]
    })

    st.table(cm_df)

    st.markdown(
        "💡 Lower threshold → Higher recall (catch more churners) but more false positives.\n\n"
        "💡 Higher threshold → Higher precision but may miss risky customers."
    )

except Exception:
    st.warning("Model prediction file not found.")

# ============================================================
# RISK DASHBOARD SECTION
# ============================================================

def categorize(score):
    if score >= 75:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk Segment"] = df["risk_score"].apply(categorize)

st.markdown("---")

risk_threshold = st.slider(
    "Minimum Risk Score Filter",
    min_value=0,
    max_value=100,
    value=60
)

filtered_df = df[df["risk_score"] >= risk_threshold]

col1, col2, col3 = st.columns(3)

col1.metric("Customers Above Threshold", len(filtered_df))

if len(filtered_df) > 0:
    col2.metric("Average Risk", round(filtered_df["risk_score"].mean(), 2))
    col3.metric("Max Risk", round(filtered_df["risk_score"].max(), 2))
else:
    col2.metric("Average Risk", 0)
    col3.metric("Max Risk", 0)

# -----------------------------
# DISTRIBUTIONS
# -----------------------------
st.subheader("📊 Risk Segment Distribution")
st.bar_chart(df["Risk Segment"].value_counts())

st.subheader("📈 Risk Score Histogram")
hist_values = np.histogram(df["risk_score"], bins=20)[0]
st.bar_chart(hist_values)

# -----------------------------
# GLOBAL FEATURE IMPORTANCE
# -----------------------------
st.subheader("🌍 Global Churn Drivers")

feature_columns = df_encoded.drop(columns=["customerID"]).columns.tolist()
importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance_values
}).sort_values("Importance", ascending=False).head(10)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# COLOR-CODED TABLE
# -----------------------------
st.subheader("⚠️ Customers at Risk")

def highlight_risk(row):
    if row["risk_score"] >= 75:
        return ['background-color: #3a1f1f; color: white'] * len(row)
    elif row["risk_score"] >= 40:
        return ['background-color: #3a331f; color: white'] * len(row)
    else:
        return ['background-color: #1f3a2a; color: white'] * len(row)

styled_df = (
    filtered_df
        .sort_values(["risk_score", "customerID"], ascending=[False, True])
        .reset_index(drop=True)
        .style.apply(highlight_risk, axis=1)
)

st.dataframe(styled_df, width="stretch")

# -----------------------------
# CUSTOMER INSIGHT + SIMULATION
# -----------------------------
st.subheader("🔍 Customer Risk Insight")

if len(filtered_df) > 0:

    selected_customer = st.selectbox(
        "Select Customer ID",
        sorted(filtered_df["customerID"].unique())
    )

    row = filtered_df[filtered_df["customerID"] == selected_customer].iloc[0]

    st.markdown(f"### Current Risk Score: **{row['risk_score']}**")
    st.markdown(f"### Risk Segment: **{row['Risk Segment']}**")
    st.markdown("**Top Risk Drivers:**")
    st.write(row["top_risk_drivers"])

    # Recommended Action
    st.markdown("### 📌 Recommended Action")

    if row["risk_score"] >= 75:
        st.error("Immediate retention action required. Offer contract upgrade or pricing incentive.")
    elif row["risk_score"] >= 40:
        st.warning("Monitor customer closely. Consider engagement campaign or personalized offer.")
    else:
        st.success("Customer is stable. Maintain engagement strategy.")

    # -----------------------------
    # WHAT-IF SIMULATION
    # -----------------------------
    st.markdown("## 🧪 Retention Strategy Simulation")

    encoded_row = df_encoded[df_encoded["customerID"] == selected_customer] \
        .drop(columns=["customerID"]).copy()

    colA, colB, colC = st.columns(3)

    with colA:
        new_contract = st.selectbox(
            "Change Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with colB:
        new_internet = st.selectbox(
            "Change Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with colC:
        new_monthly = st.slider(
            "Adjust Monthly Charges",
            min_value=0,
            max_value=150,
            value=int(encoded_row["MonthlyCharges"].values[0])
        )

    # Reset contract encoding
    for col in encoded_row.columns:
        if "Contract_" in col:
            encoded_row[col] = 0

    if new_contract == "One year" and "Contract_One year" in encoded_row.columns:
        encoded_row["Contract_One year"] = 1
    elif new_contract == "Two year" and "Contract_Two year" in encoded_row.columns:
        encoded_row["Contract_Two year"] = 1

    # Reset internet encoding
    for col in encoded_row.columns:
        if "InternetService_" in col:
            encoded_row[col] = 0

    if new_internet == "Fiber optic" and "InternetService_Fiber optic" in encoded_row.columns:
        encoded_row["InternetService_Fiber optic"] = 1
    elif new_internet == "DSL" and "InternetService_DSL" in encoded_row.columns:
        encoded_row["InternetService_DSL"] = 1

    encoded_row["MonthlyCharges"] = new_monthly

    raw_prob = model.predict_proba(encoded_row)[0][1]
    new_risk = 100 / (1 + np.exp(-6 * (raw_prob - 0.5)))
    new_risk = round(new_risk, 2)

    st.markdown("### 📈 Simulated Risk Score")
    st.metric("New Risk Score", new_risk, delta=round(new_risk - row["risk_score"], 2))

else:
    st.info("No customers meet the selected threshold.")
