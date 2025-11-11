import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---- Load model ----
@st.cache_data
def load_model():
    return joblib.load("../models/final_model.joblib")

model = load_model()

st.set_page_config(page_title="Financial Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Financial Fraud Detection System")
st.write("Upload transactions CSV or enter transaction data manually to detect potential fraud.")

# ---- Sidebar: threshold ----
threshold = st.sidebar.slider("Fraud Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# ---- File upload ----
uploaded_file = st.file_uploader("Upload CSV with transactions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Transactions")
    st.dataframe(df.head())

    # Predict probabilities
    probs = model.predict_proba(df)[:,1]
    df["Fraud Probability"] = probs
    df["Fraud Prediction"] = (probs >= threshold).astype(int)

    st.subheader("Predictions")
    st.dataframe(df.head())

    st.subheader("Fraud Stats")
    st.write("Total transactions:", len(df))
    st.write("Potential frauds detected:", df["Fraud Prediction"].sum())

    # Top 10 risky transactions
    st.subheader("Top 10 Highest Risk Transactions")
    st.dataframe(df.sort_values("Fraud Probability", ascending=False).head(10))

# ---- Manual input ----
st.subheader("Manual Transaction Input")
st.write("Enter transaction details to predict fraud.")

# Example: assume dataset has features: V1, V2, V3, ..., Amount
try:
    features = [col for col in model.feature_names_in_]
    user_input = {}
    for f in features:
        val = st.number_input(f"{f}", value=0.0)
        user_input[f] = val

    if st.button("Predict Fraud"):
        input_df = pd.DataFrame([user_input])
        prob = model.predict_proba(input_df)[:,1][0]
        prediction = int(prob >= threshold)
        st.write(f"Fraud Probability: {prob:.4f}")
        st.write("Prediction:", "Fraud" if prediction==1 else "Legitimate")

        # Optional: SHAP explanation
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)
        st.subheader("Feature Impact (SHAP)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)

except Exception as e:
    st.write("Manual input not available for this model.")
