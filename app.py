import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter feature values and predict whether the transaction is Fraud (1) or Not Fraud (0).")

# Load saved objects
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# IMPORTANT: columns order same as training dataset
# If your dataset columns are different, replace this list with df.columns (without Class)
FEATURES = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

st.subheader("🧾 Input Features")

input_data = {}
for col in FEATURES:
    input_data[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    X = np.array([[input_data[c] for c in FEATURES]], dtype=float)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])

    if pred == 1:
        st.error("🚨 Prediction: FRAUD Transaction (1)")
    else:
        st.success("✅ Prediction: NOT Fraud (0)")
