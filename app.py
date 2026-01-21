import streamlit as st
import numpy as np
import joblib

# Load saved model
data = joblib.load("model/breast_cancer_model.pkl")
model = data["model"]
scaler = data["scaler"]
feature_names = data["feature_names"]

st.set_page_config(page_title="Breast Cancer Prediction System")

st.title("🩺 Breast Cancer Prediction System")
st.write(
    "This system predicts whether a breast tumor is **Benign** or **Malignant** "
    "based on selected tumor features.\n\n"
    "**For educational purposes only. Not a medical diagnostic tool.**"
)

st.subheader("Enter Tumor Feature Values")

# Input fields (example feature set — must match training)
radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.4f")
texture_mean = st.number_input("Texture Mean", min_value=0.0, format="%.4f")
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.4f")
area_mean = st.number_input("Area Mean", min_value=0.0, format="%.4f")
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, format="%.4f")

if st.button("Predict Diagnosis"):
    input_data = np.array([
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean
    ]).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Map output
    diagnosis = "Malignant" if prediction == 1 else "Benign"

    st.success(f"🔍 Predicted Diagnosis: **{diagnosis}**")
