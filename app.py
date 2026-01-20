import streamlit as st
import numpy as np
import joblib

# Load saved model
data = joblib.load("wine_cultivar_model.pkl")
model = data['model']
scaler = data['scaler']
features = data['feature_names']
target_names = data['target_names']

st.title("🍷 Wine Cultivar Origin Prediction System")

st.write("Enter the wine chemical properties to predict its cultivar.")

# Input fields
alcohol = st.number_input("Alcohol", min_value=0.0)
malic_acid = st.number_input("Malic Acid", min_value=0.0)
ash = st.number_input("Ash", min_value=0.0)
magnesium = st.number_input("Magnesium", min_value=0.0)
flavanoids = st.number_input("Flavanoids", min_value=0.0)
color_intensity = st.number_input("Color Intensity", min_value=0.0)

if st.button("Predict Cultivar"):
    input_data = np.array([
        alcohol, malic_acid, ash,
        magnesium, flavanoids, color_intensity
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Wine Cultivar: {target_names[prediction]}")
