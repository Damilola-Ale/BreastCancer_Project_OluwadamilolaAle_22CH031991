import streamlit as st
import pandas as pd
import joblib

# Load saved model
data = joblib.load("model/wine_cultivar_model.pkl")
model = data['model']
scaler = data['scaler']
feature_names = data['feature_names']
target_names = data['target_names']

st.title("🍷 Wine Cultivar Origin Prediction System")
st.write("Enter the wine chemical properties to predict its cultivar.")

# Input fields
alcohol = st.number_input("Alcohol", min_value=0.0, value=13.0)
malic_acid = st.number_input("Malic Acid", min_value=0.0, value=2.0)
ash = st.number_input("Ash", min_value=0.0, value=2.3)
magnesium = st.number_input("Magnesium", min_value=0.0, value=100.0)
flavanoids = st.number_input("Flavanoids", min_value=0.0, value=2.0)
color_intensity = st.number_input("Color Intensity", min_value=0.0, value=5.0)

if st.button("Predict Cultivar"):
    # Build input as DataFrame with correct feature order
    input_df = pd.DataFrame([{
        'alcohol': alcohol,
        'malic_acid': malic_acid,
        'ash': ash,
        'magnesium': magnesium,
        'flavanoids': flavanoids,
        'color_intensity': color_intensity
    }])[feature_names]

    # Scale correctly
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    predicted_label = target_names[prediction]

    st.success(f"Predicted Wine Cultivar: {predicted_label}")