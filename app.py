import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
model = joblib.load("wine_quality_catboost.pkl")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction")
st.write(
    "Enter the physicochemical properties of the wine below to predict its quality."
)

# -------------------------------
# Get Feature Names from Model
# -------------------------------
try:
    feature_names = model.feature_names_
except:
    st.error("Could not retrieve feature names from the model.")
    st.stop()

# -------------------------------
# User Inputs
# -------------------------------
input_data = {}

st.subheader("Input Features")

for feature in feature_names:
    input_data[feature] = st.number_input(
        label=feature.replace("_", " ").title(),
        value=0.0,
        format="%.4f"
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Wine Quality"):
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    try:
        predicted_quality = int(prediction[0][0])
    except:
        predicted_quality = int(prediction[0])

    st.success(f"### Predicted Wine Quality: **{predicted_quality}** 🍷")