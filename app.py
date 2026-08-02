import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("SCC_best_model.pkl")

st.set_page_config(
    page_title="SCC AI Predictor",
    page_icon="🦷",
    layout="wide"
)

st.title("🦷 Oral SCC AI Predictor")

st.write(
    "Machine learning model for predicting Oral Squamous Cell Carcinoma outcomes."
)

age = st.number_input("Age", 1, 120, 45)

tumour_size = st.number_input(
    "Tumour size (cm)",
    0.0,
    20.0,
    2.5
)

if st.button("Predict"):
    
    patient = pd.DataFrame({
        "Age":[age],
        "Tumour_Size":[tumour_size]
    })
    
    prediction = model.predict(patient)
    
    st.success(f"Prediction: {prediction[0]}")
