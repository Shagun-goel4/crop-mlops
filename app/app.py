import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_crop

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

st.title("🌾 AI-Based Crop Recommendation System")
st.markdown("Enter the soil and environmental parameters below to get the best crop recommendation.")

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=50.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=50.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=50.0)
    temperature = st.number_input("Temperature (°C)", min_value=5.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=10.0, max_value=300.0, value=100.0)

if st.button("Recommend Crop", type="primary", use_container_width=True):
    with st.spinner('Analyzing data...'):
        prediction = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
        if prediction:
            st.success(f"🌱 Recommended Crop: **{prediction.capitalize()}**")
