import streamlit as st
import requests
import os

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# Backend API URL logic (connects to FastAPI)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Custom CSS for premium aesthetic
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        transition: 0.3s;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 AI-Based Crop Recommendation System")
st.markdown("Enter the soil and environmental parameters below to get the best crop recommendation.")

st.divider()

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

st.divider()

if st.button("Recommend Crop", type="primary", use_container_width=True):
    with st.spinner('Analyzing data...'):
        payload = {
            "N": n, "P": p, "K": k,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            prediction = response.json().get("recommended_crop")
            if prediction:
                st.success(f"🌱 Recommended Crop: **{prediction.capitalize()}**")
            else:
                st.error("Failed to make a prediction.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend API: {e}")
