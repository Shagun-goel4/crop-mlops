from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending crops based on soil and environmental features.",
    version="1.0"
)

# Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')

model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

class CropFeatures(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API. Use POST /predict to get recommendations."}

@app.post("/predict")
def predict_crop(features: CropFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model is not trained resolving/loaded yet.")
    
    # Extract features matching the model training order
    input_data = np.array([[
        features.N, features.P, features.K, 
        features.temperature, features.humidity, 
        features.ph, features.rainfall
    ]])
    
    # Scale and predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    return {"recommended_crop": prediction[0]}
