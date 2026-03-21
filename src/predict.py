import joblib
import numpy as np

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    try:
        model = joblib.load('model/model.pkl')
        scaler = joblib.load('model/scaler.pkl')
    except Exception as e:
        print("Error loading model. Make sure to train it first.", e)
        return None
        
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    return prediction[0]
