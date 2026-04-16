import joblib
import numpy as np
import os

def test_model_prediction():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl'))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl'))
    
    assert os.path.exists(model_path), "Model file not found"
    assert os.path.exists(scaler_path), "Scaler file not found"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    input_data = np.array([[50, 50, 50, 25.0, 60.0, 6.5, 100.0]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    assert prediction is not None
    assert isinstance(prediction[0], str)
    assert len(prediction[0]) > 0
