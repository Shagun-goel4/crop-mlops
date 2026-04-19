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
    model_name = type(model).__name__
    if model_name in ['RandomForestClassifier', 'DecisionTreeClassifier']:
        final_data = input_data
    else:
        final_data = scaler.transform(input_data)
        
    # Test probabilistic mapping behavior
    probabilities = model.predict_proba(final_data)[0]
    classes = model.classes_
    
    assert probabilities is not None
    assert len(probabilities) > 0
    assert len(classes) == len(probabilities)
