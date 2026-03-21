import pandas as pd
import numpy as np
import random
import os

def generate_mock_data(n_samples=500):
    np.random.seed(42)
    random.seed(42)
    
    crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 
             'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 
             'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 
             'rice', 'watermelon']
             
    data = {
        'N': np.random.randint(0, 140, n_samples),
        'P': np.random.randint(5, 145, n_samples),
        'K': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(8.0, 43.0, n_samples),
        'humidity': np.random.uniform(14.0, 100.0, n_samples),
        'ph': np.random.uniform(3.5, 9.9, n_samples),
        'rainfall': np.random.uniform(20.0, 298.0, n_samples),
        'label': [random.choice(crops) for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    file_path = 'data/Crop_recommendation.csv'
    df.to_csv(file_path, index=False)
    print(f"Mock dataset generated successfully at {file_path}")

if __name__ == "__main__":
    generate_mock_data()
