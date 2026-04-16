import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('data/Crop_recommendation.csv')
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Classifier with MLflow tracking...")
    
    # MLflow tracking
    mlflow.set_experiment("Crop_Recommendation_Experiment")
    
    with mlflow.start_run():
        n_estimators = 100
        random_state = 42
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {acc:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Saving model locally for FastAPI inference
        os.makedirs('model', exist_ok=True)
        joblib.dump(model, 'model/model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        print("Model and scaler saved to 'model/' directory")

if __name__ == "__main__":
    train_model()
