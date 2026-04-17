import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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
    
    print("Training models and logging to MLflow...")
    
    # MLflow tracking
    mlflow.set_experiment("Crop_Recommendation_Experiment")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    model_accuracies = {}
    best_model_name = ""
    best_accuracy = 0.0
    best_model_instance = None
    
    for model_name, model in models.items():
        print(f"--> Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            if model_name in ["Logistic Regression", "SVM"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            model_accuracies[model_name] = acc
            print(f"    Accuracy: {acc:.4f}")
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Check if this is the best model so far
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = model_name
                best_model_instance = model
                
    print("\n" + "="*40)
    print("🏆 Model Comparison Leaderboard")
    print("="*40)
    for name, acc in sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{name.ljust(25)} : {acc:.4f}")
    print("="*40)
    
    print(f"\nBest Model: {best_model_name} with Accuracy >> {best_accuracy:.4f}")
    
    # Saving best model locally for FastAPI inference
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model_instance, 'model/model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    print(f"\nSaved {best_model_name} (the highest performing model) to 'model/' directory to be served!")

if __name__ == "__main__":
    train_model()
