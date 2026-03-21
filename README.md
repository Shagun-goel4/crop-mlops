# AI-Based Crop Recommendation System with MLOps Pipeline

A complete end-to-end Machine Learning Operations (MLOps) project demonstrating training, CI/CD, and deployment of a crop recommendation system.

## 🌾 Project Overview

Farmers often struggle to decide which crop to grow based on soil nutrients and environmental conditions. This system uses ML to suggest the most suitable crop from inputs: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.

## ⚙️ Tech Stack
- **ML**: Scikit-learn (Random Forest)
- **UI**: Streamlit
- **MLOps**: Docker, GitHub Actions, Pytest

## 🚀 Running Locally

1. **Install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate / Download Data**:
   ```bash
   python data/generate_data.py
   # Or replace data/Crop_recommendation.csv with Kaggle dataset:
   # https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
   ```

3. **Train Model**:
   ```bash
   python src/train.py
   ```

4. **Run Application**:
   ```bash
   streamlit run app/app.py
   ```

## 🐳 Docker

Build and run the containerized Streamlit app:
```bash
docker build -t crop-mlops .
docker run -p 8501:8501 crop-mlops
```
