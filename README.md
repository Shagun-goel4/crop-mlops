# 🌾 AI-Based Crop Recommendation System with MLOps Pipeline

A complete end-to-end Machine Learning Operations (MLOps) project demonstrating training, experiment tracking, scalable API serving, continuous deployment (CI/CD), and a dynamic frontend UI.

## 📖 Project Overview

Farmers often struggle to decide which crop to grow based on soil nutrients and environmental conditions. This system uses applied Machine Learning to suggest the most suitable crop based on the following inputs: 
**Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.**

## ⚙️ Modern Tech Stack
- **Machine Learning**: Scikit-learn (Logistic Regression, Decision Tree, Random Forest, SVM)
- **Experiment Tracking**: MLflow
- **Backend / Server**: FastAPI & Pydantic
- **Frontend / UI**: Streamlit
- **MLOps & Infrastructure**: Docker, Docker Compose, GitHub Actions (CI/CD), AWS EC2
- **Testing**: Pytest

---

## 🏗️ Architecture

This project cleanly separates concerns into a decoupled microservices setup:
1. **Model Pipeline**: `src/train.py` scales data, evaluates multiple candidate models (Logistic Regression, Decision Tree, Random Forest, SVM), tracks parameters and scores via MLflow, and exports the highest-performing model artifact (`.pkl` file) into the `model/` directory.
2. **Backend API**: A FastAPI service (`api/main.py`) validates requests via Pydantic and loads the `.pkl` models to serve predictions natively over a REST endpoint.
3. **Frontend UI**: A Streamlit application (`app/app.py`) provides an interactive GUI to the user and communicates with the backend API over standard HTTP requests.

---

## 🚀 Running Locally (Without Docker)

1. **Install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate / Download Data**:
   Ensure you have a dataset in the `data/` folder, such as [Kaggle's Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). 

3. **Train Model & Track via MLflow**:
   ```bash
   python src/train.py
   ```
   *Logs will be automatically captured in the locally generated `mlruns/` directory.*

4. **Start the FastAPI Backend**:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```

5. **Start the Streamlit Application (In a new terminal window)**:
   ```bash
   export API_URL=http://localhost:8000/predict
   streamlit run app/app.py
   ```

---

## 🐳 Running with Docker

You can easily orchestrate both the frontend and backend microservices natively using Docker Compose.

```bash
docker compose up -d --build
```
- **Streamlit Frontend** runs on port `8501`. (http://localhost:8501)
- **FastAPI Backend** runs on port `8000`. (Read interactive API docs at http://localhost:8000/docs)

---

## ☁️ Continuous Deployment to AWS EC2

This repository is equipped with a GitHub Actions workflow (`.github/workflows/main.yml`) that automates everything:
1. **CI**: Pushing to `main` trains the model to check parameter integrations and runs Pytest validation.
2. **CD**: Upon successful CI, GitHub Actions will securely SSH into your configured AWS EC2 Instance, pull the latest code, and robustly restart the `docker compose` clusters.

> [!NOTE]
> Review `ec2_specifications.md` for EC2 server sizing & security group requirements. Simply execute `bash scripts/deploy_ec2.sh` on your server for automated prerequisite installations!
