# 🧠 Customer Churn Prediction

Predict customer attrition for a bank's credit card service using an end-to-end MLOps pipeline. This project includes data preprocessing, feature engineering, model training, automated tuning, deployment with Streamlit, and monitoring.

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

---

## 📌 Project Overview

- **Problem**: Classify customers as _Attrited_ or _Existing_ using their profile and transaction data.
- **Dataset**: [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **Best Model**: Tuned XGBoost with ~94.6% Balanced Accuracy
- **Interface**: Interactive frontend built with Streamlit

---

## 🚀 Features

✅ Preprocessing pipeline with feature engineering  
✅ Hyperparameter tuning (ANN, RF, XGBoost, Logistic Regression)  
✅ Streamlit frontend for predictions  
✅ Logging & Monitoring (prediction logs + charts)  
✅ Dockerized application  
✅ Tested with GitHub Actions + DVC support

---

## 🗂️ Project Structure

```
CustomerChurnPrediction/
│
├── app/                 # Streamlit app
├── data/                # Raw + processed data
├── models/              # Saved model artifacts
├── src/                 # Source code for training, preprocessing
├── tests/               # Unit tests
├── logs/                # Prediction logs
├── results/             # Evaluation outputs
├── ann_tuning/          # (Optional) Tuning artifacts
├── Dockerfile           # For containerization
├── requirements.txt     # Dependencies
├── run_docker.sh        # Shell script to run app in Docker
├── dvc.yaml             # DVC pipeline
├── .github/workflows/   # GitHub Actions (test.yml)
└── README.md
```

## 🛠️ Setup Instructions

# 1. Clone and Setup
```
git clone https://github.com/vish4486/CustomerChurnPrediction.git
cd CustomerChurnPrediction
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

# 2 Ensure Model Artifacts Exist

```
The repository already includes:

models/best_model.pkl

models/best_model_meta.json

data/processed/preprocessor.pkl

No need to retrain. You can immediately run the app.
```

# 3 🌐 Run the Streamlit App
```
streamlit run app/streamlit_app.py
Then open: http://localhost:8501/ in your browser.
```

# 4 🐳 Run with Docker (Optional)

```
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```

# 5 🔁 Reproduce ML Pipeline (optional)
If needed:

```
dvc pull           # fetch versioned data/models
dvc repro          # re-run pipeline if code/data changed
```

# 6 ✅ CI/CD with GitHub Actions
Whenever you push changes:

✅ Unit tests run (via pytest)
✅ Linting is checked (via flake8)
✅ DVC pipeline status is validated

Workflow file: .github/workflows/test.yml

# 7 📊 Monitoring & Logging
Every prediction is logged in logs/predictions.csv

Streamlit UI visualizes:

🥧 Prediction distribution (Attrited vs Existing)

📈 Feature value histograms

# 8 👨‍💻 Authors

Vishal Nigam – Data Engineer

Mubashar Mohammad Shahzad – Software Engineer

Sepehr – Software Developer

# 9 📄 License
This project is licensed under the MIT License. See LICENSE for details.