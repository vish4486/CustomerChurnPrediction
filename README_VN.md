# 🧠 Churn Predictor Web Application

A machine learning pipeline to predict customer churn based on bank data. This repository implements the MLOps workflow using modular code, hyperparameter tuning, DVC-based versioning, and clear team responsibilities.

---

## 👤 Role: Data Engineer (Vishal Nigam)

### Responsibilities Completed

- **Data Understanding & Cleaning**
  - Loaded and explored `BankChurners.csv` with distribution, null checks, and data types
  - Added custom feature: `Transaction_Intensity = Total_Trans_Ct * Total_Amt_Chng_Q4_Q1`

- **Preprocessing Pipeline**
  - Scaled numerical & one-hot encoded categorical variables
  - Stratified train-test split
  - Applied `ColumnTransformer` for modularity
  - Saved transformed outputs under `data/processed/`

- **Class Imbalance Handling**
  - Used `class_weight='balanced'` in traditional models
  - Passed computed class weights to ANN model

- **Model Training**
  - Implemented baseline classifiers:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Artificial Neural Network (Keras)

- **Hyperparameter Tuning**
  - Used `GridSearchCV` for sklearn models
  - Used `Keras Tuner` for ANN (5-trial random search)
  - Best models saved in `models/` directory

- **Data Version Control (DVC)**
  - Tracked raw and processed data
  - Added a DVC pipeline stage: `preprocess_data`
  - Made pipeline reproducible via `dvc repro`
  - Data decoupled from Git for clean repository state

- **Evaluation Visualizations**
  - Saved all plots (correlation heatmaps, metrics bar chart) in `results/plots/`
  - Exported model performance summary as `.tex` and `.png`

---

## 📁 Project Structure (Simplified)
```
MLOPS-PROJECT/
│
├── data/ # Contains raw and processed data
│ ├── raw/BankChurners.csv
│ └── processed/
├── models/ # All trained models (.pkl, .keras)
├── results/plots/ # All generated plots and tables
├── src/ # Core preprocessing, training and evaluation scripts
├── app/main.py # Entry-point pipeline
├── dvc.yaml + dvc.lock # DVC stage definition
├── requirements.txt # Python dependencies
```


---

## 🚀 Next Steps for Team

### 🔧 Software Developer (Sepehr)

- Build user-friendly **streamlit/flask web interface**
- Input: user form → Output: churn prediction
- Load model from `models/ann_model.keras`
- Provide frontend for metrics display using `results/plots/`

### 🧑‍💻 Software Engineer (Mubashar)

- Ensure **CI/CD pipeline** (GitHub Actions or similar)
- Wrap model training in dockerized workflow
- Use `dvc repro` as build step
- Optional: Connect to cloud storage for data artifacts

---

## 📝 Usage

```bash
# Run the full MLOps pipeline
python app/main.py

# Reproduce with data version control
dvc repro
```

## 📝 Dependencies
```
pip install -r requirements.txt
```
## 📣 Contact
Vishal Nigam (Data Engineer) – LinkedIn | Email


