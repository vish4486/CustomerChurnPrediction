import streamlit as st
import pandas as pd
import json
import joblib
from tensorflow.keras.models import load_model

# Load metadata
with open("models/best_model_meta.json", "r") as f:
    meta = json.load(f)

model_type = meta["best_model_type"]
timestamp = meta["timestamp"]
top_features = meta.get("top_features", [])

model_path = "models/best_model.pkl" if "ANN" not in model_type else f"models/ann_model_{timestamp}.keras"
model = load_model(model_path) if "ANN" in model_type else joblib.load(model_path)
preprocessor = joblib.load("data/processed/preprocessor.pkl")

st.title("Customer Churn Prediction")
st.markdown("#### Enter basic customer details:")

# === Step 1: Intuitive Inputs ===
user_input = {
    "Customer_Age": st.slider("Customer Age", 18, 100, 45),
    "Gender": st.selectbox("Gender", ["M", "F"]),
    "Marital_Status": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "Income_Category": st.selectbox("Income Category", [
        "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"
    ]),
    "Card_Category": st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"]),
}

# === Step 2: Advanced Inputs from Top Features (Dynamic) ===
advanced_inputs = {}
with st.expander("Optional: Advanced Transactional Features"):
    for feature in top_features:
        if feature.startswith("Card_Category_"):
            advanced_inputs[feature] = st.selectbox(feature, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        elif feature in ["Transaction_Intensity", "Total_Amt_Chng_Q4_Q1", "Total_Ct_Chng_Q4_Q1"]:
            advanced_inputs[feature] = st.number_input(feature, min_value=0.0, step=0.01)
        else:
            advanced_inputs[feature] = st.number_input(feature, step=1.0)

# === Step 3: Fill remaining required features with defaults ===
user_input.update(advanced_inputs)

expected_features = list(preprocessor.feature_names_in_)
default_values = {
    "Dependent_count": 2,
    "Education_Level": "Uneducated",
    "Months_on_book": 36,
    "Avg_Open_To_Buy": 8000,
    "Credit_Limit": 10000,
    "Avg_Utilization_Ratio": 0.3
}

for col in expected_features:
    if col not in user_input:
        user_input[col] = default_values.get(col, 0)

# === Step 4: Prediction Button ===
if st.button("Predict Churn"):
    input_df = pd.DataFrame([user_input])
    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)
    if "ANN" in model_type:
        prediction = (prediction > 0.5).astype(int)
    result = "Attrited Customer" if prediction[0] == 1 else "Existing Customer"
    st.success(f"### Prediction: {result}")
