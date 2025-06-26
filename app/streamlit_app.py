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

# === Step 2: Advanced Inputs from Top Features ===
with st.expander("Optional: Advanced Transactional Features"):
    user_input["Total_Trans_Ct"] = st.number_input("Total_Trans_Ct", value=50)
    user_input["Total_Revolving_Bal"] = st.number_input("Total_Revolving_Bal", value=500.0)
    user_input["Total_Relationship_Count"] = st.number_input("Total_Relationship_Count", value=3)
    user_input["Total_Trans_Amt"] = st.number_input("Total_Trans_Amt", value=5000.0)
    user_input["Transaction_Intensity"] = st.number_input("Transaction_Intensity", value=5.0)
    user_input["Total_Ct_Chng_Q4_Q1"] = st.number_input("Total_Ct_Chng_Q4_Q1", value=1.2)
    user_input["Months_Inactive_12_mon"] = st.number_input("Months_Inactive_12_mon", value=2)
    user_input["Contacts_Count_12_mon"] = st.number_input("Contacts_Count_12_mon", value=2)
    user_input["Total_Amt_Chng_Q4_Q1"] = st.number_input("Total_Amt_Chng_Q4_Q1", value=1.5)

# === Step 3: Fill remaining required features with defaults ===
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
