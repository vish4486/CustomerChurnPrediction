import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Load Model & Metadata ===
with open("models/best_model_meta.json", "r") as f:
    meta = json.load(f)

model_type = meta["best_model_type"]
timestamp = meta["timestamp"]
top_features = meta.get("top_features", [])

model_path = "models/best_model.pkl" if "ANN" not in model_type else f"models/ann_model_{timestamp}.keras"
model = load_model(model_path) if "ANN" in model_type else joblib.load(model_path)
preprocessor = joblib.load("data/processed/preprocessor.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🔍 Credit Card Customer Churn Prediction")
st.markdown("Use the form below to predict if a customer is likely to **churn**.")

# === Input Modes ===
st.subheader("Choose Input Mode")

example = st.radio("📁 Load an example input or enter manually:", ["Manual Input", "Use Existing Customer Example", "Use Attrited Customer Example"])

# === Default Values ===
default_values = {
    "Dependent_count": 2,
    "Education_Level": "Uneducated",
    "Months_on_book": 36,
    "Avg_Open_To_Buy": 8000,
    "Credit_Limit": 10000,
    "Avg_Utilization_Ratio": 0.3
}

example_inputs = {
    "Use Existing Customer Example": {
        "Customer_Age": 45, "Gender": "M", "Marital_Status": "Married", "Income_Category": "$80K - $120K", "Card_Category": "Blue",
        "Total_Trans_Ct": 80, "Total_Revolving_Bal": 500.0, "Total_Relationship_Count": 5, "Total_Trans_Amt": 12000.0,
        "Transaction_Intensity": 8.0, "Total_Ct_Chng_Q4_Q1": 1.5, "Months_Inactive_12_mon": 1,
        "Contacts_Count_12_mon": 1, "Total_Amt_Chng_Q4_Q1": 1.4
    },
    "Use Attrited Customer Example": {
        "Customer_Age": 60, "Gender": "F", "Marital_Status": "Single", "Income_Category": "Less than $40K", "Card_Category": "Platinum",
        "Total_Trans_Ct": 10, "Total_Revolving_Bal": 2000.0, "Total_Relationship_Count": 1, "Total_Trans_Amt": 500.0,
        "Transaction_Intensity": 1.0, "Total_Ct_Chng_Q4_Q1": 0.1, "Months_Inactive_12_mon": 6,
        "Contacts_Count_12_mon": 4, "Total_Amt_Chng_Q4_Q1": 0.3
    }
}

user_input = {}

# === Sidebar Input ===
with st.sidebar:
    st.header("🔧 Basic Inputs")
    if example == "Manual Input":
        user_input["Customer_Age"] = st.slider("Customer Age", 18, 100, 45)
        user_input["Gender"] = st.selectbox("Gender", ["M", "F"])
        user_input["Marital_Status"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        user_input["Income_Category"] = st.selectbox("Income Category", [
            "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"
        ])
        user_input["Card_Category"] = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
    else:
        user_input = example_inputs[example].copy()

# === Optional Advanced Section ===
with st.expander("⚙️ Advanced Transactional Features (Optional)", expanded=(example == "Manual Input")):
    for feature in top_features:
        default_val = user_input.get(feature, 1.0)

        # Fix for Streamlit numeric input error: ensure consistent float type
        if isinstance(default_val, int):
            default_val = float(default_val)

        if feature.startswith("Card_Category_"):
            index = int(default_val) if default_val in [0, 1] else 0
            user_input[feature] = st.selectbox(
                feature, [0, 1], index=index, format_func=lambda x: "Yes" if x == 1 else "No"
            )
        elif feature in ["Transaction_Intensity", "Total_Amt_Chng_Q4_Q1", "Total_Ct_Chng_Q4_Q1"]:
            user_input[feature] = st.number_input(feature, min_value=0.0, step=0.01, value=float(default_val))
        else:
            user_input[feature] = st.number_input(feature, step=1.0, value=float(default_val))

# === Final Touch: Fill All Required Fields ===
expected_features = list(preprocessor.feature_names_in_)
for col in expected_features:
    if col not in user_input:
        user_input[col] = default_values.get(col, 0)

# === Prediction ===
if st.button("📤 Predict Churn"):
    input_df = pd.DataFrame([user_input])
    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)
    if "ANN" in model_type or "XGBoost" in model_type:
        prediction = (prediction > 0.5).astype(int)

    result = "Attrited Customer" if prediction[0] == 1 else "Existing Customer"
    emoji = "❌" if prediction[0] == 1 else "✅"
    st.success(f"{emoji} Prediction: **{result}**")

    # Log
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/predictions.csv"
    input_df["Prediction"] = result
    pd.concat([pd.read_csv(log_file), input_df] if os.path.exists(log_file) else [input_df]).to_csv(log_file, index=False)

    # Dashboard
    st.markdown("---")
    st.subheader("📊 Monitoring Dashboard")
    logs = pd.read_csv(log_file)
    col1, col2 = st.columns(2)
    with col1:
        counts = logs["Prediction"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)
    with col2:
        f = top_features[0] if top_features else "Total_Trans_Ct"
        fig2, ax2 = plt.subplots()
        logs[f].hist(ax=ax2, bins=15, color="skyblue")
        ax2.set_title(f"{f} Distribution")
        st.pyplot(fig2)
