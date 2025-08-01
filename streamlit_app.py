import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the model and scaler
model = load_model("model_churn.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn.")

st.subheader("📋 Customer Details")

# Input features
SeniorCitizen = st.number_input("Senior Citizen (0 or 1)", 0, 1)
tenure = st.number_input("Tenure (in months)", 0)
MonthlyCharges = st.number_input("Monthly Charges", 0.0)
TotalCharges = st.number_input("Total Charges", 0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
])

# Mapping binary values
binary_map = {"Yes": 1, "No": 0, "Male": 0, "Female": 1}

input_data = [
    binary_map[gender],
    SeniorCitizen,
    binary_map[Partner],
    binary_map[Dependents],
    tenure,
    binary_map[PhoneService],
    binary_map[MultipleLines],
    binary_map[OnlineSecurity],
    binary_map[OnlineBackup],
    binary_map[DeviceProtection],
    binary_map[TechSupport],
    binary_map[StreamingTV],
    binary_map[StreamingMovies],
    binary_map[PaperlessBilling],
    MonthlyCharges,
    TotalCharges,

    # Internet Service One-hot
    1 if InternetService == "DSL" else 0,
    1 if InternetService == "Fiber optic" else 0,
    1 if InternetService == "No" else 0,

    # Contract One-hot
    1 if Contract == "Month-to-month" else 0,
    1 if Contract == "One year" else 0,
    1 if Contract == "Two year" else 0,

    # Payment Method One-hot
    1 if PaymentMethod == "Bank transfer (automatic)" else 0,
    1 if PaymentMethod == "Credit card (automatic)" else 0,
    1 if PaymentMethod == "Electronic check" else 0,
    1 if PaymentMethod == "Mailed check" else 0
]

if st.button("🔍 Predict Churn"):
    try:
        input_np = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_np)
        prediction = model.predict(input_scaled)[0][0]
        st.success(f"🔁 Probability of Churn: {prediction * 100:.2f}%")

        if prediction > 0.5:
            st.error("❌ Customer is likely to churn.")
        else:
            st.info("✅ Customer is likely to stay.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
