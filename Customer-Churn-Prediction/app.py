import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(page_title="Customer Churn Prediction")

# Load model
model = joblib.load("churn_model.pkl")

# Title
st.title("Customer Churn Prediction")

st.write("Enter customer details below:")

# Inputs
tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    step=1
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

# Create dataframe
input_data = pd.DataFrame({

    'tenure': [tenure],

    'Contract': [{
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }[contract]],

    'MonthlyCharges': [monthly_charges]

})

# Predict button
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer Will Churn")

    else:
        st.success("✅ Customer Will Stay")