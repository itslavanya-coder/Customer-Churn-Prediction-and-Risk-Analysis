import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction & Risk Analysis")
st.write("Machine Learning Based Customer Churn Prediction")

# Load trained model
model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Load input configuration
with open("input_config.json", "r") as f:
    config = json.load(f)

st.subheader("Enter Customer Details")

customer_data = {}

# Customer input fields
for column, details in config.items():

    if details["type"] == "number":
        customer_data[column] = st.number_input(
            column,
            value=float(details["default"])
        )

    else:
        customer_data[column] = st.selectbox(
            column,
            details["options"]
        )

# Prediction
if st.button("Predict Customer Churn"):

    input_df = pd.DataFrame([customer_data])

    # Convert categorical columns
    for column, details in config.items():

        if details["type"] == "category":
            input_df[column] = pd.Categorical(
                input_df[column],
                categories=details["options"]
            )

    input_df = pd.get_dummies(
        input_df,
        drop_first=True,
        dtype=int
    )

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer May Leave the Company")
    else:
        st.success("Customer May Stay with the Company")

    st.write(
        f"Churn Probability: {probability * 100:.2f}%"
    )

    st.subheader("Risk Level")

    if probability >= 0.7:
        st.error("High Risk")
    elif probability >= 0.4:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")

    st.progress(float(probability))
