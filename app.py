import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction & Risk Analysis")

st.write("Machine Learning Based Customer Churn Prediction")

# Load model
model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

st.subheader("Enter Customer Details")

# Remove target and ID
X = df.drop(columns=["CustomerID", "Churn"], errors="ignore")

customer_data = {}

# Create customer input fields
for column in X.columns:

    numeric_data = pd.to_numeric(
        X[column], errors="coerce"
    )

    if numeric_data.notna().all():

        customer_data[column] = st.number_input(
            column,
            value=float(numeric_data.median())
        )

    else:

        options = X[column].dropna().astype(str).unique().tolist()

        customer_data[column] = st.selectbox(
            column,
            options
        )

# Prediction
if st.button("Predict Customer Churn"):

    input_df = pd.DataFrame([customer_data])

    # Convert categorical data
    for column in X.columns:

        if column in input_df.columns:

            if not pd.to_numeric(
                X[column], errors="coerce"
            ).notna().all():

                categories = sorted(
                    X[column].dropna().astype(str).unique()
                )

                input_df[column] = pd.Categorical(
                    input_df[column],
                    categories=categories
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