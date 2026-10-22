import pandas as pd

# Step 1: Load Dataset
df = pd.read_csv("data/customer_churn.csv")

# Step 2: Display Dataset
print("Dataset Loaded Successfully!")

print("\nFirst 5 Rows:")
print(df.head())

# Step 3: Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Step 4: Column Names
print("\nColumn Names:")
print(df.columns)

# Step 5: Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 6: Churn Count
print("\nChurn Count:")
print(df["Churn"].value_counts())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check total customers
print("\nTotal Customers:")
print(len(df))

# Step 6: Customer Churn Analysis

print("\nCustomer Churn Count:")
print(df["Churn"].value_counts())

print("\nCustomer Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Step 7: Customer Churn Visualization

import matplotlib.pyplot as plt

churn_counts = df["Churn"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Stayed", "Churned"],
    churn_counts.values,
    color=["green", "red"]
)

plt.title("Customer Churn Analysis")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# Step 8: Support Calls vs Customer Churn

import matplotlib.pyplot as plt

support_churn = df.groupby("Support Calls")["Churn"].mean() * 100

plt.figure(figsize=(8, 5))

plt.bar(support_churn.index, support_churn.values)

plt.title("Support Calls vs Customer Churn")
plt.xlabel("Number of Support Calls")
plt.ylabel("Churn Percentage (%)")

plt.tight_layout()
plt.show()

# Step 9: Data Preprocessing

# Separate input features and target
X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"]

# Convert categorical data into numbers
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nData Preprocessing Completed!")

print("Input Features:", X.shape)
print("Target:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed!")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 11: Logistic Regression Model Training

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nMachine Learning Model Trained Successfully!")

# Step 12: Model Evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Predict customer churn
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation Results")

print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

# Step 13: Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
)

disp.plot(cmap="Blues")

plt.title("Customer Churn Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 14: Customer Churn Risk Prediction

# Predict churn probability
churn_probability = model.predict_proba(X_test)[:, 1]

# Create prediction results
results = X_test.copy()

results["Actual_Churn"] = y_test
results["Predicted_Churn"] = y_pred
results["Churn_Probability"] = churn_probability

# Classify customer risk
def risk_level(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

results["Risk_Level"] = results["Churn_Probability"].apply(risk_level)

print("\nCustomer Churn Risk Prediction:")
print(results[[
    "Actual_Churn",
    "Predicted_Churn",
    "Churn_Probability",
    "Risk_Level"
]].head(10))

# Step 15: Save Trained ML Model

import joblib
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/churn_model.pkl")

# Save feature column names
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nModel Saved Successfully!")