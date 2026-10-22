Customer Churn Prediction & Risk Analysis

Project Overview

This project predicts whether a customer is likely to leave a company using Machine Learning. It also calculates churn probability and classifies customers into Low, Medium, and High Risk categories.

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

Dataset

The dataset contains 64,374 customer records with information such as age, gender, tenure, support calls, payment delay, subscription type, contract length, and total spending.

Machine Learning Model

- Algorithm: Logistic Regression
- Training Data: 80%
- Testing Data: 20%
- Accuracy: 82.70%

Features

- Customer churn prediction
- Churn probability calculation
- Low, Medium, and High Risk classification
- Interactive Streamlit web application
- Data visualization and model evaluation

How to Run

1. Install the required libraries:

pip install -r requirements.txt

2. Train the model:

python train_model.py

3. Launch the Streamlit application:

python -m streamlit run app.py

Project Structure

Customer_Churn_Prediction/
├── data/
│   └── customer_churn.csv
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

Conclusion

This project demonstrates an end-to-end Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and deployment using Streamlit.