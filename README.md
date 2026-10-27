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

This project uses a Customer Churn Dataset obtained from Kaggle.

The dataset contains 64,374 customer records with information such as age, gender, tenure, support calls, payment delay, subscription type, contract length, and total spending.

### Dataset Source

- Platform: Kaggle
- Dataset: Customer Churn Dataset
- Source: Kaggle (Original dataset URL not available)

*Note:* The dataset is not included in this repository. It was obtained from Kaggle and used for educational and machine learning purposes. Dataset ownership and licensing rights remain with the original creator.
  
How to Download and Use the Dataset

1. Visit the Kaggle dataset link provided above.
2. Download the Customer Churn Dataset.
3. Extract the downloaded ZIP file if necessary.
4. Create a folder named "data" inside the project directory.
5. Place the downloaded CSV file inside the "data" folder.
6. Rename the CSV file to "customer_churn.csv" if necessary.
7. Run the model training script to train the machine learning model.

Dataset Note: The dataset is not included in this GitHub repository. Users must download it separately from the original Kaggle source and comply with its license and usage terms.

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

Step 1: Install the Required Libraries

pip install -r requirements.txt

Step 2: Download the Dataset

Download the dataset from the Kaggle source link mentioned in the Dataset section.

Place the CSV file inside the "data" folder and ensure its filename is "customer_churn.csv".

Step 3: Train the Model

python train_model.py

Step 4: Launch the Streamlit Application

python -m streamlit run app.py

The Streamlit application will open in your browser.

Project Structure

Customer_Churn_Prediction/
│
├── data/
│
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md

Note: The "data" folder should be created locally after downloading the dataset. GitHub does not preserve empty folders.

Conclusion

This project demonstrates an end-to-end Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, and deployment using Streamlit.

The project provides an interactive application for predicting customer churn probability and identifying customer risk levels.

The dataset is referenced through its original Kaggle source to respect dataset ownership and licensing requirements.
