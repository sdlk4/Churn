# Customer Churn Prediction

A Python-based project to predict customer churn using Logistic Regression, with an interactive Streamlit dashboard.

## Features
- Automated data preprocessing & feature selection
- Logistic Regression model with evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Interactive dashboard with Plotly visualizations
- CSV upload for new customer churn prediction

## Project Structure
Churnproject/
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── preprocessing_model.py
├── train_and_evaluate.py
├── app.py
├── requirements.txt
└── README.md

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Train the model:
    python train_and_evaluate.py
3. Launch the dashboard:
    streamlit run app.py

## Dataset
The project uses the Telco Customer Churn dataset from Kaggle.
This dataset contains customer details such as demographics, account information, and whether they churned or not.

## Model Performance
The Logistic Regression model was trained and tested on the dataset.
Here are the results:
Accuracy: 0.796
Precision: 0.645
Recall: 0.516
F1-Score: 0.574
ROC-AUC: 0.832

## Interpretation of results:
Accuracy of 79% means overall predictions are fairly correct.
Precision of 0.645 shows that when the model predicts churn, it’s correct ~64% of the time.
Recall of 0.516 means the model detects about 51% of actual churn cases (it misses some churners).
F1-Score of 0.574 balances both precision and recall.
ROC-AUC of 0.832 indicates the model is quite good at distinguishing churn vs. non-churn customers.

## Tools and Technologies Used
Python (Pandas, NumPy, Scikit-learn, Plotly, Streamlit)
Logistic Regression as the machine learning algorithm
Streamlit for building an interactive dashboard
Kaggle dataset as the data source

This project demonstrates how to build, train, and deploy a simple customer churn prediction system with a clean dashboard.
