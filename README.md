# Customer Churn Prediction

A Python-based project to predict customer churn using Logistic Regression, with an interactive Streamlit dashboard.

## Features
- Automated preprocessing & feature selection
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
   ```bash
   pip install -r requirements.txt
2. Train the model:
    python train_and_evaluate.py
3. Launch the dashboard:
    streamlit run app.py

## Dataset
Telco Customer Churn dataset from Kaggle.
