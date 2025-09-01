# Customer Lifetime Value (CLV) Prediction  
Machine learning model to predict customer churn using Logistic Regression. Includes a Streamlit dashboard for interactive visualization and insights.

Customer churn prediction is vital for any subscription-based business that wants to:
- Identify customers at risk of leaving before they actually do
- Implement targeted retention strategies
- Reduce customer acquisition costs by retaining existing customers
- Optimize marketing spend on high-risk customer segments

This project uses machine learning to predict which customers are likely to churn and provides an interactive dashboard to explore the results. No more reactive approaches – just proactive customer retention backed by data.

## Project Structure  
- ├── data/                       # Customer datasets
- ├── results/                    # Model outputs and analysis results
- ├── Customer Churn Prediction Model.pdf # Project documentation
- ├── README.md                   # Project documentation
- ├── app.py                      # Streamlit dashboard application
- ├── preprocessing_model.py      # Data preprocessing and feature engineering
- ├── requirements.txt            # Project dependencies
- └── train_and_evaluate.py       # Model training and evaluation

## Getting Started
## Prerequisites
You'll need Python 3.7+ and the packages listed in requirements.txt. The usual suspects are there:
- pandas, numpy for data manipulation
- logistic regression for classification
- scikit-learn for machine learning
- matplotlib, seaborn for visualization
- streamlit for interactive dashboard

# Installation
1. Clone this repository:
git clone https://github.com/sdlk/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Install the required packages:
pip install -r requirements.txt

3. Run the automated pipeline:
python preprocessing_model.py
python train_and_evaluate.py

4. Launch the interactive dashboard:
streamlit run app.py

That's it! The dashboard will open in your browser with interactive churn prediction insights.

# How It Works
1. Data Preprocessing (preprocessing_model.py)
- Cleans and prepares customer data for modeling
- Handles missing values and categorical variables
- Creates feature engineering for customer behavior metrics
- Applies data scaling and normalization

2. Model Training & Evaluation (train_and_evaluate.py)
- Trains logistic regression model for churn prediction
- Performs model validation and performance evaluation
- Generates classification reports and confusion matrices
- Saves trained model for dashboard integration

3. Interactive Dashboard (app.py)
- Streamlit-based web application for model insights
- Interactive visualizations of customer segments
- Real-time churn probability predictions
- Customer filtering and analysis tools

# Key Features
- Interactive Dashboard: Streamlit-powered web interface for exploring predictions
- Logistic Regression Model: Interpretable machine learning for churn prediction
- Customer Segmentation: Visual analysis of different customer groups
- Real-time Predictions: Input customer data and get instant churn probability
- Performance Metrics: Comprehensive model evaluation and validation
- Visualization Tools: Charts and graphs for data exploration

# Sample Dashboard Features
The Streamlit dashboard includes:
- Customer Overview: Summary statistics and churn distribution
- Prediction Interface: Input customer details for churn probability
- Feature Importance: Which factors most influence churn decisions
- Customer Segments: Visual breakdown of high-risk vs. low-risk customers
- Model Performance: Accuracy metrics, ROC curves, and validation results

# Model Performance
The logistic regression model provides:
- Interpretability: Clear understanding of which features drive churn
- Probability Scores: Not just yes/no, but likelihood percentages
- Feature Coefficients: Quantified impact of each customer attribute
- Statistical Significance: Confidence levels for predictions

# Typical performance metrics:
- Accuracy: ~82-88%
- Precision: ~79-85%
- Recall: ~76-82%
- F1-Score: ~77-83%

# Dashboard Usage
Once you run streamlit run app.py, you can:
- Explore Data: View customer distributions and patterns
- Make Predictions: Input customer characteristics for churn probability
- Analyze Segments: Compare different customer groups
- Understand Features: See which factors matter most for retention

# Data Requirements
Your customer dataset should include:
- Demographics: Age, gender, location
- Account Information: Tenure, contract type, payment method
- Service Usage: Monthly charges, total charges, service subscriptions
- Support History: Customer service calls, technical issues
- Target Variable: Churn indicator (Yes/No)

# Contributing
Found a bug or have an idea for improvement? Feel free to:
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

# Technical Notes
- The model uses logistic regression for interpretability in business contexts
- Feature engineering includes customer lifetime metrics and usage patterns
- The Streamlit dashboard supports real-time model inference
- All preprocessing steps are automated and reproducible
- Model coefficients provide clear business insights for retention strategies
