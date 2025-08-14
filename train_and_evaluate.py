# Telco Customer Churn Dataset - Model Training & Evaluation
# This script provides comprehensive model evaluation with multiple metrics and visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def create_complete_pipeline(use_feature_selection=True, k=10):
    """
    Create a complete ML pipeline with preprocessing, optional feature selection, and model
    
    Parameters:
    use_feature_selection (bool): Whether to include feature selection
    k (int): Number of features to select (if feature selection is used)
    
    Returns:
    Pipeline: Complete ML pipeline
    """
    
    # Define column types for Telco dataset
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # Create pipeline steps
    pipeline_steps = [('preprocessor', preprocessor)]
    
    if use_feature_selection:
        pipeline_steps.append(('feature_selection', SelectKBest(score_func=mutual_info_classif, k=k)))
    
    pipeline_steps.append(('classifier', LogisticRegression(random_state=42, max_iter=1000)))
    
    return Pipeline(pipeline_steps)

def load_and_prepare_data(file_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load and prepare the Telco dataset
    """
    print("Loading Telco Customer Churn dataset...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None, None
    
    # Data preprocessing
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    return X, y

def evaluate_model_comprehensive(pipeline, X_test, y_test, y_pred, y_pred_proba):
    """
    Perform comprehensive model evaluation
    
    Parameters:
    pipeline: Trained pipeline
    X_test: Test features
    y_test: True test labels
    y_pred: Predicted labels
    y_pred_proba: Predicted probabilities
    
    Returns:
    dict: Dictionary with all evaluation metrics
    """
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    return metrics

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_roc_curve(y_test, y_pred_proba, title="ROC Curve"):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_metrics_table(metrics_dict, model_name="Logistic Regression"):
    """
    Create a neat table format for metrics
    """
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [f"{metrics_dict['Accuracy']:.4f}"],
        'Precision': [f"{metrics_dict['Precision']:.4f}"],
        'Recall': [f"{metrics_dict['Recall']:.4f}"],
        'F1-Score': [f"{metrics_dict['F1-Score']:.4f}"],
        'ROC-AUC': [f"{metrics_dict['ROC-AUC']:.4f}"]
    })
    
    return metrics_df

def run_complete_model_evaluation(file_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv', 
                                 use_feature_selection=True, k=10):
    """
    Run complete model training and evaluation pipeline
    
    Parameters:
    file_path (str): Path to the CSV file
    use_feature_selection (bool): Whether to use feature selection
    k (int): Number of features to select
    
    Returns:
    dict: Complete evaluation results
    """
    
    print("TELCO CUSTOMER CHURN - MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # Load and prepare data
    X, y = load_and_prepare_data(file_path)
    if X is None:
        return None
    
    # Split data
    print(f"\nSplitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Create and train pipeline
    feature_selection_text = f"with top {k} features" if use_feature_selection else "with all features"
    print(f"\nTraining Logistic Regression model {feature_selection_text}...")
    
    pipeline = create_complete_pipeline(use_feature_selection=use_feature_selection, k=k)
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_model_comprehensive(pipeline, X_test, y_test, y_pred, y_pred_proba)
    
    # Display metrics in table format
    metrics_table = create_metrics_table(metrics)
    print(f"\nMODEL PERFORMANCE METRICS:")
    print("="*50)
    print(metrics_table.to_string(index=False))
    
    # Cross-validation scores
    print(f"\nCross-Validation Results (5-fold):")
    print("-" * 35)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    cv_scores_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    print(f"CV F1-Score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std() * 2:.4f})")
    
    # Detailed classification report
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print("="*45)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Plot confusion matrix
    print("\nGenerating Confusion Matrix...")
    cm = plot_confusion_matrix(y_test, y_pred, "Logistic Regression - Confusion Matrix")
    
    # Calculate additional confusion matrix metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"\nConfusion Matrix Analysis:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    
    # Plot ROC curve
    print("\nGenerating ROC Curve...")
    plot_roc_curve(y_test, y_pred_proba, "Logistic Regression - ROC Curve")
    
    # Plot Precision-Recall curve
    print("Generating Precision-Recall Curve...")
    plot_precision_recall_curve(y_test, y_pred_proba, "Logistic Regression - Precision-Recall Curve")
    
    # Feature importance analysis (if using feature selection)
    if use_feature_selection:
        print(f"\nFEATURE SELECTION ANALYSIS:")
        print("="*40)
        
        # Get selected features
        selector = pipeline.named_steps['feature_selection']
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_[selected_indices]
        
        # Get feature names (simplified approach)
        # Note: This is a simplified version. In practice, you'd want to track feature names through preprocessing
        print(f"Selected {len(selected_indices)} features out of total available features")
        print(f"Top feature scores: {sorted(feature_scores, reverse=True)[:5]}")
        
        # Model coefficients
        coefficients = pipeline.named_steps['classifier'].coef_[0]
        print(f"\nModel Coefficients (top 5 by absolute value):")
        coef_indices = np.argsort(np.abs(coefficients))[-5:][::-1]
        for i, idx in enumerate(coef_indices):
            print(f"{i+1}. Feature {idx}: {coefficients[idx]:.4f}")
    
    # Business insights
    print(f"\nBUSINESS INSIGHTS:")
    print("="*30)
    churn_rate = y_test.mean()
    predicted_churn_rate = y_pred.mean()
    
    print(f"Actual churn rate in test set: {churn_rate:.1%}")
    print(f"Predicted churn rate: {predicted_churn_rate:.1%}")
    print(f"Model identifies {tp} out of {tp + fn} actual churners ({tp/(tp+fn):.1%})")
    print(f"Model incorrectly flags {fp} non-churners as churners")
    
    # Cost-benefit analysis (example)
    retention_cost = 50  # Cost to retain a customer
    churn_cost = 500     # Cost of losing a customer
    
    # True positives: Correctly identified churners (we can try to retain them)
    benefit_tp = tp * (churn_cost - retention_cost)
    
    # False positives: Incorrectly identified non-churners (unnecessary retention cost)
    cost_fp = fp * retention_cost
    
    # False negatives: Missed churners (lost customers)
    cost_fn = fn * churn_cost
    
    net_benefit = benefit_tp - cost_fp - cost_fn
    
    print(f"\nCOST-BENEFIT ANALYSIS (Example):")
    print("="*35)
    print(f"Assumption: Retention cost = ${retention_cost}, Churn cost = ${churn_cost}")
    print(f"Benefit from correctly identified churners: ${benefit_tp:,.2f}")
    print(f"Cost from false positives: ${cost_fp:,.2f}")
    print(f"Cost from missed churners: ${cost_fn:,.2f}")
    print(f"Net benefit: ${net_benefit:,.2f}")
    
    # Compile all results
    results = {
        'metrics': metrics,
        'metrics_table': metrics_table,
        'confusion_matrix': cm,
        'cv_scores': cv_scores,
        'pipeline': pipeline,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'business_metrics': {
            'actual_churn_rate': churn_rate,
            'predicted_churn_rate': predicted_churn_rate,
            'net_benefit': net_benefit,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
    }
    
    return results

def compare_models_with_without_feature_selection(file_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Compare model performance with and without feature selection
    """
    print("\nCOMPARING MODELS: WITH vs WITHOUT FEATURE SELECTION")
    print("="*60)
    
    # Load data
    X, y = load_and_prepare_data(file_path)
    if X is None:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results_comparison = []
    
    # Test with different configurations
    configs = [
        {'use_feature_selection': False, 'k': None, 'name': 'All Features'},
        {'use_feature_selection': True, 'k': 5, 'name': 'Top 5 Features'},
        {'use_feature_selection': True, 'k': 10, 'name': 'Top 10 Features'},
        {'use_feature_selection': True, 'k': 15, 'name': 'Top 15 Features'},
    ]
    
    for config in configs:
        print(f"\nTraining model with {config['name']}...")
        
        pipeline = create_complete_pipeline(
            use_feature_selection=config['use_feature_selection'], 
            k=config['k']
        )
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model_comprehensive(pipeline, X_test, y_test, y_pred, y_pred_proba)
        
        result = {
            'Model': config['name'],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score'],
            'ROC-AUC': metrics['ROC-AUC']
        }
        
        results_comparison.append(result)
    
    # Create comparison table
    comparison_df = pd.DataFrame(results_comparison)
    
    print(f"\nMODEL COMPARISON RESULTS:")
    print("="*80)
    print(comparison_df.round(4).to_string(index=False))
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    print(f"\nBest performing model: {best_model['Model']} (F1-Score: {best_model['F1-Score']:.4f})")
    
    return comparison_df

# Example usage
if __name__ == "__main__":
    # Run complete model evaluation
    results = run_complete_model_evaluation(use_feature_selection=True, k=10)
    
    if results:
        print(f"\n\nEVALUATION COMPLETE!")
        print("="*30)
        print("Key takeaways:")
        print(f"- Model Accuracy: {results['metrics']['Accuracy']:.3f}")
        print(f"- Model F1-Score: {results['metrics']['F1-Score']:.3f}")
        print(f"- Model ROC-AUC: {results['metrics']['ROC-AUC']:.3f}")
        print(f"- Business Net Benefit: ${results['business_metrics']['net_benefit']:,.2f}")
        
        # Compare different approaches
        print(f"\n" + "="*60)
        comparison_results = compare_models_with_without_feature_selection()
        
        if comparison_results is not None:
            print(f"\nRecommendation: Use the model configuration with the highest F1-Score")
            print(f"for balanced performance in identifying churners while minimizing false positives.")