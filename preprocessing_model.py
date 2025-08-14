# Integrated Telco Customer Churn Pipeline - Preprocessing & Feature Selection
# This script combines data preprocessing and feature selection into one continuous workflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PREPROCESSING PIPELINE
# ============================================================================

def load_and_preprocess_telco_data(file_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load and preprocess the Telco Customer Churn dataset
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    
    # 1. Load the CSV file
    print("STEP 1: Loading Telco Customer Churn dataset...")
    print("="*50)
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        print("Please ensure the CSV file is in the data/ folder")
        return None
    
    # Display basic info about the dataset
    print(f"\nDataset Info:")
    print(f"Columns: {list(df.columns)}")
    print(f"Target variable 'Churn' distribution:")
    print(df['Churn'].value_counts())
    
    # 2. Handle TotalCharges column - convert to numeric and handle missing values
    print("\nSTEP 2: Processing TotalCharges column...")
    print("-" * 30)
    
    # Check for non-numeric values in TotalCharges
    print(f"TotalCharges data type: {df['TotalCharges'].dtype}")
    non_numeric = df[df['TotalCharges'] == ' ']['TotalCharges'].count()
    print(f"Found {non_numeric} empty/space values in TotalCharges")
    
    # Replace empty strings with NaN and convert to numeric
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    print(f"Missing values in TotalCharges after conversion: {df['TotalCharges'].isnull().sum()}")
    
    # 3. Prepare features and target
    print("\nSTEP 3: Preparing features and target variable...")
    print("-" * 30)
    
    # Drop customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode target variable (Yes/No to 1/0)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # 4. Identify column types for preprocessing
    print("\nSTEP 4: Identifying column types...")
    print("-" * 30)
    
    # Get numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # 5. Create preprocessing pipeline
    print("\nSTEP 5: Creating preprocessing pipeline...")
    print("-" * 30)
    
    # Numerical pipeline: impute missing values then standardize
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing values then one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # 6. Split data into train/test sets (80/20 split)
    print("\nSTEP 6: Splitting data into train/test sets (80/20)...")
    print("-" * 30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    # 7. Fit the preprocessor and transform the data
    print("\nSTEP 7: Fitting preprocessor and transforming data...")
    print("-" * 30)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    # Numerical features keep their names
    num_feature_names = numerical_cols
    
    # Categorical features get new names from one-hot encoding
    cat_feature_names = []
    for i, col in enumerate(categorical_cols):
        encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        if hasattr(encoder, 'categories_'):
            # Get categories for this column (excluding the first one due to drop='first')
            categories = encoder.categories_[i][1:]  # Skip first category
            cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    all_feature_names = num_feature_names + cat_feature_names
    
    # Convert to DataFrames for easier handling
    X_train_processed = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=all_feature_names)
    
    # 8. Output processed data shapes and summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")
    print(f"Number of features after preprocessing: {len(all_feature_names)}")
    print(f"Target classes: {le_target.classes_}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, all_feature_names

# ============================================================================
# PART 2: FEATURE SELECTION AUTOMATION
# ============================================================================

def apply_feature_selection(X_train, X_test, y_train, feature_names, k=10):
    """
    Apply SelectKBest feature selection with mutual information
    
    Parameters:
    X_train: Training features (after preprocessing)
    X_test: Test features (after preprocessing)
    y_train: Training target
    feature_names: List of feature names
    k: Number of top features to select
    
    Returns:
    tuple: (X_train_selected, X_test_selected, selected_features, feature_scores, selector)
    """
    
    print(f"\n\nSTEP 8: FEATURE SELECTION - Selecting top {k} features...")
    print("="*60)
    
    # Apply SelectKBest with mutual_info_classif
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature information
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    feature_scores = selector.scores_[selected_indices]
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Mutual_Info_Score': feature_scores,
        'Rank': range(1, len(selected_features) + 1)
    }).sort_values('Mutual_Info_Score', ascending=False).reset_index(drop=True)
    
    print(f"Selected {len(selected_features)} features from {len(feature_names)} total features:")
    print(f"\nTop {k} Selected Features (by Mutual Information Score):")
    print("-" * 55)
    for i, (feature, score) in enumerate(zip(selected_features, 
                                           sorted(feature_scores, reverse=True)), 1):
        print(f"{i:2d}. {feature:<35} (score: {score:.4f})")
    
    # Convert selected data back to DataFrames
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    print(f"\nFeature selection complete!")
    print(f"Training data shape after selection: {X_train_selected.shape}")
    print(f"Test data shape after selection: {X_test_selected.shape}")
    
    return X_train_selected, X_test_selected, selected_features, feature_scores, selector, feature_importance_df

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name="Logistic Regression"):
    """
    Train and evaluate a logistic regression model
    
    Parameters:
    X_train: Training features
    X_test: Test features
    y_train: Training target
    y_test: Test target
    model_name: Name for the model
    
    Returns:
    tuple: (model, accuracy, predictions)
    """
    
    print(f"\nSTEP 9: Training {model_name}...")
    print("-" * 40)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 35)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    return model, accuracy, y_pred

def create_integrated_pipeline(k=10):
    """
    Create a complete integrated pipeline with preprocessing and feature selection
    
    Parameters:
    k (int): Number of top features to select
    
    Returns:
    Pipeline: Complete integrated pipeline
    """
    
    print(f"\nSTEP 10: Creating integrated pipeline...")
    print("-" * 40)
    
    # Define column types for Telco dataset
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Numerical preprocessing pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # Create complete pipeline: preprocessing -> feature selection -> model
    integrated_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=mutual_info_classif, k=k)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print(f"Integrated pipeline created with {k} feature selection!")
    print("Pipeline steps:")
    for i, (name, step) in enumerate(integrated_pipeline.steps, 1):
        print(f"  {i}. {name}: {step.__class__.__name__}")
    
    return integrated_pipeline

def compare_feature_selection_performance(X_train_full, X_test_full, y_train, y_test, 
                                        feature_names, k_values=[5, 10, 15, 20]):
    """
    Compare model performance with different numbers of selected features
    
    Parameters:
    X_train_full: Full training features (after preprocessing)
    X_test_full: Full test features (after preprocessing)  
    y_train: Training target
    y_test: Test target
    feature_names: List of all feature names
    k_values: List of k values to test
    
    Returns:
    DataFrame: Comparison results
    """
    
    print(f"\n\nSTEP 11: Comparing performance with different k values...")
    print("="*60)
    
    results = []
    
    # Test with all features (no selection)
    print(f"Testing with ALL features ({len(feature_names)} features)...")
    model_all, accuracy_all, _ = train_and_evaluate_model(
        X_train_full, X_test_full, y_train, y_test, "All Features"
    )
    
    results.append({
        'k_features': 'All',
        'num_features': len(feature_names),
        'accuracy': accuracy_all
    })
    
    # Test with different k values
    for k in k_values:
        if k >= len(feature_names):
            print(f"Skipping k={k} (exceeds total features: {len(feature_names)})")
            continue
            
        print(f"\nTesting with k={k} features...")
        
        # Apply feature selection
        X_train_sel, X_test_sel, selected_features, _, _, _ = apply_feature_selection(
            X_train_full, X_test_full, y_train, feature_names, k=k
        )
        
        # Train and evaluate model
        model, accuracy, _ = train_and_evaluate_model(
            X_train_sel, X_test_sel, y_train, y_test, f"Top {k} Features"
        )
        
        results.append({
            'k_features': k,
            'num_features': k,
            'accuracy': accuracy
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n" + "="*50)
    print("FEATURE SELECTION COMPARISON RESULTS")
    print("="*50)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Find best performing configuration
    best_idx = results_df['accuracy'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\nBest Performance:")
    print(f"Configuration: {best_result['k_features']} features")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    
    return results_df

def run_complete_integrated_pipeline(file_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv', 
                                   k=10, compare_k_values=True):
    """
    Run the complete integrated pipeline from preprocessing to feature selection and evaluation
    
    Parameters:
    file_path (str): Path to the CSV file
    k (int): Number of features to select
    compare_k_values (bool): Whether to compare different k values
    
    Returns:
    dict: Complete results
    """
    
    print("TELCO CUSTOMER CHURN - INTEGRATED PREPROCESSING & FEATURE SELECTION PIPELINE")
    print("="*80)
    
    # Part 1: Preprocessing
    preprocessing_result = load_and_preprocess_telco_data(file_path)
    if preprocessing_result is None:
        return None
    
    X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names = preprocessing_result
    
    # Part 2: Feature Selection
    X_train_selected, X_test_selected, selected_features, feature_scores, selector, feature_importance_df = apply_feature_selection(
        X_train_processed, X_test_processed, y_train, feature_names, k=k
    )
    
    # Part 3: Model Training and Evaluation
    model, accuracy, y_pred = train_and_evaluate_model(
        X_train_selected, X_test_selected, y_train, y_test
    )
    
    # Part 4: Create Integrated Pipeline
    integrated_pipeline = create_integrated_pipeline(k=k)
    
    # Demonstrate the integrated pipeline on raw data
    print(f"\nSTEP 12: Testing integrated pipeline on raw data...")
    print("-" * 50)
    
    # Load raw data again for pipeline testing
    try:
        df_raw = pd.read_csv(file_path)
        df_raw['TotalCharges'] = df_raw['TotalCharges'].replace(' ', np.nan)
        df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
        
        if 'customerID' in df_raw.columns:
            df_raw = df_raw.drop('customerID', axis=1)
        
        X_raw = df_raw.drop('Churn', axis=1)
        y_raw = LabelEncoder().fit_transform(df_raw['Churn'])
        
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
        )
        
        # Fit and test integrated pipeline
        integrated_pipeline.fit(X_train_raw, y_train_raw)
        pipeline_accuracy = integrated_pipeline.score(X_test_raw, y_test_raw)
        
        print(f"Integrated pipeline accuracy: {pipeline_accuracy:.4f}")
        print("✅ Integrated pipeline working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing integrated pipeline: {e}")
        pipeline_accuracy = None
    
    # Part 5: Optional comparison of different k values
    comparison_results = None
    if compare_k_values:
        comparison_results = compare_feature_selection_performance(
            X_train_processed, X_test_processed, y_train, y_test, 
            feature_names, k_values=[5, 10, 15, 20]
        )
    
    # Final Summary
    print(f"\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE - FINAL SUMMARY")
    print("="*80)
    print(f"📊 Original dataset shape: {X_train_processed.shape[0] + X_test_processed.shape[0]} rows")
    print(f"🔢 Features after preprocessing: {len(feature_names)}")
    print(f"⭐ Features selected: {len(selected_features)} (top {k})")
    print(f"🎯 Model accuracy with selected features: {accuracy:.4f}")
    if pipeline_accuracy:
        print(f"🔧 Integrated pipeline accuracy: {pipeline_accuracy:.4f}")
    
    print(f"\n📋 Top 5 Selected Features:")
    top_5_features = feature_importance_df.head(5)
    for i, row in top_5_features.iterrows():
        print(f"   {i+1}. {row['Feature']} (score: {row['Mutual_Info_Score']:.4f})")
    
    # Compile all results
    results = {
        'preprocessing': {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_names': feature_names
        },
        'feature_selection': {
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector,
            'feature_importance_df': feature_importance_df
        },
        'model': {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        },
        'integrated_pipeline': integrated_pipeline,
        'pipeline_accuracy': pipeline_accuracy,
        'comparison_results': comparison_results
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Run the complete integrated pipeline
    results = run_complete_integrated_pipeline(
        file_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        k=10,
        compare_k_values=True
    )
    
    if results:
        print(f"\n🎉 SUCCESS! All pipeline steps completed successfully.")
        print(f"💾 Results stored in 'results' dictionary with keys:")
        for key in results.keys():
            print(f"   - {key}")
        
        # Example of using the integrated pipeline for new predictions
        print(f"\n🔮 The integrated pipeline is ready for production use!")
        print(f"   Use: results['integrated_pipeline'].predict(new_data)")
    else:
        print(f"❌ Pipeline execution failed. Please check your data file path.")
