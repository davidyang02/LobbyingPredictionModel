#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lobbying Prediction Model
=========================

A machine learning project that predicts lobbying success using Canadian 
lobbying registrations data. The model determines whether a lobbying attempt 
will receive government funding based on various features.

Author: David Yang
License: MIT
"""

# %% [markdown]
# # 1. Setup and Imports

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # 2. Data Loading
# 
# Load the Canadian cities reference data and the main lobbying dataset.

# %%
def load_data():
    """Load and return the datasets."""
    # Load Canadian cities for geographic mapping
    canada_cities = pd.read_csv("canadacities.csv")
    
    # Load main lobbying dataset
    # NOTE: 'merged lobby.csv' must be obtained from the Office of the 
    # Commissioner of Lobbying of Canada (https://lobbycanada.gc.ca/en/)
    df = pd.read_csv("merged lobby.csv", encoding='latin-1')
    
    print(f"Loaded {len(df):,} lobbying registrations")
    print(f"Loaded {len(canada_cities):,} Canadian cities")
    
    return df, canada_cities

# %% [markdown]
# # 3. Data Exploration

# %%
def explore_data(df):
    """Print basic data exploration information."""
    print("Dataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print("\nNumerical Statistics:")
    print(df.describe())
    print("\nMissing Values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

# %% [markdown]
# # 4. Feature Engineering
# 
# Prepare features for model training by:
# - Mapping addresses to cities using area codes as region proxy
# - Encoding the target variable (government funding indicator)
# - Dropping irrelevant columns

# %%
def prepare_features(df):
    """
    Prepare feature set and target variable for modeling.
    
    Returns:
        features: DataFrame with predictor variables
        target: Series with binary target variable
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Convert target variable to binary (Y=1, N=0)
    df['GOVT_FUND_IND_FIN_GOUV'] = df['GOVT_FUND_IND_FIN_GOUV'].replace({'Y': 1, 'N': 0})
    
    # Extract area code from phone number as region proxy
    if 'RGSTRNT_TEL_DCLRNT' in df.columns:
        df['RGSTRNT_TEL_DCLRNT'] = df['RGSTRNT_TEL_DCLRNT'].astype(str).str[:3]
    
    # Define target
    target = df['GOVT_FUND_IND_FIN_GOUV']
    
    # Columns to drop (identifiers, text fields, dates)
    columns_to_drop = [
        'ï»¿REG_ID_ENR', 'REG_TYPE_ENR', 'REG_NUM_ENR', 'VERSION_CODE',
        'EN_FIRM_NM_FIRME_AN', 'FR_FIRM_NM_FIRME', 'RGSTRNT_POS_POSTE_DCLRNT',
        'FIRM_ADDRESS_ADRESSE_FIRME', 'FIRM_TEL_FIRME', 'FIRM_FAX_FIRME',
        'RGSTRNT_NUM_DECLARANT', 'RGSTRNT_LAST_NM_DCLRNT', 
        'RGSTRNT_1ST_NM_PRENOM_DCLRNT', 'RO_POS_POSTE_AR', 'RGSTRNT_FAX_DCLRNT',
        'CLIENT_ORG_CORP_PROFIL_ID_PROFIL_CLIENT_ORG_CORP',
        'CLIENT_ORG_CORP_NUM', 'EN_CLIENT_ORG_CORP_NM_AN', 'FR_CLIENT_ORG_CORP_NM',
        'CLIENT_ORG_CORP_ADDRESS_ADRESSE_CLIENT_ORG_CORP', 'CLIENT_ORG_CORP_TEL',
        'CLIENT_ORG_CORP_FAX', 'REP_LAST_NM_REP', 'REP_1ST_NM_PRENOM_REP',
        'REP_POSITION_POSTE_REP', 'EFFECTIVE_DATE_VIGUEUR', 'END_DATE_FIN',
        'FY_END_DATE_FIN_EXERCICE', 'CONTG_FEE_IND_HON_COND',
        'PREV_REG_ID_ENR_PRECEDNT', 'POSTED_DATE_PUBLICATION',
        'GOVT_FUND_IND_FIN_GOUV',  # This is the target
        'RGSTRNT_ADDRESS_ADRESSE_DCLRNT'  # Address field after extraction
    ]
    
    # Keep only columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    features = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    print(f"Features shape: {features.shape}")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")
    
    return features, target

# %% [markdown]
# # 5. Preprocessing Pipeline
# 
# Create preprocessing pipelines for numerical and categorical features:
# - Numerical: Impute missing values with mean, then scale
# - Categorical: Impute missing values with mode, then one-hot encode

# %%
def create_preprocessor(features):
    """
    Create a preprocessing pipeline for feature transformation.
    
    Args:
        features: DataFrame with predictor variables
        
    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Identify feature types
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

# %% [markdown]
# # 6. Model Training
# 
# Train a Random Forest Classifier with random undersampling to handle 
# class imbalance (approximately 33/67 distribution).

# %%
def train_model(X_train, y_train, preprocessor):
    """
    Train a Random Forest model with undersampling for class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: ColumnTransformer for preprocessing
        
    Returns:
        Trained pipeline model
    """
    # Handle class imbalance with random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"Original training samples: {len(X_train)}")
    print(f"Resampled training samples: {len(X_resampled)}")
    print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    # Create and train the model
    model = make_pipeline(
        preprocessor, 
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    
    model.fit(X_resampled, y_resampled)
    
    return model

# %% [markdown]
# # 7. Model Evaluation
# 
# Evaluate model performance using:
# - Accuracy, Precision, Recall, F1 Score
# - Confusion Matrix visualization
# - ROC Curve and AUC

# %%
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("="*50 + "\n")
    
    return metrics, y_pred

# %%
def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()

# %%
def plot_roc_curve(model, X_test, y_test, save_path=None):
    """Plot ROC curve with AUC score."""
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.show()
    
    return roc_auc

# %% [markdown]
# # 8. Feature Importance Analysis
# 
# Analyze which features are most predictive and key patterns for 
# positive predictions (government funding approved).

# %%
def analyze_predictions(X_test, y_pred):
    """
    Analyze top predictive features for positive class predictions.
    
    Args:
        X_test: Test features
        y_pred: Model predictions
    """
    # Combine test data with predictions
    df_predictions = X_test.copy()
    df_predictions['Prediction'] = y_pred
    
    # Filter for positive predictions (Class = 1)
    df_positive = df_predictions[df_predictions['Prediction'] == 1]
    
    print("\n" + "="*60)
    print("TOP 3 MOST COMMON VALUES PER FEATURE (Government Funding = Yes)")
    print("="*60)
    
    for feature in df_positive.columns[:-1]:  # Exclude 'Prediction' column
        print(f"\n{feature}:")
        value_counts = df_positive[feature].value_counts().head(3)
        total = len(df_positive)
        for value, count in value_counts.items():
            percentage = count / total * 100
            print(f"  {value}: {count:,} ({percentage:.2f}%)")

# %% [markdown]
# # 9. Main Execution

# %%
def main():
    """Main execution pipeline."""
    
    # 1. Load data
    print("Loading data...")
    df, canada_cities = load_data()
    
    # 2. Explore data (optional)
    # explore_data(df)
    
    # 3. Prepare features and target
    print("\nPreparing features...")
    features, target = prepare_features(df)
    
    # 4. Create preprocessor
    print("\nCreating preprocessing pipeline...")
    preprocessor = create_preprocessor(features)
    
    # 5. Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # 6. Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, preprocessor)
    
    # 7. Evaluate model
    print("\nEvaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # 8. Visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    roc_auc = plot_roc_curve(model, X_test, y_test)
    
    # 9. Analyze predictions
    analyze_predictions(X_test, y_pred)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    
    return model, metrics

# %%
if __name__ == "__main__":
    model, metrics = main()
