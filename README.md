# Lobbying Prediction Model

## Overview
This project predicts lobbying success using a dataset containing lobbying registrations, financial indicators, and categorical variables such as city and province. The model aims to determine whether a lobbying attempt receives government funding.

## Features
- **Data Preprocessing**:
  - Handling missing values with imputation.
  - Encoding categorical features.
  - Feature scaling using StandardScaler.
  - Addressing class imbalance using `RandomUnderSampler`.
- **Model Training & Evaluation**:
  - Multiple machine learning models for classification.
  - Performance evaluation using accuracy, precision, recall, and F1-score.
- **Visualization & Insights**:
  - Confusion Matrix: Summarizes a classification model's performance by showing the counts of true positives, true negatives, false positives, and false negatives.
  - Cumulative Gains Chart: Shows model effectiveness in capturing true positives.
  - Lift Chart: Illustrates how much better the model performs compared to random guessing.
  - ROC Curve: Plots the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR).

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
```

## Models Used
- **Logistic Regression** – Baseline linear model.
- **Random Forest Classifier** – Ensemble learning with decision trees.
- **Gradient Boosting Classifier** – Boosting-based decision tree model.
- **XGBoost Classifier** – Optimized gradient boosting model.

## Note:
- Missing Repository is missing primary data set, due to file size.

## License
This project is open-source and available under the MIT License.



