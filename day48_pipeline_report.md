# Day 48: Customer Intelligence Data Pipeline Report

## 1. Objective
Build a reusable preprocessing pipeline for the Customer Intelligence Platform using the Telco Customer Churn dataset.

## 2. Dataset Overview
- Source: `WA_FnUseC_TelcoCustomerChurn.csv`
- Original records: 7,043
- Duplicate rows detected: 0
- Target variable: `Churn`

## 3. Preprocessing Workflow
1. Load the dataset using Pandas.
2. Remove duplicate records.
3. Convert `TotalCharges` to numeric values.
4. Remove rows with missing target labels.
5. Separate customer identifiers and target labels from model features.
6. Impute missing numeric values using the median.
7. Scale numeric features using StandardScaler.
8. Impute categorical values using the most frequent value.
9. Encode categorical features using OneHotEncoder.
10. Export the transformed dataset to CSV.

## 4. Pipeline Architecture
`CSV Dataset → Data Cleaning → Feature Separation → ColumnTransformer → Processed Features → CSV Output`

## 5. Results
- Processed dataset shape: 7,043 rows × 47 columns
- Remaining missing values: 0
- Output file: `day48_cleaned_customer_churn.csv`

## 6. Technologies
Python, Pandas, Scikit-learn, SimpleImputer, StandardScaler, OneHotEncoder, Pipeline, and ColumnTransformer.

## 7. Reliability Notes
The preprocessing steps are organized into reusable Scikit-learn pipelines. For model evaluation, preprocessing should be fitted only on training data and then applied to validation or test data to prevent data leakage.

## 8. Conclusion
The Day 48 preprocessing workflow prepares customer data for further analysis and machine learning development in the Customer Intelligence Platform.