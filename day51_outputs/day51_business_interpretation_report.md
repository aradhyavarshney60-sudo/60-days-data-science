# Day 51: Explainable AI for Customer Churn

## Objective
Explain customer churn model behavior using feature importance and SHAP.

## Evaluation
- Model: Gradient Boosting Classifier
- Hold-out test accuracy: 0.8062
- Hold-out test F1-score: 0.5895
- Training/test split: 80% / 20%
- Random state: 42

## Top SHAP Drivers
- **numeric__categorical__Contract_Month-to-month**: mean absolute SHAP = 0.6432
- **numeric__numeric__tenure**: mean absolute SHAP = 0.3617
- **numeric__numeric__MonthlyCharges**: mean absolute SHAP = 0.2512
- **numeric__categorical__OnlineSecurity_No**: mean absolute SHAP = 0.2371
- **numeric__categorical__TechSupport_No**: mean absolute SHAP = 0.2054
- **numeric__categorical__InternetService_Fiber optic**: mean absolute SHAP = 0.1926
- **numeric__categorical__PaymentMethod_Electronic check**: mean absolute SHAP = 0.1654
- **numeric__categorical__Contract_Two year**: mean absolute SHAP = 0.1461
- **numeric__numeric__TotalCharges**: mean absolute SHAP = 0.1420
- **numeric__categorical__MultipleLines_No**: mean absolute SHAP = 0.1098

## Business Interpretation
Features with larger mean absolute SHAP values have a greater average
influence on model output for the analyzed sample. This ranking does not
by itself show whether a feature increases or decreases churn probability.
Use the SHAP summary plot to inspect direction and distribution.

These are model associations, not proof of causation. Business teams should
validate findings before taking customer-facing actions.

## Generated Outputs
- `day51_feature_importance.csv`
- `day51_feature_importance.png`
- `day51_shap_summary.png`
- `day51_shap_importance.csv`
- `day51_business_interpretation_report.md`
