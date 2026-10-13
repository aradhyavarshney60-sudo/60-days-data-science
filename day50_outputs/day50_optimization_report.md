# Day 50: Model Optimization Report

## Dataset
- Rows used: 7043
- Training rows: 5634
- Test rows: 1409
- Target column: Churn

## Model Comparison
                    model  accuracy  precision   recall  f1_score
        Gradient Boosting  0.806246   0.796394 0.806246  0.797889
            Random Forest  0.766501   0.777102 0.766501  0.770848
      Logistic Regression  0.739532   0.797200 0.739532  0.753231
Tuned Logistic Regression  0.738822   0.796175 0.738822  0.752529
           Dummy Baseline  0.734564   0.539584 0.734564  0.622155

## Best Model
Selected by weighted F1-score on the held-out test set: **Gradient Boosting**

## Classification Report
              precision    recall  f1-score   support

          No       0.84      0.91      0.87      1035
         Yes       0.67      0.52      0.59       374

    accuracy                           0.81      1409
   macro avg       0.76      0.72      0.73      1409
weighted avg       0.80      0.81      0.80      1409


## Hyperparameter Tuning
- Model: Logistic Regression
- Best parameters: {'model__C': 0.1}
- Cross-validation scoring: weighted F1-score

## Notes
- Preprocessing was fitted within each training pipeline.
- The test set was held out from model fitting and hyperparameter tuning.
- Model performance may vary with data quality and class distribution.
- This comparison uses the available dataset; external validation may still be needed.
