# Day 49: Baseline Capstone Prototype Report

## Objective
Build a baseline customer churn prediction prototype.

## Dataset
- Source: `WA_FnUseC_TelcoCustomerChurn.csv`
- Records processed: 7043

## Prediction Output
- File: `day49_prediction_outputs.csv`
- Prediction records: 7043

## Initial Evaluation
- Accuracy on source dataset: 75.01%

This score is calculated on the same dataset used for this prototype.
It is not an independent test score and may be optimistic.

### Classification Report
```text
              precision    recall  f1-score   support

          No       0.91      0.73      0.81      5174
         Yes       0.52      0.80      0.63      1869

    accuracy                           0.75      7043
   macro avg       0.71      0.77      0.72      7043
weighted avg       0.81      0.75      0.76      7043

```

## Prototype Strengths
- Reuses an existing trained model.
- Generates customer-level churn predictions.
- Saves predictions to a CSV file.
- Produces an initial evaluation report.

## Limitations
- The model's training details and input compatibility should be verified.
- Evaluation on the source dataset does not establish generalization.
- A held-out test set is required for an unbiased evaluation.
- Dashboard integration and customer segmentation can be improved.

## Next Steps
- Evaluate on unseen customer records.
- Verify the model's expected input features.
- Integrate predictions into the dashboard.
- Add segmentation and monitoring features.
