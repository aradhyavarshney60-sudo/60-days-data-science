
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("day43_customer_churn_model.joblib")
DATA_PATH = Path("WA_FnUseC_TelcoCustomerChurn.csv")
OUTPUT_PATH = Path("day49_prediction_outputs.csv")
REPORT_PATH = Path("day49_prototype_report.md")


def main():
    # 1. Check required files
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        return

    if not DATA_PATH.exists():
        logger.error("Dataset not found: %s", DATA_PATH)
        return

    # 2. Load model and dataset
    logger.info("Loading model and customer dataset")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "Churn" not in df.columns:
        logger.error("Target column 'Churn' is missing.")
        return

    logger.info("Dataset shape: %s", df.shape)

    # 3. Separate customer IDs, target and input features
    if "customerID" in df.columns:
        customer_ids = df["customerID"].copy()
    else:
        customer_ids = pd.Series(df.index, name="customerID")

    y_actual = df["Churn"].astype(str).str.strip()

    X = df.drop(
        columns=[
            col for col in ["Churn", "customerID"]
            if col in df.columns
        ]
    ).copy()

    # 4. Clean numeric field
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"], errors="coerce"
        )

    # 5. Generate predictions using the existing model
    logger.info("Generating baseline churn predictions")

    try:
        predictions = model.predict(X)
    except Exception:
        logger.exception(
            "Prediction failed. The model may expect different "
            "input columns or preprocessing."
        )
        return

    # 6. Convert 0/1 predictions to No/Yes when appropriate
    y_pred = pd.Series(predictions, index=df.index)

    if set(pd.Series(y_pred).dropna().unique()).issubset({0, 1}):
        y_pred = y_pred.map({0: "No", 1: "Yes"}).astype(str)
    else:
        y_pred = y_pred.astype(str).str.strip()

    # 7. Build output file
    results = pd.DataFrame({
        "customerID": customer_ids,
        "actual_churn": y_actual,
        "predicted_churn": y_pred
    })

    # Include churn probability when supported by the model
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
            if probabilities.shape[1] == 2:
                results["churn_probability"] = probabilities[:, 1]
        except Exception:
            logger.warning("Could not generate probability scores.")

    # 8. Calculate metrics only when labels are comparable
    accuracy = accuracy_score(y_actual, y_pred)
    report_text = classification_report(
        y_actual, y_pred, zero_division=0
    )

    # 9. Save predictions after label conversion
    results.to_csv(OUTPUT_PATH, index=False)

    # 10. Create prototype analysis report
    report = f"""# Day 49: Baseline Capstone Prototype Report

## Objective
Build a baseline customer churn prediction prototype.

## Dataset
- Source: `{DATA_PATH.name}`
- Records processed: {len(results)}

## Prediction Output
- File: `{OUTPUT_PATH.name}`
- Prediction records: {len(results)}

## Initial Evaluation
- Accuracy on source dataset: {accuracy:.2%}

This score is calculated on the same dataset used for this prototype.
It is not an independent test score and may be optimistic.

### Classification Report
```text
{report_text}
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
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    logger.info("Predictions saved: %s", OUTPUT_PATH)
    logger.info("Prototype report saved: %s", REPORT_PATH)
    logger.info("Records processed: %d", len(results))
    logger.info("Source-dataset accuracy: %.2f%%", accuracy * 100)
    logger.info("Day 49 prototype completed.")


if __name__ == "__main__":
    main()
