# ============================================================
# DAY 19/60
# Boosting Model Performance with XGBoost
# ============================================================

import os
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATASET_FILE = "fraud_detection_dataset.csv"

PREDICTIONS_FILE = "day19_predictions.csv"
MODEL_RESULTS_FILE = "day19_model_results.csv"
FEATURE_IMPORTANCE_FILE = "day19_feature_importance.csv"

CONFUSION_MATRIX_FILE = "day19_confusion_matrix.png"
FEATURE_IMPORTANCE_FILE_PNG = "day19_feature_importance.png"
MODEL_COMPARISON_FILE = "day19_model_comparison.png"


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("DAY 19 - BOOSTING MODEL PERFORMANCE WITH XGBOOST")
print("=" * 70)


# ============================================================
# 3. CHECK XGBOOST
# ============================================================

try:
    import xgboost
    print("\nXGBoost version:", xgboost.__version__)
except ImportError:
    print("\nXGBoost is not installed.")
    print("Run this command in terminal:")
    print("pip install xgboost")
    raise


# ============================================================
# 4. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):

    print("\nDataset not found.")
    print("Please make sure fraud_detection_dataset.csv")
    print("is present in the project folder.")

    raise FileNotFoundError(
        f"{DATASET_FILE} not found."
    )


df = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 5. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)


print("\nColumns:")
print(list(df.columns))


# ============================================================
# 6. FIND TARGET COLUMN
# ============================================================

possible_targets = [
    "Fraud",
    "fraud",
    "Class",
    "class",
    "Is_Fraud",
    "is_fraud",
    "Fraud_Flag",
    "fraud_flag"
]

target_column = None

for column in possible_targets:

    if column in df.columns:

        target_column = column
        break


if target_column is None:

    raise ValueError(
        "Target column not found. "
        "Expected a column such as 'Fraud' or 'Class'."
    )


print("\nTarget column:", target_column)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    if df[column].isnull().any():

        df[column] = df[column].fillna(
            df[column].median()
        )


print("\nMissing values handled.")


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# Keep numeric columns only
X = X.select_dtypes(
    include=np.number
)


print("\nFeatures used:")
print(list(X.columns))


print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


rf_start = time.time()


random_forest = RandomForestClassifier(

    n_estimators=150,

    max_depth=8,

    min_samples_split=5,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    class_weight="balanced"
)


random_forest.fit(
    X_train,
    y_train
)


rf_predictions = random_forest.predict(
    X_test
)


rf_time = time.time() - rf_start


print(
    f"\nRandom Forest training completed in "
    f"{rf_time:.2f} seconds."
)


# ============================================================
# 11. XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)


xgb_start = time.time()


# Calculate class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()


if positive_count > 0:

    scale_pos_weight = (
        negative_count / positive_count
    )

else:

    scale_pos_weight = 1


xgb_model = XGBClassifier(

    n_estimators=200,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=2,

    gamma=0,

    reg_alpha=0.0,

    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    scale_pos_weight=scale_pos_weight
)


xgb_model.fit(

    X_train,

    y_train
)


xgb_predictions = xgb_model.predict(
    X_test
)


xgb_time = time.time() - xgb_start


print(
    f"\nXGBoost training completed in "
    f"{xgb_time:.2f} seconds."
)


# ============================================================
# 12. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    training_time
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "Training_Time_Seconds": training_time

    }


# ============================================================
# 13. EVALUATE RANDOM FOREST
# ============================================================

rf_results = evaluate_model(

    "Random Forest",

    y_test,

    rf_predictions,

    rf_time
)


# ============================================================
# 14. EVALUATE XGBOOST
# ============================================================

xgb_results = evaluate_model(

    "XGBoost",

    y_test,

    xgb_predictions,

    xgb_time
)


# ============================================================
# 15. CREATE MODEL COMPARISON
# ============================================================

results = pd.DataFrame([

    rf_results,

    xgb_results

])


print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results.to_string(index=False)
)


# ============================================================
# 16. DETERMINE BEST MODEL
# ============================================================

best_model_row = results.loc[
    results["F1_Score"].idxmax()
]

best_model_name = best_model_row["Model"]

print("\nBest model based on F1 Score:")

print(best_model_name)


# ============================================================
# 17. XGBOOST CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(

        y_test,

        xgb_predictions,

        zero_division=0

    )
)


# ============================================================
# 18. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(

        y_test,

        rf_predictions,

        zero_division=0

    )
)


# ============================================================
# 19. XGBOOST CONFUSION MATRIX
# ============================================================

xgb_cm = confusion_matrix(

    y_test,

    xgb_predictions

)


print("\nXGBoost Confusion Matrix:")

print(xgb_cm)


tn, fp, fn, tp = xgb_cm.ravel()


print("\nXGBoost Confusion Matrix Analysis:")

print("True Negatives :", tn)

print("False Positives:", fp)

print("False Negatives:", fn)

print("True Positives :", tp)


# ============================================================
# 20. XGBOOST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": xgb_model.feature_importances_

})


feature_importance = (

    feature_importance

    .sort_values(

        by="Importance",

        ascending=False

    )

    .reset_index(drop=True)

)


print("\n" + "=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 21. SAVE FEATURE IMPORTANCE CSV
# ============================================================

feature_importance.to_csv(

    FEATURE_IMPORTANCE_FILE,

    index=False

)


print(
    "\nFeature importance saved successfully."
)


# ============================================================
# 22. SAVE MODEL RESULTS
# ============================================================

results.to_csv(

    MODEL_RESULTS_FILE,

    index=False

)


print(
    "Model results saved successfully."
)


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================

prediction_results = X_test.copy()


prediction_results["Actual_Fraud"] = (
    y_test.values
)


prediction_results["Random_Forest_Prediction"] = (
    rf_predictions
)


prediction_results["XGBoost_Prediction"] = (
    xgb_predictions
)


prediction_results["XGBoost_Prediction_Correct"] = (

    prediction_results["Actual_Fraud"]

    ==

    prediction_results["XGBoost_Prediction"]

)


prediction_results.to_csv(

    PREDICTIONS_FILE,

    index=False

)


print(
    "Prediction results saved successfully."
)


# ============================================================
# 24. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    xgb_cm,
    interpolation="nearest"
)


plt.title(
    "XGBoost Confusion Matrix"
)


plt.colorbar()


plt.xticks(

    [0, 1],

    ["Legitimate", "Fraud"]

)


plt.yticks(

    [0, 1],

    ["Legitimate", "Fraud"]

)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "Actual Label"
)


for i in range(
    xgb_cm.shape[0]
):

    for j in range(
        xgb_cm.shape[1]
    ):

        plt.text(

            j,

            i,

            xgb_cm[i, j],

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    CONFUSION_MATRIX_FILE,

    dpi=300

)


plt.close()


print(
    "Confusion matrix created successfully."
)


# ============================================================
# 25. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.barh(

    feature_importance["Feature"],

    feature_importance["Importance"]

)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "XGBoost Feature Importance"
)


plt.gca().invert_yaxis()


plt.tight_layout()


plt.savefig(

    FEATURE_IMPORTANCE_FILE_PNG,

    dpi=300

)


plt.close()


print(
    "Feature importance visualization created successfully."
)


# ============================================================
# 26. RANDOM FOREST VS XGBOOST COMPARISON
# ============================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1_Score"

]


x = np.arange(
    len(metrics)
)


width = 0.35


rf_values = results.loc[

    results["Model"] == "Random Forest",

    metrics

].values.flatten()


xgb_values = results.loc[

    results["Model"] == "XGBoost",

    metrics

].values.flatten()


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    x - width / 2,

    rf_values,

    width,

    label="Random Forest"

)


plt.bar(

    x + width / 2,

    xgb_values,

    width,

    label="XGBoost"

)


plt.xticks(

    x,

    metrics

)


plt.ylabel(
    "Score"
)


plt.ylim(
    0,
    1
)


plt.title(
    "Random Forest vs XGBoost"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    MODEL_COMPARISON_FILE,

    dpi=300

)


plt.close()


print(
    "Model comparison visualization created successfully."
)


# ============================================================
# 27. PERFORMANCE IMPROVEMENT ANALYSIS
# ============================================================

rf_f1 = rf_results["F1_Score"]

xgb_f1 = xgb_results["F1_Score"]


f1_improvement = (
    xgb_f1 - rf_f1
)


print("\n" + "=" * 70)
print("BOOSTING PERFORMANCE ANALYSIS")
print("=" * 70)


print(
    f"Random Forest F1 Score : {rf_f1:.4f}"
)


print(
    f"XGBoost F1 Score       : {xgb_f1:.4f}"
)


print(
    f"F1 Score Improvement   : {f1_improvement:.4f}"
)


if f1_improvement > 0:

    print(
        "\nXGBoost improved the F1 Score "
        "over Random Forest."
    )

elif f1_improvement < 0:

    print(
        "\nRandom Forest achieved a higher "
        "F1 Score on this test set."
    )

else:

    print(
        "\nBoth models achieved the same "
        "F1 Score."
    )


# ============================================================
# 28. BOOSTING ADVANTAGES AND TRADE-OFFS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST ADVANTAGES")
print("=" * 70)

print(
    "1. Strong predictive performance."
)

print(
    "2. Captures complex non-linear relationships."
)

print(
    "3. Built-in regularization helps control overfitting."
)

print(
    "4. Supports feature importance analysis."
)

print(
    "5. Works efficiently with tabular datasets."
)


print("\n" + "=" * 70)
print("XGBOOST TRADE-OFFS")
print("=" * 70)

print(
    "1. More hyperparameters than simple models."
)

print(
    "2. Can overfit if not properly tuned."
)

print(
    "3. Training can be slower than a simple decision tree."
)

print(
    "4. Model interpretation is more difficult."
)

print(
    "5. Parameter tuning can require experimentation."
)


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "Day 19 XGBoost Boosting Analysis "
    "completed successfully!"
)

print("=" * 70)


print("\nGenerated files:")

print(
    f"- {PREDICTIONS_FILE}"
)

print(
    f"- {MODEL_RESULTS_FILE}"
)

print(
    f"- {FEATURE_IMPORTANCE_FILE}"
)

print(
    f"- {CONFUSION_MATRIX_FILE}"
)

print(
    f"- {FEATURE_IMPORTANCE_FILE_PNG}"
)

print(
    f"- {MODEL_COMPARISON_FILE}"
)

print("\nBest model:", best_model_name)

print("=" * 70)