# ============================================================
# DAY 18/60
# Fraud Detection Using Random Forest
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATASET_FILE = "fraud_detection_dataset.csv"

PREDICTIONS_FILE = "day18_predictions.csv"
MODEL_RESULTS_FILE = "day18_model_results.csv"
FEATURE_IMPORTANCE_FILE = "day18_feature_importance.csv"

CONFUSION_MATRIX_FILE = "day18_confusion_matrix.png"
FEATURE_IMPORTANCE_FILE_PNG = "day18_feature_importance.png"
MODEL_COMPARISON_FILE = "day18_model_comparison.png"


# ============================================================
# 2. CREATE SYNTHETIC FRAUD DATASET IF NOT AVAILABLE
# ============================================================

def create_fraud_dataset(n_samples=5000):
    """
    Creates a synthetic fraud detection dataset.

    Target:
        0 = Legitimate transaction
        1 = Fraudulent transaction
    """

    np.random.seed(RANDOM_STATE)

    amount = np.random.lognormal(
        mean=4.2,
        sigma=1.0,
        size=n_samples
    )

    transaction_hour = np.random.randint(
        0,
        24,
        size=n_samples
    )

    transaction_frequency = np.random.poisson(
        lam=5,
        size=n_samples
    )

    account_age_days = np.random.randint(
        10,
        2500,
        size=n_samples
    )

    international = np.random.binomial(
        1,
        0.15,
        size=n_samples
    )

    new_device = np.random.binomial(
        1,
        0.20,
        size=n_samples
    )

    failed_attempts = np.random.poisson(
        lam=1,
        size=n_samples
    )

    distance_from_home = np.random.exponential(
        scale=25,
        size=n_samples
    )

    online_transaction = np.random.binomial(
        1,
        0.65,
        size=n_samples
    )

    # Fraud probability
    fraud_score = (
        0.0008 * amount
        + 0.8 * international
        + 1.0 * new_device
        + 0.35 * failed_attempts
        + 0.012 * distance_from_home
        + 0.15 * online_transaction
        + 0.3 * (transaction_hour < 5)
        + 0.015 * transaction_frequency
        - 0.0002 * account_age_days
    )

    # Add randomness
    fraud_score += np.random.normal(
        0,
        0.7,
        n_samples
    )

    threshold = np.percentile(
        fraud_score,
        90
    )

    fraud = (
        fraud_score > threshold
    ).astype(int)

    df = pd.DataFrame({
        "Transaction_Amount": amount.round(2),
        "Transaction_Hour": transaction_hour,
        "Transaction_Frequency": transaction_frequency,
        "Account_Age_Days": account_age_days,
        "International_Transaction": international,
        "New_Device": new_device,
        "Failed_Attempts": failed_attempts,
        "Distance_From_Home": distance_from_home.round(2),
        "Online_Transaction": online_transaction,
        "Fraud": fraud
    })

    return df


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("DAY 18 - FRAUD DETECTION USING RANDOM FOREST")
print("=" * 60)

if os.path.exists(DATASET_FILE):

    df = pd.read_csv(DATASET_FILE)

    print("\nExisting fraud detection dataset loaded.")

else:

    print("\nFraud dataset not found.")
    print("Creating synthetic fraud detection dataset...")

    df = create_fraud_dataset(
        n_samples=5000
    )

    df.to_csv(
        DATASET_FILE,
        index=False
    )

    print(
        f"Dataset created and saved as: {DATASET_FILE}"
    )


# ============================================================
# 4. BASIC DATA UNDERSTANDING
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")

if "Fraud" in df.columns:
    print(df["Fraud"].value_counts())


# ============================================================
# 5. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)


# ============================================================
# 6. IDENTIFY TARGET COLUMN
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
        "Please use a target column named 'Fraud' or 'Class'."
    )


print(
    f"\nTarget column: {target_column}"
)


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


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# Keep only numeric features
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
# 10. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 11. DECISION TREE MODEL
# ============================================================

decision_tree = DecisionTreeClassifier(
    max_depth=6,
    random_state=RANDOM_STATE
)

decision_tree.fit(
    X_train_scaled,
    y_train
)

dt_predictions = decision_tree.predict(
    X_test_scaled
)


# ============================================================
# 12. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

random_forest.fit(
    X_train_scaled,
    y_train
)

rf_predictions = random_forest.predict(
    X_test_scaled
)


# ============================================================
# 13. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
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
        "F1_Score": f1
    }


# ============================================================
# 14. EVALUATE BOTH MODELS
# ============================================================

dt_results = evaluate_model(
    "Decision Tree",
    y_test,
    dt_predictions
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions
)

results = pd.DataFrame([
    dt_results,
    rf_results
])


print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(
    results.to_string(index=False)
)


# ============================================================
# 15. RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    rf_predictions
)

print("\nRandom Forest Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Analysis:")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nFeature Importance:")
print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

results.to_csv(
    MODEL_RESULTS_FILE,
    index=False
)


# ============================================================
# 20. SAVE PREDICTIONS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Fraud"] = (
    y_test.values
)

prediction_results["Decision_Tree_Prediction"] = (
    dt_predictions
)

prediction_results["Random_Forest_Prediction"] = (
    rf_predictions
)

prediction_results["Prediction_Correct"] = (
    prediction_results["Actual_Fraud"]
    ==
    prediction_results["Random_Forest_Prediction"]
)

prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 21. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Random Forest Confusion Matrix"
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

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300
)

plt.close()


# ============================================================
# 22. FEATURE IMPORTANCE VISUALIZATION
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
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_FILE_PNG,
    dpi=300
)

plt.close()


# ============================================================
# 23. MODEL COMPARISON VISUALIZATION
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

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    x - width / 2,
    results.loc[
        results["Model"] == "Decision Tree",
        metrics
    ].values.flatten(),
    width,
    label="Decision Tree"
)

plt.bar(
    x + width / 2,
    results.loc[
        results["Model"] == "Random Forest",
        metrics
    ].values.flatten(),
    width,
    label="Random Forest"
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
    "Decision Tree vs Random Forest"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_COMPARISON_FILE,
    dpi=300
)

plt.close()


# ============================================================
# 24. DECISION TREE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(20, 10)
)

plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=["Legitimate", "Fraud"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree Structure"
)

plt.tight_layout()

plt.savefig(
    "day18_decision_tree.png",
    dpi=200
)

plt.close()


# ============================================================
# 25. MODEL ROBUSTNESS ANALYSIS
# ============================================================

dt_f1 = dt_results["F1_Score"]
rf_f1 = rf_results["F1_Score"]

print("\n" + "=" * 60)
print("ROBUSTNESS ANALYSIS")
print("=" * 60)

if rf_f1 > dt_f1:

    print(
        "Random Forest achieved a higher F1-score "
        "than the Decision Tree."
    )

    print(
        "This indicates that the ensemble approach "
        "provided better overall classification performance."
    )

elif rf_f1 < dt_f1:

    print(
        "Decision Tree achieved a higher F1-score "
        "than Random Forest on this test split."
    )

    print(
        "Further tuning and validation may be useful."
    )

else:

    print(
        "Both models achieved the same F1-score."
    )


# ============================================================
# 26. FRAUD DETECTION CHALLENGES
# ============================================================

fraud_count = int(
    (y == 1).sum()
)

legitimate_count = int(
    (y == 0).sum()
)

print("\n" + "=" * 60)
print("FRAUD DETECTION ANALYSIS")
print("=" * 60)

print(
    f"Legitimate transactions: {legitimate_count}"
)

print(
    f"Fraudulent transactions: {fraud_count}"
)

print(
    "\nFalse Positive:",
    fp,
    "legitimate transactions were incorrectly "
    "classified as fraud."
)

print(
    "False Negative:",
    fn,
    "fraudulent transactions were incorrectly "
    "classified as legitimate."
)

print(
    "\nIn fraud detection, False Negatives are "
    "particularly important because they represent "
    "fraudulent transactions missed by the model."
)


# ============================================================
# 27. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "Fraud detection dataset loaded successfully."
)

print(
    "Decision Tree classifier trained successfully."
)

print(
    "Random Forest classifier trained successfully."
)

print(
    "Predictions generated successfully."
)

print(
    "Decision Tree vs Random Forest comparison completed."
)

print(
    "Feature importance analysis completed."
)

print(
    "Fraud detection robustness analysis completed."
)

print(
    "Confusion matrix created successfully."
)

print(
    "Decision tree visualization created successfully."
)

print(
    "Feature importance visualization created successfully."
)

print(
    "Model comparison visualization created successfully."
)

print(
    "Prediction results saved successfully."
)

print(
    "Model results saved successfully."
)

print(
    "\nDay 18 Fraud Detection using Random Forest "
    "completed successfully! 🚨🌲"
)

print("=" * 60)