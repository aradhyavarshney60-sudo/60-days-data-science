# ============================================================
# DAY 20: MODEL EVALUATION
# Why Accuracy Alone Can Mislead You
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "fraud_detection_dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 70)
print("DAY 20: MODEL EVALUATION")
print("Why Accuracy Alone Can Mislead You")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nDataset '{DATA_FILE}' not found.\n"
        "Make sure fraud_detection_dataset.csv is in the project folder."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 4. IDENTIFY TARGET COLUMN
# ============================================================

possible_targets = [
    "is_fraud",
    "fraud",
    "Fraud",
    "target",
    "Target",
    "label",
    "Label",
    "class",
    "Class",
    "y",
]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    raise ValueError(
        "\nTarget column not found automatically.\n"
        "Expected one of:\n"
        "is_fraud, fraud, Fraud, target, Target, label, Label, class, Class, y"
    )

print(f"\nTarget column detected: {target_column}")


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

df = df.copy()

numeric_columns = df.select_dtypes(include=[np.number]).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_column])
y = df[target_column]


# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Make sure all values are numeric
X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)


# Convert target to numeric if required
if not pd.api.types.is_numeric_dtype(y):

    unique_values = y.dropna().unique()

    if len(unique_values) == 2:
        mapping = {
            unique_values[0]: 0,
            unique_values[1]: 1,
        }

        y = y.map(mapping)

        print("\nTarget mapping:")
        print(mapping)

    else:
        raise ValueError(
            "Target column must contain a binary classification target."
        )


y = pd.to_numeric(y)


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_counts = y.value_counts().sort_index()

print(target_counts)

print("\nTarget percentages:")
print((target_counts / len(y) * 100).round(2))


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"Features:         {X_train.shape[1]}")


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=8,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        max_depth=12,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
prediction_results = []

confusion_matrices = {}

roc_data = {}

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions
    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test)[:, 1]

    else:
        y_probability = None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    if y_probability is not None:
        roc_auc = roc_auc_score(
            y_test,
            y_probability,
        )
    else:
        roc_auc = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    confusion_matrices[model_name] = cm

    # ROC data
    if y_probability is not None:

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_probability,
        )

        roc_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
        }

    # Store results
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
        }
    )

    # Store predictions
    prediction_results.append(
        pd.DataFrame(
            {
                "Actual": y_test.values,
                "Predicted": y_pred,
                "Probability": (
                    y_probability
                    if y_probability is not None
                    else np.nan
                ),
                "Model": model_name,
            }
        )
    )

    print(f"{model_name} completed.")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 11. CREATE MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 12. SAVE MODEL RESULTS
# ============================================================

results_df.to_csv(
    "day20_model_results.csv",
    index=False,
)

results_df.to_csv(
    "day20_metrics_comparison.csv",
    index=False,
)

print("\nModel comparison files saved successfully.")


# ============================================================
# 13. SAVE ALL PREDICTIONS
# ============================================================

all_predictions = pd.concat(
    prediction_results,
    ignore_index=True,
)

all_predictions.to_csv(
    "day20_predictions.csv",
    index=False,
)

print("Prediction results saved successfully.")


# ============================================================
# 14. CONFUSION MATRIX VISUALIZATION
# ============================================================

number_of_models = len(confusion_matrices)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
)

axes = axes.flatten()

for index, (model_name, cm) in enumerate(
    confusion_matrices.items()
):

    ax = axes[index]

    ax.imshow(cm)

    ax.set_title(
        f"{model_name}\nConfusion Matrix"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

plt.tight_layout()

plt.savefig(
    "day20_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Confusion matrix visualization created successfully.")


# ============================================================
# 15. ROC CURVE
# ============================================================

plt.figure(figsize=(10, 7))

for model_name, data in roc_data.items():

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{model_name} (AUC = {data['auc']:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison - Day 20"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "day20_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("ROC curve created successfully.")


# ============================================================
# 16. METRICS COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
]

plot_df = results_df.set_index("Model")[metrics]

ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7),
)

ax.set_title(
    "Classification Model Performance Comparison"
)

ax.set_ylabel("Score")
ax.set_xlabel("Model")

plt.xticks(
    rotation=20,
    ha="right",
)

plt.ylim(0, 1.05)

plt.legend(
    loc="lower right"
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    "day20_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Metrics comparison visualization created successfully.")


# ============================================================
# 17. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best model based on F1-Score: "
    f"{best_model['Model']}"
)

print(
    f"Accuracy : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision: {best_model['Precision']:.4f}"
)

print(
    f"Recall   : {best_model['Recall']:.4f}"
)

print(
    f"F1 Score : {best_model['F1_Score']:.4f}"
)

print(
    f"ROC-AUC  : {best_model['ROC_AUC']:.4f}"
)


# ============================================================
# 18. ACCURACY VS F1 ANALYSIS
# ============================================================

highest_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]

highest_recall = results_df.loc[
    results_df["Recall"].idxmax()
]

highest_precision = results_df.loc[
    results_df["Precision"].idxmax()
]

highest_f1 = results_df.loc[
    results_df["F1_Score"].idxmax()
]

highest_auc = results_df.loc[
    results_df["ROC_AUC"].idxmax()
]

print("\n" + "=" * 70)
print("METRIC ANALYSIS")
print("=" * 70)

print(
    f"\nHighest Accuracy : "
    f"{highest_accuracy['Model']}"
)

print(
    f"Highest Precision: "
    f"{highest_precision['Model']}"
)

print(
    f"Highest Recall   : "
    f"{highest_recall['Model']}"
)

print(
    f"Highest F1 Score : "
    f"{highest_f1['Model']}"
)

print(
    f"Highest ROC-AUC  : "
    f"{highest_auc['Model']}"
)


# ============================================================
# 19. CLASSIFICATION REPORT FOR BEST MODEL
# ============================================================

best_model_name = best_model["Model"]

best_estimator = models[best_model_name]

best_predictions = best_estimator.predict(X_test)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0,
    )
)


# ============================================================
# 20. FRAUD DETECTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WHY ACCURACY ALONE CAN MISLEAD YOU")
print("=" * 70)

fraud_percentage = (
    y.value_counts(normalize=True)
    .get(1, 0)
    * 100
)

print(
    f"\nPositive/Fraud class percentage: "
    f"{fraud_percentage:.2f}%"
)

if fraud_percentage < 20:

    print(
        "\nThe dataset is highly imbalanced."
    )

    print(
        "Accuracy alone may give a misleading picture "
        "of model performance."
    )

    print(
        "For fraud detection, Recall and F1-Score "
        "are especially important."
    )

else:

    print(
        "\nThe target classes are not extremely imbalanced."
    )

    print(
        "However, Accuracy should still be considered "
        "together with Precision, Recall, F1-Score and ROC-AUC."
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "DAY 20 MODEL EVALUATION COMPLETED SUCCESSFULLY!"
)

print("=" * 70)

print("\nGenerated files:")

print("1. day20.py")
print("2. day20_model_results.csv")
print("3. day20_metrics_comparison.csv")
print("4. day20_predictions.csv")
print("5. day20_confusion_matrix.png")
print("6. day20_roc_curve.png")
print("7. day20_metrics_comparison.png")

print("\nKey learning:")

print(
    "Accuracy alone is not enough for evaluating "
    "classification models, especially with imbalanced data."
)

print(
    "Precision, Recall, F1-Score and ROC-AUC provide "
    "a more complete view of model performance."
)

print("\n" + "=" * 70)