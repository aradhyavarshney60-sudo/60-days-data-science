# ============================================================
# DAY 23 - FEATURE SELECTION
# 60 Days Data Science Challenge
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET = "fraud_detection_dataset.csv"

OUTPUT_IMPORTANCE_CSV = "day23_feature_importance.csv"
OUTPUT_IMPORTANCE_PNG = "day23_feature_importance.png"
OUTPUT_COMPARISON_CSV = "day23_model_comparison.csv"
OUTPUT_COMPARISON_PNG = "day23_model_comparison.png"
OUTPUT_REPORT = "day23_feature_selection_report.txt"
OUTPUT_REFLECTION = "day23_reflection.txt"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("DAY 23 - FEATURE SELECTION")
print("=" * 70)

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"{DATASET} not found. Make sure it is in the project folder."
    )

df = pd.read_csv(DATASET)

print("\nDataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. IDENTIFY TARGET COLUMN
# ============================================================

possible_targets = [
    "target",
    "Target",
    "fraud",
    "Fraud",
    "is_fraud",
    "Is_Fraud",
    "label",
    "Label",
    "class",
    "Class",
    "y"
]

target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    # Try last column if it looks like a classification target
    last_col = df.columns[-1]

    if df[last_col].nunique() <= 10:
        target_column = last_col

if target_column is None:
    raise ValueError(
        "Target column could not be identified. "
        "Please set target_column manually."
    )

print(f"\nTarget column: {target_column}")


# ============================================================
# 4. BASIC CLEANING
# ============================================================

df = df.drop_duplicates().copy()

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Fill missing numeric values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print(f"\nDataset after cleaning: {df.shape}")


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 6. HANDLE CATEGORICAL FEATURES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

print("\nCategorical features:")
print(categorical_columns)

print("\nNumerical features:")
print(numeric_columns)

if categorical_columns:
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

print(f"\nFeatures after encoding: {X.shape[1]}")


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. BASELINE MODEL USING ALL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL - ALL FEATURES")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

baseline_model.fit(X_train_scaled, y_train)

baseline_pred = baseline_model.predict(X_test_scaled)
baseline_prob = baseline_model.predict_proba(X_test_scaled)[:, 1]

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(
    y_test,
    baseline_pred,
    zero_division=0
)
baseline_recall = recall_score(
    y_test,
    baseline_pred,
    zero_division=0
)
baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    zero_division=0
)

try:
    baseline_roc_auc = roc_auc_score(y_test, baseline_prob)
except Exception:
    baseline_roc_auc = 0.0

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1 Score : {baseline_f1:.4f}")
print(f"ROC-AUC  : {baseline_roc_auc:.4f}")


# ============================================================
# 9. FEATURE IMPORTANCE USING RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

importance_df.to_csv(
    OUTPUT_IMPORTANCE_CSV,
    index=False
)

print("\nTop 15 important features:")

print(
    importance_df.head(15).to_string(index=False)
)


# ============================================================
# 10. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance")

plt.tight_layout()
plt.savefig(
    OUTPUT_IMPORTANCE_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

correlation_df = X.copy()
correlation_df[target_column] = y.values

correlations = correlation_df.corr(numeric_only=True)[
    target_column
].drop(target_column)

correlations = correlations.abs().sort_values(
    ascending=False
)

print("\nTop correlated features with target:")

print(correlations.head(15))


# ============================================================
# 12. SELECT IMPORTANT FEATURES
# ============================================================

# Select features based on Random Forest importance
importance_threshold = importance_df["Importance"].quantile(0.50)

selected_features = importance_df[
    importance_df["Importance"] >= importance_threshold
]["Feature"].tolist()

# Make sure at least 5 features are selected
if len(selected_features) < 5:
    selected_features = importance_df.head(
        min(5, len(importance_df))
    )["Feature"].tolist()

print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

print(f"\nTotal features before selection: {X.shape[1]}")
print(f"Selected features: {len(selected_features)}")

print("\nSelected features:")
for feature in selected_features:
    print(f"  - {feature}")

removed_features = [
    feature for feature in X.columns
    if feature not in selected_features
]

print("\nRemoved features:")
for feature in removed_features:
    print(f"  - {feature}")


# ============================================================
# 13. TRAIN MODEL USING SELECTED FEATURES
# ============================================================

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

selected_scaler = StandardScaler()

X_train_selected_scaled = selected_scaler.fit_transform(
    X_train_selected
)

X_test_selected_scaled = selected_scaler.transform(
    X_test_selected
)

selected_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

selected_model.fit(
    X_train_selected_scaled,
    y_train
)

selected_pred = selected_model.predict(
    X_test_selected_scaled
)

selected_prob = selected_model.predict_proba(
    X_test_selected_scaled
)[:, 1]


# ============================================================
# 14. SELECTED MODEL METRICS
# ============================================================

selected_accuracy = accuracy_score(
    y_test,
    selected_pred
)

selected_precision = precision_score(
    y_test,
    selected_pred,
    zero_division=0
)

selected_recall = recall_score(
    y_test,
    selected_pred,
    zero_division=0
)

selected_f1 = f1_score(
    y_test,
    selected_pred,
    zero_division=0
)

try:
    selected_roc_auc = roc_auc_score(
        y_test,
        selected_prob
    )
except Exception:
    selected_roc_auc = 0.0


# ============================================================
# 15. PERFORMANCE COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression - All Features",
        "Logistic Regression - Selected Features"
    ],
    "Feature_Count": [
        X.shape[1],
        len(selected_features)
    ],
    "Accuracy": [
        baseline_accuracy,
        selected_accuracy
    ],
    "Precision": [
        baseline_precision,
        selected_precision
    ],
    "Recall": [
        baseline_recall,
        selected_recall
    ],
    "F1_Score": [
        baseline_f1,
        selected_f1
    ],
    "ROC_AUC": [
        baseline_roc_auc,
        selected_roc_auc
    ]
})

comparison_df.to_csv(
    OUTPUT_COMPARISON_CSV,
    index=False
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(index=False)
)


# ============================================================
# 16. PERFORMANCE VISUALIZATION
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    comparison_df.iloc[0][metrics],
    width,
    label="All Features"
)

plt.bar(
    x + width / 2,
    comparison_df.iloc[1][metrics],
    width,
    label="Selected Features"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.05)

plt.title(
    "Model Performance Before vs After Feature Selection"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_COMPARISON_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. GENERATE REPORT
# ============================================================

report = []

report.append("DAY 23 - FEATURE SELECTION REPORT")
report.append("=" * 60)
report.append("")

report.append(f"Dataset: {DATASET}")
report.append(f"Target Column: {target_column}")
report.append(f"Total Rows: {len(df)}")
report.append(f"Features Before Selection: {X.shape[1]}")
report.append(f"Features After Selection: {len(selected_features)}")
report.append("")

report.append("SELECTED FEATURES")
report.append("-" * 60)

for feature in selected_features:
    importance_value = importance_df.loc[
        importance_df["Feature"] == feature,
        "Importance"
    ].iloc[0]

    report.append(
        f"{feature}: {importance_value:.6f}"
    )

report.append("")
report.append("REMOVED FEATURES")
report.append("-" * 60)

for feature in removed_features:
    report.append(feature)

report.append("")
report.append("MODEL PERFORMANCE")
report.append("-" * 60)

report.append(
    f"Baseline Accuracy: {baseline_accuracy:.4f}"
)

report.append(
    f"Selected Accuracy: {selected_accuracy:.4f}"
)

report.append(
    f"Baseline Precision: {baseline_precision:.4f}"
)

report.append(
    f"Selected Precision: {selected_precision:.4f}"
)

report.append(
    f"Baseline Recall: {baseline_recall:.4f}"
)

report.append(
    f"Selected Recall: {selected_recall:.4f}"
)

report.append(
    f"Baseline F1 Score: {baseline_f1:.4f}"
)

report.append(
    f"Selected F1 Score: {selected_f1:.4f}"
)

report.append(
    f"Baseline ROC-AUC: {baseline_roc_auc:.4f}"
)

report.append(
    f"Selected ROC-AUC: {selected_roc_auc:.4f}"
)

report.append("")
report.append("ANALYSIS")
report.append("-" * 60)

if selected_f1 >= baseline_f1:
    report.append(
        "Feature selection maintained or improved F1 Score."
    )
else:
    report.append(
        "Feature selection reduced F1 Score, showing that "
        "some removed features contained useful predictive information."
    )

report.append(
    "Feature selection can reduce model complexity, "
    "remove noisy variables, and improve interpretability."
)

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(report))


# ============================================================
# 18. REFLECTION
# ============================================================

reflection = f"""
DAY 23 REFLECTION
=================

Today I learned about Feature Selection in Machine Learning.

The main objective was to identify important prediction signals
and remove low-impact or unnecessary features.

I analyzed feature importance using Random Forest and also
studied correlations between features and the target variable.

The baseline Logistic Regression model used
{X.shape[1]} features.

After feature selection, the model used
{len(selected_features)} features.

The F1 Score before feature selection was
{baseline_f1:.4f}.

The F1 Score after feature selection was
{selected_f1:.4f}.

This exercise helped me understand that more features do not
always mean a better Machine Learning model.

Removing irrelevant features can reduce model complexity,
improve interpretability, and make the model easier to analyze.

I also learned that feature selection should be evaluated
using multiple metrics instead of relying only on accuracy.

Day 23 helped me understand how identifying the right signals
in data can improve the Machine Learning workflow.
"""

with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:
    file.write(reflection.strip())


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DAY 23 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(f"1. {OUTPUT_IMPORTANCE_CSV}")
print(f"2. {OUTPUT_IMPORTANCE_PNG}")
print(f"3. {OUTPUT_COMPARISON_CSV}")
print(f"4. {OUTPUT_COMPARISON_PNG}")
print(f"5. {OUTPUT_REPORT}")
print(f"6. {OUTPUT_REFLECTION}")

print("\nFeature Selection Summary:")
print(f"Features Before: {X.shape[1]}")
print(f"Features After : {len(selected_features)}")

print("\nF1 Score:")
print(f"Before Selection: {baseline_f1:.4f}")
print(f"After Selection : {selected_f1:.4f}")

print("\n" + "=" * 70)