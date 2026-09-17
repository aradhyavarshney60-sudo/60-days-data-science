# ============================================================
# DAY 24 - PCA (PRINCIPAL COMPONENT ANALYSIS)
# 60 DAYS DATA SCIENCE
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "feature_engineered_train.csv"

OUTPUT_DATASET = "day24_pca_dataset.csv"
OUTPUT_COMPARISON = "day24_pca_comparison.csv"
OUTPUT_PLOT = "day24_pca_visualization.png"
OUTPUT_VARIANCE = "day24_explained_variance.png"
OUTPUT_REPORT = "day24_pca_performance_report.txt"
OUTPUT_REFLECTION = "day24_reflection.txt"

RANDOM_STATE = 42


# ============================================================
# 2. START
# ============================================================

print("=" * 70)
print("DAY 24 - PCA DIMENSIONALITY REDUCTION")
print("=" * 70)


# ============================================================
# 3. LOAD DATASET
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"\n'{INPUT_FILE}' not found.\n"
        "Make sure the CSV file is in the same folder as day24.py."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset Loaded Successfully!")
print(f"Dataset Shape: {df.shape}")

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")
print(list(df.columns))

print("\nMissing Values:")
print(df.isnull().sum())


# Remove completely empty columns
df = df.dropna(axis=1, how="all")


# ============================================================
# 5. IDENTIFY TARGET COLUMN
# ============================================================

possible_targets = [
    "target",
    "Target",
    "label",
    "Label",
    "y",
    "Y",
    "loan_approved",
    "Loan_Approved",
    "approved",
    "Approved",
    "fraud",
    "Fraud",
    "class",
    "Class"
]

target_column = None

for col in possible_targets:

    if col in df.columns:

        target_column = col
        break


# If no common target is found,
# use the last column
if target_column is None:

    target_column = df.columns[-1]


print(f"\nTarget Column Selected: {target_column}")


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_columns = X.select_dtypes(
    include=np.number
).columns

categorical_columns = X.select_dtypes(
    exclude=np.number
).columns


# Numerical missing values
for col in numeric_columns:

    X[col] = X[col].fillna(
        X[col].median()
    )


# Categorical missing values
for col in categorical_columns:

    if X[col].isna().any():

        mode_value = X[col].mode()

        if len(mode_value) > 0:

            X[col] = X[col].fillna(
                mode_value.iloc[0]
            )

        else:

            X[col] = X[col].fillna("Unknown")


# ============================================================
# 8. ENCODE TARGET
# ============================================================

target_encoder = LabelEncoder()

if (
    y.dtype == "object"
    or str(y.dtype).startswith("category")
):

    y = target_encoder.fit_transform(y)

else:

    y = y.to_numpy()


y = np.asarray(y)

print("\nTarget Classes:")
print(np.unique(y))


# ============================================================
# 9. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE PREPROCESSING")
print("=" * 70)

print(
    f"\nNumerical Features: "
    f"{len(numeric_columns)}"
)

print(
    f"Categorical Features: "
    f"{len(categorical_columns)}"
)


if len(categorical_columns) > 0:

    X = pd.get_dummies(
        X,
        columns=list(categorical_columns),
        drop_first=True
    )


# Convert boolean columns to integers
for col in X.columns:

    if X[col].dtype == bool:

        X[col] = X[col].astype(int)


# Convert everything to float
X = X.astype(float)


print(
    f"\nFeatures After Encoding: "
    f"{X.shape[1]}"
)


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"\nTraining Samples: "
    f"{X_train.shape[0]}"
)

print(
    f"Testing Samples : "
    f"{X_test.shape[0]}"
)


# ============================================================
# 11. STANDARDIZATION
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("Standardization Completed.")


# ============================================================
# 12. BASELINE MODEL BEFORE PCA
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL - BEFORE PCA")
print("=" * 70)


baseline_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)


baseline_model.fit(
    X_train_scaled,
    y_train
)


baseline_predictions = baseline_model.predict(
    X_test_scaled
)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="weighted",
    zero_division=0
)


print(
    f"\nAccuracy : "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{baseline_precision:.4f}"
)

print(
    f"Recall   : "
    f"{baseline_recall:.4f}"
)

print(
    f"F1 Score : "
    f"{baseline_f1:.4f}"
)


# ============================================================
# 13. PCA WITH 2 COMPONENTS
# ============================================================

print("\n" + "=" * 70)
print("APPLYING PCA - 2 COMPONENTS")
print("=" * 70)


pca_2 = PCA(
    n_components=2,
    svd_solver="randomized",
    random_state=RANDOM_STATE
)


X_train_pca = pca_2.fit_transform(
    X_train_scaled
)

X_test_pca = pca_2.transform(
    X_test_scaled
)


explained_variance_2 = (
    pca_2.explained_variance_ratio_
)


pc1_variance = (
    explained_variance_2[0] * 100
)

pc2_variance = (
    explained_variance_2[1] * 100
)

total_variance_2 = (
    explained_variance_2.sum() * 100
)


print(
    f"\nPC1 Explained Variance: "
    f"{pc1_variance:.2f}%"
)

print(
    f"PC2 Explained Variance: "
    f"{pc2_variance:.2f}%"
)

print(
    f"Total Variance Retained: "
    f"{total_variance_2:.2f}%"
)


# ============================================================
# 14. MODEL AFTER PCA
# ============================================================

print("\n" + "=" * 70)
print("MODEL AFTER PCA")
print("=" * 70)


pca_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)


pca_model.fit(
    X_train_pca,
    y_train
)


pca_predictions = pca_model.predict(
    X_test_pca
)


pca_accuracy = accuracy_score(
    y_test,
    pca_predictions
)

pca_precision = precision_score(
    y_test,
    pca_predictions,
    average="weighted",
    zero_division=0
)

pca_recall = recall_score(
    y_test,
    pca_predictions,
    average="weighted",
    zero_division=0
)

pca_f1 = f1_score(
    y_test,
    pca_predictions,
    average="weighted",
    zero_division=0
)


print(
    f"\nAccuracy : "
    f"{pca_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{pca_precision:.4f}"
)

print(
    f"Recall   : "
    f"{pca_recall:.4f}"
)

print(
    f"F1 Score : "
    f"{pca_f1:.4f}"
)


# ============================================================
# 15. SAVE PCA DATASET
# ============================================================

print("\nSaving PCA dataset...")


pca_dataset = pd.DataFrame(
    X_train_pca,
    columns=[
        "Principal_Component_1",
        "Principal_Component_2"
    ]
)

pca_dataset["Target"] = y_train


pca_dataset.to_csv(
    OUTPUT_DATASET,
    index=False
)


# ============================================================
# 16. PERFORMANCE COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression - Before PCA",
        "Logistic Regression - After PCA"
    ],

    "Features": [
        X.shape[1],
        2
    ],

    "Accuracy": [
        baseline_accuracy,
        pca_accuracy
    ],

    "Precision": [
        baseline_precision,
        pca_precision
    ],

    "Recall": [
        baseline_recall,
        pca_recall
    ],

    "F1_Score": [
        baseline_f1,
        pca_f1
    ]
})


comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 17. PCA VISUALIZATION
# ============================================================

print("\nCreating PCA visualization...")


plt.figure(
    figsize=(10, 7)
)


unique_classes = np.unique(
    y_train
)


for class_value in unique_classes:

    mask = (
        y_train == class_value
    )

    plt.scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
        label=f"Class {class_value}",
        alpha=0.7
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "PCA Visualization - 2 Principal Components"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 18. EXPLAINED VARIANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXPLAINED VARIANCE ANALYSIS")
print("=" * 70)


# IMPORTANT:
# Do NOT calculate PCA for all 13,373 features.
# We only calculate up to 50 components.

variance_components = min(
    50,
    X_train_scaled.shape[1],
    X_train_scaled.shape[0]
)


print(
    f"\nCalculating explained variance "
    f"for {variance_components} components..."
)


pca_variance = PCA(
    n_components=variance_components,
    svd_solver="randomized",
    random_state=RANDOM_STATE
)


pca_variance.fit(
    X_train_scaled
)


explained_variance = (
    pca_variance.explained_variance_ratio_
)


cumulative_variance = np.cumsum(
    explained_variance
)


# Check 90% variance
if np.any(
    cumulative_variance >= 0.90
):

    components_90 = (
        np.argmax(
            cumulative_variance >= 0.90
        ) + 1
    )

else:

    components_90 = (
        "Not reached within calculated components"
    )


# Check 95% variance
if np.any(
    cumulative_variance >= 0.95
):

    components_95 = (
        np.argmax(
            cumulative_variance >= 0.95
        ) + 1
    )

else:

    components_95 = (
        "Not reached within calculated components"
    )


print(
    f"\nVariance calculated for first "
    f"{variance_components} components."
)

print(
    f"\nVariance retained by first 2 components: "
    f"{total_variance_2:.2f}%"
)

print(
    f"Variance retained by first "
    f"{variance_components} components: "
    f"{cumulative_variance[-1] * 100:.2f}%"
)

print(
    f"\nComponents for 90% variance: "
    f"{components_90}"
)

print(
    f"Components for 95% variance: "
    f"{components_95}"
)


# ============================================================
# 19. EXPLAINED VARIANCE PLOT
# ============================================================

print("\nCreating explained variance plot...")


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance,
    marker="o"
)


plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% Variance"
)


plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% Variance"
)


plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Explained Variance Analysis"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_VARIANCE,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 20. PERFORMANCE REPORT
# ============================================================

feature_reduction = (
    (
        X.shape[1] - 2
    )
    /
    X.shape[1]
) * 100


report = f"""
DAY 24 - PCA PERFORMANCE REPORT
================================

Dataset
-------
Input File: {INPUT_FILE}

Original Features: {X.shape[1]}

PCA Features: 2

Feature Reduction:
{feature_reduction:.2f}%


PCA EXPLAINED VARIANCE
----------------------

PC1 Variance:
{pc1_variance:.2f}%

PC2 Variance:
{pc2_variance:.2f}%

Total Variance Retained by 2 Components:
{total_variance_2:.2f}%


Variance Analysis
-----------------

Components calculated:
{variance_components}

Variance retained by calculated components:
{cumulative_variance[-1] * 100:.2f}%

Components required for 90% variance:
{components_90}

Components required for 95% variance:
{components_95}


MODEL PERFORMANCE
=================

BEFORE PCA
----------

Features:
{X.shape[1]}

Accuracy:
{baseline_accuracy:.4f}

Precision:
{baseline_precision:.4f}

Recall:
{baseline_recall:.4f}

F1 Score:
{baseline_f1:.4f}


AFTER PCA
---------

Features:
2

Accuracy:
{pca_accuracy:.4f}

Precision:
{pca_precision:.4f}

Recall:
{pca_recall:.4f}

F1 Score:
{pca_f1:.4f}


CONCLUSION
==========

PCA reduced the feature space from
{X.shape[1]} features to 2 principal components.

The first two principal components retained
{total_variance_2:.2f}% of the total variance.

Model performance was compared before and after
dimensionality reduction.

PCA can simplify high-dimensional datasets,
reduce computational complexity and make data
easier to visualize.

However, reducing the number of components too
aggressively may remove useful predictive information.

Therefore, the number of PCA components should be
selected based on the amount of variance retained
and the requirements of the ML problem.
"""


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# 21. REFLECTION
# ============================================================

reflection = f"""
DAY 24 REFLECTION - PCA
=======================

Today I learned about Principal Component Analysis (PCA)
and how it can be used for dimensionality reduction.

The dataset originally contained {X.shape[1]} features.

I standardized the features and applied PCA to transform
the data into 2 principal components.

The first two principal components retained
{total_variance_2:.2f}% of the total variance.

Before PCA, Logistic Regression achieved an
F1 Score of {baseline_f1:.4f}.

After PCA, the F1 Score was {pca_f1:.4f}.

This experiment helped me understand that PCA can
compress high-dimensional data while preserving
important patterns.

I also learned that choosing too few components can
remove useful information, so explained variance and
model performance should both be considered.

Another step completed in my Data Science journey.
"""


with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(reflection)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DAY 24 COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nGenerated Files:")

print(
    f"1. {OUTPUT_DATASET}"
)

print(
    f"2. {OUTPUT_COMPARISON}"
)

print(
    f"3. {OUTPUT_PLOT}"
)

print(
    f"4. {OUTPUT_VARIANCE}"
)

print(
    f"5. {OUTPUT_REPORT}"
)

print(
    f"6. {OUTPUT_REFLECTION}"
)


print("\n" + "=" * 70)
print("PCA SUMMARY")
print("=" * 70)


print(
    f"Original Features : "
    f"{X.shape[1]}"
)

print(
    f"PCA Features      : 2"
)

print(
    f"Variance Retained : "
    f"{total_variance_2:.2f}%"
)


print("\nF1 Score:")

print(
    f"Before PCA: "
    f"{baseline_f1:.4f}"
)

print(
    f"After PCA : "
    f"{pca_f1:.4f}"
)


print("\n" + "=" * 70)
print("ALL DAY 24 TASKS COMPLETED")
print("=" * 70)