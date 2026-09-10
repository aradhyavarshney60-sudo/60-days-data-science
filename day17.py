# ============================================================
# DAY 17 - LOAN APPROVAL PREDICTION USING DECISION TREES
# 60 Days Data Science Challenge
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_FILE = "loan_approval_dataset.csv"

PREDICTIONS_FILE = "day17_predictions.csv"
MODEL_RESULTS_FILE = "day17_model_results.csv"
FEATURE_IMPORTANCE_FILE = "day17_feature_importance.csv"

CONFUSION_MATRIX_FILE = "day17_confusion_matrix.png"
DECISION_TREE_FILE = "day17_decision_tree.png"
FEATURE_IMPORTANCE_IMAGE = "day17_feature_importance.png"
OVERFITTING_IMAGE = "day17_overfitting_analysis.png"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("DAY 17 - LOAN APPROVAL PREDICTION")
print("Using Decision Trees")
print("=" * 60)

print("\nLoading loan approval dataset...")

if not os.path.exists(DATASET_FILE):
    print("\nDataset file not found.")
    print(f"Please place '{DATASET_FILE}' in the project folder.")
    raise FileNotFoundError(DATASET_FILE)

df = pd.read_csv(DATASET_FILE)

print("Loan approval dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. BASIC DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nRemoved {duplicate_count} duplicate rows.")
else:
    print("\nNo duplicate rows found.")


# ============================================================
# 6. IDENTIFY TARGET COLUMN
# ============================================================

possible_targets = [
    "loan_status",
    "loan_approved",
    "loan_approval",
    "approved",
    "approval",
    "target",
    "status"
]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    raise ValueError(
        "Target column not found. Expected one of: "
        + ", ".join(possible_targets)
    )

print(f"\nTarget column identified: {target_column}")


# ============================================================
# 7. CLEAN TARGET COLUMN
# ============================================================

df = df.dropna(subset=[target_column]).copy()

# Remove extra spaces from string target values
if df[target_column].dtype == "object":
    df[target_column] = df[target_column].astype(str).str.strip()


# ============================================================
# 8. CONVERT TARGET TO NUMERIC
# ============================================================

target_values = df[target_column].unique()

print("\nTarget values:")
print(target_values)

# Common loan approval values:
# Y / N
# Yes / No
# Approved / Rejected
# 1 / 0

if df[target_column].dtype == "object":

    target_lower = df[target_column].astype(str).str.lower().str.strip()

    mapping = {
        "y": 1,
        "yes": 1,
        "approved": 1,
        "approve": 1,
        "accepted": 1,
        "accept": 1,
        "1": 1,
        "true": 1,

        "n": 0,
        "no": 0,
        "rejected": 0,
        "reject": 0,
        "declined": 0,
        "decline": 0,
        "0": 0,
        "false": 0
    }

    mapped_target = target_lower.map(mapping)

    # If all values were successfully mapped
    if mapped_target.notna().all():
        df[target_column] = mapped_target.astype(int)

    else:
        # Generic binary encoding
        unique_values = list(df[target_column].dropna().unique())

        if len(unique_values) == 2:
            generic_mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            df[target_column] = (
                df[target_column]
                .map(generic_mapping)
                .astype(int)
            )
        else:
            raise ValueError(
                "Target column must contain two classes "
                "for binary loan approval classification."
            )

else:
    # Numeric target
    unique_values = sorted(df[target_column].dropna().unique())

    if len(unique_values) == 2:
        if set(unique_values) != {0, 1}:
            target_mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            df[target_column] = df[target_column].map(
                target_mapping
            ).astype(int)

    else:
        raise ValueError(
            "Target column must contain exactly two classes."
        )


print("\nTarget distribution:")
print(df[target_column].value_counts())


# ============================================================
# 9. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()


# ============================================================
# 10. REMOVE ID-LIKE COLUMNS
# ============================================================

id_like_columns = []

for column in X.columns:

    column_lower = column.lower()

    if (
        column_lower in ["loan_id", "id", "customer_id", "customerid"]
        or column_lower.endswith("_id")
    ):
        id_like_columns.append(column)

if id_like_columns:
    X = X.drop(columns=id_like_columns)
    print("\nRemoved ID-like columns:")
    print(id_like_columns)


# ============================================================
# 11. CONVERT NUMERIC-LIKE COLUMNS
# ============================================================

for column in X.columns:

    if X[column].dtype == "object":

        cleaned = (
            X[column]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
        )

        numeric_version = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        # Convert to numeric if most values are numeric
        valid_ratio = numeric_version.notna().mean()

        if valid_ratio >= 0.80:
            X[column] = numeric_version


# ============================================================
# 12. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 13. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nData split completed.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 14. DATA PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 15. CREATE DECISION TREE MODEL
# ============================================================

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            decision_tree
        )
    ]
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DECISION TREE")
print("=" * 60)

model.fit(X_train, y_train)

print("Decision Tree classifier trained successfully.")


# ============================================================
# 17. GENERATE PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Predictions generated successfully.")


# ============================================================
# 18. MODEL EVALUATION
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Day 17 - Loan Approval Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix created successfully.")


# ============================================================
# 21. FEATURE NAMES AFTER PREPROCESSING
# ============================================================

preprocessor_fitted = model.named_steps["preprocessor"]

feature_names = []

# Numeric features
feature_names.extend(numeric_features)

# Categorical one-hot features
if len(categorical_features) > 0:

    onehot_encoder = (
        preprocessor_fitted
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )

    categorical_feature_names = (
        onehot_encoder
        .get_feature_names_out(categorical_features)
        .tolist()
    )

    feature_names.extend(
        categorical_feature_names
    )


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

classifier = model.named_steps["classifier"]

importances = classifier.feature_importances_

feature_importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
)

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

feature_importance_df.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)

print("\nFeature importance analysis completed.")

print("\nTop 10 important features:")
print(
    feature_importance_df.head(10)
)


# ============================================================
# 23. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = feature_importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Day 17 - Decision Tree Feature Importance"
)

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Feature importance visualization created successfully.")


# ============================================================
# 24. DECISION TREE VISUALIZATION
# ============================================================

X_train_processed = preprocessor_fitted.transform(
    X_train
)

plt.figure(figsize=(24, 14))

plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=[
        "Rejected",
        "Approved"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Day 17 - Loan Approval Decision Tree"
)

plt.tight_layout()

plt.savefig(
    DECISION_TREE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision tree visualization created successfully.")


# ============================================================
# 25. OVERFITTING ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS")
print("=" * 60)

depth_values = list(range(1, 16))

training_scores = []
testing_scores = []

for depth in depth_values:

    temp_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    temp_tree.fit(
        X_train_processed,
        y_train
    )

    train_score = temp_tree.score(
        X_train_processed,
        y_train
    )

    X_test_processed = preprocessor_fitted.transform(
        X_test
    )

    test_score = temp_tree.score(
        X_test_processed,
        y_test
    )

    training_scores.append(
        train_score
    )

    testing_scores.append(
        test_score
    )


plt.figure(figsize=(10, 6))

plt.plot(
    depth_values,
    training_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_values,
    testing_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")

plt.title(
    "Day 17 - Decision Tree Overfitting Analysis"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OVERFITTING_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Overfitting analysis completed.")


# ============================================================
# 26. IDENTIFY POSSIBLE OVERFITTING
# ============================================================

accuracy_gap = train_accuracy - test_accuracy

print(f"\nTraining Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")
print(f"Accuracy Gap      : {accuracy_gap:.4f}")

if accuracy_gap > 0.10:

    overfitting_status = "Possible Overfitting"

elif accuracy_gap > 0.05:

    overfitting_status = "Mild Overfitting"

else:

    overfitting_status = "No Significant Overfitting"

print(
    f"Overfitting Status: {overfitting_status}"
)


# ============================================================
# 27. SAVE PREDICTION RESULTS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Loan_Status"] = (
    y_test.values
)

prediction_results["Predicted_Loan_Status"] = (
    y_test_pred
)

prediction_results["Prediction_Correct"] = (
    prediction_results["Actual_Loan_Status"]
    ==
    prediction_results["Predicted_Loan_Status"]
)

prediction_results.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print("\nPrediction results saved successfully.")


# ============================================================
# 28. SAVE MODEL RESULTS
# ============================================================

model_results = pd.DataFrame(
    {
        "Metric": [
            "Training Accuracy",
            "Testing Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Accuracy Gap",
            "Overfitting Status",
            "Tree Depth",
            "Number of Training Samples",
            "Number of Testing Samples"
        ],
        "Value": [
            train_accuracy,
            test_accuracy,
            precision,
            recall,
            f1,
            accuracy_gap,
            overfitting_status,
            decision_tree.max_depth,
            len(X_train),
            len(X_test)
        ]
    }
)

model_results.to_csv(
    MODEL_RESULTS_FILE,
    index=False
)

print("Model results saved successfully.")


# ============================================================
# 29. DISPLAY ERROR ANALYSIS
# ============================================================

false_positives = (
    (y_test == 0)
    &
    (y_test_pred == 1)
).sum()

false_negatives = (
    (y_test == 1)
    &
    (y_test_pred == 0)
).sum()

true_positives = (
    (y_test == 1)
    &
    (y_test_pred == 1)
).sum()

true_negatives = (
    (y_test == 0)
    &
    (y_test_pred == 0)
).sum()


print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print(f"True Positives  : {true_positives}")
print(f"True Negatives  : {true_negatives}")
print(f"False Positives : {false_positives}")
print(f"False Negatives : {false_negatives}")


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("Loan approval dataset loaded successfully.")
print("Decision Tree classifier trained successfully.")
print("Predictions generated successfully.")
print("Decision tree visualization created successfully.")
print("Feature importance analysis completed.")
print("Overfitting analysis completed.")
print("Confusion matrix created successfully.")
print("Prediction results saved successfully.")
print("Model results saved successfully.")

print(
    "\nDay 17 Loan Approval Prediction "
    "using Decision Trees completed successfully! 🌳"
)

print("=" * 60)