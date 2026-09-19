# ============================================================
# DAY 26 - HYPERPARAMETER TUNING
# Optimizing ML Systems with Hyperparameter Tuning
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
)
from sklearn.utils.multiclass import type_of_target

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DATASET = "day25_cv_dataset.csv"

# Fallback dataset if Day 25 dataset is not available
FALLBACK_DATASET = "train.csv"

OUTPUT_DATASET = "day26_tuning_dataset.csv"
OUTPUT_COMPARISON = "day26_model_comparison.csv"
OUTPUT_PLOT = "day26_performance_comparison.png"
OUTPUT_PARAMS = "day26_best_parameters.txt"
OUTPUT_REPORT = "day26_hyperparameter_report.txt"
OUTPUT_REFLECTION = "day26_reflection.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def clean_dataset(df):
    """Clean basic missing and infinite values."""

    df = df.copy()

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Fill numeric missing values
    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(exclude=np.number).columns

    for column in categorical_columns:
        if df[column].isna().any():
            mode = df[column].mode()

            if len(mode) > 0:
                df[column] = df[column].fillna(mode.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

    return df


def encode_features(df):
    """Convert categorical features into numerical features."""

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) > 0:
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )

    return df


def convert_numeric(df):
    """Convert remaining values to numeric where possible."""

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# LOAD DATASET
# ============================================================

print_header("DAY 26 - HYPERPARAMETER TUNING")

if os.path.exists(INPUT_DATASET):
    print(f"Loading Day 25 dataset: {INPUT_DATASET}")
    df = pd.read_csv(INPUT_DATASET)
else:
    print(
        f"{INPUT_DATASET} not found."
    )
    print(
        f"Using fallback dataset: {FALLBACK_DATASET}"
    )

    if not os.path.exists(FALLBACK_DATASET):
        raise FileNotFoundError(
            "Neither day25_cv_dataset.csv nor train.csv was found."
        )

    df = pd.read_csv(FALLBACK_DATASET)


print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# CLEAN DATA
# ============================================================

df = clean_dataset(df)

print(f"Cleaned dataset shape: {df.shape}")


# ============================================================
# TARGET SELECTION
# ============================================================

# The final column is used as the target,
# matching the approach used in the previous tasks.

TARGET_COLUMN = df.columns[-1]

print(f"\nTarget column: {TARGET_COLUMN}")

print("\nTarget data type:")
print(df[TARGET_COLUMN].dtype)

print("\nTarget unique values:")
print(df[TARGET_COLUMN].nunique())


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].copy()


# ============================================================
# ENCODE FEATURES
# ============================================================

X = encode_features(X)

X = convert_numeric(X)

# Convert target if necessary
if not pd.api.types.is_numeric_dtype(y):
    y = pd.factorize(y)[0]

y = pd.Series(y).replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove rows where target is missing
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Fill any remaining feature NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Final sample count : {X.shape[0]}")


# ============================================================
# DETECT PROBLEM TYPE
# ============================================================

target_type = type_of_target(y)

if target_type in [
    "binary",
    "multiclass",
    "multiclass-multioutput"
]:
    PROBLEM_TYPE = "classification"
else:
    PROBLEM_TYPE = "regression"

print(f"\nDetected problem type: {PROBLEM_TYPE}")


# ============================================================
# SAVE PROCESSED DATASET
# ============================================================

processed_df = X.copy()
processed_df[TARGET_COLUMN] = y.values

processed_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(f"\nProcessed dataset saved: {OUTPUT_DATASET}")


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

if PROBLEM_TYPE == "classification":

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


print_header("TRAIN-TEST SPLIT")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# BASELINE MODEL
# ============================================================

print_header("BASELINE MODEL")

if PROBLEM_TYPE == "classification":

    baseline_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

else:

    baseline_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            Ridge(alpha=1.0)
        )
    ])


print("Training baseline model...")

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)


# ============================================================
# BASELINE PERFORMANCE
# ============================================================

if PROBLEM_TYPE == "classification":

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        average="weighted"
    )

    print(f"\nBaseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline F1 Score : {baseline_f1:.4f}")

else:

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    print(f"\nBaseline R2   : {baseline_r2:.4f}")
    print(f"Baseline RMSE : {baseline_rmse:.4f}")
    print(f"Baseline MAE  : {baseline_mae:.4f}")


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

print_header("HYPERPARAMETER TUNING")

if PROBLEM_TYPE == "classification":

    tuning_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs"]
    }

    scoring_metric = "f1_weighted"

else:

    tuning_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    param_grid = {
        "model__alpha": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0
        ],
        "model__solver": [
            "auto",
            "lsqr"
        ]
    }

    scoring_metric = "r2"


print(f"Cross-validation folds: {CV_FOLDS}")
print(f"Scoring metric: {scoring_metric}")

print("\nSearching best hyperparameters...")
print("Please wait...")


grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    cv=CV_FOLDS,
    scoring=scoring_metric,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)


# ============================================================
# BEST PARAMETERS
# ============================================================

best_model = grid_search.best_estimator_
best_parameters = grid_search.best_params_
best_cv_score = grid_search.best_score_


print_header("BEST HYPERPARAMETERS")

print("Best Parameters:")

for parameter, value in best_parameters.items():
    print(f"{parameter}: {value}")

print(f"\nBest CV Score: {best_cv_score:.4f}")


# ============================================================
# TUNED MODEL TEST PERFORMANCE
# ============================================================

tuned_predictions = best_model.predict(X_test)


if PROBLEM_TYPE == "classification":

    tuned_accuracy = accuracy_score(
        y_test,
        tuned_predictions
    )

    tuned_f1 = f1_score(
        y_test,
        tuned_predictions,
        average="weighted"
    )

    improvement = tuned_f1 - baseline_f1

    print_header("TUNED MODEL PERFORMANCE")

    print(f"Tuned Accuracy: {tuned_accuracy:.4f}")
    print(f"Tuned F1 Score : {tuned_f1:.4f}")
    print(f"F1 Improvement : {improvement:+.4f}")

else:

    tuned_r2 = r2_score(
        y_test,
        tuned_predictions
    )

    tuned_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            tuned_predictions
        )
    )

    tuned_mae = mean_absolute_error(
        y_test,
        tuned_predictions
    )

    improvement = tuned_r2 - baseline_r2

    print_header("TUNED MODEL PERFORMANCE")

    print(f"Tuned R2   : {tuned_r2:.4f}")
    print(f"Tuned RMSE : {tuned_rmse:.4f}")
    print(f"Tuned MAE  : {tuned_mae:.4f}")
    print(f"R2 Improvement: {improvement:+.4f}")


# ============================================================
# MODEL COMPARISON
# ============================================================

print_header("MODEL COMPARISON")

if PROBLEM_TYPE == "classification":

    comparison = pd.DataFrame({
        "Model": [
            "Baseline Logistic Regression",
            "Tuned Logistic Regression"
        ],
        "Accuracy": [
            baseline_accuracy,
            tuned_accuracy
        ],
        "F1 Score": [
            baseline_f1,
            tuned_f1
        ]
    })

else:

    comparison = pd.DataFrame({
        "Model": [
            "Baseline Ridge Regression",
            "Tuned Ridge Regression"
        ],
        "R2 Score": [
            baseline_r2,
            tuned_r2
        ],
        "RMSE": [
            baseline_rmse,
            tuned_rmse
        ],
        "MAE": [
            baseline_mae,
            tuned_mae
        ]
    })


print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# ============================================================
# PERFORMANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 6))

if PROBLEM_TYPE == "classification":

    models = comparison["Model"]
    scores = comparison["F1 Score"]

    plt.bar(models, scores)
    plt.ylabel("F1 Score")
    plt.title("Baseline vs Tuned Model Performance")
    plt.ylim(
        max(0, min(scores) - 0.05),
        min(1, max(scores) + 0.05)
    )

else:

    models = comparison["Model"]
    scores = comparison["R2 Score"]

    plt.bar(models, scores)
    plt.ylabel("R2 Score")
    plt.title("Baseline vs Tuned Model Performance")

plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=150
)

plt.close()


# ============================================================
# SAVE BEST PARAMETERS
# ============================================================

with open(
    OUTPUT_PARAMS,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 26 - BEST HYPERPARAMETERS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Problem Type: {PROBLEM_TYPE}\n\n"
    )

    file.write(
        "Best Parameters:\n"
    )

    for parameter, value in best_parameters.items():
        file.write(
            f"{parameter}: {value}\n"
        )

    file.write(
        f"\nBest Cross-Validation Score: "
        f"{best_cv_score:.4f}\n"
    )


# ============================================================
# PERFORMANCE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 26 - HYPERPARAMETER TUNING REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Dataset: {INPUT_DATASET}\n"
    )

    file.write(
        f"Target: {TARGET_COLUMN}\n"
    )

    file.write(
        f"Problem Type: {PROBLEM_TYPE}\n"
    )

    file.write(
        f"Cross-Validation: {CV_FOLDS}-Fold\n\n"
    )

    file.write(
        "BEST PARAMETERS\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    for parameter, value in best_parameters.items():

        file.write(
            f"{parameter}: {value}\n"
        )

    file.write(
        f"\nBest CV Score: {best_cv_score:.4f}\n\n"
    )

    file.write(
        "MODEL COMPARISON\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        comparison.to_string(index=False)
    )

    file.write(
        "\n\n"
    )

    file.write(
        "OPTIMIZATION OBSERVATION\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    if improvement > 0:

        file.write(
            "Hyperparameter tuning improved the "
            "test performance of the model.\n"
        )

    elif improvement < 0:

        file.write(
            "The tuned model did not improve the "
            "test metric compared with the baseline.\n"
        )

    else:

        file.write(
            "The tuned and baseline models produced "
            "the same test metric.\n"
        )


# ============================================================
# REFLECTION
# ============================================================

with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 26 - HYPERPARAMETER TUNING REFLECTION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "Today I worked on hyperparameter tuning "
        "to optimize Machine Learning model performance.\n\n"
    )

    file.write(
        "I used GridSearchCV with cross-validation "
        "to systematically evaluate different "
        "hyperparameter combinations.\n\n"
    )

    file.write(
        "The main goal was to compare a baseline "
        "model with a tuned model and understand "
        "how hyperparameter selection affects "
        "model performance.\n\n"
    )

    file.write(
        "The best parameters were selected using "
        "cross-validation rather than relying on "
        "a single train-test split.\n\n"
    )

    file.write(
        "This task helped me understand that model "
        "optimization involves balancing predictive "
        "performance, validation reliability and "
        "computational cost.\n\n"
    )

    file.write(
        f"Best CV Score: {best_cv_score:.4f}\n"
    )

    file.write(
        f"Test Metric Improvement: {improvement:+.4f}\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("GENERATED FILES")

print(f"1. {OUTPUT_DATASET}")
print(f"2. {OUTPUT_COMPARISON}")
print(f"3. {OUTPUT_PLOT}")
print(f"4. {OUTPUT_PARAMS}")
print(f"5. {OUTPUT_REPORT}")
print(f"6. {OUTPUT_REFLECTION}")

print("\n" + "=" * 70)
print("ALL DAY 26 TASKS COMPLETED")
print("=" * 70)