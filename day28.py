# ============================================================
# DAY 28 - BUILDING YOUR MOST OPTIMIZED ML SYSTEM YET
# 60 DAYS DATA SCIENCE
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import is_classifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DATASET = "day28_optimized_dataset.csv"
OUTPUT_COMPARISON = "day28_model_comparison.csv"
OUTPUT_PLOT = "day28_model_performance.png"
OUTPUT_REPORT = "day28_final_performance_report.txt"
OUTPUT_REFLECTION = "day28_sprint_reflection.txt"

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_dataset():
    """
    Try to use the dataset created during previous days.
    """

    candidates = [
        "day27_bias_variance_analysis.csv",
        "day26_tuning_dataset.csv",
        "day25_cv_dataset.csv",
        "day24_pca_dataset.csv",
        "train.csv"
    ]

    for file in candidates:
        if os.path.exists(file):
            print(f"Dataset found: {file}")
            return file

    raise FileNotFoundError(
        "No suitable dataset found. "
        "Please keep day26_tuning_dataset.csv or train.csv "
        "in the same folder as day28.py."
    )


def detect_target(df):
    """
    Automatically detect target column.
    """

    possible_targets = [
        "target",
        "Target",
        "TARGET",
        "price",
        "Price",
        "SalePrice",
        "sale_price",
        "label",
        "Label",
        "y",
        "Y",
        "output",
        "Output"
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    # Common classification targets
    for col in df.columns:
        unique_values = df[col].nunique()

        if unique_values == 2:
            print(f"Binary target candidate detected: {col}")
            return col

    # Last column fallback
    target = df.columns[-1]

    print(
        f"No standard target name found. "
        f"Using last column as target: {target}"
    )

    return target


def detect_problem_type(y):
    """
    Determine regression vs classification.
    """

    if y.dtype == "object":
        return "classification"

    if y.nunique() <= 10:
        return "classification"

    return "regression"


def clean_dataset(df):
    """
    Basic cleaning.
    """

    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Removed duplicate rows: {before - after}")

    return df


# ============================================================
# LOAD DATA
# ============================================================

print_header("DAY 28 - OPTIMIZED ML SYSTEM")

DATASET = find_dataset()

df = pd.read_csv(DATASET)

print(f"\nDataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

df = clean_dataset(df)

target_column = detect_target(df)

print(f"\nTarget column: {target_column}")

# Save optimized dataset
df.to_csv(OUTPUT_DATASET, index=False)


# ============================================================
# TARGET / FEATURES
# ============================================================

X = df.drop(columns=[target_column])
y = df[target_column]

problem_type = detect_problem_type(y)

print(f"Problem type: {problem_type}")

print("\nTarget statistics:")
print(y.describe(include="all"))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

if problem_type == "classification":

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# MODEL DEFINITIONS
# ============================================================

if problem_type == "regression":

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=5,
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

else:

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


# ============================================================
# MODEL TRAINING AND COMPARISON
# ============================================================

results = []

print_header("MODEL COMPARISON")

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if problem_type == "regression":

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}")

        results.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    else:

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

print_header("FINAL MODEL COMPARISON")

print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

if problem_type == "regression":

    best_index = results_df["R2"].idxmax()
    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]

else:

    best_index = results_df["F1"].idxmax()
    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]


best_model = models[best_model_name]

print(f"\nSelected final model: {best_model_name}")


# ============================================================
# FINAL OPTIMIZED PIPELINE
# ============================================================

final_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", best_model)
    ]
)

print("\nTraining final optimized pipeline...")

final_pipeline.fit(
    X_train,
    y_train
)

final_predictions = final_pipeline.predict(X_test)


# ============================================================
# FINAL METRICS
# ============================================================

print_header("FINAL PERFORMANCE")

if problem_type == "regression":

    final_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            final_predictions
        )
    )

    final_mae = mean_absolute_error(
        y_test,
        final_predictions
    )

    final_r2 = r2_score(
        y_test,
        final_predictions
    )

    print(f"Final Model: {best_model_name}")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"MAE : {final_mae:.4f}")
    print(f"R2  : {final_r2:.4f}")

else:

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    final_accuracy = accuracy_score(
        y_test,
        final_predictions
    )

    final_precision = precision_score(
        y_test,
        final_predictions,
        average="weighted",
        zero_division=0
    )

    final_recall = recall_score(
        y_test,
        final_predictions,
        average="weighted",
        zero_division=0
    )

    final_f1 = f1_score(
        y_test,
        final_predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Final Model: {best_model_name}")
    print(f"Accuracy : {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall   : {final_recall:.4f}")
    print(f"F1 Score : {final_f1:.4f}")


# ============================================================
# PERFORMANCE VISUALIZATION
# ============================================================

print("\nCreating performance visualization...")

plt.figure(figsize=(10, 6))

if problem_type == "regression":

    plt.bar(
        results_df["Model"],
        results_df["R2"]
    )

    plt.ylabel("R² Score")
    plt.title("Day 28 - Model Performance Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

else:

    plt.bar(
        results_df["Model"],
        results_df["F1"]
    )

    plt.ylabel("F1 Score")
    plt.title("Day 28 - Model Performance Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()


plt.savefig(
    OUTPUT_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL PERFORMANCE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("DAY 28 - FINAL ML PERFORMANCE REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Dataset: {DATASET}\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Target Column: {target_column}\n")
    f.write(f"Problem Type: {problem_type}\n\n")

    f.write("FEATURE INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Numeric Features: {len(numeric_features)}\n"
    )
    f.write(
        f"Categorical Features: {len(categorical_features)}\n"
    )
    f.write(
        f"Training Samples: {len(X_train)}\n"
    )
    f.write(
        f"Testing Samples: {len(X_test)}\n\n"
    )

    f.write("MODEL COMPARISON\n")
    f.write("-" * 70 + "\n")

    f.write(
        results_df.to_string(index=False)
    )

    f.write("\n\n")
    f.write("FINAL SELECTED MODEL\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Model: {best_model_name}\n"
    )

    if problem_type == "regression":

        f.write(
            f"RMSE: {final_rmse:.4f}\n"
        )

        f.write(
            f"MAE: {final_mae:.4f}\n"
        )

        f.write(
            f"R2: {final_r2:.4f}\n"
        )

    else:

        f.write(
            f"Accuracy: {final_accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {final_precision:.4f}\n"
        )

        f.write(
            f"Recall: {final_recall:.4f}\n"
        )

        f.write(
            f"F1 Score: {final_f1:.4f}\n"
        )

    f.write("\n")
    f.write("ENGINEERING TRADEOFFS\n")
    f.write("-" * 70 + "\n")
    f.write(
        "The final pipeline combines preprocessing, "
        "missing-value handling, categorical encoding, "
        "feature scaling, model training and evaluation "
        "into a reproducible workflow.\n"
    )


# ============================================================
# SPRINT REFLECTION
# ============================================================

with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("DAY 28 - WEEK 4 OPTIMIZATION REFLECTION\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. What I built\n")
    f.write("-" * 70 + "\n")
    f.write(
        "I built an optimized machine learning pipeline "
        "combining preprocessing, feature handling, "
        "model comparison and final evaluation.\n\n"
    )

    f.write("2. What I learned\n")
    f.write("-" * 70 + "\n")
    f.write(
        "I learned that model performance depends not only "
        "on the algorithm but also on preprocessing, "
        "validation strategy and model configuration.\n\n"
    )

    f.write("3. Optimization process\n")
    f.write("-" * 70 + "\n")
    f.write(
        "Multiple models were trained and evaluated using "
        "the same preprocessing pipeline. The final model "
        "was selected based on the evaluation metric "
        "appropriate for the problem type.\n\n"
    )

    f.write("4. Engineering tradeoffs\n")
    f.write("-" * 70 + "\n")
    f.write(
        "More complex ensemble models can improve predictive "
        "performance but may require more computation and "
        "reduce interpretability compared with simpler models.\n\n"
    )

    f.write("5. Key takeaway\n")
    f.write("-" * 70 + "\n")
    f.write(
        "A reliable ML system requires a reproducible pipeline "
        "rather than simply selecting a model with a high score.\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("GENERATED FILES")

print(f"1. {OUTPUT_DATASET}")
print(f"2. {OUTPUT_COMPARISON}")
print(f"3. {OUTPUT_PLOT}")
print(f"4. {OUTPUT_REPORT}")
print(f"5. {OUTPUT_REFLECTION}")

print("\n" + "=" * 70)
print("ALL DAY 28 TASKS COMPLETED")
print("=" * 70)