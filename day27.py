# ============================================================
# DAY 27 - BIAS VS VARIANCE IN ML SYSTEMS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DATASET = "day26_tuning_dataset.csv"

OUTPUT_ANALYSIS = "day27_bias_variance_analysis.csv"
OUTPUT_LEARNING_CURVE = "day27_learning_curve.png"
OUTPUT_COMPARISON = "day27_performance_comparison.png"
OUTPUT_REPORT = "day27_report.txt"
OUTPUT_REFLECTION = "day27_reflection.txt"

RANDOM_STATE = 42

# To make the large Day 26 dataset practical to run
MAX_ROWS = 5000
MAX_FEATURES = 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def detect_target(df):
    """
    Try to detect the target column automatically.
    """

    possible_targets = [
        "target",
        "Target",
        "TARGET",
        "y",
        "Y",
        "label",
        "Label",
        "price",
        "Price",
        "SalePrice",
        "sale_price"
    ]

    for column in possible_targets:
        if column in df.columns:
            return column

    # If target cannot be detected by name,
    # use the last numeric column.
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) < 2:
        raise ValueError(
            "Could not automatically identify the target column."
        )

    return numeric_columns[-1]


# ============================================================
# LOAD DATA
# ============================================================

print_header("DAY 27 - BIAS VS VARIANCE ANALYSIS")

print("Loading dataset...")
print(f"Dataset: {INPUT_DATASET}")

if not os.path.exists(INPUT_DATASET):
    raise FileNotFoundError(
        f"{INPUT_DATASET} not found in the current folder."
    )


# Read only a manageable sample from the large dataset
df = pd.read_csv(
    INPUT_DATASET,
    nrows=MAX_ROWS
)

print(f"\nLoaded rows     : {df.shape[0]}")
print(f"Loaded columns  : {df.shape[1]}")


# ============================================================
# TARGET DETECTION
# ============================================================

TARGET_COLUMN = detect_target(df)

print(f"Target column   : {TARGET_COLUMN}")


# ============================================================
# DATA PREPARATION
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=[TARGET_COLUMN])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# Keep numeric features only
X = X.select_dtypes(include=np.number)

# Fill missing values
X = X.fillna(X.median(numeric_only=True))

print(f"Numeric features: {X.shape[1]}")


# ============================================================
# FEATURE REDUCTION FOR PRACTICAL EXECUTION
# ============================================================

if X.shape[1] > MAX_FEATURES:

    print(
        f"\nDataset contains {X.shape[1]} numeric features."
    )

    print(
        f"Selecting top {MAX_FEATURES} features by variance "
        "for bias-variance analysis..."
    )

    variances = X.var()

    selected_features = (
        variances
        .sort_values(ascending=False)
        .head(MAX_FEATURES)
        .index
    )

    X = X[selected_features]

print(f"Features used   : {X.shape[1]}")


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTRAIN-TEST SPLIT")
print("-" * 50)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# BIAS VS VARIANCE ANALYSIS
# ============================================================

print_header("BIAS VS VARIANCE ANALYSIS")

depths = [2, 4, 6, 8, 12, 16, None]

results = []


for depth in depths:

    depth_name = "Unlimited" if depth is None else str(depth)

    print(f"\nEvaluating Decision Tree | max_depth = {depth_name}")

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_r2 = r2_score(y_train, train_prediction)
    test_r2 = r2_score(y_test, test_prediction)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_prediction)
    )

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    r2_gap = train_r2 - test_r2

    if train_r2 < 0.70 and test_r2 < 0.70:
        behavior = "Underfitting"

    elif r2_gap > 0.15:
        behavior = "Overfitting"

    else:
        behavior = "Better Generalization"

    results.append({
        "max_depth": depth_name,
        "train_r2": train_r2,
        "validation_r2": test_r2,
        "r2_gap": r2_gap,
        "train_rmse": train_rmse,
        "validation_rmse": test_rmse,
        "train_mae": train_mae,
        "validation_mae": test_mae,
        "behavior": behavior
    })

    print(f"Training R2    : {train_r2:.4f}")
    print(f"Validation R2  : {test_r2:.4f}")
    print(f"R2 Gap         : {r2_gap:.4f}")
    print(f"Training RMSE  : {train_rmse:.4f}")
    print(f"Validation RMSE: {test_rmse:.4f}")
    print(f"Behavior       : {behavior}")


results_df = pd.DataFrame(results)


# ============================================================
# SAVE ANALYSIS
# ============================================================

results_df.to_csv(
    OUTPUT_ANALYSIS,
    index=False
)

print(f"\nSaved: {OUTPUT_ANALYSIS}")


# ============================================================
# FIND BEST GENERALIZATION MODEL
# ============================================================

best_index = results_df["validation_r2"].idxmax()

best_row = results_df.loc[best_index]

best_depth = best_row["max_depth"]

print_header("BEST GENERALIZATION MODEL")

print(f"Best max_depth : {best_depth}")
print(f"Validation R2  : {best_row['validation_r2']:.4f}")
print(f"Training R2    : {best_row['train_r2']:.4f}")
print(f"R2 Gap         : {best_row['r2_gap']:.4f}")
print(f"Behavior       : {best_row['behavior']}")


# ============================================================
# PERFORMANCE COMPARISON PLOT
# ============================================================

plt.figure(figsize=(10, 6))

x_labels = results_df["max_depth"].astype(str)

plt.plot(
    x_labels,
    results_df["train_r2"],
    marker="o",
    label="Training R²"
)

plt.plot(
    x_labels,
    results_df["validation_r2"],
    marker="o",
    label="Validation R²"
)

plt.xlabel("Decision Tree Max Depth")
plt.ylabel("R² Score")
plt.title("Bias-Variance Tradeoff: Training vs Validation R²")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_COMPARISON,
    dpi=150
)

plt.close()

print(f"Saved: {OUTPUT_COMPARISON}")


# ============================================================
# LEARNING CURVE
# ============================================================

print_header("LEARNING CURVE")

best_depth_value = (
    None
    if best_depth == "Unlimited"
    else int(best_depth)
)

learning_model = DecisionTreeRegressor(
    max_depth=best_depth_value,
    random_state=RANDOM_STATE
)

print("Generating learning curve...")
print("Please wait...")


train_sizes, train_scores, validation_scores = learning_curve(
    learning_model,
    X_train,
    y_train,
    cv=3,
    scoring="r2",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

validation_mean = validation_scores.mean(axis=1)
validation_std = validation_scores.std(axis=1)


# ============================================================
# LEARNING CURVE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training R²"
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation R²"
)

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15
)

plt.fill_between(
    train_sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    alpha=0.15
)

plt.xlabel("Training Set Size")
plt.ylabel("R² Score")

plt.title(
    "Learning Curve - Bias vs Variance"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_LEARNING_CURVE,
    dpi=150
)

plt.close()

print(f"Saved: {OUTPUT_LEARNING_CURVE}")


# ============================================================
# REPORT
# ============================================================

report = f"""
DAY 27 - BIAS VS VARIANCE REPORT
================================

Dataset
-------
Input Dataset       : {INPUT_DATASET}
Rows Used           : {len(df)}
Features Used       : {X.shape[1]}
Target Column       : {TARGET_COLUMN}

Train-Test Split
----------------
Training Samples    : {len(X_train)}
Testing Samples     : {len(X_test)}
Test Size           : 20%
Random State        : {RANDOM_STATE}

Bias-Variance Analysis
----------------------

Decision Tree models were evaluated using different
maximum depths.

"""

for _, row in results_df.iterrows():

    report += f"""
Max Depth           : {row['max_depth']}
Training R2         : {row['train_r2']:.4f}
Validation R2       : {row['validation_r2']:.4f}
R2 Gap              : {row['r2_gap']:.4f}
Training RMSE       : {row['train_rmse']:.4f}
Validation RMSE     : {row['validation_rmse']:.4f}
Training MAE        : {row['train_mae']:.4f}
Validation MAE      : {row['validation_mae']:.4f}
Behavior            : {row['behavior']}
"""


report += f"""

Best Generalization
-------------------
Best Max Depth      : {best_depth}
Training R2         : {best_row['train_r2']:.4f}
Validation R2       : {best_row['validation_r2']:.4f}
R2 Gap              : {best_row['r2_gap']:.4f}
Behavior            : {best_row['behavior']}

Key Observations
----------------
1. Lower model complexity can lead to underfitting.
2. Increasing model complexity can improve training performance.
3. Excessive complexity can create a large training-validation gap.
4. The training-validation gap is useful for identifying overfitting.
5. Learning curves help analyze model generalization behavior.
6. The objective is to find a model that generalizes well
   rather than simply maximizing training performance.
"""


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print(f"Saved: {OUTPUT_REPORT}")


# ============================================================
# REFLECTION
# ============================================================

reflection = f"""
DAY 27 REFLECTION
=================

Today I studied the Bias-Variance Tradeoff in Machine Learning.

I compared Decision Tree models with different levels of
complexity using training and validation performance.

The analysis showed how model complexity affects
generalization.

A model with low complexity can underfit the data because
it may not capture enough patterns.

As complexity increases, training performance generally
improves. However, excessive complexity can increase the
difference between training and validation performance,
which is a common sign of overfitting.

For this task, the model with max_depth = {best_depth}
achieved the highest validation R2 of
{best_row['validation_r2']:.4f} among the tested models.

I also generated a learning curve to understand how model
performance changes as the training dataset grows.

The main lesson from Day 27 is that good Machine Learning
models should not only perform well on training data.
They should also generalize well to unseen data.

This helped me understand the practical importance of
balancing bias and variance while building ML systems.
"""


with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(reflection)


print(f"Saved: {OUTPUT_REFLECTION}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("GENERATED FILES")

print(f"1. {OUTPUT_ANALYSIS}")
print(f"2. {OUTPUT_LEARNING_CURVE}")
print(f"3. {OUTPUT_COMPARISON}")
print(f"4. {OUTPUT_REPORT}")
print(f"5. {OUTPUT_REFLECTION}")

print("\n" + "=" * 70)
print("ALL DAY 27 TASKS COMPLETED")
print("=" * 70)