# ============================================================
# DAY 25/60
# BUILDING RELIABLE MODELS WITH CROSS-VALIDATION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_DATASET = "train.csv"

OUTPUT_DATASET = "day25_cv_dataset.csv"
OUTPUT_COMPARISON = "day25_cv_comparison.csv"
OUTPUT_PLOT = "day25_cv_visualization.png"
OUTPUT_STABILITY = "day25_cv_stability.png"
OUTPUT_REPORT = "day25_cv_performance_report.txt"
OUTPUT_REFLECTION = "day25_cv_reflection.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Important for high-dimensional dataset
SVD_COMPONENTS = 50


# ============================================================
# 2. HEADER
# ============================================================

print("\n" + "=" * 70)
print("DAY 25/60 - BUILDING RELIABLE MODELS WITH CROSS-VALIDATION")
print("=" * 70)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(INPUT_DATASET):
    raise FileNotFoundError(
        f"Dataset '{INPUT_DATASET}' was not found."
    )

df = pd.read_csv(INPUT_DATASET)

print("\nDataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 4. TARGET COLUMN
# ============================================================

# This dataset uses "count" as the regression target.
if "count" in df.columns:
    TARGET_COLUMN = "count"
else:
    TARGET_COLUMN = df.columns[-1]

print("\n" + "-" * 70)
print("TARGET ANALYSIS")
print("-" * 70)

print(f"\nTarget column: {TARGET_COLUMN}")
print(f"Target dtype : {df[TARGET_COLUMN].dtype}")
print(f"Unique values: {df[TARGET_COLUMN].nunique()}")


# ============================================================
# 5. REMOVE MISSING TARGET
# ============================================================

before_rows = len(df)

df = df.dropna(
    subset=[TARGET_COLUMN]
).reset_index(drop=True)

after_rows = len(df)

if before_rows != after_rows:
    print(
        f"\nRemoved {before_rows - after_rows} "
        "rows with missing target."
    )


# ============================================================
# 6. FORCE REGRESSION
# ============================================================

PROBLEM_TYPE = "regression"

print(f"\nProblem type: {PROBLEM_TYPE.upper()}")


# ============================================================
# 7. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(
    f"\nProcessed dataset saved: {OUTPUT_DATASET}"
)


# ============================================================
# 8. FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET_COLUMN]
)

y = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

# Remove rows where target could not be converted
valid_rows = y.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)


print("\n" + "-" * 70)
print("FEATURE INFORMATION")
print("-" * 70)

print(f"\nTotal features: {X.shape[1]}")


# ============================================================
# 9. FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()

print(
    f"Numeric features    : {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 11. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


print("\n" + "-" * 70)
print("TRAIN-TEST SPLIT")
print("-" * 70)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# 12. CROSS-VALIDATION SETUP
# ============================================================

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print(
    f"\nCross-Validation: {N_SPLITS}-Fold"
)


# ============================================================
# 13. MODELS
# ============================================================

models = {

    "Ridge Regression":
        Ridge(
            alpha=1.0
        ),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

    "Gradient Boosting Regressor":
        GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )
}


# ============================================================
# 14. RESULTS STORAGE
# ============================================================

results = []
fold_results = []


print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Evaluating: {model_name}")
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            # Reduce 13,000+ dimensions to 50
            (
                "svd",
                TruncatedSVD(
                    n_components=SVD_COMPONENTS,
                    n_iter=3,
                    random_state=RANDOM_STATE
                )
            ),

            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    print("Running 5-Fold Cross-Validation...")

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1
    )

    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()

    print(
        f"Fold Scores : {np.round(cv_scores, 4)}"
    )

    print(
        f"Mean CV R2  : {mean_cv:.4f}"
    )

    print(
        f"Std CV R2   : {std_cv:.4f}"
    )


    # Save individual fold results

    for fold_number, score in enumerate(
        cv_scores,
        start=1
    ):

        fold_results.append({

            "Model": model_name,

            "Fold": fold_number,

            "R2_Score": score
        })


    # --------------------------------------------------------
    # TRAIN-TEST EVALUATION
    # --------------------------------------------------------

    print("\nTraining on training set...")

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )


    test_r2 = r2_score(
        y_test,
        predictions
    )

    test_mae = mean_absolute_error(
        y_test,
        predictions
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    print(
        f"Test R2     : {test_r2:.4f}"
    )

    print(
        f"Test MAE    : {test_mae:.4f}"
    )

    print(
        f"Test RMSE   : {test_rmse:.4f}"
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Problem_Type": PROBLEM_TYPE,

        "CV_Mean_R2": mean_cv,

        "CV_Std_R2": std_cv,

        "Test_R2": test_r2,

        "Test_MAE": test_mae,

        "Test_RMSE": test_rmse

    })


# ============================================================
# 16. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

fold_df = pd.DataFrame(
    fold_results
)


# ============================================================
# 17. SAVE COMPARISON
# ============================================================

results_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)


print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. BEST MODEL
# ============================================================

best_index = results_df[
    "CV_Mean_R2"
].idxmax()

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_cv_score = results_df.loc[
    best_index,
    "CV_Mean_R2"
]

best_cv_std = results_df.loc[
    best_index,
    "CV_Std_R2"
]

best_test_r2 = results_df.loc[
    best_index,
    "Test_R2"
]

best_test_mae = results_df.loc[
    best_index,
    "Test_MAE"
]

best_test_rmse = results_df.loc[
    best_index,
    "Test_RMSE"
]


# ============================================================
# 19. PERFORMANCE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

x = np.arange(
    len(results_df)
)

width = 0.35

plt.bar(
    x - width / 2,
    results_df["CV_Mean_R2"],
    width,
    label="Mean CV R2"
)

plt.bar(
    x + width / 2,
    results_df["Test_R2"],
    width,
    label="Test R2"
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=15
)

plt.ylabel("R2 Score")

plt.title(
    "Day 25 - Cross-Validation vs Test Performance"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=150
)

plt.close()


# ============================================================
# 20. STABILITY VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for model_name in fold_df["Model"].unique():

    model_data = fold_df[
        fold_df["Model"] == model_name
    ]

    plt.plot(
        model_data["Fold"],
        model_data["R2_Score"],
        marker="o",
        label=model_name
    )


plt.xlabel("Fold")

plt.ylabel("R2 Score")

plt.title(
    "Day 25 - Cross-Validation Stability"
)

plt.xticks(
    range(1, N_SPLITS + 1)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_STABILITY,
    dpi=150
)

plt.close()


# ============================================================
# 21. PERFORMANCE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 25 - CROSS-VALIDATION PERFORMANCE REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
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
        f"Cross-Validation: {N_SPLITS}-Fold\n"
    )

    file.write(
        f"SVD Components: {SVD_COMPONENTS}\n\n"
    )

    file.write(
        "MODEL COMPARISON\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        results_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\nBEST MODEL BASED ON MEAN CV R2\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"Model: {best_model_name}\n"
    )

    file.write(
        f"Mean CV R2: {best_cv_score:.4f}\n"
    )

    file.write(
        f"CV Standard Deviation: {best_cv_std:.4f}\n"
    )

    file.write(
        f"Test R2: {best_test_r2:.4f}\n"
    )

    file.write(
        f"Test MAE: {best_test_mae:.4f}\n"
    )

    file.write(
        f"Test RMSE: {best_test_rmse:.4f}\n"
    )


# ============================================================
# 22. REFLECTION
# ============================================================

with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 25 - CROSS-VALIDATION REFLECTION\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        "Objective:\n"
    )

    file.write(
        "Evaluate model reliability using 5-fold "
        "cross-validation and compare it with "
        "train-test performance.\n\n"
    )

    file.write(
        "Observations:\n"
    )

    file.write(
        "1. Cross-validation provides performance "
        "across multiple data splits.\n"
    )

    file.write(
        "2. Mean CV R2 represents average model "
        "performance across folds.\n"
    )

    file.write(
        "3. CV standard deviation indicates "
        "performance variation between folds.\n"
    )

    file.write(
        "4. The test R2 provides performance on "
        "the held-out test dataset.\n"
    )

    file.write(
        "5. Comparing CV and test performance helps "
        "understand generalization behavior.\n"
    )

    file.write(
        "6. Dimensionality reduction was used because "
        "the dataset contains a very large number "
        "of encoded features.\n\n"
    )

    file.write(
        f"Model with highest mean CV R2: "
        f"{best_model_name}\n"
    )

    file.write(
        f"Mean CV R2: {best_cv_score:.4f}\n"
    )

    file.write(
        f"CV Standard Deviation: {best_cv_std:.4f}\n"
    )


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    f"\nCross-Validation : {N_SPLITS}-Fold"
)

print(
    f"Best Model       : {best_model_name}"
)

print(
    f"Mean CV R2       : {best_cv_score:.4f}"
)

print(
    f"CV Standard Dev  : {best_cv_std:.4f}"
)

print(
    f"Test R2          : {best_test_r2:.4f}"
)

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
    f"4. {OUTPUT_STABILITY}"
)

print(
    f"5. {OUTPUT_REPORT}"
)

print(
    f"6. {OUTPUT_REFLECTION}"
)

print("\n" + "=" * 70)
print("ALL DAY 25 TASKS COMPLETED")
print("=" * 70)