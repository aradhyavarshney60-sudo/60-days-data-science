
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "day48_cleaned_customer_churn.csv"
OUTPUT_DIR = BASE_DIR / "day50_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET = "Churn"


def find_target(df):
    """Find a likely churn target column."""
    candidates = ["Churn", "churn", "Exited", "exited", "target"]
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Target column not found. Available columns: {list(df.columns)}"
    )


def main():
    logger.info("Loading customer dataset")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    target_column = find_target(df)
    df = df.dropna(subset=[target_column]).copy()

    # Remove identifiers that should not be predictive features.
    id_columns = [
        col for col in df.columns
        if col.lower() in {"customerid", "customer_id", "id"}
    ]

    y = df[target_column]
    X = df.drop(columns=[target_column] + [
        col for col in id_columns if col in df.columns
    ])

    if X.empty:
        raise ValueError("No feature columns available.")

    # Normalize common Yes/No labels.
    if y.dtype == "object":
        y = y.astype(str).str.strip()
        y = y.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0})

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    # Split BEFORE fitting any preprocessing/model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        col for col in X.columns if col not in numeric_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():
        logger.info("Training %s", name)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
        })

        fitted_models[name] = pipeline

    # Tune Logistic Regression using cross-validation on training data only.
    logger.info("Tuning Logistic Regression")

    tuning_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced"
        )),
    ])

    search = GridSearchCV(
        tuning_pipeline,
        param_grid={
            "model__C": [0.1, 1.0, 10.0],
        },
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    tuned_predictions = search.best_estimator_.predict(X_test)

    results.append({
        "model": "Tuned Logistic Regression",
        "accuracy": accuracy_score(y_test, tuned_predictions),
        "precision": precision_score(
            y_test, tuned_predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, tuned_predictions, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            y_test, tuned_predictions, average="weighted", zero_division=0
        ),
    })

    comparison = pd.DataFrame(results).sort_values(
        "f1_score", ascending=False
    )
    comparison.to_csv(
        OUTPUT_DIR / "day50_model_comparison.csv", index=False
    )

    # Best model according to weighted F1 on the held-out test set.
    best_name = comparison.iloc[0]["model"]
    best_model = (
        search.best_estimator_
        if best_name == "Tuned Logistic Regression"
        else fitted_models[best_name]
    )

    best_predictions = best_model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test, best_predictions, xticks_rotation="vertical"
    )
    plt.title(f"Day 50 Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "day50_confusion_matrix.png", dpi=150
    )
    plt.close()

    report = f"""# Day 50: Model Optimization Report

## Dataset
- Rows used: {len(df)}
- Training rows: {len(X_train)}
- Test rows: {len(X_test)}
- Target column: {target_column}

## Model Comparison
{comparison.to_string(index=False)}

## Best Model
Selected by weighted F1-score on the held-out test set: **{best_name}**

## Classification Report
{classification_report(y_test, best_predictions, zero_division=0)}

## Hyperparameter Tuning
- Model: Logistic Regression
- Best parameters: {search.best_params_}
- Cross-validation scoring: weighted F1-score

## Notes
- Preprocessing was fitted within each training pipeline.
- The test set was held out from model fitting and hyperparameter tuning.
- Model performance may vary with data quality and class distribution.
- This comparison uses the available dataset; external validation may still be needed.
"""

    (OUTPUT_DIR / "day50_optimization_report.md").write_text(
        report, encoding="utf-8"
    )

    logger.info("Best model: %s", best_name)
    logger.info("Results saved to %s", OUTPUT_DIR.resolve())
    logger.info("\n%s", comparison.to_string(index=False))
    logger.info("Day 50 optimization completed successfully.")


if __name__ == "__main__":
    main()
