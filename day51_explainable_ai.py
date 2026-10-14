
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_PATH = Path("day48_cleaned_customer_churn.csv")
OUTPUT_DIR = Path("day51_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' was not found.")

    # Remove rows with missing target values.
    df = df.dropna(subset=["Churn"]).copy()

    # Encode target as 0/1.
    y = df["Churn"].astype(str).str.strip().str.lower()
    y = y.map({"yes": 1, "no": 0, "1": 1, "0": 0})

    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # Remove target and identifier-like columns from features.
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])

    logger.info("Training Gradient Boosting model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Feature names after preprocessing.
    fitted_preprocessor = model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()

    trained_model = model.named_steps["model"]
    importances = trained_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "day51_feature_importance.csv", index=False
    )

    top_features = importance_df.head(15).sort_values("importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Customer Churn Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "day51_feature_importance.png", dpi=150)
    plt.close()

    # SHAP explanations on a manageable test sample.
    X_sample = X_test.sample(
        n=min(100, len(X_test)), random_state=42
    )
    X_transformed = fitted_preprocessor.transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(X_transformed)
    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X_transformed)

    # Binary classifiers may return a list for each class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Some SHAP versions return (rows, features, classes).
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "day51_shap_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": mean_abs_shap,
    }).sort_values("mean_absolute_shap", ascending=False)

    shap_df.to_csv(OUTPUT_DIR / "day51_shap_importance.csv", index=False)

    top_drivers = shap_df.head(10)
    driver_lines = "\n".join(
        f"- **{row.feature}**: mean absolute SHAP = "
        f"{row.mean_absolute_shap:.4f}"
        for row in top_drivers.itertuples()
    )

    report = f"""# Day 51: Explainable AI for Customer Churn

## Objective
Explain customer churn model behavior using feature importance and SHAP.

## Evaluation
- Model: Gradient Boosting Classifier
- Hold-out test accuracy: {accuracy:.4f}
- Hold-out test F1-score: {f1:.4f}
- Training/test split: 80% / 20%
- Random state: 42

## Top SHAP Drivers
{driver_lines}

## Business Interpretation
Features with larger mean absolute SHAP values have a greater average
influence on model output for the analyzed sample. This ranking does not
by itself show whether a feature increases or decreases churn probability.
Use the SHAP summary plot to inspect direction and distribution.

These are model associations, not proof of causation. Business teams should
validate findings before taking customer-facing actions.

## Generated Outputs
- `day51_feature_importance.csv`
- `day51_feature_importance.png`
- `day51_shap_summary.png`
- `day51_shap_importance.csv`
- `day51_business_interpretation_report.md`
"""
    (OUTPUT_DIR / "day51_business_interpretation_report.md").write_text(
        report, encoding="utf-8"
    )

    logger.info("Accuracy: %.2f%%", accuracy * 100)
    logger.info("F1-score: %.4f", f1)
    logger.info("Top feature: %s", importance_df.iloc[0]["feature"])
    logger.info("Day 51 outputs saved in: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
