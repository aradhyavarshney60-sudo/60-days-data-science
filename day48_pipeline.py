
import logging
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("WA_FnUseC_TelcoCustomerChurn.csv")
OUTPUT_PATH = Path("day48_cleaned_customer_churn.csv")


def build_preprocessing_pipeline(numeric_columns, categorical_columns):
    """Build a reusable preprocessing pipeline."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])


def main():
    if not DATA_PATH.exists():
        logger.error("Dataset not found: %s", DATA_PATH)
        return

    logger.info("Loading customer churn dataset")
    df = pd.read_csv(DATA_PATH)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    logger.info("Dataset shape: %s", df.shape)
    logger.info("Duplicate rows: %s", df.duplicated().sum())

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Clean TotalCharges, which may contain blank strings
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Remove rows without the target label
    target = "Churn"
    if target not in df.columns:
        logger.error("Target column '%s' is missing", target)
        return

    df = df.dropna(subset=[target]).copy()

    # Keep customer ID separate from model features
    if "customerID" in df.columns:
        customer_ids = df["customerID"].copy()
        X = df.drop(columns=["customerID", target])
    else:
        customer_ids = None
        X = df.drop(columns=[target])

    y = df[target].copy()

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    logger.info("Numeric features: %s", numeric_columns)
    logger.info("Categorical feature count: %s", len(categorical_columns))

    preprocessor = build_preprocessing_pipeline(
        numeric_columns, categorical_columns
    )

    # Fit and transform features for the capstone dataset
    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray")
        else X_processed,
        columns=feature_names,
        index=df.index
    )

    # Save transformed features with target and optional customer ID
    processed_df["Churn"] = y
    if customer_ids is not None:
        processed_df.insert(0, "customerID", customer_ids.loc[df.index])

    processed_df.to_csv(OUTPUT_PATH, index=False)

    logger.info("Cleaned dataset saved to %s", OUTPUT_PATH)
    logger.info("Processed dataset shape: %s", processed_df.shape)
    logger.info("Missing values remaining: %s", processed_df.isna().sum().sum())


if __name__ == "__main__":
    main()
