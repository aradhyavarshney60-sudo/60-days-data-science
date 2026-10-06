# ============================================================
# DAY 43/60 - TURNING CUSTOMER INTELLIGENCE MODEL INTO AN API
# ============================================================

import os
import warnings
from typing import List

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "WA_FnUseC_TelcoCustomerChurn.csv"

MODEL_FILE = "day43_customer_churn_model.joblib"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Day 43 Data Science Project - "
        "Machine Learning API for Customer Churn Prediction"
    ),
    version="1.0.0",
)


# ============================================================
# GLOBAL MODEL INFORMATION
# ============================================================

model = None
model_metrics = {}
feature_columns = []


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the Telco Customer Churn dataset.
    """

    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    print("\n" + "=" * 60)
    print("DATASET LOADED")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# DATA PREPROCESSING
# ============================================================

def prepare_data(df):
    """
    Clean dataset and prepare features and target.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Remove customer ID
    # --------------------------------------------------------

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --------------------------------------------------------
    # Convert TotalCharges to numeric
    # --------------------------------------------------------

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Target column
    # --------------------------------------------------------

    target_column = "Churn"

    if target_column not in df.columns:
        raise ValueError(
            "Churn column was not found in the dataset."
        )

    # --------------------------------------------------------
    # Remove missing target values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[target_column]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column].astype(str).str.strip().str.lower()

    # --------------------------------------------------------
    # Convert Yes/No to 1/0
    # --------------------------------------------------------

    target_mapping = {
        "no": 0,
        "yes": 1
    }

    y = y.map(target_mapping)

    if y.isna().any():
        raise ValueError(
            "Churn target contains unexpected values. "
            "Expected only Yes and No."
        )

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    print(
        y.value_counts()
    )

    print("\nTarget mapping:")
    print("No  -> 0")
    print("Yes -> 1")

    return X, y


# ============================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================

def build_pipeline(X):
    """
    Build preprocessing and Logistic Regression pipeline.
    """

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Combine preprocessing
    # --------------------------------------------------------

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
        ]
    )

    # --------------------------------------------------------
    # Complete ML pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    return pipeline


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():
    """
    Train the customer churn prediction model.
    """

    global model
    global model_metrics
    global feature_columns

    df = load_dataset()

    X, y = prepare_data(df)

    feature_columns = X.columns.tolist()

    print("\n" + "=" * 60)
    print("FEATURE INFORMATION")
    print("=" * 60)

    print(
        f"Number of features: {len(feature_columns)}"
    )

    print(
        "\nFeatures:"
    )

    for column in feature_columns:
        print(f"- {column}")

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print(
        f"Training rows : {len(X_train)}"
    )

    print(
        f"Testing rows  : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_pipeline(X)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model training completed successfully."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    model_metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4)
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Saved as: {MODEL_FILE}"
    )

    return model


# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================

def initialize_model():
    """
    Load saved model if available.
    Otherwise train a new model.
    """

    global model

    if os.path.exists(MODEL_FILE):

        try:

            model = joblib.load(
                MODEL_FILE
            )

            print("\n" + "=" * 60)
            print("SAVED MODEL LOADED")
            print("=" * 60)

            print(
                f"Model file: {MODEL_FILE}"
            )

            return

        except Exception as error:

            print(
                f"Could not load saved model: {error}"
            )

            print(
                "Training a new model..."
            )

    train_model()


# ============================================================
# PYDANTIC INPUT MODEL
# ============================================================

class CustomerData(BaseModel):

    gender: str = "Male"

    SeniorCitizen: int = Field(
        default=0,
        ge=0,
        le=1
    )

    Partner: str = "Yes"

    Dependents: str = "No"

    tenure: int = Field(
        default=12,
        ge=0
    )

    PhoneService: str = "Yes"

    MultipleLines: str = "No"

    InternetService: str = "DSL"

    OnlineSecurity: str = "No"

    OnlineBackup: str = "No"

    DeviceProtection: str = "No"

    TechSupport: str = "No"

    StreamingTV: str = "No"

    StreamingMovies: str = "No"

    Contract: str = "Month-to-month"

    PaperlessBilling: str = "Yes"

    PaymentMethod: str = "Electronic check"

    MonthlyCharges: float = Field(
        default=70.0,
        ge=0
    )

    TotalCharges: float = Field(
        default=840.0,
        ge=0
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Customer Churn Prediction API",
        "day": "43/60",
        "status": "running",
        "docs": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_FILE
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "Logistic Regression",
        "problem": "Binary Customer Churn Classification",
        "target": "Churn",
        "target_mapping": {
            "No": 0,
            "Yes": 1
        },
        "features": feature_columns,
        "metrics": model_metrics
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_customer(
    customer: CustomerData
):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    # --------------------------------------------------------
    # Convert request to DataFrame
    # --------------------------------------------------------

    customer_dict = customer.model_dump()

    input_df = pd.DataFrame(
        [customer_dict]
    )

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(
            input_df
        )[0]
    )

    probability = float(
        model.predict_proba(
            input_df
        )[0][1]
    )

    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    if prediction == 1:

        risk_level = "High"

        recommendation = (
            "Customer is likely to churn. "
            "Consider retention offers, personalized "
            "support, or contract incentives."
        )

    else:

        risk_level = "Low"

        recommendation = (
            "Customer is unlikely to churn. "
            "Continue normal engagement and service."
        )

    return {
        "prediction": prediction,
        "churn_prediction": (
            "Yes" if prediction == 1 else "No"
        ),
        "churn_probability": round(
            probability,
            4
        ),
        "risk_level": risk_level,
        "recommendation": recommendation
    }


# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-batch")
def predict_batch(
    customers: List[CustomerData]
):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    if len(customers) == 0:

        raise HTTPException(
            status_code=400,
            detail="Customer list cannot be empty."
        )

    # --------------------------------------------------------
    # Convert customers to DataFrame
    # --------------------------------------------------------

    data = [
        customer.model_dump()
        for customer in customers
    ]

    input_df = pd.DataFrame(
        data
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        input_df
    )

    probabilities = model.predict_proba(
        input_df
    )[:, 1]

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):

        prediction = int(
            prediction
        )

        probability = float(
            probability
        )

        results.append(
            {
                "prediction": prediction,
                "churn_prediction": (
                    "Yes"
                    if prediction == 1
                    else "No"
                ),
                "churn_probability": round(
                    probability,
                    4
                )
            }
        )

    return {
        "total_customers": len(results),
        "predictions": results
    }


# ============================================================
# STARTUP
# ============================================================

initialize_model()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "day43_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )