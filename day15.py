# ============================================================
# DAY 15 - PREDICTING CUSTOMER CHURN WITH LOGISTIC REGRESSION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD CUSTOMER CHURN DATASET
# ============================================================

print("=" * 60)
print("DAY 15 - CUSTOMER CHURN PREDICTION")
print("=" * 60)

data_url = (
    "https://raw.githubusercontent.com/"
    "IBM/watsonx-ai-samples/master/"
    "cpd4.5/data/customer_churn/"
    "WA_FnUseC_TelcoCustomerChurn.csv"
)

try:
    df = pd.read_csv("WA_FnUseC_TelcoCustomerChurn.csv")
    print("\nLocal dataset loaded successfully!")
except FileNotFoundError:
    print("\nLocal dataset not found.")
    print("Downloading customer churn dataset...")
    df = pd.read_csv(data_url)
    df.to_csv("WA_FnUseC_TelcoCustomerChurn.csv", index=False)
    print("Dataset downloaded and saved successfully!")


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# Remove customer ID because it is only an identifier
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Clean TotalCharges
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# ============================================================
# 4. IDENTIFY TARGET
# ============================================================

target_column = "Churn"

if target_column not in df.columns:
    raise ValueError(
        "Churn column was not found in the dataset."
    )

print("\nTarget Column:")
print(target_column)

print("\nTarget Distribution:")
print(df[target_column].value_counts())


# ============================================================
# 5. CONVERT TARGET INTO 0 AND 1
# ============================================================

df[target_column] = df[target_column].map({
    "No": 0,
    "Yes": 1
})

# Remove rows where target could not be converted
df = df.dropna(subset=[target_column])

df[target_column] = df[target_column].astype(int)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("0 = Customer did not churn")
print("1 = Customer churned")


# ============================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 9. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
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
    ]
)


# ============================================================
# 10. BUILD LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ============================================================
# 12. GENERATE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully!")


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

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

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nTrue Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)


# ============================================================
# 16. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION ERROR ANALYSIS")
print("=" * 60)

print(f"\nFalse Positives: {fp}")
print(f"False Negatives: {fn}")

print("\nBusiness Interpretation:")

print(
    "\nFalse Positive:"
    "\nThe model predicted that a customer would churn,"
    "\nbut the customer actually did not churn."
)

print(
    "\nBusiness Impact:"
    "\nThe company may spend unnecessary money on"
    "\nretention offers or promotional campaigns."
)

print(
    "\nFalse Negative:"
    "\nThe model predicted that a customer would not churn,"
    "\nbut the customer actually churned."
)

print(
    "\nBusiness Impact:"
    "\nThe company may fail to identify an at-risk customer"
    "\nand lose an opportunity to retain them."
)


# ============================================================
# 17. CREATE PREDICTION OUTPUT
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Churn"] = y_test.values

prediction_results["Predicted_Churn"] = y_pred

prediction_results["Churn_Probability"] = y_probability

prediction_results["Prediction_Result"] = np.where(
    y_test.values == y_pred,
    "Correct",
    "Incorrect"
)

prediction_results.to_csv(
    "day15_predictions.csv",
    index=False
)

print(
    "\nPrediction results saved successfully!"
)


# ============================================================
# 18. SAVE MODEL RESULTS
# ============================================================

model_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp
    ]
})

model_results.to_csv(
    "day15_model_results.csv",
    index=False
)

print(
    "Model evaluation results saved successfully!"
)


# ============================================================
# 19. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Day 15 - Customer Churn Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Predicted No Churn", "Predicted Churn"]
)

plt.yticks(
    [0, 1],
    ["Actual No Churn", "Actual Churn"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "day15_confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved successfully!"
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 15 SUMMARY")
print("=" * 60)

print("Customer churn dataset loaded successfully.")
print("Logistic Regression model trained successfully.")
print("Predictions generated successfully.")
print("False Positive and False Negative analysis completed.")
print("Confusion matrix created successfully.")
print("Prediction results saved successfully.")
print("Model results saved successfully.")

print(
    "\nDay 15 Predicting Customer Churn "
    "with Logistic Regression completed successfully! 🚀"
)