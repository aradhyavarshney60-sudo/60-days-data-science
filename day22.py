# ============================================================
# DAY 22/60 - FEATURE ENCODING
# Turning Raw Categories into Machine Learning Signals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "fraud_detection_dataset.csv"

OUTPUT_DATASET = "day22_encoded_dataset.csv"
OUTPUT_RESULTS = "day22_encoding_comparison.csv"
OUTPUT_PNG = "day22_encoding_comparison.png"
OUTPUT_REPORT = "day22_performance_report.txt"
OUTPUT_REFLECTION = "day22_reflection.txt"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("DAY 22 - FEATURE ENCODING")
print("=" * 70)

print("\nLoading dataset...")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"\nERROR: {INPUT_FILE} not found.")
    print("Make sure the CSV file is in the same folder as day22.py")
    exit()


print("\nDataset loaded successfully!")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. FIND TARGET COLUMN
# ============================================================

possible_targets = [
    "is_fraud",
    "fraud",
    "Fraud",
    "target",
    "Target",
    "class",
    "Class",
    "label",
    "Label"
]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break


if target_column is None:
    print("\nTarget column automatically selected as last column.")
    target_column = df.columns[-1]


print(f"\nTarget column: {target_column}")


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)


# ============================================================
# 6. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print("\nCategorical columns:")
if categorical_columns:
    for column in categorical_columns:
        print(f"  - {column}")
else:
    print("  No categorical columns found.")

print("\nNumerical columns:")
if numerical_columns:
    for column in numerical_columns:
        print(f"  - {column}")
else:
    print("  No numerical columns found.")


# ============================================================
# 7. BEFORE ENCODING
# ============================================================

before_shape = X.shape

print("\n" + "=" * 70)
print("BEFORE ENCODING")
print("=" * 70)

print(f"\nRows    : {before_shape[0]}")
print(f"Columns : {before_shape[1]}")


# ============================================================
# 8. LABEL ENCODING
# ============================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

X_label = X.copy()

label_encoders = {}

for column in categorical_columns:

    le = LabelEncoder()

    X_label[column] = X_label[column].fillna("Missing").astype(str)

    X_label[column] = le.fit_transform(X_label[column])

    label_encoders[column] = le

    print(f"\n{column}:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))


print("\nLabel Encoding completed.")


# ============================================================
# 9. SAVE LABEL ENCODED DATASET
# ============================================================

label_encoded_dataset = X_label.copy()
label_encoded_dataset[target_column] = y.values

label_encoded_dataset.to_csv(
    "day22_label_encoded_dataset.csv",
    index=False
)

print("\nSaved: day22_label_encoded_dataset.csv")


# ============================================================
# 10. ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 70)
print("ONE-HOT ENCODING")
print("=" * 70)

if categorical_columns:

    X_onehot = pd.get_dummies(
        X,
        columns=categorical_columns,
        dummy_na=True
    )

else:

    X_onehot = X.copy()


print("\nOne-Hot Encoding completed.")

print(f"\nBefore Encoding columns : {X.shape[1]}")
print(f"After Encoding columns  : {X_onehot.shape[1]}")


# ============================================================
# 11. SAVE ONE-HOT ENCODED DATASET
# ============================================================

onehot_encoded_dataset = X_onehot.copy()
onehot_encoded_dataset[target_column] = y.values

onehot_encoded_dataset.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(f"\nSaved: {OUTPUT_DATASET}")


# ============================================================
# 12. DATASET STRUCTURE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("DATASET STRUCTURE COMPARISON")
print("=" * 70)

print("\nOriginal dataset:")
print(f"Rows    : {X.shape[0]}")
print(f"Columns : {X.shape[1]}")

print("\nLabel Encoded dataset:")
print(f"Rows    : {X_label.shape[0]}")
print(f"Columns : {X_label.shape[1]}")

print("\nOne-Hot Encoded dataset:")
print(f"Rows    : {X_onehot.shape[0]}")
print(f"Columns : {X_onehot.shape[1]}")


# ============================================================
# 13. PREPARE DATA FOR MODEL
# ============================================================

print("\n" + "=" * 70)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 70)

# Convert target to numeric if necessary
if y.dtype == "object" or str(y.dtype) == "category":

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))

else:

    y_encoded = y.values


# Check binary classification
unique_classes = np.unique(y_encoded)

if len(unique_classes) != 2:
    print(
        f"\nWARNING: Target has {len(unique_classes)} classes."
    )

print("Target classes:", unique_classes)


# ============================================================
# 14. TRAIN TEST SPLIT - LABEL ENCODING
# ============================================================

X_train_label, X_test_label, y_train, y_test = train_test_split(
    X_label,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ============================================================
# 15. LABEL ENCODED MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL - LABEL ENCODING")
print("=" * 70)

label_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

label_model.fit(X_train_label, y_train)

label_predictions = label_model.predict(X_test_label)


# ============================================================
# 16. LABEL ENCODING METRICS
# ============================================================

label_accuracy = accuracy_score(
    y_test,
    label_predictions
)

label_precision = precision_score(
    y_test,
    label_predictions,
    average="weighted",
    zero_division=0
)

label_recall = recall_score(
    y_test,
    label_predictions,
    average="weighted",
    zero_division=0
)

label_f1 = f1_score(
    y_test,
    label_predictions,
    average="weighted",
    zero_division=0
)


print("\nLabel Encoding Results:")
print(f"Accuracy  : {label_accuracy:.4f}")
print(f"Precision : {label_precision:.4f}")
print(f"Recall    : {label_recall:.4f}")
print(f"F1 Score  : {label_f1:.4f}")


# ============================================================
# 17. TRAIN TEST SPLIT - ONE-HOT ENCODING
# ============================================================

X_train_onehot, X_test_onehot, y_train2, y_test2 = train_test_split(
    X_onehot,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ============================================================
# 18. ONE-HOT ENCODED MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL - ONE-HOT ENCODING")
print("=" * 70)

onehot_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

onehot_model.fit(
    X_train_onehot,
    y_train2
)

onehot_predictions = onehot_model.predict(
    X_test_onehot
)


# ============================================================
# 19. ONE-HOT ENCODING METRICS
# ============================================================

onehot_accuracy = accuracy_score(
    y_test2,
    onehot_predictions
)

onehot_precision = precision_score(
    y_test2,
    onehot_predictions,
    average="weighted",
    zero_division=0
)

onehot_recall = recall_score(
    y_test2,
    onehot_predictions,
    average="weighted",
    zero_division=0
)

onehot_f1 = f1_score(
    y_test2,
    onehot_predictions,
    average="weighted",
    zero_division=0
)


print("\nOne-Hot Encoding Results:")
print(f"Accuracy  : {onehot_accuracy:.4f}")
print(f"Precision : {onehot_precision:.4f}")
print(f"Recall    : {onehot_recall:.4f}")
print(f"F1 Score  : {onehot_f1:.4f}")


# ============================================================
# 20. PERFORMANCE COMPARISON
# ============================================================

results = pd.DataFrame({
    "Encoding": [
        "Label Encoding",
        "One-Hot Encoding"
    ],
    "Accuracy": [
        label_accuracy,
        onehot_accuracy
    ],
    "Precision": [
        label_precision,
        onehot_precision
    ],
    "Recall": [
        label_recall,
        onehot_recall
    ],
    "F1 Score": [
        label_f1,
        onehot_f1
    ],
    "Number of Features": [
        X_label.shape[1],
        X_onehot.shape[1]
    ]
})


print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print("\n")
print(results.to_string(index=False))


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_RESULTS,
    index=False
)

print(f"\nSaved: {OUTPUT_RESULTS}")


# ============================================================
# 22. PERFORMANCE VISUALIZATION
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    results.iloc[0][metrics].values,
    width,
    label="Label Encoding"
)

plt.bar(
    x + width / 2,
    results.iloc[1][metrics].values,
    width,
    label="One-Hot Encoding"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.xlabel("Evaluation Metrics")
plt.title("Label Encoding vs One-Hot Encoding")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved: {OUTPUT_PNG}")


# ============================================================
# 23. FIND BEST ENCODING
# ============================================================

if onehot_f1 > label_f1:

    best_encoding = "One-Hot Encoding"
    best_f1 = onehot_f1

elif label_f1 > onehot_f1:

    best_encoding = "Label Encoding"
    best_f1 = label_f1

else:

    best_encoding = "Both performed similarly"
    best_f1 = label_f1


# ============================================================
# 24. PERFORMANCE REPORT
# ============================================================

report = f"""
============================================================
DAY 22 - FEATURE ENCODING PERFORMANCE REPORT
============================================================

Dataset:
{INPUT_FILE}

Target Column:
{target_column}

Original Number of Features:
{X.shape[1]}

Label Encoded Features:
{X_label.shape[1]}

One-Hot Encoded Features:
{X_onehot.shape[1]}


------------------------------------------------------------
LABEL ENCODING
------------------------------------------------------------

Accuracy  : {label_accuracy:.4f}
Precision : {label_precision:.4f}
Recall    : {label_recall:.4f}
F1 Score  : {label_f1:.4f}


------------------------------------------------------------
ONE-HOT ENCODING
------------------------------------------------------------

Accuracy  : {onehot_accuracy:.4f}
Precision : {onehot_precision:.4f}
Recall    : {onehot_recall:.4f}
F1 Score  : {onehot_f1:.4f}


------------------------------------------------------------
BEST ENCODING
------------------------------------------------------------

{best_encoding}

Best F1 Score:
{best_f1:.4f}


------------------------------------------------------------
KEY OBSERVATION
------------------------------------------------------------

Categorical variables cannot be directly understood by most
Machine Learning algorithms.

Feature encoding converts categorical information into numerical
representations that Machine Learning models can process.

Label Encoding assigns an integer to each category.

One-Hot Encoding creates separate binary columns for categories
and avoids implying an artificial numerical order between them.

The appropriate encoding technique depends on the type of
categorical feature and the Machine Learning algorithm being used.

============================================================
"""

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print(f"\nSaved: {OUTPUT_REPORT}")


# ============================================================
# 25. LINKEDIN REFLECTION
# ============================================================

reflection = f"""
Day 22/60

Today I worked on Feature Encoding as part of my 60 Days of
Data Science journey.

I learned how categorical variables can be transformed into
numerical representations that Machine Learning models can
understand.

I implemented both Label Encoding and One-Hot Encoding and
compared their impact on a Logistic Regression baseline model.

The comparison was performed using Accuracy, Precision, Recall,
and F1 Score.

One of the important things I learned is that preprocessing
decisions can directly affect Machine Learning model performance.

I also learned that Label Encoding can introduce an artificial
ordering between categories, while One-Hot Encoding represents
categories independently.

Today's work helped me better understand the connection between
data preprocessing, feature representation, and model performance.

Day 22/60 completed.

Best Encoding based on F1 Score:
{best_encoding}

F1 Score:
{best_f1:.4f}

#60DaysOfDataScience
#DataScienceJourney
#Python
#DataScience
#MachineLearning
#FeatureEngineering
#FeatureEncoding
#LabelEncoding
#OneHotEncoding
#ScikitLearn
#DataPreprocessing
#MachineLearning
#LearningInPublic
#BuildInPublic
#GitHub
"""

with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:
    file.write(reflection)


print(f"Saved: {OUTPUT_REFLECTION}")


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DAY 22 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(f"1. day22_label_encoded_dataset.csv")
print(f"2. {OUTPUT_DATASET}")
print(f"3. {OUTPUT_RESULTS}")
print(f"4. {OUTPUT_PNG}")
print(f"5. {OUTPUT_REPORT}")
print(f"6. {OUTPUT_REFLECTION}")

print("\nBest Encoding:")
print(best_encoding)

print(f"\nBest F1 Score: {best_f1:.4f}")

print("\n" + "=" * 70)