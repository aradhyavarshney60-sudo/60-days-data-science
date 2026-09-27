# ============================================================
# DAY 33 - DETECTING UNUSUAL CUSTOMER BEHAVIOR
# 60 Days of Data Science
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILES = [
    "day31_customer_personas.csv",
    "day32_customer_similarity.csv",
    "day30_optimized_segments.csv",
    "feature_engineered_train.csv",
]

OUTPUT_ANOMALIES = "day33_customer_anomalies.csv"
OUTPUT_VISUALIZATION = "day33_anomaly_visualization.png"
OUTPUT_COMPARISON = "day33_behavior_comparison.csv"
OUTPUT_RISK = "day33_business_risk_analysis.txt"
OUTPUT_REFLECTION = "day33_reflection.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_input_file():
    for file in INPUT_FILES:
        if os.path.exists(file):
            return file

    raise FileNotFoundError(
        "No suitable input CSV found. "
        "Expected one of: " + ", ".join(INPUT_FILES)
    )


def clean_numeric_data(df):
    """
    Select numeric columns and clean missing/infinite values.
    """

    numeric_df = df.select_dtypes(include=np.number).copy()

    # Remove columns that are completely empty
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Replace infinite values
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values using median
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Remove constant columns
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    return numeric_df


# ============================================================
# LOAD DATA
# ============================================================

header("DAY 33 - ANOMALY DETECTION")

input_file = find_input_file()

print(f"Input dataset: {input_file}")

df = pd.read_csv(input_file)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# DATA INSPECTION
# ============================================================

header("DATA INSPECTION")

print("\nColumn names:")
print(list(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# SELECT BEHAVIORAL FEATURES
# ============================================================

header("SELECTING NUMERIC BEHAVIORAL FEATURES")

features = clean_numeric_data(df)

if features.shape[1] < 2:
    raise ValueError(
        "At least two numeric features are required for anomaly detection."
    )

print(f"\nSelected {features.shape[1]} numeric features:")

for column in features.columns:
    print(f"- {column}")


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

header("FEATURE STANDARDIZATION")

scaler = StandardScaler()

X = scaler.fit_transform(features)

print("Numeric behavioral features standardized successfully.")


# ============================================================
# ISOLATION FOREST
# ============================================================

header("ANOMALY DETECTION USING ISOLATION FOREST")

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X)

# Prediction:
# 1  = normal
# -1 = anomaly
predictions = model.predict(X)

# Anomaly score
scores = model.decision_function(X)

df["anomaly_label"] = predictions
df["anomaly_score"] = scores

df["behavior_status"] = df["anomaly_label"].map({
    1: "Normal",
    -1: "Anomaly"
})

print("\nAnomaly detection completed.")


# ============================================================
# ANOMALY SUMMARY
# ============================================================

header("ANOMALY SUMMARY")

normal_count = int((df["anomaly_label"] == 1).sum())
anomaly_count = int((df["anomaly_label"] == -1).sum())
total_count = len(df)

anomaly_percentage = (
    anomaly_count / total_count * 100
    if total_count > 0
    else 0
)

print(f"Total customers : {total_count}")
print(f"Normal customers: {normal_count}")
print(f"Anomalies       : {anomaly_count}")
print(f"Anomaly rate    : {anomaly_percentage:.2f}%")


# ============================================================
# SAVE ANOMALY DATASET
# ============================================================

df.to_csv(OUTPUT_ANOMALIES, index=False)

print(f"\nSaved: {OUTPUT_ANOMALIES}")


# ============================================================
# NORMAL VS ANOMALOUS BEHAVIOR COMPARISON
# ============================================================

header("NORMAL VS ANOMALOUS BEHAVIOR")

normal_data = df[df["anomaly_label"] == 1][features.columns]
anomaly_data = df[df["anomaly_label"] == -1][features.columns]

comparison_rows = []

for column in features.columns:

    normal_mean = normal_data[column].mean()
    anomaly_mean = anomaly_data[column].mean()

    comparison_rows.append({
        "feature": column,
        "normal_mean": normal_mean,
        "anomaly_mean": anomaly_mean,
        "difference": anomaly_mean - normal_mean
    })

comparison_df = pd.DataFrame(comparison_rows)

print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

print(f"\nSaved: {OUTPUT_COMPARISON}")


# ============================================================
# VISUALIZATION
# ============================================================

header("CREATING ANOMALY VISUALIZATION")

plt.figure(figsize=(12, 7))

# Use first two standardized features for visualization
x_axis = X[:, 0]
y_axis = X[:, 1]

normal_mask = df["anomaly_label"] == 1
anomaly_mask = df["anomaly_label"] == -1

plt.scatter(
    x_axis[normal_mask],
    y_axis[normal_mask],
    alpha=0.6,
    label="Normal Customers"
)

plt.scatter(
    x_axis[anomaly_mask],
    y_axis[anomaly_mask],
    marker="X",
    s=100,
    label="Anomalies"
)

plt.title("Customer Behavior Anomaly Detection")
plt.xlabel(features.columns[0])
plt.ylabel(features.columns[1])
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_VISUALIZATION,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {OUTPUT_VISUALIZATION}")


# ============================================================
# BUSINESS RISK ANALYSIS
# ============================================================

header("BUSINESS RISK ANALYSIS")

risk_lines = []

risk_lines.append("DAY 33 - BUSINESS RISK ANALYSIS")
risk_lines.append("=" * 60)
risk_lines.append("")
risk_lines.append("Objective:")
risk_lines.append(
    "Identify unusual customer behavior patterns using anomaly detection."
)
risk_lines.append("")

risk_lines.append("Detection Summary:")
risk_lines.append(f"Total customers: {total_count}")
risk_lines.append(f"Normal customers: {normal_count}")
risk_lines.append(f"Anomalous customers: {anomaly_count}")
risk_lines.append(f"Anomaly rate: {anomaly_percentage:.2f}%")
risk_lines.append("")

risk_lines.append("Potential Business Risks:")
risk_lines.append(
    "1. Unusual spending patterns may require additional investigation."
)
risk_lines.append(
    "2. Suspicious customer activity may indicate potential fraud or misuse."
)
risk_lines.append(
    "3. Sudden changes in engagement may indicate unusual customer behavior."
)
risk_lines.append(
    "4. Operational teams can use anomaly alerts for further investigation."
)
risk_lines.append(
    "5. Anomalies should be reviewed with business context before taking action."
)
risk_lines.append("")

risk_lines.append("Recommended Analytical Workflow:")
risk_lines.append(
    "Detect -> Investigate -> Validate -> Take appropriate business action"
)

with open(OUTPUT_RISK, "w", encoding="utf-8") as file:
    file.write("\n".join(risk_lines))

print(f"Saved: {OUTPUT_RISK}")


# ============================================================
# REFLECTION
# ============================================================

header("CREATING REFLECTION")

reflection = f"""
DAY 33 - REFLECTION
============================================================

Topic:
Detecting Unusual Customer Behavior

Today I worked on anomaly detection using customer behavioral data.

Key work completed:
- Inspected customer behavioral data
- Selected numerical features
- Standardized behavioral features
- Applied Isolation Forest
- Identified normal and anomalous customers
- Compared normal and anomalous behavior
- Created anomaly visualizations
- Documented potential business risks

Key Learning:

I learned how anomaly detection can be used to identify unusual
customer behavior patterns.

I also learned that machine learning can help businesses identify
potentially suspicious or unusual observations that may require
additional investigation.

Important Insight:

An anomaly does not automatically mean that a customer is fraudulent
or problematic. It indicates that the behavior is different from the
patterns learned by the model and should be investigated using
appropriate business context.

Day 33 completed successfully.

Total customers analyzed: {total_count}
Anomalies detected: {anomaly_count}
Anomaly rate: {anomaly_percentage:.2f}%
"""

with open(OUTPUT_REFLECTION, "w", encoding="utf-8") as file:
    file.write(reflection.strip())


# ============================================================
# FINAL SUMMARY
# ============================================================

header("GENERATED FILES")

print(f"1. {OUTPUT_ANOMALIES}")
print(f"2. {OUTPUT_COMPARISON}")
print(f"3. {OUTPUT_VISUALIZATION}")
print(f"4. {OUTPUT_RISK}")
print(f"5. {OUTPUT_REFLECTION}")

print("\n" + "=" * 70)
print("ALL DAY 33 TASKS COMPLETED")
print("=" * 70)