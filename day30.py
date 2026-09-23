# ============================================================
# DAY 30 - FINDING THE IDEAL NUMBER OF CUSTOMER SEGMENTS
# K-Means + Elbow Method + Silhouette Score
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "day29_customer_segments.csv"

OUTPUT_DATASET = "day30_optimized_segments.csv"
OUTPUT_ELBOW = "day30_elbow_curve.png"
OUTPUT_SILHOUETTE = "day30_silhouette_curve.png"
OUTPUT_REPORT = "day30_segmentation_strategy.txt"
OUTPUT_COMPARISON = "day30_cluster_comparison.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_numeric_features(df):
    """
    Automatically select useful numerical columns.
    Removes obvious ID/cluster columns where possible.
    """

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    exclude_words = [
        "id",
        "index",
        "cluster",
        "label",
        "segment",
        "prediction"
    ]

    selected = []

    for col in numeric_columns:
        col_lower = col.lower()

        if any(word == col_lower or col_lower.endswith("_" + word)
               for word in exclude_words):
            continue

        selected.append(col)

    # If too few features remain, use all numeric columns
    if len(selected) < 2:
        selected = numeric_columns

    return selected


# ============================================================
# 1. LOAD DATA
# ============================================================

header("DAY 30 - CUSTOMER SEGMENTATION OPTIMIZATION")

print(f"\nLoading dataset: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. "
        "Make sure your Day 29 output file is in the same folder."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. SELECT NUMERICAL FEATURES
# ============================================================

header("SELECTING NUMERICAL FEATURES")

features = find_numeric_features(df)

print("\nSelected features:")

for feature in features:
    print(f" - {feature}")

if len(features) < 2:
    raise ValueError(
        "At least 2 numerical features are required for clustering."
    )


# ============================================================
# 3. CLEAN DATA
# ============================================================

X = df[features].copy()

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values with median
X = X.fillna(X.median(numeric_only=True))

# Make sure all values are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Final missing value handling
X = X.fillna(X.median(numeric_only=True))


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

header("FEATURE STANDARDIZATION")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Features standardized successfully.")


# ============================================================
# 5. ELBOW METHOD
# ============================================================

header("ELBOW METHOD")

k_values = range(2, 11)

inertias = []

for k in k_values:

    print(f"Training K-Means with K={k}...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

# Create Elbow Plot

plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_ELBOW,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nElbow plot saved: {OUTPUT_ELBOW}")


# ============================================================
# 6. SILHOUETTE SCORE
# ============================================================

header("SILHOUETTE SCORE ANALYSIS")

silhouette_scores = []

# Large datasets can make silhouette calculation very expensive.
# Therefore use a representative sample.

MAX_SAMPLE_SIZE = 10000

if len(X_scaled) > MAX_SAMPLE_SIZE:

    rng = np.random.RandomState(42)

    sample_indices = rng.choice(
        len(X_scaled),
        size=MAX_SAMPLE_SIZE,
        replace=False
    )

    X_sample = X_scaled[sample_indices]

    print(
        f"Dataset is large. Using {MAX_SAMPLE_SIZE:,} "
        "samples for Silhouette Score."
    )

else:

    X_sample = X_scaled

    print(
        f"Using all {len(X_sample):,} rows "
        "for Silhouette Score."
    )


for k in k_values:

    print(f"Calculating Silhouette Score for K={k}...")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_sample)

    score = silhouette_score(
        X_sample,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"   K={k} -> Silhouette Score={score:.4f}"
    )


# ============================================================
# 7. SILHOUETTE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_SILHOUETTE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nSilhouette plot saved: {OUTPUT_SILHOUETTE}"
)


# ============================================================
# 8. FIND BEST K USING SILHOUETTE SCORE
# ============================================================

best_index = np.argmax(silhouette_scores)

best_k = list(k_values)[best_index]

best_score = silhouette_scores[best_index]

print("\n" + "-" * 70)

print(
    f"Best K according to Silhouette Score: {best_k}"
)

print(
    f"Best Silhouette Score: {best_score:.4f}"
)

print("-" * 70)


# ============================================================
# 9. COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertias,
    "Silhouette_Score": silhouette_scores
})

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

print(
    f"\nCluster comparison saved: {OUTPUT_COMPARISON}"
)


# ============================================================
# 10. TRAIN FINAL K-MEANS MODEL
# ============================================================

header("TRAINING FINAL OPTIMIZED MODEL")

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_labels = final_model.fit_predict(X_scaled)

df["Customer_Segment"] = final_labels

print(
    f"Final model trained with K={best_k}"
)


# ============================================================
# 11. SAVE OPTIMIZED DATASET
# ============================================================

df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(
    f"\nOptimized dataset saved: {OUTPUT_DATASET}"
)


# ============================================================
# 12. CLUSTER SUMMARY
# ============================================================

header("CUSTOMER SEGMENT SUMMARY")

cluster_summary = df.groupby(
    "Customer_Segment"
)[features].mean()

cluster_counts = df[
    "Customer_Segment"
].value_counts().sort_index()

cluster_summary.insert(
    0,
    "Customer_Count",
    cluster_counts
)

cluster_summary[
    "Percentage"
] = (
    cluster_summary["Customer_Count"]
    / len(df)
    * 100
)

print("\nCluster Summary:\n")

print(
    cluster_summary.round(2)
)


# ============================================================
# 13. SAVE CLUSTER SUMMARY
# ============================================================

cluster_summary.round(2).to_csv(
    "day30_cluster_summary.csv"
)


# ============================================================
# 14. GENERATE BUSINESS SEGMENTATION STRATEGY
# ============================================================

header("GENERATING BUSINESS STRATEGY")

report_lines = []

report_lines.append(
    "DAY 30 - CUSTOMER SEGMENTATION STRATEGY"
)

report_lines.append("=" * 60)

report_lines.append("")

report_lines.append(
    f"Dataset Size: {len(df):,} customers"
)

report_lines.append(
    f"Features Used: {', '.join(features)}"
)

report_lines.append(
    f"Tested K Values: 2 to 10"
)

report_lines.append(
    f"Selected K: {best_k}"
)

report_lines.append(
    f"Best Silhouette Score: {best_score:.4f}"
)

report_lines.append("")

report_lines.append(
    "CLUSTER DISTRIBUTION"
)

report_lines.append("-" * 60)


for cluster_id in sorted(df["Customer_Segment"].unique()):

    count = int(
        (df["Customer_Segment"] == cluster_id).sum()
    )

    percentage = (
        count / len(df)
    ) * 100

    report_lines.append(
        f"Cluster {cluster_id}: "
        f"{count:,} customers "
        f"({percentage:.2f}%)"
    )


report_lines.append("")

report_lines.append(
    "CLUSTER CHARACTERISTICS"
)

report_lines.append("-" * 60)

for cluster_id in sorted(
    df["Customer_Segment"].unique()
):

    report_lines.append("")

    report_lines.append(
        f"Cluster {cluster_id}"
    )

    cluster_data = cluster_summary.loc[
        cluster_id
    ]

    for feature in features:

        value = cluster_data[feature]

        report_lines.append(
            f"  {feature}: {value:.2f}"
        )


report_lines.append("")

report_lines.append(
    "BUSINESS INTERPRETATION"
)

report_lines.append("-" * 60)

report_lines.append(
    "The customer base was divided into "
    f"{best_k} segments using K-Means clustering."
)

report_lines.append(
    "The Elbow Method was used to examine "
    "the relationship between cluster count "
    "and within-cluster variation."
)

report_lines.append(
    "Silhouette Score was used to evaluate "
    "cluster separation and cohesion."
)

report_lines.append(
    "The selected clustering configuration "
    "provides a data-driven segmentation "
    "framework for customer analysis."
)

report_lines.append("")

report_lines.append(
    "POTENTIAL BUSINESS USES"
)

report_lines.append("-" * 60)

report_lines.append(
    "1. Personalized marketing campaigns"
)

report_lines.append(
    "2. Customer targeting"
)

report_lines.append(
    "3. Customer retention strategies"
)

report_lines.append(
    "4. Product recommendations"
)

report_lines.append(
    "5. Customer behavior analysis"
)

report_lines.append(
    "6. Segment-specific business strategies"
)


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


print(
    f"\nStrategy report saved: {OUTPUT_REPORT}"
)


# ============================================================
# 15. FINAL OUTPUT SUMMARY
# ============================================================

header("GENERATED FILES")

print(
    f"1. {OUTPUT_DATASET}"
)

print(
    f"2. {OUTPUT_COMPARISON}"
)

print(
    f"3. day30_cluster_summary.csv"
)

print(
    f"4. {OUTPUT_ELBOW}"
)

print(
    f"5. {OUTPUT_SILHOUETTE}"
)

print(
    f"6. {OUTPUT_REPORT}"
)

print("\n" + "=" * 70)

print(
    "ALL DAY 30 TASKS COMPLETED"
)

print("=" * 70)