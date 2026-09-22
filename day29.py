# ============================================================
# DAY 29 - CUSTOMER SEGMENTATION WITH K-MEANS CLUSTERING
# 60 Days of Data Science
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

RANDOM_STATE = 42

OUTPUT_DATASET = "day29_customer_segments.csv"
OUTPUT_COMPARISON = "day29_cluster_comparison.csv"
OUTPUT_PLOT = "day29_customer_clusters.png"
OUTPUT_ELBOW = "day29_elbow_curve.png"
OUTPUT_REPORT = "day29_business_insights.txt"
OUTPUT_REFLECTION = "day29_reflection.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_dataset():
    """
    Try to locate a suitable customer behavior dataset.
    """

    possible_files = [
        "customer_behavior.csv",
        "customer_behaviour.csv",
        "customers.csv",
        "customer_data.csv",
        "Mall_Customers.csv",
        "mall_customers.csv",
        "customer_segmentation.csv",
        "customer_segmentation_dataset.csv",
        "WA_FnUseC_TelcoCustomerChurn.csv",
        "feature_engineered_train.csv",
        "train.csv"
    ]

    for file in possible_files:
        if os.path.exists(file):
            print(f"Dataset found: {file}")
            return file

    # Search all CSV files if predefined names are not found
    csv_files = [
        file for file in os.listdir(".")
        if file.lower().endswith(".csv")
    ]

    if csv_files:
        print("Available CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")

        # Prefer files containing customer-related names
        customer_files = [
            file for file in csv_files
            if any(
                keyword in file.lower()
                for keyword in [
                    "customer",
                    "churn",
                    "mall",
                    "segment"
                ]
            )
        ]

        if customer_files:
            print(f"Using customer-related dataset: {customer_files[0]}")
            return customer_files[0]

        print(f"Using first available CSV: {csv_files[0]}")
        return csv_files[0]

    return None


def clean_numeric_data(df):
    """
    Select numerical columns and clean missing/infinite values.
    """

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Remove obvious ID columns
    id_keywords = [
        "id",
        "customerid",
        "customer_id",
        "userid",
        "user_id"
    ]

    columns_to_remove = []

    for column in numeric_df.columns:
        lower_column = column.lower().replace(" ", "").replace("-", "_")

        if any(
            lower_column == keyword or
            lower_column.endswith(keyword)
            for keyword in id_keywords
        ):
            columns_to_remove.append(column)

    numeric_df.drop(
        columns=columns_to_remove,
        errors="ignore",
        inplace=True
    )

    # Remove columns with almost no variation
    for column in numeric_df.columns:
        if numeric_df[column].nunique(dropna=True) <= 1:
            numeric_df.drop(columns=[column], inplace=True)

    numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    numeric_df.dropna(axis=1, how="all", inplace=True)

    # Fill missing values using median
    for column in numeric_df.columns:
        if numeric_df[column].isna().any():
            numeric_df[column] = numeric_df[column].fillna(
                numeric_df[column].median()
            )

    return numeric_df


def select_best_k(X_scaled, k_values):
    """
    Calculate inertia and silhouette score for different K values.
    """

    results = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        if len(set(labels)) > 1:
            silhouette = silhouette_score(
                X_scaled,
                labels
            )
        else:
            silhouette = np.nan

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

        print(
            f"K={k:<3} | "
            f"Inertia={inertia:.2f} | "
            f"Silhouette={silhouette:.4f}"
        )

    results_df = pd.DataFrame(results)

    return results_df


def generate_business_insights(
    clustered_df,
    feature_columns
):
    """
    Generate basic business-oriented cluster descriptions.
    """

    cluster_summary = clustered_df.groupby(
        "Cluster"
    )[feature_columns].mean()

    overall_mean = clustered_df[
        feature_columns
    ].mean()

    insights = []

    for cluster in cluster_summary.index:

        row = cluster_summary.loc[cluster]

        high_features = []
        low_features = []

        for feature in feature_columns:

            if overall_mean[feature] == 0:
                continue

            ratio = row[feature] / overall_mean[feature]

            if ratio >= 1.20:
                high_features.append(feature)

            elif ratio <= 0.80:
                low_features.append(feature)

        insight = f"Cluster {cluster}:\n"

        if high_features:
            insight += (
                "  Higher-than-average features: "
                + ", ".join(high_features)
                + "\n"
            )

        if low_features:
            insight += (
                "  Lower-than-average features: "
                + ", ".join(low_features)
                + "\n"
            )

        if not high_features and not low_features:
            insight += (
                "  Characteristics are relatively close "
                "to the overall customer average.\n"
            )

        insights.append(insight)

    return cluster_summary, insights


# ============================================================
# START
# ============================================================

print_header(
    "DAY 29 - CUSTOMER SEGMENTATION WITH K-MEANS"
)

print("Goal:")
print("Identify hidden customer groups using unsupervised learning.")


# ============================================================
# 1. LOAD DATASET
# ============================================================

print_header("1. LOADING DATASET")

dataset_file = find_dataset()

if dataset_file is None:
    print("ERROR: No CSV dataset found.")
    print("Please place a customer dataset in the project folder.")
    raise SystemExit

df = pd.read_csv(dataset_file)

print(f"Dataset: {dataset_file}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print_header("2. DATASET INFORMATION")

print("\nColumn names:")
print(list(df.columns))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. SELECT NUMERICAL FEATURES
# ============================================================

print_header("3. SELECTING NUMERICAL FEATURES")

numeric_df = clean_numeric_data(df)

if numeric_df.shape[1] < 2:
    print(
        "ERROR: At least two numerical features "
        "are required for meaningful clustering."
    )
    raise SystemExit

print("Selected numerical features:")

for column in numeric_df.columns:
    print(f"  - {column}")

print(f"\nNumber of features: {numeric_df.shape[1]}")


# ============================================================
# 4. SCALE FEATURES
# ============================================================

print_header("4. FEATURE SCALING")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    numeric_df
)

print("StandardScaler applied successfully.")
print("Mean of scaled features:")
print(np.round(X_scaled.mean(axis=0), 4))

print("\nStandard deviation:")
print(np.round(X_scaled.std(axis=0), 4))


# ============================================================
# 5. TEST DIFFERENT CLUSTER COUNTS
# ============================================================

print_header("5. TESTING DIFFERENT CLUSTER COUNTS")

max_k = min(10, len(numeric_df) - 1)

if max_k < 3:
    print("ERROR: Dataset is too small for clustering.")
    raise SystemExit

k_values = range(2, max_k + 1)

comparison_df = select_best_k(
    X_scaled,
    k_values
)

comparison_df.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

print(
    f"\nSaved cluster comparison: "
    f"{OUTPUT_COMPARISON}"
)


# ============================================================
# 6. SELECT K USING SILHOUETTE SCORE
# ============================================================

print_header("6. SELECTING OPTIMAL K")

best_row = comparison_df.loc[
    comparison_df["silhouette_score"].idxmax()
]

best_k = int(best_row["k"])

print(f"Selected K: {best_k}")
print(
    f"Best silhouette score: "
    f"{best_row['silhouette_score']:.4f}"
)


# ============================================================
# 7. FINAL K-MEANS MODEL
# ============================================================

print_header("7. TRAINING FINAL K-MEANS MODEL")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_scaled
)

print("K-Means model trained successfully.")
print(f"Number of clusters: {best_k}")


# ============================================================
# 8. ADD CLUSTERS TO DATASET
# ============================================================

print_header("8. CREATING CUSTOMER SEGMENTS")

clustered_df = df.copy()

# Align numerical rows with original dataframe
clean_numeric_original = df[
    numeric_df.columns
].copy()

valid_mask = ~clean_numeric_original.isnull().any(axis=1)

# Recreate scaled matrix based on rows used
clustered_df["Cluster"] = np.nan

clustered_df.loc[
    valid_mask,
    "Cluster"
] = clusters

clustered_df["Cluster"] = clustered_df[
    "Cluster"
].astype("Int64")

clustered_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(
    f"Customer segmentation saved to: "
    f"{OUTPUT_DATASET}"
)


# ============================================================
# 9. CLUSTER DISTRIBUTION
# ============================================================

print_header("9. CLUSTER DISTRIBUTION")

cluster_counts = clustered_df[
    "Cluster"
].value_counts().sort_index()

print(cluster_counts)

print("\nCluster percentages:")

cluster_percentages = (
    cluster_counts /
    cluster_counts.sum() *
    100
)

for cluster, percentage in cluster_percentages.items():

    print(
        f"Cluster {cluster}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# 10. CLUSTER PROFILE
# ============================================================

print_header("10. CUSTOMER CLUSTER PROFILES")

cluster_profile = clustered_df.groupby(
    "Cluster"
)[numeric_df.columns].mean()

print(
    cluster_profile.round(2)
)


# ============================================================
# 11. ELBOW CURVE
# ============================================================

print_header("11. GENERATING ELBOW CURVE")

plt.figure(figsize=(10, 6))

plt.plot(
    comparison_df["k"],
    comparison_df["inertia"],
    marker="o"
)

plt.title(
    "K-Means Elbow Curve"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_ELBOW,
    dpi=150
)

plt.close()

print(
    f"Saved elbow curve: "
    f"{OUTPUT_ELBOW}"
)


# ============================================================
# 12. CLUSTER VISUALIZATION
# ============================================================

print_header("12. VISUALIZING CUSTOMER CLUSTERS")

# Use first two scaled features for visualization
feature_1 = numeric_df.columns[0]
feature_2 = numeric_df.columns[1]

feature_1_index = list(
    numeric_df.columns
).index(feature_1)

feature_2_index = list(
    numeric_df.columns
).index(feature_2)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_scaled[:, feature_1_index],
    X_scaled[:, feature_2_index],
    c=clusters,
    cmap="viridis",
    alpha=0.7,
    s=45
)

plt.xlabel(
    f"{feature_1} (scaled)"
)

plt.ylabel(
    f"{feature_2} (scaled)"
)

plt.title(
    f"Customer Segmentation using K-Means (K={best_k})"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=150
)

plt.close()

print(
    f"Saved cluster visualization: "
    f"{OUTPUT_PLOT}"
)


# ============================================================
# 13. BUSINESS INSIGHTS
# ============================================================

print_header("13. GENERATING BUSINESS INSIGHTS")

cluster_summary, insights = generate_business_insights(
    clustered_df.dropna(subset=["Cluster"]),
    list(numeric_df.columns)
)

print("\nCluster summary:")
print(
    cluster_summary.round(2)
)

print("\nBusiness insights:")

for insight in insights:
    print(insight)


# ============================================================
# 14. SAVE BUSINESS REPORT
# ============================================================

print_header("14. SAVING BUSINESS INSIGHT REPORT")

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 29 - CUSTOMER SEGMENTATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Dataset: {dataset_file}\n"
    )

    file.write(
        f"Number of customers: {len(df)}\n"
    )

    file.write(
        f"Number of numerical features: "
        f"{len(numeric_df.columns)}\n"
    )

    file.write(
        f"Selected number of clusters: {best_k}\n"
    )

    file.write(
        f"Best silhouette score: "
        f"{best_row['silhouette_score']:.4f}\n\n"
    )

    file.write(
        "FEATURES USED\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for feature in numeric_df.columns:
        file.write(
            f"- {feature}\n"
        )

    file.write("\n")

    file.write(
        "CLUSTER DISTRIBUTION\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    for cluster, count in cluster_counts.items():

        percentage = cluster_percentages[
            cluster
        ]

        file.write(
            f"Cluster {cluster}: "
            f"{count} customers "
            f"({percentage:.2f}%)\n"
        )

    file.write("\n")

    file.write(
        "CLUSTER PROFILES\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        cluster_summary.round(2).to_string()
    )

    file.write("\n\n")

    file.write(
        "BUSINESS INSIGHTS\n"
    )

    file.write(
        "-" * 70 + "\n\n"
    )

    for insight in insights:
        file.write(
            insight + "\n"
        )

print(
    f"Business report saved: "
    f"{OUTPUT_REPORT}"
)


# ============================================================
# 15. REFLECTION
# ============================================================

print_header("15. CREATING REFLECTION")

reflection = f"""
DAY 29 REFLECTION
=================

Today I worked on customer segmentation using K-Means clustering.

The main objective was to understand how unsupervised learning
can be used to identify hidden groups of customers based on
their numerical behavior patterns.

What I worked on:
- Loaded and explored a customer behavior dataset
- Selected relevant numerical features
- Scaled the features using StandardScaler
- Tested multiple K values
- Compared inertia and silhouette scores
- Selected K = {best_k}
- Applied the final K-Means model
- Assigned customers to different clusters
- Visualized the customer segments
- Generated cluster profiles
- Created business-oriented insights

Key learning:

K-Means clustering can group customers without requiring
predefined customer labels.

Feature scaling is important because variables with larger
numerical ranges can otherwise influence the clustering process.

Testing different cluster counts helped me understand that
choosing K is an important part of clustering analysis.

The silhouette score provided a way to compare the quality
of different clustering configurations.

I also learned that clustering results need to be interpreted
from a business perspective rather than looking only at the
cluster labels.

Overall, Day 29 helped me understand the practical use of
unsupervised learning for customer segmentation and business
analysis.

Day 29/60 completed.
"""

with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        reflection.strip()
    )

print(
    f"Reflection saved: "
    f"{OUTPUT_REFLECTION}"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print_header("GENERATED FILES")

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
    f"4. {OUTPUT_ELBOW}"
)

print(
    f"5. {OUTPUT_REPORT}"
)

print(
    f"6. {OUTPUT_REFLECTION}"
)

print("\n" + "=" * 70)

print(
    "ALL DAY 29 TASKS COMPLETED"
)

print("=" * 70)