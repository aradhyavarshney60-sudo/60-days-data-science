# ============================================================
# DAY 32 - PERSONALIZED PRODUCT RECOMMENDATION ENGINE
# 60 DAYS OF DATA SCIENCE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILES = [
    "day30_optimized_segments.csv",
    "day31_customer_personas.csv",
    "day29_customer_segments.csv",
    "day30_cluster_summary.csv",
]

OUTPUT_SIMILARITY = "day32_customer_similarity.csv"
OUTPUT_RECOMMENDATIONS = "day32_recommendations.csv"
OUTPUT_VISUALIZATION = "day32_similarity_visualization.png"
OUTPUT_REPORT = "day32_recommendation_report.txt"
OUTPUT_REFLECTION = "day32_reflection.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_input_file():
    """Find the first available customer dataset."""
    
    for file in INPUT_FILES:
        if os.path.exists(file):
            return file
    
    return None


def create_sample_dataset():
    """
    Create a fallback customer dataset if no previous
    customer dataset is available.
    """
    
    np.random.seed(42)

    n = 100

    data = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "annual_income": np.random.randint(20000, 120000, n),
        "spending_score": np.random.randint(1, 101, n),
        "purchase_frequency": np.random.randint(1, 30, n),
        "avg_order_value": np.random.randint(10, 500, n),
        "website_visits": np.random.randint(1, 50, n)
    })

    return data


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DAY 32 - PERSONALIZED PRODUCT RECOMMENDATION ENGINE")
print("=" * 70)

input_file = find_input_file()

if input_file:
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_csv(input_file)
else:
    print("\nNo previous customer dataset found.")
    print("Creating fallback customer dataset...")
    df = create_sample_dataset()

print(f"Dataset shape: {df.shape}")


# ============================================================
# DATA CLEANING
# ============================================================

df = df.copy()

# Remove completely empty columns
df = df.dropna(axis=1, how="all")

# Remove duplicate rows
df = df.drop_duplicates()

# Reset index
df = df.reset_index(drop=True)

print(f"Cleaned dataset shape: {df.shape}")


# ============================================================
# CUSTOMER ID
# ============================================================

customer_id_column = None

possible_id_columns = [
    "customer_id",
    "CustomerID",
    "customer",
    "id",
    "ID"
]

for col in possible_id_columns:
    if col in df.columns:
        customer_id_column = col
        break

if customer_id_column is None:
    df["customer_id"] = range(1, len(df) + 1)
    customer_id_column = "customer_id"


# ============================================================
# SELECT NUMERICAL FEATURES
# ============================================================

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# Remove customer ID from features
feature_columns = [
    col for col in numeric_columns
    if col != customer_id_column
]

# Remove columns with no variation
feature_columns = [
    col for col in feature_columns
    if df[col].nunique() > 1
]

if len(feature_columns) < 2:
    print("\nNot enough numerical features found.")
    print("Creating additional customer behavior features...")

    sample = create_sample_dataset()

    df = sample
    customer_id_column = "customer_id"

    feature_columns = [
        "annual_income",
        "spending_score",
        "purchase_frequency",
        "avg_order_value",
        "website_visits"
    ]


print("\nFeatures used for recommendation:")
for feature in feature_columns:
    print(f" - {feature}")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(
        df[column].median()
    )


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[feature_columns]
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(X_scaled)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df[customer_id_column],
    columns=df[customer_id_column]
)


# ============================================================
# CREATE CUSTOMER SIMILARITY OUTPUT
# ============================================================

similarity_records = []

customer_ids = df[customer_id_column].tolist()

for i, customer_id in enumerate(customer_ids):

    similarities = similarity_matrix[i].copy()

    # Do not recommend the customer themselves
    similarities[i] = -1

    top_indices = np.argsort(
        similarities
    )[-5:][::-1]

    for rank, index in enumerate(top_indices, start=1):

        similarity_records.append({
            "customer_id": customer_id,
            "similar_customer_id": customer_ids[index],
            "similarity_score": round(
                similarities[index],
                4
            ),
            "rank": rank
        })


similarity_output = pd.DataFrame(
    similarity_records
)

similarity_output.to_csv(
    OUTPUT_SIMILARITY,
    index=False
)


# ============================================================
# PRODUCT RECOMMENDATION LOGIC
# ============================================================

# Product catalog based on customer behavior
products = [
    "Premium Membership",
    "Budget Bundle",
    "Electronics Collection",
    "Fashion Collection",
    "Home & Lifestyle Bundle",
    "Travel Package",
    "Fitness Package",
    "Entertainment Subscription",
    "Technology Upgrade",
    "Loyalty Rewards"
]


def recommend_products(row):
    """
    Generate personalized product recommendations
    using customer behavioral characteristics.
    """

    recommendations = []

    row_lower = row.copy()

    # Find columns dynamically
    spending = None
    income = None
    frequency = None

    for col in row_lower.index:

        name = col.lower()

        if "spending" in name or "score" in name:
            spending = row_lower[col]

        if "income" in name:
            income = row_lower[col]

        if "frequency" in name or "purchase" in name:
            frequency = row_lower[col]

    # Recommendation rules

    if spending is not None:

        if spending >= 70:
            recommendations.append(
                "Premium Membership"
            )

        elif spending <= 30:
            recommendations.append(
                "Budget Bundle"
            )

    if frequency is not None:

        if frequency >= 15:
            recommendations.append(
                "Loyalty Rewards"
            )

        else:
            recommendations.append(
                "Entertainment Subscription"
            )

    if income is not None:

        if income >= 80000:
            recommendations.append(
                "Technology Upgrade"
            )

        elif income <= 40000:
            recommendations.append(
                "Home & Lifestyle Bundle"
            )

    # Default recommendations
    for product in products:

        if product not in recommendations:
            recommendations.append(product)

        if len(recommendations) >= 5:
            break

    return recommendations[:5]


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

recommendation_records = []

for _, row in df.iterrows():

    customer_id = row[customer_id_column]

    recommendations = recommend_products(
        row[feature_columns]
    )

    for rank, product in enumerate(
        recommendations,
        start=1
    ):

        recommendation_records.append({
            "customer_id": customer_id,
            "recommendation_rank": rank,
            "recommended_product": product,
            "recommendation_score": round(
                1 / rank,
                4
            )
        })


recommendations_df = pd.DataFrame(
    recommendation_records
)

recommendations_df.to_csv(
    OUTPUT_RECOMMENDATIONS,
    index=False
)


# ============================================================
# SIMILARITY ANALYSIS
# ============================================================

average_similarity = (
    similarity_output[
        "similarity_score"
    ].mean()
)

max_similarity = (
    similarity_output[
        "similarity_score"
    ].max()
)

min_similarity = (
    similarity_output[
        "similarity_score"
    ].min()
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    similarity_matrix,
    aspect="auto"
)

plt.colorbar(
    label="Cosine Similarity"
)

plt.title(
    "Customer Similarity Matrix"
)

plt.xlabel(
    "Customers"
)

plt.ylabel(
    "Customers"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_VISUALIZATION,
    dpi=150
)

plt.close()


# ============================================================
# RECOMMENDATION REPORT
# ============================================================

report = f"""
DAY 32 - PERSONALIZED PRODUCT RECOMMENDATION ENGINE
====================================================

OBJECTIVE
---------
Build a similarity-based recommendation system that
personalizes product suggestions using customer behavior.

DATASET
-------
Input file:
{input_file if input_file else "Generated fallback dataset"}

Number of customers:
{len(df)}

Features used:
{", ".join(feature_columns)}

METHOD
------
1. Customer behavioral data was loaded.
2. Numerical customer features were selected.
3. Missing values were handled.
4. Features were standardized using StandardScaler.
5. Customer similarity was calculated using cosine similarity.
6. Top similar customers were identified.
7. Personalized product recommendations were generated.
8. Similarity results were visualized.

SIMILARITY ANALYSIS
-------------------
Average similarity score:
{average_similarity:.4f}

Maximum similarity score:
{max_similarity:.4f}

Minimum similarity score:
{min_similarity:.4f}

RECOMMENDATION STRATEGY
-----------------------
Recommendations are generated using customer behavioral
characteristics such as spending, income and purchase
frequency when those features are available.

OUTPUT FILES
------------
1. {OUTPUT_SIMILARITY}
2. {OUTPUT_RECOMMENDATIONS}
3. {OUTPUT_VISUALIZATION}
4. {OUTPUT_REPORT}
5. {OUTPUT_REFLECTION}

BUSINESS APPLICATION
--------------------
A recommendation engine can help businesses:

- Personalize customer experiences
- Improve product discovery
- Increase customer engagement
- Support targeted marketing
- Improve customer retention
- Identify relevant product opportunities

CONCLUSION
----------
The recommendation system demonstrates how customer
similarity can be used to generate personalized product
suggestions.
"""


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# REFLECTION
# ============================================================

reflection = """
DAY 32 REFLECTION
=================

Today I worked on building a personalized product
recommendation engine using customer behavioral data.

KEY LEARNINGS
-------------

1. Customer similarity can be used to personalize
   product recommendations.

2. Feature scaling is important when calculating
   similarity between customers.

3. Cosine similarity provides a useful way to compare
   customer behavior patterns.

4. Recommendation systems can improve customer
   engagement and personalization.

5. Data science models can be connected with practical
   business use cases.

6. Recommendation quality depends strongly on the
   quality and relevance of customer features.

BUSINESS INSIGHT
----------------

A recommendation system can help businesses deliver
more relevant products to different customers instead
of showing the same products to everyone.

NEXT STEP
---------

Future improvements could include collaborative
filtering, product-level purchase history, user-item
matrices, recommendation evaluation metrics and
machine learning based ranking systems.

Day 32 completed successfully.
"""


with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(reflection)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("GENERATED FILES")

print("=" * 70)

print(f"1. {OUTPUT_SIMILARITY}")
print(f"2. {OUTPUT_RECOMMENDATIONS}")
print(f"3. {OUTPUT_VISUALIZATION}")
print(f"4. {OUTPUT_REPORT}")
print(f"5. {OUTPUT_REFLECTION}")

print("\n" + "=" * 70)

print("ALL DAY 32 TASKS COMPLETED")

print("=" * 70)