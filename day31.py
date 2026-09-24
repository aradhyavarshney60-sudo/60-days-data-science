# ============================================================
# DAY 31 - BUILDING CUSTOMER PERSONAS FROM BEHAVIORAL DATA
# 60 DAYS OF DATA SCIENCE
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "day30_optimized_segments.csv"

OUTPUT_PERSONAS = "day31_customer_personas.csv"
OUTPUT_SUMMARY = "day31_persona_summary.csv"
OUTPUT_VISUAL = "day31_persona_visualization.png"
OUTPUT_RECOMMENDATIONS = "day31_business_recommendations.txt"
OUTPUT_REFLECTION = "day31_reflection.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_column(df, possible_names):
    """
    Find a column using exact name first,
    then partial matching.
    """
    columns_lower = {str(col).lower(): col for col in df.columns}

    # Exact match
    for name in possible_names:
        if name.lower() in columns_lower:
            return columns_lower[name.lower()]

    # Partial match
    for col in df.columns:
        col_lower = str(col).lower()
        for name in possible_names:
            if name.lower() in col_lower:
                return col

    return None


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


# ============================================================
# LOAD DATA
# ============================================================

header("DAY 31 - CUSTOMER PERSONA ANALYSIS")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: {INPUT_FILE} not found.")
    print("Make sure day30_optimized_segments.csv is in the same folder.")
    raise SystemExit


df = pd.read_csv(INPUT_FILE)

print(f"Input file: {INPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nAvailable columns:")
for col in df.columns:
    print(f" - {col}")


# ============================================================
# IDENTIFY CLUSTER COLUMN
# ============================================================

cluster_col = find_column(
    df,
    [
        "cluster",
        "cluster_label",
        "cluster_id",
        "segment",
        "clusterid"
    ]
)

if cluster_col is None:
    print("\nERROR: Could not identify the cluster column.")
    print("Expected something like: cluster, cluster_label, cluster_id, segment")
    raise SystemExit


print(f"\nCluster column detected: {cluster_col}")


# ============================================================
# IDENTIFY IMPORTANT BEHAVIORAL COLUMNS
# ============================================================

spending_col = find_column(
    df,
    [
        "total_spend",
        "spending",
        "spend",
        "annual_spending",
        "purchase_amount",
        "monetary",
        "amount",
        "income"
    ]
)

engagement_col = find_column(
    df,
    [
        "engagement",
        "engagement_score",
        "frequency",
        "purchase_frequency",
        "visits",
        "sessions",
        "activity"
    ]
)

recency_col = find_column(
    df,
    [
        "recency",
        "days_since_purchase",
        "last_purchase"
    ]
)


print(f"Spending column:   {spending_col}")
print(f"Engagement column: {engagement_col}")
print(f"Recency column:    {recency_col}")


# ============================================================
# CLEAN CLUSTER DATA
# ============================================================

df = df.dropna(subset=[cluster_col]).copy()

# Convert cluster labels to string for safe grouping
df[cluster_col] = df[cluster_col].astype(str)

clusters = sorted(df[cluster_col].unique())

print(f"\nNumber of customer clusters: {len(clusters)}")
print(f"Clusters: {clusters}")


# ============================================================
# CREATE NUMERIC DATA FOR ANALYSIS
# ============================================================

num_cols = numeric_columns(df)

print("\nNumeric columns detected:")
for col in num_cols:
    print(f" - {col}")


# ============================================================
# CLUSTER SUMMARY
# ============================================================

header("CUSTOMER CLUSTER ANALYSIS")

summary_rows = []

for cluster in clusters:

    cluster_data = df[df[cluster_col] == cluster]

    row = {
        "cluster": cluster,
        "customer_count": len(cluster_data),
        "percentage_of_customers":
            round((len(cluster_data) / len(df)) * 100, 2)
    }

    if spending_col is not None:
        row["avg_spending"] = round(
            pd.to_numeric(
                cluster_data[spending_col],
                errors="coerce"
            ).mean(),
            2
        )

    if engagement_col is not None:
        row["avg_engagement"] = round(
            pd.to_numeric(
                cluster_data[engagement_col],
                errors="coerce"
            ).mean(),
            2
        )

    if recency_col is not None:
        row["avg_recency"] = round(
            pd.to_numeric(
                cluster_data[recency_col],
                errors="coerce"
            ).mean(),
            2
        )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

print("\nCluster Summary:")
print(summary.to_string(index=False))


# ============================================================
# NORMALIZE BEHAVIOR SCORES
# ============================================================

def min_max_score(series):
    series = pd.to_numeric(series, errors="coerce")

    if series.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            np.ones(len(series)) * 0.5,
            index=series.index
        )

    return (series - min_value) / (max_value - min_value)


# Calculate cluster-level normalized metrics

if spending_col is not None:
    summary["spending_score"] = min_max_score(
        summary["avg_spending"]
    )

if engagement_col is not None:
    summary["engagement_score"] = min_max_score(
        summary["avg_engagement"]
    )


# ============================================================
# PERSONA ASSIGNMENT
# ============================================================

header("ASSIGNING CUSTOMER PERSONAS")


def assign_persona(row):

    spending = row.get("spending_score", 0.5)
    engagement = row.get("engagement_score", 0.5)

    if spending >= 0.70 and engagement >= 0.70:
        return "High-Value Loyal Customers"

    elif spending >= 0.70 and engagement < 0.70:
        return "High-Spending Occasional Customers"

    elif spending < 0.40 and engagement >= 0.70:
        return "Highly Engaged Budget Customers"

    elif spending < 0.40 and engagement < 0.40:
        return "Low-Value Low-Engagement Customers"

    elif spending >= 0.40 and engagement >= 0.40:
        return "Balanced Active Customers"

    elif spending >= 0.40:
        return "Moderate-Spending Customers"

    else:
        return "Developing Customers"


summary["persona"] = summary.apply(assign_persona, axis=1)


# ============================================================
# PERSONA DESCRIPTION
# ============================================================

def persona_description(persona):

    descriptions = {

        "High-Value Loyal Customers":
            "Customers with strong spending and high engagement. "
            "They represent valuable and active customer relationships.",

        "High-Spending Occasional Customers":
            "Customers who spend significantly but show comparatively "
            "lower engagement or purchase frequency.",

        "Highly Engaged Budget Customers":
            "Customers who interact frequently but have relatively "
            "lower spending levels.",

        "Low-Value Low-Engagement Customers":
            "Customers with relatively low spending and low engagement.",

        "Balanced Active Customers":
            "Customers showing a relatively balanced combination "
            "of spending and engagement.",

        "Moderate-Spending Customers":
            "Customers with moderate spending behavior who may have "
            "potential for stronger engagement.",

        "Developing Customers":
            "Customers who may require targeted engagement and "
            "personalized offers to increase activity."
    }

    return descriptions.get(
        persona,
        "Customer segment identified through behavioral analysis."
    )


summary["persona_description"] = summary["persona"].apply(
    persona_description
)


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

def recommendation(persona):

    recommendations = {

        "High-Value Loyal Customers":
            "Focus on retention, loyalty rewards, premium experiences, "
            "early access and personalized offers.",

        "High-Spending Occasional Customers":
            "Use personalized campaigns, reminders and loyalty incentives "
            "to increase purchase frequency.",

        "Highly Engaged Budget Customers":
            "Use affordable bundles, discounts and cross-selling "
            "opportunities to increase average spending.",

        "Low-Value Low-Engagement Customers":
            "Use re-engagement campaigns, limited-time offers and "
            "personalized recommendations.",

        "Balanced Active Customers":
            "Maintain engagement through personalized recommendations, "
            "loyalty programs and relevant promotions.",

        "Moderate-Spending Customers":
            "Encourage repeat purchases with targeted offers, bundles "
            "and loyalty incentives.",

        "Developing Customers":
            "Improve engagement using onboarding campaigns, personalized "
            "content and introductory offers."
    }

    return recommendations.get(
        persona,
        "Use targeted marketing based on customer behavior."
    )


summary["business_recommendation"] = summary["persona"].apply(
    recommendation
)


# ============================================================
# PRINT PERSONA RESULTS
# ============================================================

print("\nCustomer Personas:")

for _, row in summary.iterrows():

    print("\n" + "-" * 65)
    print(f"Cluster: {row['cluster']}")
    print(f"Persona: {row['persona']}")
    print(f"Customers: {row['customer_count']:,}")
    print(f"Share: {row['percentage_of_customers']}%")

    if "avg_spending" in row:
        print(f"Average Spending: {row['avg_spending']}")

    if "avg_engagement" in row:
        print(f"Average Engagement: {row['avg_engagement']}")

    print(f"Description: {row['persona_description']}")
    print(f"Strategy: {row['business_recommendation']}")


# ============================================================
# MERGE PERSONAS BACK INTO CUSTOMER DATA
# ============================================================

persona_mapping = summary[
    ["cluster", "persona", "persona_description"]
].copy()

df = df.merge(
    persona_mapping,
    left_on=cluster_col,
    right_on="cluster",
    how="left"
)

df.drop(
    columns=["cluster"],
    inplace=True,
    errors="ignore"
)


# ============================================================
# SAVE CUSTOMER PERSONAS
# ============================================================

df.to_csv(
    OUTPUT_PERSONAS,
    index=False
)

print(f"\nSaved: {OUTPUT_PERSONAS}")


# ============================================================
# SAVE PERSONA SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print(f"Saved: {OUTPUT_SUMMARY}")


# ============================================================
# VISUALIZATION
# ============================================================

header("CREATING PERSONA VISUALIZATION")

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111)

x_labels = summary["persona"].tolist()

values = summary["customer_count"].tolist()

bars = ax.bar(
    range(len(x_labels)),
    values
)

ax.set_title(
    "Customer Persona Distribution",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Customer Persona")
ax.set_ylabel("Number of Customers")

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(
    x_labels,
    rotation=35,
    ha="right"
)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_VISUAL,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {OUTPUT_VISUAL}")


# ============================================================
# SECOND VISUALIZATION
# ============================================================

if (
    spending_col is not None
    and engagement_col is not None
):

    plt.figure(figsize=(11, 8))

    for _, row in summary.iterrows():

        plt.scatter(
            row["avg_spending"],
            row["avg_engagement"],
            s=300,
            alpha=0.75
        )

        plt.annotate(
            row["persona"],
            (
                row["avg_spending"],
                row["avg_engagement"]
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9
        )

    plt.title(
        "Customer Personas: Spending vs Engagement",
        fontsize=17,
        fontweight="bold"
    )

    plt.xlabel("Average Spending")
    plt.ylabel("Average Engagement")

    plt.grid(alpha=0.25)

    plt.tight_layout()

    scatter_file = "day31_spending_vs_engagement.png"

    plt.savefig(
        scatter_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {scatter_file}")


# ============================================================
# BUSINESS RECOMMENDATIONS REPORT
# ============================================================

header("CREATING BUSINESS RECOMMENDATIONS REPORT")

with open(
    OUTPUT_RECOMMENDATIONS,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 31 - CUSTOMER PERSONA BUSINESS INSIGHTS\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        "Objective:\n"
        "Transform customer clustering results into meaningful "
        "customer personas and business strategies.\n\n"
    )

    for _, row in summary.iterrows():

        file.write("-" * 70 + "\n")

        file.write(
            f"Cluster: {row['cluster']}\n"
        )

        file.write(
            f"Persona: {row['persona']}\n"
        )

        file.write(
            f"Customer Count: {row['customer_count']}\n"
        )

        file.write(
            f"Customer Share: "
            f"{row['percentage_of_customers']}%\n"
        )

        file.write(
            f"Description: "
            f"{row['persona_description']}\n"
        )

        file.write(
            f"Recommendation: "
            f"{row['business_recommendation']}\n\n"
        )


print(f"Saved: {OUTPUT_RECOMMENDATIONS}")


# ============================================================
# REFLECTION
# ============================================================

reflection = f"""
DAY 31 REFLECTION
=================

Today I worked on building customer personas from behavioral data.

The main objective was to transform customer clustering results into
business-friendly customer personas.

Key tasks completed:

1. Analyzed customer clusters.
2. Examined spending and engagement behavior.
3. Assigned meaningful persona names.
4. Created a customer persona summary.
5. Visualized persona distribution.
6. Compared spending and engagement patterns.
7. Developed business recommendations for each persona.
8. Generated structured output files for further analysis.

Key Learning:

I learned that machine learning outputs become more useful when they
are translated into understandable business insights.

Customer clustering identifies groups of similar customers, while
customer personas provide a clearer way to communicate the behavior
and needs of those groups.

This helped me understand how data science can connect technical
machine learning results with practical business decision-making.

Day 31 completed.
"""


with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(reflection.strip())


print(f"Saved: {OUTPUT_REFLECTION}")


# ============================================================
# FINAL SUMMARY
# ============================================================

header("GENERATED FILES")

print(f"1. {OUTPUT_PERSONAS}")
print(f"2. {OUTPUT_SUMMARY}")
print(f"3. {OUTPUT_VISUAL}")

if spending_col is not None and engagement_col is not None:
    print("4. day31_spending_vs_engagement.png")

print(f"5. {OUTPUT_RECOMMENDATIONS}")
print(f"6. {OUTPUT_REFLECTION}")

print("\n" + "=" * 70)
print("ALL DAY 31 TASKS COMPLETED")
print("=" * 70)