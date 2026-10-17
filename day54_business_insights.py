
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "WA_FnUseC_TelcoCustomerChurn.csv"
OUTPUT_DIR = "day54_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prepare data
df = pd.read_csv(DATA_FILE)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df = df.dropna(subset=["Churn", "MonthlyCharges", "tenure"])

total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = int((df["Churn"] == "No").sum())
churn_rate = churned / total_customers * 100

# Executive KPIs
avg_monthly_charges = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

print("\n" + "=" * 55)
print("DAY 54: EXECUTIVE CUSTOMER CHURN INSIGHTS")
print("=" * 55)
print(f"Total Customers       : {total_customers:,}")
print(f"Churned Customers     : {churned:,}")
print(f"Retained Customers    : {retained:,}")
print(f"Overall Churn Rate    : {churn_rate:.2f}%")
print(f"Average Monthly Charge: ${avg_monthly_charges:.2f}")
print(f"Average Tenure        : {avg_tenure:.1f} months")

# Segment-level churn rates
segment_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
]

segment_results = []

for column in segment_columns:
    grouped = (
        df.groupby(column, dropna=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    for segment, rate in grouped.items():
        segment_results.append({
            "Feature": column,
            "Segment": str(segment),
            "ChurnRatePercent": round(rate, 2),
        })

segments = pd.DataFrame(segment_results)
segments.to_csv(
    os.path.join(OUTPUT_DIR, "day54_segment_churn_analysis.csv"),
    index=False
)

print("\nHighest churn segments by category:")
for column in segment_columns:
    subset = segments[segments["Feature"] == column]
    highest = subset.loc[subset["ChurnRatePercent"].idxmax()]
    print(
        f"- {column}: {highest['Segment']} "
        f"({highest['ChurnRatePercent']:.2f}%)"
    )

# Chart 1: Churn distribution
churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)

plt.figure(figsize=(7, 5))
churn_counts.plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "day54_churn_distribution.png"),
    dpi=150
)
plt.close()

# Chart 2: Churn rate by contract
contract_rates = (
    df.groupby("Contract")["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
contract_rates.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "day54_contract_churn.png"),
    dpi=150
)
plt.close()

# Chart 3: Churn rate by tenure group
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"]
)

tenure_rates = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
)

plt.figure(figsize=(8, 5))
tenure_rates.plot(kind="bar")
plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "day54_tenure_churn.png"),
    dpi=150
)
plt.close()

# Business recommendations based on observed data
highest_contract = contract_rates.idxmax()
highest_contract_rate = contract_rates.max()
lowest_contract = contract_rates.idxmin()
lowest_contract_rate = contract_rates.min()

recommendations = [
    (
        f"Prioritize retention analysis for {highest_contract} customers, "
        f"which has the highest observed contract-level churn rate "
        f"({highest_contract_rate:.2f}%)."
    ),
    (
        f"Study retention practices among {lowest_contract} customers, "
        f"which has the lowest observed contract-level churn rate "
        f"({lowest_contract_rate:.2f}%)."
    ),
    (
        "Review early-tenure customer journeys and onboarding to identify "
        "opportunities to improve customer retention."
    ),
    (
        "Test targeted retention interventions and evaluate their impact "
        "before scaling them across customer segments."
    ),
]

report_path = os.path.join(
    OUTPUT_DIR, "day54_executive_report.md"
)

with open(report_path, "w", encoding="utf-8") as report:
    report.write("# Day 54: Executive Customer Churn Insights\n\n")
    report.write("## Executive Summary\n\n")
    report.write(
        f"The dataset contains **{total_customers:,} customers**, "
        f"including **{churned:,} churned customers**. "
        f"The observed churn rate is **{churn_rate:.2f}%**.\n\n"
    )
    report.write("## Key Business Metrics\n\n")
    report.write(f"- Total customers: {total_customers:,}\n")
    report.write(f"- Churned customers: {churned:,}\n")
    report.write(f"- Retained customers: {retained:,}\n")
    report.write(f"- Churn rate: {churn_rate:.2f}%\n")
    report.write(
        f"- Average monthly charges: ${avg_monthly_charges:.2f}\n"
    )
    report.write(f"- Average tenure: {avg_tenure:.1f} months\n\n")

    report.write("## Highest-Churn Segments\n\n")
    for column in segment_columns:
        subset = segments[segments["Feature"] == column]
        highest = subset.loc[subset["ChurnRatePercent"].idxmax()]
        report.write(
            f"- **{column}:** {highest['Segment']} "
            f"({highest['ChurnRatePercent']:.2f}% churn)\n"
        )

    report.write("\n## Recommended Business Actions\n\n")
    for index, item in enumerate(recommendations, start=1):
        report.write(f"{index}. {item}\n")

    report.write(
        "\n## Interpretation Note\n\n"
        "These results describe observed associations in the supplied "
        "dataset. They do not establish causation or guarantee future "
        "customer behavior. Validate proposed interventions with testing.\n"
    )

with open(
    os.path.join(OUTPUT_DIR, "day54_business_recommendations.md"),
    "w",
    encoding="utf-8",
) as file:
    file.write("# Day 54: Business Recommendations\n\n")
    for index, item in enumerate(recommendations, start=1):
        file.write(f"{index}. {item}\n")

print("\nCharts and reports saved in:", OUTPUT_DIR)
print("Generated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nDay 54 business insights completed successfully!")
