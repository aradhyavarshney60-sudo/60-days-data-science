# ============================================================
# DAY 34 - EXECUTIVE CUSTOMER ANALYTICS DASHBOARD
# 60 Days of Data Science
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Analytics Dashboard - Day 34",
    page_icon="📊",
    layout="wide"
)

np.random.seed(42)

OUTPUT_DATA = "day34_customer_dashboard_data.csv"
OUTPUT_INSIGHTS = "day34_business_insights.txt"
OUTPUT_REFLECTION = "day34_reflection.txt"


# ============================================================
# DATA GENERATION
# ============================================================

def generate_customer_data(n=500):

    customer_ids = [f"CUST_{i:04d}" for i in range(1, n + 1)]

    age = np.random.randint(18, 65, n)

    income = np.random.randint(20000, 150000, n)

    total_spend = np.round(
        np.random.gamma(shape=4, scale=500, size=n),
        2
    )

    purchases = np.random.randint(1, 30, n)

    engagement_score = np.round(
        np.random.uniform(20, 100, n),
        2
    )

    satisfaction_score = np.round(
        np.random.uniform(1, 10, n),
        2
    )

    tenure_months = np.random.randint(1, 72, n)

    last_activity_days = np.random.randint(1, 120, n)

    # Churn probability based on customer behavior
    churn_probability = (
        0.15
        + (last_activity_days > 60) * 0.30
        + (engagement_score < 40) * 0.20
        + (satisfaction_score < 5) * 0.15
    )

    churn_probability = np.clip(churn_probability, 0, 0.95)

    churn = np.random.binomial(1, churn_probability)

    # Customer segments
    def assign_segment(row):

        if row["Total_Spend"] >= 2500 and row["Engagement_Score"] >= 70:
            return "High Value"

        elif row["Engagement_Score"] >= 65:
            return "Engaged"

        elif row["Total_Spend"] >= 1200:
            return "Regular"

        else:
            return "Low Value"

    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": age,
        "Income": income,
        "Total_Spend": total_spend,
        "Purchases": purchases,
        "Engagement_Score": engagement_score,
        "Satisfaction_Score": satisfaction_score,
        "Tenure_Months": tenure_months,
        "Last_Activity_Days": last_activity_days,
        "Churn": churn
    })

    df["Segment"] = df.apply(assign_segment, axis=1)

    df["Churn_Status"] = df["Churn"].map({
        0: "Active",
        1: "Churn Risk"
    })

    return df


# ============================================================
# LOAD / CREATE DATA
# ============================================================

if os.path.exists(OUTPUT_DATA):
    df = pd.read_csv(OUTPUT_DATA)
else:
    df = generate_customer_data()
    df.to_csv(OUTPUT_DATA, index=False)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Executive Customer Analytics Dashboard")

st.markdown(
    """
    ### Day 34/60 | Customer Analytics & Business Intelligence

    This dashboard provides an executive-level overview of
    customer behavior, segmentation, engagement and churn risk.
    """
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Dashboard Filters")

segments = sorted(df["Segment"].unique())

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    segments,
    default=segments
)

churn_filter = st.sidebar.multiselect(
    "Customer Status",
    ["Active", "Churn Risk"],
    default=["Active", "Churn Risk"]
)

min_spend = st.sidebar.slider(
    "Minimum Total Spend",
    0,
    int(df["Total_Spend"].max()),
    0
)

filtered_df = df[
    (df["Segment"].isin(selected_segments))
    & (df["Churn_Status"].isin(churn_filter))
    & (df["Total_Spend"] >= min_spend)
]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

total_revenue = filtered_df["Total_Spend"].sum()

avg_spend = (
    filtered_df["Total_Spend"].mean()
    if total_customers > 0
    else 0
)

churn_rate = (
    filtered_df["Churn"].mean() * 100
    if total_customers > 0
    else 0
)

avg_engagement = (
    filtered_df["Engagement_Score"].mean()
    if total_customers > 0
    else 0
)


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📌 Executive KPIs")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Total Revenue",
    f"${total_revenue:,.0f}"
)

col3.metric(
    "Average Spend",
    f"${avg_spend:,.2f}"
)

col4.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%"
)

col5.metric(
    "Avg Engagement",
    f"{avg_engagement:.1f}"
)


st.divider()


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

st.subheader("👥 Customer Segmentation")

col1, col2 = st.columns(2)

segment_counts = (
    filtered_df["Segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Segment", "Customers"]

fig_segment = px.bar(
    segment_counts,
    x="Segment",
    y="Customers",
    title="Customers by Segment",
    text="Customers"
)

fig_segment.update_traces(
    textposition="outside"
)

col1.plotly_chart(
    fig_segment,
    use_container_width=True
)


# Revenue by segment

segment_revenue = (
    filtered_df
    .groupby("Segment")["Total_Spend"]
    .sum()
    .reset_index()
    .sort_values("Total_Spend", ascending=False)
)

fig_revenue = px.pie(
    segment_revenue,
    names="Segment",
    values="Total_Spend",
    title="Revenue Contribution by Segment",
    hole=0.4
)

col2.plotly_chart(
    fig_revenue,
    use_container_width=True
)


# ============================================================
# CHURN ANALYSIS
# ============================================================

st.subheader("⚠️ Churn & Customer Risk Analysis")

col1, col2 = st.columns(2)

churn_counts = (
    filtered_df["Churn_Status"]
    .value_counts()
    .reset_index()
)

churn_counts.columns = ["Status", "Customers"]

fig_churn = px.bar(
    churn_counts,
    x="Status",
    y="Customers",
    title="Active Customers vs Churn Risk",
    text="Customers"
)

fig_churn.update_traces(
    textposition="outside"
)

col1.plotly_chart(
    fig_churn,
    use_container_width=True
)


# Churn by segment

churn_segment = (
    filtered_df
    .groupby("Segment")["Churn"]
    .mean()
    .reset_index()
)

churn_segment["Churn_Rate"] = churn_segment["Churn"] * 100

fig_churn_segment = px.bar(
    churn_segment,
    x="Segment",
    y="Churn_Rate",
    title="Churn Rate by Customer Segment",
    text="Churn_Rate"
)

fig_churn_segment.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig_churn_segment.update_yaxes(
    title="Churn Rate (%)"
)

col2.plotly_chart(
    fig_churn_segment,
    use_container_width=True
)


# ============================================================
# ENGAGEMENT ANALYSIS
# ============================================================

st.subheader("📈 Customer Engagement Analysis")

fig_engagement = px.scatter(
    filtered_df,
    x="Engagement_Score",
    y="Total_Spend",
    size="Purchases",
    color="Segment",
    hover_data=[
        "Customer_ID",
        "Satisfaction_Score",
        "Churn_Status"
    ],
    title="Engagement vs Customer Spend"
)

st.plotly_chart(
    fig_engagement,
    use_container_width=True
)


# ============================================================
# CUSTOMER SATISFACTION
# ============================================================

col1, col2 = st.columns(2)

fig_satisfaction = px.histogram(
    filtered_df,
    x="Satisfaction_Score",
    nbins=10,
    title="Customer Satisfaction Distribution"
)

col1.plotly_chart(
    fig_satisfaction,
    use_container_width=True
)


# Activity vs churn

fig_activity = px.scatter(
    filtered_df,
    x="Last_Activity_Days",
    y="Engagement_Score",
    color="Churn_Status",
    hover_data=["Customer_ID", "Segment"],
    title="Last Activity vs Engagement"
)

col2.plotly_chart(
    fig_activity,
    use_container_width=True
)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Business Insights")

if len(filtered_df) > 0:

    highest_revenue_segment = (
        filtered_df.groupby("Segment")["Total_Spend"]
        .sum()
        .idxmax()
    )

    highest_churn_segment = (
        filtered_df.groupby("Segment")["Churn"]
        .mean()
        .idxmax()
    )

    highest_engagement_segment = (
        filtered_df.groupby("Segment")["Engagement_Score"]
        .mean()
        .idxmax()
    )

    insights = [
        f"• The segment contributing the highest total revenue is {highest_revenue_segment}.",
        f"• The segment with the highest observed churn rate is {highest_churn_segment}.",
        f"• The segment with the highest average engagement is {highest_engagement_segment}.",
        f"• Overall observed churn rate is {churn_rate:.1f}%.",
        f"• Average customer engagement score is {avg_engagement:.1f}.",
        f"• Average customer spend is ${avg_spend:,.2f}."
    ]

    for insight in insights:
        st.write(insight)

else:
    insights = ["No customers match the selected filters."]
    st.warning("No customers match the selected filters.")


# ============================================================
# EXECUTIVE DECISION SUPPORT
# ============================================================

st.subheader("🎯 Executive Decision Support")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        **Customer Growth**

        Monitor segment sizes and revenue contribution
        to understand customer distribution.
        """
    )

with col2:
    st.warning(
        """
        **Churn Monitoring**

        Customers showing lower engagement and recent
        inactivity can be monitored as potential churn-risk groups.
        """
    )

with col3:
    st.success(
        """
        **Engagement Strategy**

        High-engagement customer segments can be analyzed
        for retention and expansion opportunities.
        """
    )


# ============================================================
# CUSTOMER TABLE
# ============================================================

st.subheader("📋 Customer Data")

display_columns = [
    "Customer_ID",
    "Segment",
    "Total_Spend",
    "Purchases",
    "Engagement_Score",
    "Satisfaction_Score",
    "Last_Activity_Days",
    "Churn_Status"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    height=350
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.subheader("📥 Export Dashboard Data")

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Customer Data",
    data=csv_data,
    file_name="day34_filtered_customer_data.csv",
    mime="text/csv"
)


# ============================================================
# SAVE BUSINESS INSIGHTS
# ============================================================

insight_text = """
DAY 34 - BUSINESS INSIGHT SUMMARY
===================================

Executive Customer Analytics Dashboard

Key Findings
------------

"""

for insight in insights:
    insight_text += insight + "\n"

insight_text += """

Dashboard Components
--------------------
1. Executive KPI cards
2. Customer segmentation analysis
3. Revenue contribution by segment
4. Churn analysis
5. Churn rate by segment
6. Engagement vs spending analysis
7. Customer satisfaction distribution
8. Activity and engagement analysis
9. Customer-level data table
10. Business decision support

Purpose
-------
The dashboard converts customer-level analytics into
business-friendly visual insights for executive decision-making.
"""

with open(OUTPUT_INSIGHTS, "w", encoding="utf-8") as file:
    file.write(insight_text)


# ============================================================
# REFLECTION FILE
# ============================================================

reflection = """
DAY 34 REFLECTION
=================

Today I worked on designing an executive-level customer
analytics dashboard.

Key Learning
------------

1. Learned how to create KPI cards for business metrics.
2. Learned how to visualize customer segmentation.
3. Learned how to analyze churn patterns.
4. Learned how to build interactive Plotly visualizations.
5. Learned how Streamlit can be used to create analytics dashboards.
6. Learned how to convert data science outputs into
   business-friendly insights.

Business Understanding
----------------------

The dashboard demonstrates how customer behavior,
segmentation, engagement and churn metrics can be presented
in a format that supports business analysis and decision-making.

Tools Used
----------

Python
Pandas
NumPy
Streamlit
Plotly

Day 34/60 completed.
"""

with open(OUTPUT_REFLECTION, "w", encoding="utf-8") as file:
    file.write(reflection)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 34/60 | 60 Days of Data Science | "
    "Executive Customer Analytics Dashboard"
)