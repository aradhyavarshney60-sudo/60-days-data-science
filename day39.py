import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Executive KPI Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📊 Executive KPI Monitoring System")
st.markdown(
    "Customer, Revenue & Retention performance monitoring dashboard"
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Dashboard Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    # Demo dataset
    np.random.seed(42)

    dates = pd.date_range(
        start="2025-01-01",
        periods=180,
        freq="D"
    )

    customers = np.random.randint(80, 250, 180)
    revenue = np.random.randint(10000, 50000, 180)
    retained = np.random.randint(60, 220, 180)
    churned = np.random.randint(5, 40, 180)

    df = pd.DataFrame({
        "Date": dates,
        "Customers": customers,
        "Revenue": revenue,
        "Retained": retained,
        "Churned": churned
    })

# ============================================================
# DATA PREPARATION
# ============================================================

df.columns = df.columns.str.strip()

# Try to detect important columns

date_col = None
revenue_col = None
customer_col = None
retained_col = None
churn_col = None

for col in df.columns:

    col_lower = col.lower()

    if "date" in col_lower or "time" in col_lower:
        date_col = col

    if "revenue" in col_lower or "sales" in col_lower:
        revenue_col = col

    if "customer" in col_lower:
        customer_col = col

    if "retained" in col_lower or "retention" in col_lower:
        retained_col = col

    if "churn" in col_lower:
        churn_col = col


# ============================================================
# DATE CONVERSION
# ============================================================

if date_col:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = df.dropna(subset=[date_col])

    df = df.sort_values(date_col)

# ============================================================
# KPI CALCULATIONS
# ============================================================

# Revenue

if revenue_col:

    total_revenue = df[revenue_col].sum()
    avg_revenue = df[revenue_col].mean()

else:

    total_revenue = 0
    avg_revenue = 0


# Customers

if customer_col:

    total_customers = df[customer_col].sum()
    avg_customers = df[customer_col].mean()

else:

    total_customers = 0
    avg_customers = 0


# Retention

if retained_col and customer_col:

    retention_rate = (
        df[retained_col].sum()
        / df[customer_col].sum()
    ) * 100

else:

    retention_rate = 0


# Churn

if churn_col and customer_col:

    churn_rate = (
        df[churn_col].sum()
        / df[customer_col].sum()
    ) * 100

else:

    churn_rate = 0


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="💰 Total Revenue",
        value=f"₹{total_revenue:,.0f}"
    )

with col2:

    st.metric(
        label="👥 Total Customers",
        value=f"{total_customers:,.0f}"
    )

with col3:

    st.metric(
        label="🔄 Retention Rate",
        value=f"{retention_rate:.2f}%"
    )

with col4:

    st.metric(
        label="📉 Churn Rate",
        value=f"{churn_rate:.2f}%"
    )


st.divider()

# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("🔍 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

# ============================================================
# REVENUE TREND
# ============================================================

st.subheader("💰 Revenue Performance")

if date_col and revenue_col:

    revenue_fig = px.line(
        df,
        x=date_col,
        y=revenue_col,
        title="Revenue Trend Over Time",
        markers=True
    )

    revenue_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified"
    )

    st.plotly_chart(
        revenue_fig,
        use_container_width=True
    )

else:

    st.warning(
        "Date and Revenue columns are required for revenue trend."
    )

# ============================================================
# CUSTOMER TREND
# ============================================================

st.subheader("👥 Customer Performance")

if date_col and customer_col:

    customer_fig = px.line(
        df,
        x=date_col,
        y=customer_col,
        title="Customer Trend Over Time",
        markers=True
    )

    customer_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Customers",
        hovermode="x unified"
    )

    st.plotly_chart(
        customer_fig,
        use_container_width=True
    )

else:

    st.warning(
        "Date and Customer columns are required for customer trend."
    )

# ============================================================
# RETENTION & CHURN
# ============================================================

st.subheader("🔄 Customer Retention & Churn")

col1, col2 = st.columns(2)

# Retention chart

with col1:

    if date_col and retained_col:

        retention_fig = px.line(
            df,
            x=date_col,
            y=retained_col,
            title="Retained Customers",
            markers=True
        )

        retention_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Retained Customers"
        )

        st.plotly_chart(
            retention_fig,
            use_container_width=True
        )

    else:

        st.info(
            "Retained customer data not available."
        )


# Churn chart

with col2:

    if date_col and churn_col:

        churn_fig = px.bar(
            df,
            x=date_col,
            y=churn_col,
            title="Customer Churn",
        )

        churn_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Churned Customers"
        )

        st.plotly_chart(
            churn_fig,
            use_container_width=True
        )

    else:

        st.info(
            "Churn data not available."
        )

# ============================================================
# KPI SUMMARY TABLE
# ============================================================

st.subheader("📋 KPI Summary")

kpi_data = pd.DataFrame({

    "KPI": [
        "Total Revenue",
        "Average Revenue",
        "Total Customers",
        "Average Customers",
        "Retention Rate",
        "Churn Rate"
    ],

    "Value": [
        f"₹{total_revenue:,.0f}",
        f"₹{avg_revenue:,.0f}",
        f"{total_customers:,.0f}",
        f"{avg_customers:,.0f}",
        f"{retention_rate:.2f}%",
        f"{churn_rate:.2f}%"
    ]

})

st.dataframe(
    kpi_data,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.subheader("💡 Executive Business Insights")

insights = []

if total_revenue > 0:

    insights.append(
        f"Total recorded revenue is ₹{total_revenue:,.0f}."
    )

if avg_revenue > 0:

    insights.append(
        f"Average revenue per record is ₹{avg_revenue:,.0f}."
    )

if retention_rate > 0:

    insights.append(
        f"The calculated retention rate is {retention_rate:.2f}%."
    )

if churn_rate > 0:

    insights.append(
        f"The calculated churn rate is {churn_rate:.2f}%."
    )

if len(insights) == 0:

    insights.append(
        "Additional customer and revenue fields are required "
        "to generate detailed business insights."
    )

for insight in insights:

    st.write("•", insight)


# ============================================================
# DATA QUALITY
# ============================================================

st.subheader("🔎 Data Quality Overview")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with quality_col2:

    st.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )

with quality_col3:

    missing_values = df.isnull().sum().sum()

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )


# ============================================================
# EXPORT
# ============================================================

st.subheader("📥 Export KPI Data")

csv_data = kpi_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download KPI Summary CSV",
    data=csv_data,
    file_name="day39_kpi_summary.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 39/60 | 60 Days of Data Science | "
    "Executive KPI Monitoring System"
)