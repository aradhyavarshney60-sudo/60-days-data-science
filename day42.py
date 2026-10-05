import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Day 42 - Unified Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 20px;
        color: #777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .insight-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border-left: 5px solid #4f46e5;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE DISCOVERY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_csv_files():
    return list(BASE_DIR.glob("*.csv"))


CSV_FILES = find_csv_files()


# ============================================================
# DATA LOADING HELPERS
# ============================================================

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_file(keywords):
    """
    Finds the first CSV whose filename contains
    any of the supplied keywords.
    """
    keywords = [k.lower() for k in keywords]

    for file in CSV_FILES:
        filename = file.name.lower()

        for keyword in keywords:
            if keyword in filename:
                return file

    return None


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False),
        errors="coerce"
    )


# ============================================================
# LOAD AVAILABLE DATASETS
# ============================================================

forecast_file = find_file([
    "forecast",
    "customer_growth_forecast"
])

kpi_file = find_file([
    "kpi_summary",
    "kpi"
])

risk_file = find_file([
    "risk",
    "churn_prediction",
    "customer_risk"
])

retention_file = find_file([
    "retention",
    "customer_growth",
    "churn"
])

telco_file = find_file([
    "telco",
    "wa_fnuse"
])


forecast_df = load_csv(forecast_file) if forecast_file else None
kpi_df = load_csv(kpi_file) if kpi_file else None
risk_df = load_csv(risk_file) if risk_file else None
retention_df = load_csv(retention_file) if retention_file else None
telco_df = load_csv(telco_file) if telco_file else None


# ============================================================
# CUSTOMER DATA
# ============================================================

customer_df = None

for df in [
    risk_df,
    retention_df,
    telco_df
]:
    if df is not None and len(df) > 0:
        customer_df = df.copy()
        break


# ============================================================
# GENERATE FALLBACK CUSTOMER DATA
# ============================================================

if customer_df is None:

    np.random.seed(42)

    n = 1000

    customer_df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "MonthlyRevenue": np.random.uniform(50, 500, n),
        "TenureMonths": np.random.randint(1, 72, n),
        "SupportTickets": np.random.poisson(2, n),
        "MonthlyUsage": np.random.uniform(10, 100, n),
        "Churn": np.random.choice(
            [0, 1],
            size=n,
            p=[0.74, 0.26]
        )
    })


# ============================================================
# NORMALIZE CHURN COLUMN
# ============================================================

def detect_churn_column(df):

    possible = [
        "Churn",
        "churn",
        "Exited",
        "exited",
        "Churned",
        "churned",
        "is_churn",
        "target"
    ]

    for col in possible:
        if col in df.columns:
            return col

    return None


churn_col = detect_churn_column(customer_df)


if churn_col:

    churn_values = (
        customer_df[churn_col]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    customer_df["_churn_flag"] = churn_values.map({
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "churned": 1,
        "retained": 0
    })

    customer_df["_churn_flag"] = pd.to_numeric(
        customer_df["_churn_flag"],
        errors="coerce"
    )

else:

    customer_df["_churn_flag"] = np.random.choice(
        [0, 1],
        len(customer_df),
        p=[0.74, 0.26]
    )


customer_df["_churn_flag"] = customer_df["_churn_flag"].fillna(0)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 Customer Intelligence")

st.sidebar.markdown(
    """
    **Day 42/60**

    Unified Business Intelligence Decision System
    """
)

st.sidebar.divider()

st.sidebar.subheader("Dashboard Modules")

show_forecast = st.sidebar.checkbox(
    "📈 Forecasting",
    value=True
)

show_kpi = st.sidebar.checkbox(
    "📊 KPI Monitoring",
    value=True
)

show_retention = st.sidebar.checkbox(
    "🔄 Retention Analytics",
    value=True
)

show_risk = st.sidebar.checkbox(
    "⚠️ Customer Risk",
    value=True
)

show_architecture = st.sidebar.checkbox(
    "🏗️ Architecture",
    value=True
)

show_report = st.sidebar.checkbox(
    "📝 Business Report",
    value=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Day 42 | Unified Customer Intelligence Decision System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Integrating forecasting, KPI tracking, retention analytics and predictive risk scoring'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EXECUTIVE KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📌 Executive Overview</div>',
    unsafe_allow_html=True
)

total_customers = len(customer_df)

churned_customers = int(
    customer_df["_churn_flag"].sum()
)

retained_customers = total_customers - churned_customers

churn_rate = (
    churned_customers / total_customers
    if total_customers > 0
    else 0
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

with col4:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2%}"
    )


# ============================================================
# KPI MONITORING
# ============================================================

if show_kpi:

    st.markdown(
        '<div class="section-title">📊 KPI Monitoring</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    avg_revenue = None

    revenue_columns = [
        "MonthlyRevenue",
        "MonthlyCharges",
        "Revenue",
        "TotalRevenue",
        "revenue"
    ]

    for col in revenue_columns:

        if col in customer_df.columns:

            revenue = clean_numeric(
                customer_df[col]
            )

            if revenue.notna().any():

                avg_revenue = revenue.mean()
                break

    if avg_revenue is None:
        avg_revenue = 0

    with k1:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2%}"
        )

    with k2:
        st.metric(
            "Retention Rate",
            f"{1 - churn_rate:.2%}"
        )

    with k3:
        st.metric(
            "Average Revenue",
            f"{avg_revenue:,.2f}"
        )

    with k4:
        st.metric(
            "Customers at Risk",
            f"{churned_customers:,}"
        )

    if churn_col:

        chart_df = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [
                retained_customers,
                churned_customers
            ]
        })

        fig = px.bar(
            chart_df,
            x="Status",
            y="Customers",
            title="Customer Retention vs Churn"
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# FORECASTING
# ============================================================

if show_forecast:

    st.markdown(
        '<div class="section-title">📈 Forecasting Module</div>',
        unsafe_allow_html=True
    )

    if forecast_df is not None and len(forecast_df) > 0:

        st.success(
            f"Forecast dataset loaded: {forecast_file.name}"
        )

        st.dataframe(
            forecast_df.head(10),
            use_container_width=True
        )

        numeric_cols = forecast_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) > 0:

            forecast_value_col = numeric_cols[-1]

            fig = px.line(
                forecast_df,
                y=forecast_value_col,
                title=f"Forecast Trend: {forecast_value_col}"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "No forecast CSV was found. Showing an illustrative customer trend."
        )

        np.random.seed(42)

        periods = 12

        months = pd.date_range(
            end=pd.Timestamp.today(),
            periods=periods,
            freq="ME"
        )

        base = 5000

        values = []

        for i in range(periods):
            value = (
                base
                + i * 180
                + np.random.randint(-150, 150)
            )

            values.append(value)

        forecast_demo = pd.DataFrame({
            "Month": months,
            "Customer_Count": values
        })

        fig = px.line(
            forecast_demo,
            x="Month",
            y="Customer_Count",
            markers=True,
            title="Customer Growth Forecast"
        )

        fig.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# RETENTION ANALYTICS
# ============================================================

if show_retention:

    st.markdown(
        '<div class="section-title">🔄 Retention Analytics</div>',
        unsafe_allow_html=True
    )

    retention_col1, retention_col2 = st.columns(2)

    with retention_col1:

        retention_rate = 1 - churn_rate

        st.metric(
            "Current Retention Rate",
            f"{retention_rate:.2%}"
        )

        st.markdown(
            """
            <div class="insight-box">
            <b>Retention Insight</b><br>
            Customers who are not showing churn signals represent the
            retained customer base. Monitoring behavior changes can help
            identify customers who may require proactive engagement.
            </div>
            """,
            unsafe_allow_html=True
        )

    with retention_col2:

        status_df = pd.DataFrame({
            "Customer Status": [
                "Retained",
                "Churned"
            ],
            "Count": [
                retained_customers,
                churned_customers
            ]
        })

        fig = px.pie(
            status_df,
            names="Customer Status",
            values="Count",
            title="Customer Retention Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PREDICTIVE CUSTOMER RISK
# ============================================================

risk_predictions = None
model_auc = None

if show_risk:

    st.markdown(
        '<div class="section-title">⚠️ Predictive Customer Risk</div>',
        unsafe_allow_html=True
    )

    model_df = customer_df.copy()

    target = "_churn_flag"

    X = model_df.drop(
        columns=[target],
        errors="ignore"
    )

    y = model_df[target]

    # Remove obvious ID columns
    id_columns = []

    for col in X.columns:

        if (
            "id" in col.lower()
            or "customerid" in col.lower()
            or "customer_id" in col.lower()
        ):
            id_columns.append(col)

    X = X.drop(
        columns=id_columns,
        errors="ignore"
    )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if len(numeric_features) + len(categorical_features) > 0:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers = []

        if numeric_features:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                )
            )

        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers
        )

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ]
        )

        try:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model.fit(
                X_train,
                y_train
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            try:
                model_auc = roc_auc_score(
                    y_test,
                    probabilities
                )
            except Exception:
                model_auc = None

            st.success(
                f"Risk model trained successfully | Accuracy: {accuracy:.2%}"
            )

            if model_auc is not None:

                st.metric(
                    "Model ROC-AUC",
                    f"{model_auc:.3f}"
                )

            # Predict all customers
            all_probabilities = model.predict_proba(
                X
            )[:, 1]

            risk_predictions = customer_df.copy()

            risk_predictions["Risk_Score"] = (
                all_probabilities
            )

            risk_predictions["Risk_Level"] = pd.cut(
                risk_predictions["Risk_Score"],
                bins=[
                    -0.01,
                    0.20,
                    0.50,
                    1.01
                ],
                labels=[
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ]
            )

            risk_predictions = risk_predictions.sort_values(
                "Risk_Score",
                ascending=False
            )

            high_risk = risk_predictions[
                risk_predictions["Risk_Level"] == "High Risk"
            ]

            medium_risk = risk_predictions[
                risk_predictions["Risk_Level"] == "Medium Risk"
            ]

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "High Risk",
                    f"{len(high_risk):,}"
                )

            with r2:
                st.metric(
                    "Medium Risk",
                    f"{len(medium_risk):,}"
                )

            with r3:
                st.metric(
                    "Low Risk",
                    f"{len(risk_predictions) - len(high_risk) - len(medium_risk):,}"
                )

            risk_chart = (
                risk_predictions["Risk_Level"]
                .value_counts()
                .reset_index()
            )

            risk_chart.columns = [
                "Risk Level",
                "Customers"
            ]

            fig = px.bar(
                risk_chart,
                x="Risk Level",
                y="Customers",
                title="Customer Risk Distribution"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "Top 20 High-Risk Customers"
            )

            display_columns = [
                col for col in [
                    "CustomerID",
                    "customerID",
                    "customer_id",
                    "Risk_Score",
                    "Risk_Level"
                ]
                if col in risk_predictions.columns
            ]

            if not display_columns:
                display_columns = [
                    "Risk_Score",
                    "Risk_Level"
                ]

            st.dataframe(
                risk_predictions[
                    display_columns
                ].head(20),
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Risk model could not be trained: {e}"
            )


# ============================================================
# BUSINESS DECISION MATRIX
# ============================================================

if show_risk:

    st.markdown(
        '<div class="section-title">🎯 Business Decision Matrix</div>',
        unsafe_allow_html=True
    )

    decision_df = pd.DataFrame({
        "Risk Level": [
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],
        "Business Action": [
            "Immediate retention campaign + personal outreach",
            "Targeted offers + engagement monitoring",
            "Regular engagement + loyalty programs"
        ],
        "Priority": [
            "Critical",
            "High",
            "Normal"
        ]
    })

    st.dataframe(
        decision_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ARCHITECTURE WORKFLOW
# ============================================================

if show_architecture:

    st.markdown(
        '<div class="section-title">🏗️ Unified BI Architecture</div>',
        unsafe_allow_html=True
    )

    labels = [
        "Customer Data",
        "Data Preparation",
        "KPI Analytics",
        "Forecasting",
        "Retention Analytics",
        "Risk Prediction",
        "Executive Dashboard",
        "Business Decision"
    ]

    source = [
        0,
        1,
        1,
        1,
        1,
        1,
        2,
        3,
        4,
        5,
        6
    ]

    target = [
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        6,
        6,
        6,
        7
    ]

    value = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=labels,
                pad=20,
                thickness=25
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )
    )

    fig.update_layout(
        title="Customer Intelligence Decision Workflow",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
        **Workflow:**

        Customer Data → Data Preparation → Analytics Modules →
        Executive Dashboard → Business Decision
        """
    )


# ============================================================
# BUSINESS REPORT
# ============================================================

if show_report:

    st.markdown(
        '<div class="section-title">📝 Business Intelligence Report</div>',
        unsafe_allow_html=True
    )

    if churn_rate >= 0.30:

        risk_summary = (
            "The current churn rate is relatively high. "
            "The business should prioritize retention campaigns "
            "and proactive outreach to high-risk customers."
        )

    elif churn_rate >= 0.20:

        risk_summary = (
            "The current churn rate indicates a meaningful "
            "retention opportunity. Customer risk scoring should "
            "be used to prioritize targeted interventions."
        )

    else:

        risk_summary = (
            "The current churn rate is comparatively controlled. "
            "The business should continue monitoring customer behavior "
            "and maintain proactive retention programs."
        )

    report = f"""
DAY 42 BUSINESS INTELLIGENCE REPORT
===================================

Executive Summary
-----------------

The unified customer intelligence system combines KPI monitoring,
forecasting, retention analytics and predictive customer risk scoring
into one decision-support workflow.

Customer Metrics
----------------

Total Customers: {total_customers:,}
Churned Customers: {churned_customers:,}
Retained Customers: {retained_customers:,}
Churn Rate: {churn_rate:.2%}
Retention Rate: {(1 - churn_rate):.2%}

Business Insight
----------------

{risk_summary}

Decision Recommendations
------------------------

1. Monitor customer churn and retention KPIs regularly.

2. Prioritize high-risk customers for proactive retention actions.

3. Use forecasting to identify future changes in customer demand
   or business performance.

4. Combine KPI trends with customer-level risk signals before
   making major business decisions.

5. Maintain a centralized dashboard so executives can access
   important business intelligence from one location.

Architecture
------------

Customer Data
      |
      v
Data Preparation
      |
      +---- KPI Analytics
      |
      +---- Forecasting
      |
      +---- Retention Analytics
      |
      +---- Risk Prediction
      |
      v
Executive Dashboard
      |
      v
Business Decision

Key Tradeoffs
-------------

Accuracy vs Interpretability:
Complex predictive models may improve prediction performance,
while simpler models are easier for business stakeholders to
understand.

Centralization vs Complexity:
A unified dashboard improves decision visibility but increases
the complexity of maintaining multiple analytics modules.

Automation vs Human Judgment:
Automated risk scores support faster decisions, but business
teams should still consider customer context before taking action.

Week 6 Reflection
-----------------

This week focused on moving from individual analytics tasks toward
an integrated business intelligence system. The major learning was
understanding how forecasting, KPI monitoring, retention analytics
and predictive modeling can work together to support real-world
business decisions.
"""

    st.text_area(
        "Business Intelligence Report",
        report,
        height=500
    )

    st.download_button(
        label="📥 Download Business Report",
        data=report,
        file_name="day42_business_intelligence_report.txt",
        mime="text/plain"
    )


# ============================================================
# DATA SOURCES
# ============================================================

st.markdown(
    '<div class="section-title">📂 Data Sources Detected</div>',
    unsafe_allow_html=True
)

source_data = []

for name, file in [
    ("Forecasting", forecast_file),
    ("KPI", kpi_file),
    ("Risk", risk_file),
    ("Retention", retention_file),
    ("Customer Dataset", telco_file)
]:

    source_data.append({
        "Module": name,
        "Dataset": file.name if file else "Not found"
    })

st.dataframe(
    pd.DataFrame(source_data),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 42/60 | 60 Days of Data Science | "
    "Unified Customer Intelligence Decision System"
)

st.caption(
    "Built with Python, Pandas, Scikit-learn, Plotly and Streamlit"
)