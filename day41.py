# ============================================================
# DAY 41 | PREDICTING HIGH-RISK CUSTOMERS BEFORE THEY CHURN
# 60 Days of Data Science
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Day 41 - Customer Churn Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 20px;
    color: #9ca3af;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

.high-risk {
    color: #ff4b4b;
    font-weight: 700;
}

.medium-risk {
    color: #ffa500;
    font-weight: 700;
}

.low-risk {
    color: #21c354;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        "WA_Fn-UseC_TelcoCustomerChurn.csv",
        "WA_Fn-UseC_TelcoCustomerChurn.csv",
        "WA_FnUseC_TelcoCustomerChurn.csv",
        "telco_customer_churn.csv",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ]

    for file in possible_files:
        try:
            df = pd.read_csv(file)

            if len(df) > 0:
                return df

        except FileNotFoundError:
            continue

    return None


df = load_data()


# ============================================================
# DATA NOT FOUND
# ============================================================

if df is None:

    st.error(
        "❌ Dataset not found.\n\n"
        "Please keep the Telco Customer Churn CSV file "
        "in the same folder as day41.py."
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.copy()

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Fill missing numerical values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())


# ============================================================
# TARGET
# ============================================================

if "Churn" not in df.columns:

    st.error(
        "❌ 'Churn' column was not found in the dataset."
    )

    st.stop()


# Convert target to 0 / 1
df["Churn_Flag"] = (
    df["Churn"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    })
)

df = df.dropna(subset=["Churn_Flag"])

df["Churn_Flag"] = df["Churn_Flag"].astype(int)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Day 41 | Predicting High-Risk Customers Before They Churn</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predictive Customer Risk Scoring & Retention Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Customer Risk Settings")

high_threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05
)

medium_threshold = st.sidebar.slider(
    "Medium Risk Threshold",
    min_value=0.10,
    max_value=0.50,
    value=0.20,
    step=0.05
)

if medium_threshold >= high_threshold:
    st.sidebar.warning(
        "Medium threshold should be lower than High threshold."
    )

st.sidebar.divider()

st.sidebar.info(
    "High Risk customers have a predicted churn probability "
    f"above {high_threshold:.0%}."
)


# ============================================================
# CUSTOMER OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📌 Customer Overview</div>',
    unsafe_allow_html=True
)

total_customers = len(df)

churned_customers = int(df["Churn_Flag"].sum())

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
# PREPARE FEATURES
# ============================================================

drop_columns = [
    "Churn",
    "Churn_Flag"
]

# CustomerID is an identifier, not a useful predictive feature
if "customerID" in df.columns:
    drop_columns.append("customerID")

X = df.drop(
    columns=drop_columns,
    errors="ignore"
)

y = df["Churn_Flag"]


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# PREPROCESSOR
# ============================================================

# IMPORTANT:
# The previous error happened because "onehot" was passed
# as a string instead of an actual transformer.
#
# Correct:
# OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner("Training customer churn prediction model..."):

    model.fit(
        X_train,
        y_train
    )


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# MODEL PERFORMANCE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

with m2:
    st.metric(
        "Precision",
        f"{precision:.2%}"
    )

with m3:
    st.metric(
        "Recall",
        f"{recall:.2%}"
    )

with m4:
    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

with m5:
    st.metric(
        "ROC-AUC",
        f"{auc:.2%}"
    )


# ============================================================
# PREDICT ALL CUSTOMERS
# ============================================================

all_probabilities = model.predict_proba(
    X
)[:, 1]

df["Churn_Probability"] = all_probabilities


# ============================================================
# RISK CATEGORY
# ============================================================

def assign_risk(probability):

    if probability >= high_threshold:
        return "High Risk"

    elif probability >= medium_threshold:
        return "Medium Risk"

    else:
        return "Low Risk"


df["Risk_Category"] = df[
    "Churn_Probability"
].apply(assign_risk)


# ============================================================
# RISK COUNTS
# ============================================================

risk_counts = (
    df["Risk_Category"]
    .value_counts()
    .reindex(
        ["High Risk", "Medium Risk", "Low Risk"],
        fill_value=0
    )
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">🚦 Customer Risk Distribution</div>',
    unsafe_allow_html=True
)

risk_col1, risk_col2 = st.columns(2)


with risk_col1:

    risk_chart_df = pd.DataFrame({
        "Risk Category": risk_counts.index,
        "Customers": risk_counts.values
    })

    fig_risk = px.bar(
        risk_chart_df,
        x="Risk Category",
        y="Customers",
        title="Customers by Risk Category",
        text="Customers"
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


with risk_col2:

    fig_pie = px.pie(
        risk_chart_df,
        names="Risk Category",
        values="Customers",
        title="Risk Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


# ============================================================
# HIGH RISK CUSTOMERS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 High-Risk Customers</div>',
    unsafe_allow_html=True
)

high_risk_df = df[
    df["Risk_Category"] == "High Risk"
].copy()

high_risk_df = high_risk_df.sort_values(
    "Churn_Probability",
    ascending=False
)


# ============================================================
# RISK SUMMARY
# ============================================================

risk1, risk2, risk3 = st.columns(3)

with risk1:
    st.metric(
        "High Risk Customers",
        f"{len(high_risk_df):,}"
    )

with risk2:

    high_risk_percentage = (
        len(high_risk_df) / total_customers
        if total_customers > 0
        else 0
    )

    st.metric(
        "High Risk %",
        f"{high_risk_percentage:.2%}"
    )

with risk3:

    if len(high_risk_df) > 0:
        avg_high_risk = high_risk_df[
            "Churn_Probability"
        ].mean()
    else:
        avg_high_risk = 0

    st.metric(
        "Average High Risk Probability",
        f"{avg_high_risk:.2%}"
    )


# ============================================================
# DISPLAY CUSTOMER RANKING
# ============================================================

display_columns = []

if "customerID" in high_risk_df.columns:
    display_columns.append("customerID")

for col in [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod"
]:

    if col in high_risk_df.columns:
        display_columns.append(col)

display_columns += [
    "Churn_Probability",
    "Risk_Category"
]

available_display_columns = [
    col for col in display_columns
    if col in high_risk_df.columns
]

ranking_df = high_risk_df[
    available_display_columns
].head(100).copy()


if "Churn_Probability" in ranking_df.columns:

    ranking_df["Churn_Probability"] = (
        ranking_df["Churn_Probability"]
        .map(lambda x: f"{x:.2%}")
    )


st.dataframe(
    ranking_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CHURN PROBABILITY DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Churn Probability Distribution</div>',
    unsafe_allow_html=True
)

fig_hist = px.histogram(
    df,
    x="Churn_Probability",
    nbins=30,
    title="Predicted Churn Probability Across Customers",
    labels={
        "Churn_Probability": "Predicted Churn Probability"
    }
)

fig_hist.add_vline(
    x=high_threshold,
    line_dash="dash",
    annotation_text="High Risk Threshold"
)

fig_hist.add_vline(
    x=medium_threshold,
    line_dash="dot",
    annotation_text="Medium Risk Threshold"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# ============================================================
# TENURE VS CHURN RISK
# ============================================================

if "tenure" in df.columns:

    st.markdown(
        '<div class="section-title">👥 Customer Tenure vs Churn Risk</div>',
        unsafe_allow_html=True
    )

    fig_tenure = px.scatter(
        df,
        x="tenure",
        y="Churn_Probability",
        color="Risk_Category",
        hover_data=[
            col
            for col in ["customerID", "Contract", "MonthlyCharges"]
            if col in df.columns
        ],
        title="Customer Tenure vs Predicted Churn Probability"
    )

    fig_tenure.add_hline(
        y=high_threshold,
        line_dash="dash"
    )

    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )


# ============================================================
# MONTHLY CHARGES VS CHURN RISK
# ============================================================

if "MonthlyCharges" in df.columns:

    st.markdown(
        '<div class="section-title">💰 Monthly Charges vs Churn Risk</div>',
        unsafe_allow_html=True
    )

    fig_charges = px.scatter(
        df,
        x="MonthlyCharges",
        y="Churn_Probability",
        color="Risk_Category",
        title="Monthly Charges vs Predicted Churn Probability",
        opacity=0.65
    )

    st.plotly_chart(
        fig_charges,
        use_container_width=True
    )


# ============================================================
# CONTRACT ANALYSIS
# ============================================================

if "Contract" in df.columns:

    st.markdown(
        '<div class="section-title">📄 Contract Type Risk Analysis</div>',
        unsafe_allow_html=True
    )

    contract_analysis = (
        df.groupby("Contract")
        .agg(
            Customers=("Churn_Flag", "count"),
            Churn_Rate=("Churn_Flag", "mean"),
            Avg_Risk=("Churn_Probability", "mean")
        )
        .reset_index()
    )

    contract_analysis["Churn_Rate"] = (
        contract_analysis["Churn_Rate"] * 100
    )

    contract_analysis["Avg_Risk"] = (
        contract_analysis["Avg_Risk"] * 100
    )

    st.dataframe(
        contract_analysis,
        use_container_width=True,
        hide_index=True
    )

    fig_contract = px.bar(
        contract_analysis,
        x="Contract",
        y="Churn_Rate",
        title="Churn Rate by Contract Type",
        text="Churn_Rate"
    )

    fig_contract.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )


# ============================================================
# RETENTION STRATEGIES
# ============================================================

st.markdown(
    '<div class="section-title">💡 Recommended Retention Strategies</div>',
    unsafe_allow_html=True
)

high_count = len(
    df[df["Risk_Category"] == "High Risk"]
)

medium_count = len(
    df[df["Risk_Category"] == "Medium Risk"]
)

low_count = len(
    df[df["Risk_Category"] == "Low Risk"]
)


strategy_col1, strategy_col2, strategy_col3 = st.columns(3)


with strategy_col1:

    st.markdown("### 🔴 High Risk")

    st.write(
        f"**{high_count:,} customers** require immediate attention."
    )

    st.write(
        "Recommended actions:"
    )

    st.write(
        "• Personal retention offer"
    )

    st.write(
        "• Proactive customer support"
    )

    st.write(
        "• Contract upgrade incentive"
    )

    st.write(
        "• Targeted discount campaign"
    )


with strategy_col2:

    st.markdown("### 🟠 Medium Risk")

    st.write(
        f"**{medium_count:,} customers** need proactive engagement."
    )

    st.write(
        "Recommended actions:"
    )

    st.write(
        "• Engagement campaign"
    )

    st.write(
        "• Product education"
    )

    st.write(
        "• Usage-based recommendations"
    )

    st.write(
        "• Satisfaction check-in"
    )


with strategy_col3:

    st.markdown("### 🟢 Low Risk")

    st.write(
        f"**{low_count:,} customers** are relatively stable."
    )

    st.write(
        "Recommended actions:"
    )

    st.write(
        "• Loyalty programs"
    )

    st.write(
        "• Cross-selling"
    )

    st.write(
        "• Referral campaigns"
    )

    st.write(
        "• Long-term engagement"
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">📌 Business Insights</div>',
    unsafe_allow_html=True
)

insights = []

insights.append(
    f"The overall customer churn rate is {churn_rate:.2%}."
)

insights.append(
    f"The model identifies {high_count:,} customers "
    f"as high risk using the current threshold."
)

if "Contract" in df.columns:

    contract_churn = (
        df.groupby("Contract")["Churn_Flag"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(contract_churn) > 0:

        highest_contract = contract_churn.index[0]

        insights.append(
            f"{highest_contract} customers show the highest "
            "observed churn rate among contract groups."
        )


if "MonthlyCharges" in df.columns:

    high_charge_risk = df[
        df["Risk_Category"] == "High Risk"
    ]["MonthlyCharges"]

    if len(high_charge_risk) > 0:

        insights.append(
            f"High-risk customers have an average monthly charge "
            f"of ${high_charge_risk.mean():.2f}."
        )


for i, insight in enumerate(insights, start=1):

    st.write(
        f"**{i}.** {insight}"
    )


# ============================================================
# MODEL CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🧮 Confusion Matrix</div>',
    unsafe_allow_html=True
)

cm = confusion_matrix(
    y_test,
    y_pred
)

fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted No Churn", "Predicted Churn"],
        y=["Actual No Churn", "Actual Churn"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues"
    )
)

fig_cm.update_layout(
    title="Model Confusion Matrix"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with st.expander("📋 View Classification Report"):

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Retained",
            "Churned"
        ],
        zero_division=0
    )

    st.code(report)


# ============================================================
# DOWNLOAD HIGH-RISK CUSTOMERS
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Export Risk Analysis</div>',
    unsafe_allow_html=True
)


download_df = df.copy()

download_df["Churn_Probability"] = (
    download_df["Churn_Probability"]
    .round(4)
)

csv_data = download_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Complete Risk Analysis CSV",
    data=csv_data,
    file_name="day41_customer_churn_risk_analysis.csv",
    mime="text/csv"
)


# ============================================================
# HIGH RISK ONLY DOWNLOAD
# ============================================================

high_risk_csv = high_risk_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download High-Risk Customers CSV",
    data=high_risk_csv,
    file_name="day41_high_risk_customers.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 41/60 | 60 Days of Data Science | "
    "Predictive Customer Risk Scoring System"
)

st.caption(
    "Built with Python, Pandas, Scikit-learn, Plotly and Streamlit"
)