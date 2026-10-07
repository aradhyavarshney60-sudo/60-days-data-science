import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Intelligence Dashboard")
st.caption("Day 44 | Interactive Customer Analytics & Churn Insights")


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data(uploaded_file=None):

    # User uploaded CSV
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

    # Try existing customer dataset
    possible_files = [
        "WA_FnUseC_TelcoCustomerChurn.csv",
        "day34_customer_dashboard_data.csv",
        "customer_data.csv"
    ]

    for file in possible_files:
        try:
            df = pd.read_csv(file)
            return df
        except FileNotFoundError:
            continue

    # Fallback demo data
    np.random.seed(42)

    n = 500

    df = pd.DataFrame({
        "CustomerID": [f"C{i:04d}" for i in range(1, n + 1)],
        "Age": np.random.randint(18, 70, n),
        "MonthlyCharges": np.round(np.random.uniform(20, 150, n), 2),
        "Tenure": np.random.randint(1, 72, n),
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            n
        ),
        "Churn": np.random.choice(
            ["Yes", "No"],
            n,
            p=[0.27, 0.73]
        )
    })

    return df


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

df = load_data(uploaded_file)


# =========================================================
# DATA CLEANING
# =========================================================

df.columns = df.columns.str.strip()

# Convert TotalCharges if available
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Convert SeniorCitizen to readable values
if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].map({
        0: "No",
        1: "Yes"
    }).fillna(df["SeniorCitizen"].astype(str))


# =========================================================
# DETECT IMPORTANT COLUMNS
# =========================================================

churn_column = None

for col in ["Churn", "churn", "Exited", "Attrition"]:
    if col in df.columns:
        churn_column = col
        break


customer_id_column = None

for col in ["customerID", "CustomerID", "customer_id", "Customer Id"]:
    if col in df.columns:
        customer_id_column = col
        break


# =========================================================
# CHURN STANDARDIZATION
# =========================================================

if churn_column is not None:

    df["_Churn"] = (
        df[churn_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    churn_yes_values = [
        "yes",
        "y",
        "1",
        "true",
        "churned"
    ]

    df["_ChurnBinary"] = df["_Churn"].isin(
        churn_yes_values
    ).astype(int)

else:

    # Demo fallback
    df["_ChurnBinary"] = 0


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.subheader("🔎 Filters")

filtered_df = df.copy()


# Contract filter
if "Contract" in df.columns:

    contracts = sorted(
        df["Contract"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_contracts = st.sidebar.multiselect(
        "Contract",
        contracts,
        default=contracts
    )

    filtered_df = filtered_df[
        filtered_df["Contract"]
        .astype(str)
        .isin(selected_contracts)
    ]


# Gender filter
if "gender" in df.columns:

    genders = sorted(
        df["gender"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        genders,
        default=genders
    )

    filtered_df = filtered_df[
        filtered_df["gender"]
        .astype(str)
        .isin(selected_gender)
    ]


# Senior citizen filter
if "SeniorCitizen" in df.columns:

    senior_values = sorted(
        df["SeniorCitizen"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_senior = st.sidebar.multiselect(
        "Senior Citizen",
        senior_values,
        default=senior_values
    )

    filtered_df = filtered_df[
        filtered_df["SeniorCitizen"]
        .astype(str)
        .isin(selected_senior)
    ]


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_customers = len(filtered_df)

churned_customers = int(
    filtered_df["_ChurnBinary"].sum()
)

if total_customers > 0:
    churn_rate = (
        churned_customers /
        total_customers
    ) * 100
else:
    churn_rate = 0


# Revenue / charges
monthly_revenue = 0

if "MonthlyCharges" in filtered_df.columns:

    monthly_revenue = pd.to_numeric(
        filtered_df["MonthlyCharges"],
        errors="coerce"
    ).fillna(0).sum()


# Average tenure
average_tenure = 0

if "tenure" in filtered_df.columns:

    average_tenure = pd.to_numeric(
        filtered_df["tenure"],
        errors="coerce"
    ).mean()

elif "Tenure" in filtered_df.columns:

    average_tenure = pd.to_numeric(
        filtered_df["Tenure"],
        errors="coerce"
    ).mean()


# =========================================================
# KPI CARDS
# =========================================================

st.subheader("📌 Key Performance Indicators")

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
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

with col4:
    st.metric(
        "Monthly Charges",
        f"${monthly_revenue:,.2f}"
    )


st.divider()


# =========================================================
# CHURN DISTRIBUTION
# =========================================================

st.subheader("📉 Customer Churn Analysis")

col1, col2 = st.columns(2)


with col1:

    if churn_column is not None:

        churn_counts = (
            filtered_df[churn_column]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        churn_counts.columns = [
            "Churn",
            "Customers"
        ]

        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            title="Customer Churn Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No churn column found in the uploaded dataset."
        )


# =========================================================
# CONTRACT ANALYSIS
# =========================================================

with col2:

    if "Contract" in filtered_df.columns:

        contract_data = (
            filtered_df
            .groupby("Contract")["_ChurnBinary"]
            .agg(["count", "sum"])
            .reset_index()
        )

        contract_data["Churn Rate"] = (
            contract_data["sum"] /
            contract_data["count"]
        ) * 100

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate",
            title="Churn Rate by Contract",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Contract column not available."
        )


# =========================================================
# MONTHLY CHARGES VS CHURN
# =========================================================

if "MonthlyCharges" in filtered_df.columns:

    st.subheader("💰 Monthly Charges vs Churn")

    plot_df = filtered_df.copy()

    plot_df["MonthlyCharges"] = pd.to_numeric(
        plot_df["MonthlyCharges"],
        errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=["MonthlyCharges"]
    )

    if len(plot_df) > 0:

        if churn_column is not None:

            fig = px.box(
                plot_df,
                x=churn_column,
                y="MonthlyCharges",
                title="Monthly Charges Distribution by Churn"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# TENURE ANALYSIS
# =========================================================

tenure_column = None

if "tenure" in filtered_df.columns:
    tenure_column = "tenure"

elif "Tenure" in filtered_df.columns:
    tenure_column = "Tenure"


if tenure_column is not None:

    st.subheader("📅 Tenure Analysis")

    tenure_data = filtered_df.copy()

    tenure_data[tenure_column] = pd.to_numeric(
        tenure_data[tenure_column],
        errors="coerce"
    )

    tenure_data = tenure_data.dropna(
        subset=[tenure_column]
    )

    fig = px.histogram(
        tenure_data,
        x=tenure_column,
        color=churn_column if churn_column else None,
        nbins=20,
        title="Customer Tenure Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# SERVICE ANALYSIS
# =========================================================

service_columns = [
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "OnlineBackup"
]

available_service_columns = [
    col
    for col in service_columns
    if col in filtered_df.columns
]


if available_service_columns:

    st.subheader("🛠️ Customer Service Analysis")

    selected_service = st.selectbox(
        "Select service dimension",
        available_service_columns
    )

    service_data = (
        filtered_df
        .groupby(selected_service)["_ChurnBinary"]
        .agg(["count", "sum"])
        .reset_index()
    )

    service_data["Churn Rate"] = (
        service_data["sum"] /
        service_data["count"]
    ) * 100

    fig = px.bar(
        service_data,
        x=selected_service,
        y="Churn Rate",
        title=f"Churn Rate by {selected_service}",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# CUSTOMER TABLE
# =========================================================

st.subheader("👥 Customer Data")

display_df = filtered_df.copy()

# Hide internal columns
columns_to_hide = [
    "_Churn",
    "_ChurnBinary"
]

display_df = display_df.drop(
    columns=[
        col
        for col in columns_to_hide
        if col in display_df.columns
    ]
)

st.dataframe(
    display_df,
    use_container_width=True,
    height=400
)


# =========================================================
# DOWNLOAD FILTERED DATA
# =========================================================

csv_data = display_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Data",
    data=csv_data,
    file_name="filtered_customer_data.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Day 44 | Customer Intelligence Dashboard | "
    "Built with Streamlit, Pandas and Plotly"
)