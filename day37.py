import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ARIMA
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Day 37 | Revenue Forecasting",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Day 37 - Predicting Future Revenue with Time Series Models")

st.markdown(
    """
    **Objective:** Forecast future business revenue using historical
    time-series data and an ARIMA model.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=7,
    max_value=60,
    value=30
)

arima_p = st.sidebar.slider(
    "ARIMA p",
    min_value=0,
    max_value=5,
    value=1
)

arima_d = st.sidebar.slider(
    "ARIMA d",
    min_value=0,
    max_value=2,
    value=1
)

arima_q = st.sidebar.slider(
    "ARIMA q",
    min_value=0,
    max_value=5,
    value=1
)


# ============================================================
# DATA LOADING
# ============================================================

st.header("1️⃣ Load Revenue Data")

uploaded_file = st.file_uploader(
    "Upload revenue CSV",
    type=["csv"]
)


def create_demo_data():

    np.random.seed(42)

    dates = pd.date_range(
        start="2025-01-01",
        periods=365,
        freq="D"
    )

    trend = np.linspace(1000, 1800, len(dates))

    seasonal = (
        150 * np.sin(
            np.arange(len(dates)) * 2 * np.pi / 30
        )
    )

    noise = np.random.normal(
        0,
        80,
        len(dates)
    )

    revenue = trend + seasonal + noise

    revenue = np.maximum(
        revenue,
        100
    )

    df = pd.DataFrame({
        "date": dates,
        "revenue": revenue
    })

    return df


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("✅ CSV uploaded successfully!")

else:

    st.info(
        "No CSV uploaded. Demo revenue data is being used."
    )

    df = create_demo_data()


# ============================================================
# FIND DATE COLUMN
# ============================================================

date_columns = [
    col for col in df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "date",
            "day",
            "time",
            "timestamp"
        ]
    )
]


if len(date_columns) > 0:

    date_col = date_columns[0]

else:

    date_col = df.columns[0]


# ============================================================
# FIND REVENUE COLUMN
# ============================================================

revenue_columns = [
    col for col in df.columns
    if any(
        keyword in col.lower()
        for keyword in [
            "revenue",
            "sales",
            "income",
            "amount",
            "total"
        ]
    )
]


if len(revenue_columns) > 0:

    revenue_col = revenue_columns[0]

else:

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) > 0:

        revenue_col = numeric_columns[0]

    else:

        st.error(
            "No numeric revenue column found."
        )

        st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df[[date_col, revenue_col]].copy()

df.columns = [
    "date",
    "revenue"
]

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["revenue"] = pd.to_numeric(
    df["revenue"],
    errors="coerce"
)

df = df.dropna()

df = df.sort_values(
    "date"
)

df = df.drop_duplicates(
    subset="date"
)


# ============================================================
# DAILY RESAMPLING
# ============================================================

df = (
    df.set_index("date")
    .resample("D")
    .sum()
    .fillna(0)
)


# Remove zero-revenue leading rows if present
df = df[df["revenue"] > 0]


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("📊 Dataset Preview")

st.dataframe(
    df.tail(10),
    use_container_width=True
)


# ============================================================
# BASIC STATISTICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

total_revenue = df["revenue"].sum()

average_revenue = df["revenue"].mean()

max_revenue = df["revenue"].max()

min_revenue = df["revenue"].min()


col1.metric(
    "Total Revenue",
    f"₹{total_revenue:,.0f}"
)

col2.metric(
    "Average Daily Revenue",
    f"₹{average_revenue:,.0f}"
)

col3.metric(
    "Highest Revenue",
    f"₹{max_revenue:,.0f}"
)

col4.metric(
    "Lowest Revenue",
    f"₹{min_revenue:,.0f}"
)


# ============================================================
# HISTORICAL REVENUE
# ============================================================

st.header("2️⃣ Historical Revenue Trend")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    df.index,
    df["revenue"],
    label="Historical Revenue"
)

ax.set_title(
    "Historical Daily Revenue"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Revenue"
)

ax.grid(
    alpha=0.3
)

ax.legend()

st.pyplot(fig)


# ============================================================
# MOVING AVERAGE
# ============================================================

st.header("3️⃣ Revenue Trend Analysis")

df["7_day_avg"] = (
    df["revenue"]
    .rolling(7)
    .mean()
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    df.index,
    df["revenue"],
    alpha=0.5,
    label="Daily Revenue"
)

ax.plot(
    df.index,
    df["7_day_avg"],
    label="7-Day Moving Average"
)

ax.set_title(
    "Revenue with 7-Day Moving Average"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Revenue"
)

ax.grid(
    alpha=0.3
)

ax.legend()

st.pyplot(fig)


# ============================================================
# ARIMA MODEL
# ============================================================

st.header("4️⃣ ARIMA Revenue Forecast")

revenue_series = df["revenue"].copy()

# Need enough data for ARIMA
if len(revenue_series) < 30:

    st.error(
        "At least 30 days of revenue data are recommended "
        "for forecasting."
    )

    st.stop()


with st.spinner("Training ARIMA model..."):

    try:

        model = ARIMA(
            revenue_series,
            order=(
                arima_p,
                arima_d,
                arima_q
            )
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(
            steps=forecast_days
        )

    except Exception as e:

        st.error(
            f"ARIMA model error: {e}"
        )

        st.stop()


# ============================================================
# FORECAST DATAFRAME
# ============================================================

last_date = revenue_series.index[-1]

forecast_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    periods=forecast_days,
    freq="D"
)

forecast_df = pd.DataFrame({

    "date": forecast_dates,

    "predicted_revenue": forecast.values

})

forecast_df["predicted_revenue"] = (
    forecast_df["predicted_revenue"]
    .clip(lower=0)
)


# ============================================================
# FORECAST VISUALIZATION
# ============================================================

st.subheader(
    f"📈 Next {forecast_days} Days Revenue Forecast"
)

fig, ax = plt.subplots(
    figsize=(14, 6)
)

# Last 90 historical days
history_to_plot = df[
    "revenue"
].tail(90)

ax.plot(
    history_to_plot.index,
    history_to_plot.values,
    label="Historical Revenue"
)

ax.plot(
    forecast_df["date"],
    forecast_df["predicted_revenue"],
    linestyle="--",
    label="Forecast Revenue"
)

ax.axvline(
    last_date,
    linestyle=":"
)

ax.set_title(
    "Historical Revenue vs Future Forecast"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Revenue"
)

ax.grid(
    alpha=0.3
)

ax.legend()

st.pyplot(fig)


# ============================================================
# FORECAST TABLE
# ============================================================

st.subheader("📋 Forecast Output")

display_forecast = forecast_df.copy()

display_forecast[
    "predicted_revenue"
] = display_forecast[
    "predicted_revenue"
].round(2)

st.dataframe(
    display_forecast,
    use_container_width=True
)


# ============================================================
# FORECAST SUMMARY
# ============================================================

forecast_total = (
    forecast_df[
        "predicted_revenue"
    ].sum()
)

forecast_average = (
    forecast_df[
        "predicted_revenue"
    ].mean()
)

recent_average = (
    revenue_series
    .tail(30)
    .mean()
)


growth_percentage = (
    (
        forecast_average
        - recent_average
    )
    / recent_average
) * 100


col1, col2, col3 = st.columns(3)


col1.metric(
    "Forecast Total Revenue",
    f"₹{forecast_total:,.0f}"
)

col2.metric(
    "Forecast Daily Average",
    f"₹{forecast_average:,.0f}"
)

col3.metric(
    "Expected Change vs Recent Avg",
    f"{growth_percentage:.2f}%"
)


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.header("5️⃣ Business Interpretation")

if growth_percentage > 5:

    st.success(
        f"""
        **Positive revenue trend detected.**

        The forecasted average daily revenue is approximately
        {growth_percentage:.2f}% higher than the recent 30-day
        average.

        Businesses may use this forecast for:
        - Resource planning
        - Inventory planning
        - Marketing allocation
        - Financial planning
        """
    )

elif growth_percentage < -5:

    st.warning(
        f"""
        **Potential revenue decline detected.**

        The forecasted average daily revenue is approximately
        {abs(growth_percentage):.2f}% lower than the recent
        30-day average.

        Businesses should review:
        - Customer demand
        - Sales performance
        - Marketing campaigns
        - Pricing strategy
        """
    )

else:

    st.info(
        f"""
        **Revenue appears relatively stable.**

        The forecast differs from the recent 30-day average
        by approximately {growth_percentage:.2f}%.
        """
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("6️⃣ Model Information")

st.write(
    f"""
    **Model:** ARIMA

    **Parameters:** ({arima_p}, {arima_d}, {arima_q})

    **Forecast Horizon:** {forecast_days} days

    **Historical Data Points:** {len(revenue_series)}

    ARIMA uses historical time-series patterns to estimate
    future values. Forecast accuracy depends on the quality,
    length and stability of the historical data.
    """
)


# ============================================================
# DOWNLOAD FORECAST
# ============================================================

st.header("7️⃣ Export Forecast")

csv_data = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Revenue Forecast CSV",
    data=csv_data,
    file_name="day37_revenue_forecast.csv",
    mime="text/csv"
)


# ============================================================
# REFLECTION
# ============================================================

st.header("8️⃣ Day 37 Reflection")

st.markdown(
    """
    ### What I learned

    - Learned how time-series data can be used for business forecasting.
    - Learned how to prepare historical revenue data for forecasting.
    - Learned how ARIMA models can predict future revenue.
    - Learned how to visualize historical and forecasted revenue.
    - Learned how forecasting can support business planning and decision-making.
    - Learned that forecasting results depend heavily on historical data quality and patterns.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 37/60 | 60 Days of Data Science | Revenue Forecasting with ARIMA"
)