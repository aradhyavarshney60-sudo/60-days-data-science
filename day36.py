# ============================================================
# DAY 36/60 - FORECASTING CUSTOMER GROWTH TRENDS
# 60 Days of Data Science
# ============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Day 36 - Customer Growth Forecast",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Day 36/60 | Forecasting Customer Growth Trends")

st.markdown("""
### Time Series Forecasting

This dashboard analyzes historical customer activity, identifies
growth trends, and generates a **30-day customer growth forecast**.
""")


# ============================================================
# 1. LOAD HISTORICAL DATA
# ============================================================

st.header("1️⃣ Loading Historical Customer Data")


def find_dataset():
    """
    Try to find a suitable customer/revenue CSV from the project.
    """

    preferred_files = [
        "day34_customer_dashboard_data.csv",
        "day33_customer_anomalies.csv",
        "day32_recommendations.csv",
        "customer_data.csv",
        "customers.csv",
        "revenue.csv"
    ]

    for file in preferred_files:
        if os.path.exists(file):
            return file

    # Search all CSV files
    csv_files = glob.glob("*.csv")

    if csv_files:
        return csv_files[0]

    return None


dataset_path = find_dataset()


# ============================================================
# CREATE FALLBACK DATA IF NO DATASET EXISTS
# ============================================================

if dataset_path is None:

    st.warning(
        "No suitable CSV dataset found. "
        "A sample historical customer dataset will be generated."
    )

    np.random.seed(42)

    dates = pd.date_range(
        start="2025-01-01",
        periods=180,
        freq="D"
    )

    base_customers = 1000
    trend = np.arange(len(dates)) * 2.5
    seasonality = 30 * np.sin(
        np.arange(len(dates)) * 2 * np.pi / 30
    )
    noise = np.random.normal(0, 15, len(dates))

    customers = (
        base_customers
        + trend
        + seasonality
        + noise
    )

    customers = np.maximum(customers, 0)

    df = pd.DataFrame({
        "date": dates,
        "customers": customers.astype(int)
    })

else:

    st.success(f"Dataset loaded: `{dataset_path}`")

    df = pd.read_csv(dataset_path)


# ============================================================
# 2. DISPLAY RAW DATA
# ============================================================

st.subheader("Historical Data Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# 3. IDENTIFY DATE COLUMN
# ============================================================

date_column = None

possible_date_columns = [
    "date",
    "Date",
    "datetime",
    "Datetime",
    "timestamp",
    "Timestamp",
    "created_at",
    "order_date",
    "purchase_date"
]

for column in possible_date_columns:

    if column in df.columns:
        date_column = column
        break


# If no date column is available
if date_column is None:

    st.info(
        "No date column detected. "
        "A sequential daily date index will be created."
    )

    df["date"] = pd.date_range(
        start="2025-01-01",
        periods=len(df),
        freq="D"
    )

    date_column = "date"


# Convert date column
df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df = df.dropna(
    subset=[date_column]
)


# ============================================================
# 4. IDENTIFY CUSTOMER / REVENUE COLUMN
# ============================================================

possible_value_columns = [
    "customers",
    "customer_count",
    "customer_count_total",
    "total_customers",
    "revenue",
    "Revenue",
    "sales",
    "Sales",
    "amount",
    "Amount"
]

value_column = None

for column in possible_value_columns:

    if column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            value_column = column
            break


# Automatically find numeric column if required
if value_column is None:

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) > 0:

        value_column = numeric_columns[0]


# ============================================================
# IF NO NUMERIC DATA EXISTS
# ============================================================

if value_column is None:

    st.warning(
        "No suitable numeric customer/revenue column found. "
        "Generating customer count from row data."
    )

    daily_data = (
        df.groupby(date_column)
        .size()
        .reset_index(name="customers")
    )

    value_column = "customers"

else:

    daily_data = (
        df.groupby(date_column)[value_column]
        .sum()
        .reset_index()
    )


# ============================================================
# 5. CLEAN AND SORT DATA
# ============================================================

daily_data = daily_data.sort_values(
    date_column
).reset_index(drop=True)


daily_data.columns = [
    "date",
    "customers"
]


daily_data["customers"] = pd.to_numeric(
    daily_data["customers"],
    errors="coerce"
)

daily_data = daily_data.dropna(
    subset=["customers"]
)


# ============================================================
# 6. CREATE CONTINUOUS DAILY TIME SERIES
# ============================================================

daily_data = daily_data.set_index("date")

daily_data = daily_data.asfreq("D")

daily_data["customers"] = (
    daily_data["customers"]
    .interpolate()
    .ffill()
    .bfill()
)

daily_data = daily_data.reset_index()


# ============================================================
# 7. HISTORICAL TREND VISUALIZATION
# ============================================================

st.header("2️⃣ Historical Customer Growth Trend")


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    daily_data["date"],
    daily_data["customers"],
    label="Historical Customers"
)

ax.set_title(
    "Historical Customer Growth Trend"
)

ax.set_xlabel("Date")
ax.set_ylabel("Customer Count")

ax.grid(True, alpha=0.3)

ax.legend()

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig)


# Save visualization
historical_chart = "day36_historical_customer_trend.png"

fig.savefig(
    historical_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 8. MOVING AVERAGE
# ============================================================

st.header("3️⃣ Trend Analysis")


daily_data["7_day_moving_average"] = (
    daily_data["customers"]
    .rolling(window=7)
    .mean()
)


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    daily_data["date"],
    daily_data["customers"],
    alpha=0.5,
    label="Daily Customers"
)

ax.plot(
    daily_data["date"],
    daily_data["7_day_moving_average"],
    linewidth=2,
    label="7-Day Moving Average"
)

ax.set_title(
    "Customer Growth with 7-Day Moving Average"
)

ax.set_xlabel("Date")
ax.set_ylabel("Customers")

ax.grid(True, alpha=0.3)

ax.legend()

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 9. GROWTH ANALYSIS
# ============================================================

first_value = daily_data["customers"].iloc[0]
last_value = daily_data["customers"].iloc[-1]

absolute_growth = last_value - first_value

if first_value != 0:

    percentage_growth = (
        absolute_growth / first_value
    ) * 100

else:

    percentage_growth = 0


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Starting Customers",
        f"{first_value:,.0f}"
    )

with col2:
    st.metric(
        "Latest Customers",
        f"{last_value:,.0f}"
    )

with col3:
    st.metric(
        "Overall Growth",
        f"{percentage_growth:.2f}%"
    )


# ============================================================
# 10. BASELINE FORECASTING MODEL
# ============================================================

st.header("4️⃣ 30-Day Customer Growth Forecast")


# Use recent observations for trend estimation
forecast_days = 30

recent_window = min(
    60,
    len(daily_data)
)

recent_data = daily_data.tail(
    recent_window
).copy()


# Convert dates into numeric values
recent_data["day_number"] = np.arange(
    len(recent_data)
)


# Linear trend model
slope, intercept = np.polyfit(
    recent_data["day_number"],
    recent_data["customers"],
    1
)


# ============================================================
# 11. GENERATE FUTURE DATES
# ============================================================

last_date = daily_data["date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)


future_day_numbers = np.arange(
    len(recent_data),
    len(recent_data) + forecast_days
)


forecast_values = (
    intercept
    + slope * future_day_numbers
)


forecast_values = np.maximum(
    forecast_values,
    0
)


# ============================================================
# 12. ADD SIMPLE SEASONALITY
# ============================================================

# Small weekly adjustment based on recent weekday behavior

recent_data["weekday"] = (
    recent_data["date"].dt.dayofweek
)

weekday_average = (
    recent_data
    .groupby("weekday")["customers"]
    .mean()
)

overall_average = (
    recent_data["customers"].mean()
)


seasonality_factor = []

for date in future_dates:

    weekday = date.dayofweek

    if weekday in weekday_average.index:

        factor = (
            weekday_average[weekday]
            / overall_average
        )

    else:

        factor = 1.0

    seasonality_factor.append(
        factor
    )


seasonality_factor = np.array(
    seasonality_factor
)


forecast_values = (
    forecast_values
    * seasonality_factor
)


forecast_values = np.maximum(
    forecast_values,
    0
)


# ============================================================
# 13. FORECAST DATAFRAME
# ============================================================

forecast_df = pd.DataFrame({

    "date": future_dates,

    "forecast_customers":
        np.round(forecast_values).astype(int)

})


# ============================================================
# 14. FORECAST VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)


ax.plot(
    daily_data["date"],
    daily_data["customers"],
    label="Historical"
)


ax.plot(
    forecast_df["date"],
    forecast_df["forecast_customers"],
    linestyle="--",
    linewidth=2,
    label="30-Day Forecast"
)


ax.axvline(
    last_date,
    linestyle=":"
)


ax.set_title(
    "Customer Growth: Historical Data + 30-Day Forecast"
)

ax.set_xlabel("Date")

ax.set_ylabel(
    "Customer Count"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()


st.pyplot(fig)


# Save forecast visualization
forecast_chart = (
    "day36_customer_growth_forecast.png"
)

fig.savefig(
    forecast_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. DISPLAY FORECAST
# ============================================================

st.subheader(
    "Next 30 Days Customer Forecast"
)

st.dataframe(
    forecast_df,
    use_container_width=True
)


# ============================================================
# 16. FORECAST SUMMARY
# ============================================================

forecast_start = (
    forecast_df["forecast_customers"].iloc[0]
)

forecast_end = (
    forecast_df["forecast_customers"].iloc[-1]
)


forecast_growth = (
    forecast_end - forecast_start
)


if forecast_start != 0:

    forecast_growth_percentage = (
        forecast_growth
        / forecast_start
    ) * 100

else:

    forecast_growth_percentage = 0


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Forecast Day 1",
        f"{forecast_start:,.0f}"
    )


with col2:

    st.metric(
        "Forecast Day 30",
        f"{forecast_end:,.0f}"
    )


with col3:

    st.metric(
        "Expected Growth",
        f"{forecast_growth_percentage:.2f}%"
    )


# ============================================================
# 17. BUSINESS INSIGHTS
# ============================================================

st.header("5️⃣ Business Insights")


if forecast_growth_percentage > 0:

    trend_message = (
        "The forecast indicates a positive customer growth trend."
    )

elif forecast_growth_percentage < 0:

    trend_message = (
        "The forecast indicates a potential decline in customer growth."
    )

else:

    trend_message = (
        "The forecast indicates relatively stable customer growth."
    )


st.write(
    f"📊 **Trend Observation:** {trend_message}"
)


st.write(
    """
📌 **Business Applications:**

- Customer growth forecasting can support resource planning.
- Marketing teams can use projected growth for campaign planning.
- Businesses can estimate future customer demand.
- Forecasts can help with staffing and operational planning.
- Historical trends can support data-driven business decisions.
"""
)


# ============================================================
# 18. FORECAST RISKS
# ============================================================

st.header("6️⃣ Forecasting Risks & Limitations")


st.write(
    """
⚠️ **Important limitations:**

- Forecasts are based on historical patterns.
- Unexpected business events may change future customer behavior.
- Market conditions can affect actual customer growth.
- The baseline model does not capture every external factor.
- Forecast values should be treated as estimates, not guaranteed outcomes.
"""
)


# ============================================================
# 19. SAVE FORECAST CSV
# ============================================================

forecast_file = (
    "day36_customer_growth_forecast.csv"
)

forecast_df.to_csv(
    forecast_file,
    index=False
)


# ============================================================
# 20. SAVE ANALYSIS DATA
# ============================================================

analysis_file = (
    "day36_customer_growth_analysis.csv"
)

daily_data.to_csv(
    analysis_file,
    index=False
)


# ============================================================
# 21. CREATE REFLECTION FILE
# ============================================================

reflection_text = f"""
DAY 36/60 - FORECASTING CUSTOMER GROWTH TRENDS
================================================

Objective:
Analyze historical customer growth and forecast future customer
growth for the next 30 days.

What I worked on:
- Loaded historical customer data
- Processed date-based information
- Created a continuous daily time series
- Visualized customer growth trends
- Calculated a 7-day moving average
- Analyzed overall customer growth
- Built a baseline linear trend forecasting model
- Added simple weekday seasonality
- Generated a 30-day customer growth forecast
- Visualized historical and forecasted values
- Documented business applications and forecasting risks

Historical Analysis:
Starting customer value: {first_value:.2f}
Latest customer value: {last_value:.2f}
Overall growth: {percentage_growth:.2f}%

Forecast Analysis:
Forecast Day 1: {forecast_start:.2f}
Forecast Day 30: {forecast_end:.2f}
Expected forecast growth: {forecast_growth_percentage:.2f}%

Key Learning:
I learned how time series data can be used to identify historical
growth patterns and generate future business forecasts.

I also learned that forecasting is an estimation process and
future results can differ because of unexpected market conditions,
customer behavior, and external business factors.

Business Application:
Customer growth forecasting can support resource planning,
marketing decisions, staffing, demand planning, and strategic
business decisions.

Day 36/60 completed.
"""


reflection_file = (
    "day36_reflection.txt"
)

with open(
    reflection_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        reflection_text
    )


# ============================================================
# 22. DOWNLOAD BUTTONS
# ============================================================

st.header("7️⃣ Export Files")


with open(
    forecast_file,
    "rb"
) as file:

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=file,
        file_name=forecast_file,
        mime="text/csv"
    )


with open(
    reflection_file,
    "rb"
) as file:

    st.download_button(
        label="⬇️ Download Reflection",
        data=file,
        file_name=reflection_file,
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 36/60 | 60 Days of Data Science | "
    "Forecasting Customer Growth Trends"
)

st.caption(
    "Time Series Analysis • Forecasting • Python • Pandas • "
    "Data Visualization • Business Analytics"
)