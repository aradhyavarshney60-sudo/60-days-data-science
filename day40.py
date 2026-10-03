# ============================================================
# DAY 40 | 60 DAYS OF DATA SCIENCE
# Evaluating Business Decisions with A/B Testing
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Day 40 | A/B Testing",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧪 Day 40 | Evaluating Business Decisions with A/B Testing")

st.markdown("""
### A/B Testing Analytics Dashboard

This project simulates an A/B testing experiment and evaluates whether
a new business strategy performs differently from the existing control group.
""")

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Experiment Settings")

sample_size = st.sidebar.slider(
    "Users per Group",
    min_value=500,
    max_value=10000,
    value=5000,
    step=500
)

control_conversion = st.sidebar.slider(
    "Control Conversion Rate (%)",
    min_value=1.0,
    max_value=30.0,
    value=10.0,
    step=0.5
)

experiment_conversion = st.sidebar.slider(
    "Experiment Conversion Rate (%)",
    min_value=1.0,
    max_value=30.0,
    value=12.0,
    step=0.5
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    max_value=9999,
    value=42
)

alpha = st.sidebar.slider(
    "Significance Level (α)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01
)


# ============================================================
# DATA GENERATION
# ============================================================

np.random.seed(random_seed)

control_converted = np.random.binomial(
    1,
    control_conversion / 100,
    sample_size
)

experiment_converted = np.random.binomial(
    1,
    experiment_conversion / 100,
    sample_size
)

control_revenue = np.where(
    control_converted == 1,
    np.random.normal(500, 100, sample_size),
    0
)

experiment_revenue = np.where(
    experiment_converted == 1,
    np.random.normal(520, 100, sample_size),
    0
)

control_revenue = np.maximum(control_revenue, 0)
experiment_revenue = np.maximum(experiment_revenue, 0)

control_df = pd.DataFrame({
    "Group": "Control",
    "Converted": control_converted,
    "Revenue": control_revenue
})

experiment_df = pd.DataFrame({
    "Group": "Experiment",
    "Converted": experiment_converted,
    "Revenue": experiment_revenue
})

df = pd.concat(
    [control_df, experiment_df],
    ignore_index=True
)


# ============================================================
# BASIC METRICS
# ============================================================

control_users = len(control_df)
experiment_users = len(experiment_df)

control_conversions = control_df["Converted"].sum()
experiment_conversions = experiment_df["Converted"].sum()

control_rate = control_conversions / control_users
experiment_rate = experiment_conversions / experiment_users

absolute_difference = experiment_rate - control_rate

if control_rate != 0:
    relative_lift = (
        (experiment_rate - control_rate)
        / control_rate
    ) * 100
else:
    relative_lift = 0

control_total_revenue = control_df["Revenue"].sum()
experiment_total_revenue = experiment_df["Revenue"].sum()

control_avg_revenue = control_df["Revenue"].mean()
experiment_avg_revenue = experiment_df["Revenue"].mean()


# ============================================================
# HEADER METRICS
# ============================================================

st.subheader("📊 Experiment Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Control Conversion",
        f"{control_rate * 100:.2f}%"
    )

with col2:
    st.metric(
        "Experiment Conversion",
        f"{experiment_rate * 100:.2f}%",
        f"{absolute_difference * 100:+.2f} pp"
    )

with col3:
    st.metric(
        "Relative Lift",
        f"{relative_lift:+.2f}%"
    )

with col4:
    st.metric(
        "Total Users",
        f"{control_users + experiment_users:,}"
    )


st.divider()


# ============================================================
# GROUP SUMMARY
# ============================================================

st.subheader("📋 Group Performance")

summary_df = pd.DataFrame({
    "Metric": [
        "Users",
        "Conversions",
        "Conversion Rate",
        "Total Revenue",
        "Average Revenue per User"
    ],
    "Control": [
        control_users,
        int(control_conversions),
        f"{control_rate * 100:.2f}%",
        f"₹{control_total_revenue:,.2f}",
        f"₹{control_avg_revenue:,.2f}"
    ],
    "Experiment": [
        experiment_users,
        int(experiment_conversions),
        f"{experiment_rate * 100:.2f}%",
        f"₹{experiment_total_revenue:,.2f}",
        f"₹{experiment_avg_revenue:,.2f}"
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONVERSION RATE VISUALIZATION
# ============================================================

st.subheader("📈 Conversion Rate Comparison")

conversion_chart = pd.DataFrame({
    "Group": ["Control", "Experiment"],
    "Conversion Rate": [
        control_rate * 100,
        experiment_rate * 100
    ]
})

fig_conversion = px.bar(
    conversion_chart,
    x="Group",
    y="Conversion Rate",
    text="Conversion Rate",
    title="Control vs Experiment Conversion Rate"
)

fig_conversion.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_conversion.update_layout(
    yaxis_title="Conversion Rate (%)",
    xaxis_title="Group"
)

st.plotly_chart(
    fig_conversion,
    use_container_width=True
)


# ============================================================
# CONVERSION COUNTS
# ============================================================

st.subheader("👥 Conversion Distribution")

conversion_counts = pd.DataFrame({
    "Group": [
        "Control",
        "Control",
        "Experiment",
        "Experiment"
    ],
    "Status": [
        "Converted",
        "Not Converted",
        "Converted",
        "Not Converted"
    ],
    "Users": [
        control_conversions,
        control_users - control_conversions,
        experiment_conversions,
        experiment_users - experiment_conversions
    ]
})

fig_counts = px.bar(
    conversion_counts,
    x="Group",
    y="Users",
    color="Status",
    barmode="group",
    title="Converted vs Non-Converted Users"
)

st.plotly_chart(
    fig_counts,
    use_container_width=True
)


# ============================================================
# REVENUE ANALYSIS
# ============================================================

st.subheader("💰 Revenue Analysis")

revenue_chart = pd.DataFrame({
    "Group": ["Control", "Experiment"],
    "Total Revenue": [
        control_total_revenue,
        experiment_total_revenue
    ]
})

fig_revenue = px.bar(
    revenue_chart,
    x="Group",
    y="Total Revenue",
    text="Total Revenue",
    title="Total Revenue by Group"
)

fig_revenue.update_traces(
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig_revenue,
    use_container_width=True
)


# ============================================================
# REVENUE DISTRIBUTION
# ============================================================

fig_distribution = px.box(
    df[df["Revenue"] > 0],
    x="Group",
    y="Revenue",
    color="Group",
    title="Revenue Distribution Among Converted Users"
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================

st.subheader("🧮 Statistical Significance")

st.markdown("""
We use a **two-proportion z-test** to determine whether the difference
between the Control and Experiment conversion rates is statistically significant.
""")

# Calculate pooled conversion rate
total_conversions = control_conversions + experiment_conversions
total_users = control_users + experiment_users

pooled_rate = total_conversions / total_users

standard_error = np.sqrt(
    pooled_rate
    * (1 - pooled_rate)
    * (
        (1 / control_users)
        + (1 / experiment_users)
    )
)

if standard_error != 0:

    z_score = (
        experiment_rate - control_rate
    ) / standard_error

    p_value = 2 * (
        1 - stats.norm.cdf(abs(z_score))
    )

else:
    z_score = 0
    p_value = 1


stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    st.metric(
        "Z-Score",
        f"{z_score:.4f}"
    )

with stat_col2:
    st.metric(
        "P-Value",
        f"{p_value:.6f}"
    )

with stat_col3:
    st.metric(
        "Significance Level",
        f"{alpha:.2f}"
    )


# ============================================================
# SIGNIFICANCE INTERPRETATION
# ============================================================

if p_value < alpha:

    st.success(
        f"""
        ✅ Statistically Significant

        The p-value ({p_value:.6f}) is below the significance level
        ({alpha:.2f}). The observed difference between the two groups
        is statistically significant under this test.
        """
    )

else:

    st.warning(
        f"""
        ⚠️ Not Statistically Significant

        The p-value ({p_value:.6f}) is greater than the significance level
        ({alpha:.2f}). The observed difference is not statistically
        significant under this test.
        """
    )


# ============================================================
# HYPOTHESIS TEST
# ============================================================

st.subheader("🔬 Hypothesis Testing")

st.markdown("""
**Null Hypothesis (H₀):**  
There is no difference in conversion rates between the Control
and Experiment groups.

**Alternative Hypothesis (H₁):**  
There is a difference in conversion rates between the Control
and Experiment groups.
""")

if p_value < alpha:
    st.info(
        "Result: Reject H₀ based on the selected significance level."
    )
else:
    st.info(
        "Result: Fail to reject H₀ based on the selected significance level."
    )


# ============================================================
# LIFT ANALYSIS
# ============================================================

st.subheader("🚀 Experiment Lift")

lift_fig = go.Figure()

lift_fig.add_trace(
    go.Indicator(
        mode="number+delta",
        value=experiment_rate * 100,
        delta={
            "reference": control_rate * 100,
            "relative": False,
            "valueformat": ".2f"
        },
        title={
            "text": "Experiment Conversion Rate vs Control"
        },
        number={
            "suffix": "%"
        }
    )
)

lift_fig.update_layout(
    height=300
)

st.plotly_chart(
    lift_fig,
    use_container_width=True
)


# ============================================================
# BUSINESS RECOMMENDATION
# ============================================================

st.subheader("💡 Business Interpretation")

if p_value < alpha and experiment_rate > control_rate:

    st.success("""
    The experiment group shows a higher conversion rate and the
    difference is statistically significant at the selected significance
    level.

    From an analytical perspective, the experiment provides evidence
    of improved conversion performance. Business teams can consider
    the result alongside cost, implementation risk, and other relevant
    business factors before making a rollout decision.
    """)

elif p_value < alpha and experiment_rate < control_rate:

    st.error("""
    The experiment group shows a lower conversion rate and the
    difference is statistically significant.

    The experiment provides evidence that the new variation performs
    differently from the control, with a lower observed conversion rate.
    Further investigation can help identify the reason for the result.
    """)

else:

    st.warning("""
    The experiment shows a difference in observed conversion rates,
    but the difference is not statistically significant at the selected
    significance level.

    More data or additional experimentation may be useful before
    drawing a strong conclusion.
    """)


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

st.subheader("📝 Experiment Summary")

summary_text = f"""
### A/B Testing Results

**Control Group**
- Users: {control_users:,}
- Conversions: {int(control_conversions):,}
- Conversion Rate: {control_rate * 100:.2f}%

**Experiment Group**
- Users: {experiment_users:,}
- Conversions: {int(experiment_conversions):,}
- Conversion Rate: {experiment_rate * 100:.2f}%

**Experiment Difference**
- Absolute Difference: {absolute_difference * 100:.2f} percentage points
- Relative Lift: {relative_lift:.2f}%
- Z-Score: {z_score:.4f}
- P-Value: {p_value:.6f}
- Significance Level: {alpha:.2f}
"""

st.markdown(summary_text)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.subheader("📥 Download Experiment Data")

csv_data = df.to_csv(index=False)

st.download_button(
    label="Download A/B Testing Dataset",
    data=csv_data,
    file_name="day40_ab_testing_data.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 40/60 | 60 Days of Data Science | "
    "A/B Testing & Business Experimentation"
)