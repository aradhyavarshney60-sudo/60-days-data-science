# ============================================================
# DAY 35/60
# COMBINING CUSTOMER INTELLIGENCE MODULES
# INTO ONE ANALYTICS SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import os
import glob


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Intelligence Analytics System")

st.markdown(
    """
    ### Day 35/60 | Integrated Customer Analytics

    This application combines the customer intelligence modules
    developed during Week 5 into one unified analytics workflow.

    **Segmentation → Recommendations → Anomaly Detection → Dashboard**
    """
)

st.divider()


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_file(patterns):
    """
    Find the first matching file from the provided patterns.
    """

    for pattern in patterns:

        files = glob.glob(pattern)

        if files:
            return files[0]

    return None


def load_csv(patterns):

    file_path = find_file(patterns)

    if file_path is None:
        return None, None

    try:

        data = pd.read_csv(file_path)

        return data, file_path

    except Exception as error:

        st.warning(
            f"Could not read {file_path}: {error}"
        )

        return None, file_path


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📂 Data Modules")

st.sidebar.write(
    """
    This system integrates the outputs created
    during Days 31-34.
    """
)


# ============================================================
# LOAD DAY 31 DATA
# ============================================================

day31_data, day31_file = load_csv(
    [
        "day31_customer_persona*.csv",
        "day31_customer_persona*.CSV",
        "day31_customer_segment*.csv",
        "day31_customer_segment*.CSV"
    ]
)


# ============================================================
# LOAD DAY 32 DATA
# ============================================================

day32_data, day32_file = load_csv(
    [
        "day32_recommendati*.csv",
        "day32_recommendati*.CSV"
    ]
)


# ============================================================
# LOAD DAY 33 DATA
# ============================================================

day33_data, day33_file = load_csv(
    [
        "day33_customer_anomal*.csv",
        "day33_customer_anomal*.CSV",
        "day33_anomal*.csv",
        "day33_anomal*.CSV"
    ]
)


# ============================================================
# LOAD DAY 34 DATA
# ============================================================

day34_data, day34_file = load_csv(
    [
        "day34_customer_dashboard_data.csv",
        "day34_customer_dashboard_data.CSV",
        "day34_*.csv"
    ]
)


# ============================================================
# MODULE STATUS
# ============================================================

st.header("🔗 Integrated Module Status")

col1, col2, col3, col4 = st.columns(4)


with col1:

    if day31_data is not None:

        st.success("Day 31\nLoaded")

    else:

        st.error("Day 31\nNot Found")


with col2:

    if day32_data is not None:

        st.success("Day 32\nLoaded")

    else:

        st.error("Day 32\nNot Found")


with col3:

    if day33_data is not None:

        st.success("Day 33\nLoaded")

    else:

        st.error("Day 33\nNot Found")


with col4:

    if day34_data is not None:

        st.success("Day 34\nLoaded")

    else:

        st.error("Day 34\nNot Found")


st.divider()


# ============================================================
# OVERALL DATA SUMMARY
# ============================================================

st.header("📈 System Overview")


total_records = 0

if day31_data is not None:
    total_records = len(day31_data)

elif day34_data is not None:
    total_records = len(day34_data)

elif day33_data is not None:
    total_records = len(day33_data)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Customer Records",
        total_records
    )


with col2:

    st.metric(
        "Modules Integrated",
        sum(
            [
                day31_data is not None,
                day32_data is not None,
                day33_data is not None,
                day34_data is not None
            ]
        )
    )


with col3:

    if day31_data is not None:

        st.metric(
            "Day 31 Features",
            len(day31_data.columns)
        )

    else:

        st.metric(
            "Day 31 Features",
            "N/A"
        )


with col4:

    if day34_data is not None:

        st.metric(
            "Dashboard Features",
            len(day34_data.columns)
        )

    else:

        st.metric(
            "Dashboard Features",
            "N/A"
        )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "👥 Segmentation",
        "💡 Recommendations",
        "🚨 Anomalies",
        "📊 Dashboard",
        "🔬 Integrated Analysis"
    ]
)


# ============================================================
# TAB 1
# CUSTOMER SEGMENTATION
# ============================================================

with tab1:

    st.header("👥 Customer Segmentation")

    st.write(
        """
        This section integrates the customer segmentation
        and persona analysis created during Day 31.
        """
    )

    if day31_data is not None:

        st.success(
            f"Source: {day31_file}"
        )

        st.subheader("Customer Segment Data")

        st.dataframe(
            day31_data,
            use_container_width=True
        )

        st.subheader("Segment Distribution")

        # Try to identify segment/persona column

        segment_column = None

        for column in day31_data.columns:

            column_lower = column.lower()

            if (
                "segment" in column_lower
                or "persona" in column_lower
                or "cluster" in column_lower
            ):

                segment_column = column

                break

        if segment_column:

            segment_counts = (
                day31_data[segment_column]
                .value_counts()
            )

            st.bar_chart(
                segment_counts
            )

        else:

            st.info(
                "Segment/persona column was not automatically detected."
            )

    else:

        st.warning(
            "Day 31 customer segmentation file was not found."
        )


# ============================================================
# TAB 2
# RECOMMENDATIONS
# ============================================================

with tab2:

    st.header("💡 Customer Recommendations")

    st.write(
        """
        This section integrates the recommendation module
        developed during Day 32.
        """
    )

    if day32_data is not None:

        st.success(
            f"Source: {day32_file}"
        )

        st.dataframe(
            day32_data,
            use_container_width=True
        )

        st.subheader("Recommendation Summary")

        st.write(
            f"Total recommendation records: **{len(day32_data)}**"
        )

        # Try to identify recommendation column

        recommendation_column = None

        for column in day32_data.columns:

            column_lower = column.lower()

            if (
                "recommend" in column_lower
                or "product" in column_lower
                or "action" in column_lower
            ):

                recommendation_column = column

                break

        if recommendation_column:

            st.subheader(
                "Most Common Recommendations"
            )

            recommendation_counts = (
                day32_data[recommendation_column]
                .value_counts()
                .head(10)
            )

            st.bar_chart(
                recommendation_counts
            )

    else:

        st.warning(
            "Day 32 recommendation file was not found."
        )


# ============================================================
# TAB 3
# ANOMALY DETECTION
# ============================================================

with tab3:

    st.header("🚨 Customer Behavior Anomaly Detection")

    st.write(
        """
        This section integrates the anomaly detection
        module created during Day 33.
        """
    )

    if day33_data is not None:

        st.success(
            f"Source: {day33_file}"
        )

        st.dataframe(
            day33_data,
            use_container_width=True
        )

        # Try to identify anomaly column

        anomaly_column = None

        for column in day33_data.columns:

            column_lower = column.lower()

            if (
                "anomaly" in column_lower
                or "outlier" in column_lower
                or "status" in column_lower
            ):

                anomaly_column = column

                break

        if anomaly_column:

            st.subheader(
                "Anomaly Distribution"
            )

            anomaly_counts = (
                day33_data[anomaly_column]
                .value_counts()
            )

            st.bar_chart(
                anomaly_counts
            )

            st.write(
                """
                **Important:** An anomaly represents an unusual
                behavioral pattern. It does not automatically
                mean fraud or harmful activity.
                """
            )

    else:

        st.warning(
            "Day 33 anomaly detection file was not found."
        )


# ============================================================
# TAB 4
# DASHBOARD
# ============================================================

with tab4:

    st.header("📊 Customer Analytics Dashboard")

    st.write(
        """
        This section integrates the dashboard data produced
        during Day 34.
        """
    )

    if day34_data is not None:

        st.success(
            f"Source: {day34_file}"
        )

        st.subheader("Dashboard Dataset")

        st.dataframe(
            day34_data,
            use_container_width=True
        )

        st.subheader("Numerical Analytics")

        numeric_data = (
            day34_data
            .select_dtypes(include="number")
        )

        if not numeric_data.empty:

            st.dataframe(
                numeric_data.describe().round(2),
                use_container_width=True
            )

        else:

            st.info(
                "No numerical columns available."
            )

    else:

        st.warning(
            "Day 34 dashboard dataset was not found."
        )


# ============================================================
# TAB 5
# INTEGRATED ANALYSIS
# ============================================================

with tab5:

    st.header("🔬 Integrated Customer Analysis")

    st.write(
        """
        The purpose of Day 35 is to bring the separate
        analytics modules together into one decision-support
        workflow.
        """
    )

    st.subheader("System Architecture")

    st.markdown(
        """
        ```text
                         CUSTOMER DATA
                              │
                              ▼
                    DATA PREPROCESSING
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        SEGMENTATION    RECOMMENDATIONS   ANOMALY DETECTION
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                    CUSTOMER DASHBOARD
                              │
                              ▼
                     BUSINESS INSIGHTS
        ```
        """
    )

    st.subheader("Integrated Workflow")

    workflow = pd.DataFrame(
        {
            "Stage": [
                "1. Customer Data",
                "2. Segmentation",
                "3. Recommendations",
                "4. Anomaly Detection",
                "5. Dashboard",
                "6. Business Insights"
            ],
            "Purpose": [
                "Collect and prepare customer information",
                "Group customers into meaningful segments",
                "Generate customer-specific recommendations",
                "Identify unusual behavioral patterns",
                "Present analytics in one interface",
                "Support data-driven business decisions"
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Production-Oriented Thinking")

    st.markdown(
        """
        ### Key Integration Considerations

        **Data Consistency**
        - All modules should work with compatible customer records.
        - Feature names and customer identifiers should remain consistent.

        **Scalability**
        - Separate modules make the system easier to maintain.
        - New analytics modules can be added later.

        **Interpretability**
        - Business users should be able to understand segmentation,
          recommendations and anomaly results.

        **Model Dependency**
        - Each analytical module may have different assumptions
          and processing requirements.

        **Business Context**
        - Machine learning outputs should be interpreted together
          with business rules and customer context.
        """
    )


# ============================================================
# EXPORT
# ============================================================

st.divider()

st.header("💾 Export Integrated Data")


# Combine available datasets vertically where possible

datasets = []

if day31_data is not None:

    temp = day31_data.copy()
    temp["source_module"] = "Day 31 - Segmentation"
    datasets.append(temp)


if day32_data is not None:

    temp = day32_data.copy()
    temp["source_module"] = "Day 32 - Recommendations"
    datasets.append(temp)


if day33_data is not None:

    temp = day33_data.copy()
    temp["source_module"] = "Day 33 - Anomaly Detection"
    datasets.append(temp)


if day34_data is not None:

    temp = day34_data.copy()
    temp["source_module"] = "Day 34 - Dashboard"
    datasets.append(temp)


if datasets:

    integrated_data = pd.concat(
        datasets,
        ignore_index=True,
        sort=False
    )

    csv_data = integrated_data.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Integrated Dataset",
        data=csv_data,
        file_name="day35_integrated_customer_analytics.csv",
        mime="text/csv"
    )

else:

    st.warning(
        "No Day 31-34 datasets available for export."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Day 35/60 | 60 Days of Data Science | "
    "Customer Intelligence Integration"
)