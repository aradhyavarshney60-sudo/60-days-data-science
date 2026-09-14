# ============================================================
# DAY 21 - CHOOSING THE BEST MODEL LIKE A REAL DATA SCIENTIST
# 60 Days of Data Science
# ============================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_CSV = "day21_model_comparison.csv"
OUTPUT_PNG = "day21_model_comparison.png"
OUTPUT_REPORT = "day21_best_model_analysis.txt"
OUTPUT_REFLECTION = "day21_week3_reflection.txt"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_column_name(column):
    """
    Convert column names into a simple comparable format.
    """
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def find_metric_column(df, possible_names):
    """
    Find a metric column even if the CSV uses a slightly
    different column name.
    """
    cleaned_columns = {
        clean_column_name(col): col
        for col in df.columns
    }

    for name in possible_names:
        clean_name = clean_column_name(name)

        if clean_name in cleaned_columns:
            return cleaned_columns[clean_name]

    # Partial matching
    for cleaned_col, original_col in cleaned_columns.items():
        for name in possible_names:
            clean_name = clean_column_name(name)

            if clean_name in cleaned_col or cleaned_col in clean_name:
                return original_col

    return None


def find_model_column(df):
    """
    Find the model name column.
    """
    possible_names = [
        "Model",
        "model",
        "Model Name",
        "model_name",
        "Classifier",
        "Algorithm",
        "Name"
    ]

    column = find_metric_column(df, possible_names)

    if column:
        return column

    return None


def load_csv_if_exists(filename):
    """
    Load CSV if it exists.
    """
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)

            if not df.empty:
                print(f"Loaded: {filename}")
                return df

        except Exception as error:
            print(f"Could not read {filename}: {error}")

    return None


# ============================================================
# 3. LOAD PREVIOUS RESULTS
# ============================================================

print("=" * 70)
print("DAY 21 - MODEL COMPARISON & SELECTION")
print("=" * 70)

files_to_check = [
    "day20_metrics_comparison.csv",
    "day19_model_results.csv",
    "day18_model_results.csv",
    "day17_model_results.csv"
]

loaded_data = []

for filename in files_to_check:
    df = load_csv_if_exists(filename)

    if df is not None:
        loaded_data.append((filename, df))


# ============================================================
# 4. CHECK WHETHER DATA WAS FOUND
# ============================================================

if not loaded_data:
    print("\nERROR: No previous model result CSV files were found.")

    print("\nExpected files:")
    for filename in files_to_check:
        print(f"- {filename}")

    print("\nMake sure Day 17, Day 18, Day 19 and Day 20 files")
    print("are present in the same folder as day21.py.")

    raise SystemExit


# ============================================================
# 5. DISPLAY LOADED FILE INFORMATION
# ============================================================

print("\nFiles loaded successfully:")
print("-" * 50)

for filename, df in loaded_data:
    print(f"{filename}: {df.shape[0]} rows, {df.shape[1]} columns")

    print("Columns:")
    print(list(df.columns))
    print()


# ============================================================
# 6. METRIC NAMES
# ============================================================

metric_definitions = {
    "Accuracy": [
        "accuracy",
        "acc"
    ],

    "Precision": [
        "precision"
    ],

    "Recall": [
        "recall",
        "sensitivity"
    ],

    "F1 Score": [
        "f1",
        "f1score",
        "f1_score"
    ],

    "ROC-AUC": [
        "rocauc",
        "roc_auc",
        "rocaucscore",
        "auc"
    ]
}


# ============================================================
# 7. CREATE STANDARDIZED MODEL TABLE
# ============================================================

model_rows = []


for filename, df in loaded_data:

    model_column = find_model_column(df)

    # --------------------------------------------------------
    # Find metric columns
    # --------------------------------------------------------

    metric_columns = {}

    for standard_metric, possible_names in metric_definitions.items():

        found_column = find_metric_column(
            df,
            possible_names
        )

        if found_column:
            metric_columns[standard_metric] = found_column

    # --------------------------------------------------------
    # If no metrics found, skip file
    # --------------------------------------------------------

    if not metric_columns:
        print(
            f"Skipping {filename}: "
            "no recognizable evaluation metrics found."
        )
        continue

    # --------------------------------------------------------
    # Process every row
    # --------------------------------------------------------

    for index, row in df.iterrows():

        # Get model name
        if model_column:
            model_name = str(row[model_column])
        else:
            model_name = os.path.splitext(filename)[0]

        # Clean model name
        model_name = model_name.strip()

        if model_name == "" or model_name.lower() == "nan":
            model_name = os.path.splitext(filename)[0]

        result = {
            "Model": model_name,
            "Source": filename
        }

        # ----------------------------------------------------
        # Add metrics
        # ----------------------------------------------------

        for metric in metric_definitions.keys():

            if metric in metric_columns:

                value = pd.to_numeric(
                    row[metric_columns[metric]],
                    errors="coerce"
                )

                result[metric] = value

            else:
                result[metric] = float("nan")

        model_rows.append(result)


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

if not model_rows:
    print("\nERROR: No usable model results were found.")
    raise SystemExit


comparison_df = pd.DataFrame(model_rows)


# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

comparison_df = comparison_df.drop_duplicates(
    subset=["Model"],
    keep="last"
)


# ============================================================
# 10. CONVERT METRICS TO NUMERIC
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for metric in metric_names:
    comparison_df[metric] = pd.to_numeric(
        comparison_df[metric],
        errors="coerce"
    )


# ============================================================
# 11. HANDLE PERCENTAGE VALUES
# ============================================================

for metric in metric_names:

    if comparison_df[metric].notna().any():

        max_value = comparison_df[metric].max()

        # If values are percentages such as 95.5
        # convert them to decimal form.
        if max_value > 1:
            comparison_df[metric] = (
                comparison_df[metric] / 100
            )


# ============================================================
# 12. CALCULATE MODEL SCORE
# ============================================================

available_metrics = [
    metric
    for metric in metric_names
    if comparison_df[metric].notna().any()
]

print("\nMetrics available for comparison:")
print("-" * 50)

for metric in available_metrics:
    print(f"✓ {metric}")


if not available_metrics:
    print("\nERROR: No valid evaluation metrics found.")
    raise SystemExit


# Equal-weight average of available metrics
comparison_df["Overall Score"] = comparison_df[
    available_metrics
].mean(axis=1)


# ============================================================
# 13. SORT MODELS
# ============================================================

comparison_df = comparison_df.sort_values(
    by="Overall Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 14. RANK MODELS
# ============================================================

comparison_df.insert(
    0,
    "Rank",
    range(1, len(comparison_df) + 1)
)


# ============================================================
# 15. SAVE COMPARISON CSV
# ============================================================

comparison_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nModel comparison saved successfully: "
    f"{OUTPUT_CSV}"
)


# ============================================================
# 16. PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

display_columns = [
    "Rank",
    "Model"
] + available_metrics + [
    "Overall Score"
]

print(
    comparison_df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# 17. SELECT BEST MODEL
# ============================================================

best_model_row = comparison_df.iloc[0]

best_model = best_model_row["Model"]
best_score = best_model_row["Overall Score"]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best Model: {best_model}")
print(f"Overall Score: {best_score:.4f}")


# ============================================================
# 18. BEST MODEL METRICS
# ============================================================

print("\nBest Model Metrics:")
print("-" * 50)

for metric in available_metrics:

    value = best_model_row[metric]

    if pd.notna(value):
        print(f"{metric}: {value:.4f}")


# ============================================================
# 19. MODEL STRENGTHS & WEAKNESSES
# ============================================================

print("\n" + "=" * 70)
print("MODEL ANALYSIS")
print("=" * 70)

analysis_lines = []

analysis_lines.append(
    "DAY 21 - BEST MODEL ANALYSIS"
)

analysis_lines.append(
    "=" * 60
)

analysis_lines.append(
    f"\nSelected Best Model: {best_model}"
)

analysis_lines.append(
    f"Overall Score: {best_score:.4f}"
)

analysis_lines.append(
    "\nEvaluation Metrics:"
)

for metric in available_metrics:

    value = best_model_row[metric]

    if pd.notna(value):

        analysis_lines.append(
            f"- {metric}: {value:.4f}"
        )


# ============================================================
# 20. FIND BEST MODEL FOR EACH METRIC
# ============================================================

analysis_lines.append(
    "\nBest Model by Individual Metric:"
)

for metric in available_metrics:

    valid_data = comparison_df[
        comparison_df[metric].notna()
    ]

    if not valid_data.empty:

        best_index = valid_data[metric].idxmax()

        metric_best_model = comparison_df.loc[
            best_index,
            "Model"
        ]

        metric_best_value = comparison_df.loc[
            best_index,
            metric
        ]

        analysis_lines.append(
            f"- {metric}: "
            f"{metric_best_model} "
            f"({metric_best_value:.4f})"
        )


# ============================================================
# 21. STRENGTHS
# ============================================================

analysis_lines.append(
    "\nWhy this model was selected:"
)

analysis_lines.append(
    "1. It achieved the highest overall evaluation score."
)

analysis_lines.append(
    "2. It provides a strong balance across multiple metrics."
)

analysis_lines.append(
    "3. Model selection was based on more than accuracy alone."
)

analysis_lines.append(
    "4. Multiple evaluation metrics were considered for "
    "a more reliable decision."
)


# ============================================================
# 22. WEAKNESSES / TRADE-OFFS
# ============================================================

analysis_lines.append(
    "\nImportant Model Selection Trade-offs:"
)

analysis_lines.append(
    "- A model with high accuracy may still have poor recall."
)

analysis_lines.append(
    "- Low recall can be a serious issue in fraud detection "
    "because fraudulent transactions may be missed."
)

analysis_lines.append(
    "- High precision is useful when false positives are costly."
)

analysis_lines.append(
    "- F1 Score provides a balance between precision and recall."
)

analysis_lines.append(
    "- ROC-AUC helps evaluate the model's ability to "
    "distinguish between classes."
)

analysis_lines.append(
    "- Interpretability and business requirements should "
    "also be considered before deployment."
)


# ============================================================
# 23. FRAUD DETECTION RECOMMENDATION
# ============================================================

analysis_lines.append(
    "\nFraud Detection Recommendation:"
)

analysis_lines.append(
    "For fraud detection, Recall and Precision are especially "
    "important because both missed fraud and false alarms "
    "can have business consequences."
)

analysis_lines.append(
    "Therefore, the final production model should be selected "
    "using business requirements along with F1 Score, Recall, "
    "Precision and ROC-AUC."
)


# ============================================================
# 24. SAVE BEST MODEL ANALYSIS
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(analysis_lines)
    )


print(
    f"\nBest model analysis saved successfully: "
    f"{OUTPUT_REPORT}"
)


# ============================================================
# 25. CREATE MODEL COMPARISON VISUALIZATION
# ============================================================

plot_df = comparison_df.copy()

# ------------------------------------------------------------
# Keep only models with valid metric values
# ------------------------------------------------------------

plot_metrics = [
    metric
    for metric in metric_names
    if metric in plot_df.columns
    and plot_df[metric].notna().any()
]


if not plot_metrics:
    print(
        "\nNo metrics available for visualization."
    )

else:

    ax = plot_df.set_index("Model")[
        plot_metrics
    ].plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_title(
        "Day 21 - Machine Learning Model Comparison"
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(0, 1.05)

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.legend(
        title="Metrics"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Model comparison visualization saved successfully: "
        f"{OUTPUT_PNG}"
    )


# ============================================================
# 26. WEEK 3 ENGINEERING REFLECTION
# ============================================================

reflection = f"""
DAY 21 - WEEK 3 ENGINEERING REFLECTION
======================================

During Week 3, I worked with multiple Machine Learning
classification models and learned how different algorithms
approach prediction problems.

I worked with Decision Tree, Random Forest and XGBoost
models and compared their performance using multiple
evaluation metrics.

One of the most important lessons was that model selection
should not depend only on accuracy.

Precision, Recall, F1 Score and ROC-AUC can provide a much
better understanding of model performance, especially for
imbalanced classification problems such as fraud detection.

For Day 21, I compared the available model results and
selected:

Best Model: {best_model}

Overall Score: {best_score:.4f}

The model selection process helped me understand the
importance of balancing model performance with practical
requirements.

I also learned that different models have different
strengths and weaknesses. A model that performs well on
one metric may not necessarily be the best choice for
another business requirement.

For fraud detection, missing fraudulent transactions can
be costly, so Recall is important. At the same time,
excessive false positives can negatively affect legitimate
customers, making Precision important as well.

Overall, Week 3 improved my understanding of ensemble
learning, boosting, model evaluation and practical model
selection.

The main lesson I learned is:

"Choosing the right model is not just about the highest
score. It is about choosing the model that best fits the
problem and its real-world requirements."
"""


with open(
    OUTPUT_REFLECTION,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        reflection.strip()
    )


print(
    f"Week 3 reflection saved successfully: "
    f"{OUTPUT_REFLECTION}"
)


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DAY 21 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nGenerated files:")

print(f"1. {OUTPUT_CSV}")
print(f"2. {OUTPUT_PNG}")
print(f"3. {OUTPUT_REPORT}")
print(f"4. {OUTPUT_REFLECTION}")

print("\nBest Model:")
print(f"   {best_model}")

print(f"\nOverall Score:")
print(f"   {best_score:.4f}")

print("\nKey Learning:")
print(
    "Model selection should consider multiple evaluation "
    "metrics and real-world business requirements."
)

print("\n" + "=" * 70)