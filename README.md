# Sales Data Analysis Project

## Problem Statement

The goal of this project is to analyze sales data and understand the overall performance of a business.

Using the dataset, I will explore different aspects of sales such as:

- Total Sales
- Sales by Category
- Sales by Region
- Top Selling Products
- Customer Segments

The objective is to find useful insights from the data and understand which categories, regions, and products contribute the most to sales.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Dataset

The dataset contains information about orders, customers, products, sales, quantity, discounts, and other business-related details.

## Project Goal

The main goal of this project is to practice the complete Data Science workflow, including data loading, cleaning, analysis, visualization, and extracting insights.

## Day 9: Data Cleaning

### Data Cleaning Steps

1. Loaded the `train.csv` dataset using Pandas.
2. Checked the dataset for missing values.
3. Found 11 missing values in the `Postal Code` column.
4. Filled the missing `Postal Code` values with `0`.
5. Checked for duplicate records and removed duplicates.
6. Checked the data types of all columns.
7. Converted `Order Date` and `Ship Date` into datetime format.
8. Saved the cleaned dataset as `cleaned_train.csv`.

### Missing Values

Before cleaning, the `Postal Code` column contained 11 missing values.

After cleaning, the missing values were handled by replacing them with `0`.

### Data Type Cleaning

The `Order Date` and `Ship Date` columns were converted from text/object format to datetime format.

### Output

The cleaned dataset was successfully saved as:

`cleaned_train.csv`

### Conclusion

The dataset was cleaned by handling missing values, removing duplicate records, checking data types, and fixing date formats. The cleaned dataset is now ready for further Data Science analysis.

---

# Day 44: Interactive Customer Intelligence Dashboard

## Objective

Built an interactive customer intelligence dashboard using Streamlit, Pandas and Plotly.

The dashboard is designed to help understand customer behavior, business KPIs, customer segmentation and churn-related insights.

## Features

- Interactive customer analytics dashboard
- KPI cards for business performance
- Customer segmentation analysis
- Churn analysis
- Interactive Plotly visualizations
- Customer data filtering
- CSV data upload support
- Download filtered customer data
- Business-friendly dashboard interface

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly

## How to Run

Install the required libraries:

```bash
pip install streamlit pandas numpy plotly
## Day 47: Customer Intelligence Capstone Planning 🚀

Started planning the end-to-end Customer Intelligence Platform capstone project.

### What I Completed
- Defined the business problem and project objectives.
- Created the capstone proposal document.
- Documented the proposed system architecture.
- Prepared the Day 47–60 project roadmap.
- Identified the main modules: churn prediction, customer segmentation, API, dashboard, and monitoring.

### Project Files
- `day47_capstone_proposal.md` — Business problem, objectives, scope, and success criteria.
- `day47_architecture.md` — System components, architecture flow, and technology stack.
- `day47_roadmap.md` — Planned development milestones from Day 47 to Day 60.

### Technologies
Python, Pandas, NumPy, Scikit-learn, FastAPI, Streamlit, Plotly, Git, and GitHub.

### Key Learning
A successful data science product starts with a clearly defined business problem, a planned architecture, and a realistic implementation roadmap.
## Day 48: Customer Intelligence Data Pipeline 🚀

Built a reusable data preprocessing pipeline for the Customer Intelligence Platform.

### What I Completed
- Loaded and inspected the Telco Customer Churn dataset.
- Checked and removed duplicate records.
- Converted `TotalCharges` to a numeric data type.
- Handled missing values using `SimpleImputer`.
- Scaled numerical features using `StandardScaler`.
- Encoded categorical features using `OneHotEncoder`.
- Built reusable preprocessing workflows with `Pipeline` and `ColumnTransformer`.
- Exported the processed dataset to CSV.

### Results
- Dataset records processed: 7,043
- Duplicate rows detected: 0
- Processed dataset shape: 7,043 rows × 47 columns
- Missing values remaining: 0

### Project Files
- `day48_pipeline.py` — Preprocessing pipeline implementation.
- `day48_pipeline_report.md` — Pipeline architecture and results.
- `day48_cleaned_customer_churn.csv` — Processed dataset.

### Technologies
Python, Pandas, Scikit-learn, SimpleImputer, StandardScaler, OneHotEncoder, Pipeline, and ColumnTransformer.

### Key Learning
Reusable preprocessing pipelines improve consistency and make machine learning workflows easier to maintain. For reliable model evaluation, preprocessing must be fitted on training data only.
## Day 49: Customer Intelligence Baseline Prototype 🚀

Built the first working baseline prototype for the Customer Intelligence Platform.

### What I Completed
- Loaded the existing churn prediction model.
- Generated predictions for 7,043 customer records.
- Saved customer-level prediction outputs.
- Created a prototype analysis report.
- Calculated an initial source-dataset accuracy of 75.01%.

### Project Files
- `day49_capstone.py` — Baseline prediction prototype.
- `day49_prediction_outputs.csv` — Customer-level predictions.
- `day49_prototype_report.md` — Prototype analysis and limitations.

### Key Learning
Integrating existing ML models into a working prototype is an important step toward building an end-to-end analytics system. Independent testing is still required to evaluate generalization.

**Tools:** Python, Pandas, Scikit-learn, Joblib, Git & GitHub.
## Day 50: Customer Intelligence Model Optimization 🚀

Optimized and compared multiple Machine Learning models for the Customer Intelligence Platform.

### What I Completed
- Built a reusable preprocessing and modeling pipeline.
- Compared Dummy Classifier, Logistic Regression, Random Forest, and Gradient Boosting.
- Applied GridSearchCV to tune Logistic Regression.
- Evaluated models using accuracy, precision, recall, and weighted F1-score.
- Generated a model comparison report and confusion matrix.

### Results
- Best model: Gradient Boosting
- Test accuracy: 80.62%
- Weighted F1-score: 79.79%

### Output Files
- `day50_model_comparison.csv`
- `day50_confusion_matrix.png`
- `day50_optimization_report.md`

### Key Learning
Model selection requires comparing multiple algorithms using held-out data and appropriate evaluation metrics. Hyperparameter tuning does not guarantee improved performance.

**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Git & GitHub.
## Day 51: Explainable AI for Customer Churn 🔍

Implemented Explainable AI techniques to understand the key factors influencing customer churn predictions.

### What I Completed
- Trained a Gradient Boosting classifier.
- Analyzed feature importance.
- Used SHAP to explain model behavior.
- Generated feature importance and SHAP visualizations.
- Created a business interpretation report.

### Output Files
- `day51_outputs/day51_feature_importance.csv`
- `day51_outputs/day51_feature_importance.png`
- `day51_outputs/day51_shap_importance.csv`
- `day51_outputs/day51_shap_summary.png`
- `day51_outputs/day51_business_interpretation_report.md`

### Key Learning
Explainable AI helps interpret model predictions and identify important features. SHAP insights can support business decisions, but they do not prove causation.

**Tools:** Python, Pandas, Scikit-learn, SHAP, Matplotlib, Git & GitHub.
# Customer Intelligence Platform

An end-to-end Customer Intelligence Platform focused on customer churn prediction, analytics, explainable AI, and business recommendations.

## Project Overview

This project combines machine learning and analytics workflows to help analyze customer churn patterns and support data-driven retention decisions.

## Key Features

- **Churn Prediction API:** FastAPI-based prediction workflow.
- **Analytics Dashboard:** Streamlit-based customer analytics.
- **Model Optimization:** Comparison of machine learning models and evaluation metrics.
- **Explainable AI:** Feature importance and SHAP-based analysis.
- **Real-Time Prediction Interface:** Interactive customer inputs with API-connected predictions.
- **Business Insights:** Churn visualizations, executive reporting, and recommendations.
- **Performance Testing:** API latency, batch prediction, and invalid-input validation.

## Technology Stack

Python, Pandas, Scikit-learn, Streamlit, FastAPI, Requests, Matplotlib, SHAP, Git, and GitHub.

## Repository Structure

| File | Purpose |
|---|---|
| `day43_api.py` | Churn prediction API |
| `day44.py` | Streamlit analytics dashboard |
| `day49_capstone.py` | Churn prediction prototype |
| `day50_optimization.py` | Model optimization and comparison |
| `day51_explainable_ai.py` | Explainable AI analysis |
| `day53.py` | Real-time prediction interface |
| `day54_business_insights.py` | Business insights and visualizations |
| `day55_performance_test.py` | API performance and reliability tests |

Additional reports and generated outputs are available in the corresponding project files and output folders.

## Prerequisites

- Python 3.10 or later
- Git
- pip

## Installation

Clone the repository:

```bash
git clone https://github.com/aradhyavarshney60-sudo/60-days-data-science.git
cd 60-days-data-science
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If a required package is missing, install it separately using pip.

## Running the Applications

### Streamlit Dashboard

```bash
python -m streamlit run day44.py
```

### Real-Time Prediction Interface

```bash
python -m streamlit run day53.py
```

### FastAPI Prediction Service

```bash
python -m uvicorn day43_api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

Run the API in a separate terminal before using the real-time prediction interface.

## API Usage

The API exposes a prediction endpoint used by the application. Open `/docs` to inspect the actual request schema and available endpoints before sending requests.

Example request structure will depend on the schema defined in `day43_api.py`.

## Data and Model

The project uses the Telco Customer Churn dataset and an existing trained model artifact. Ensure the required dataset and model files are available locally before running the relevant scripts.

## Testing and Validation

The performance testing workflow checks API health, individual predictions, batch requests, and invalid-input handling.

Benchmark results are environment-dependent and should not be treated as production performance guarantees.

## Limitations

- Predictions are estimates, not guarantees.
- Model performance should be validated on representative unseen data.
- Local latency tests do not establish production scalability.
- Deployment requires environment-specific configuration and additional testing.

## Future Improvements

- Expand automated testing.
- Improve deployment and configuration management.
- Add monitoring and structured logging.
- Evaluate performance under concurrent load.
- Improve dashboard usability and documentation.

## Author

Developed as part of the 60 Days of Data Science Challenge.

## Repository

[GitHub Repository](https://github.com/aradhyavarshney60-sudo/60-days-data-science)