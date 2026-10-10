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