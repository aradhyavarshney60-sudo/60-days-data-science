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