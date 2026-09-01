import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Basic information
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Sales distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Sales by Category
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Sales", estimator="sum")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()

# Sales by Region
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Region", y="Sales", estimator="sum")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()

# Business Insights
print("\n5 Business Insights:")

print("1. Sales values show a wide range, indicating differences in order values.")

print("2. The dataset can be analyzed across different product categories to identify high-performing categories.")

print("3. Regional analysis helps identify which regions contribute more to total sales.")

print("4. The Sales distribution helps identify common sales values and possible high-value orders.")

print("5. Correlation analysis helps understand relationships between numerical variables.")