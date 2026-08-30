import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

print("Original Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

# Handle missing values
df["Postal Code"] = df["Postal Code"].fillna(df["Postal Code"].median())

# Create new features
df["Sales_per_Quantity"] = df["Sales"] / df["Quantity"]

df["Order_Year"] = pd.to_datetime(df["Order Date"]).dt.year

print("\nCleaned Dataset:")
print(df.head())

print("\nNew Features:")
print("Sales_per_Quantity")
print("Order_Year")

# Save cleaned dataset
df.to_csv("cleaned_train.csv", index=False)

print("\nCleaned dataset saved as cleaned_train.csv")