import pandas as pd

df = pd.read_csv("train.csv")

print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
df["Postal Code"] = df["Postal Code"].fillna(0)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Check data types
print("\nData Types:")
print(df.dtypes)

# Fix date formats
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, format="mixed")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, format="mixed")

print("\nDate Types After Cleaning:")
print(df[["Order Date", "Ship Date"]].dtypes)

# Save cleaned dataset
df.to_csv("cleaned_train.csv", index=False)

print("\nCleaned dataset saved successfully!")