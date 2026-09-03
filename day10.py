import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_train.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())
# Identify numerical and categorical features

numerical_features = df.select_dtypes(include="number").columns.tolist()
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)
# Convert date columns to datetime

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Create derived date features

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month

print("\nNew Derived Features:")
print(df[["Order Date", "Order Year", "Order Month"]].head())
# Apply one-hot encoding to categorical features

encoding_columns = [
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category"
]

df_encoded = pd.get_dummies(
    df,
    columns=encoding_columns,
    drop_first=True
)

print("\nDataset Shape After Encoding:")
print(df_encoded.shape)

print("\nEncoded Columns:")
print(df_encoded.columns.tolist())
# Scale numerical features
from sklearn.preprocessing import StandardScaler

# Select numerical columns
scaling_columns = ["Sales", "Postal Code"]

scaler = StandardScaler()

df_scaled = df.copy()

df_scaled[scaling_columns] = scaler.fit_transform(
    df_scaled[scaling_columns]
)

print("\nScaled Numerical Features:")
print(df_scaled[scaling_columns].head())


# Compare model readiness before and after feature engineering

print("\nBefore Feature Engineering:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing Values:", df.isnull().sum().sum())

print("\nAfter Feature Engineering:")
print("Rows:", df_encoded.shape[0])
print("Columns:", df_encoded.shape[1])
print("Missing Values:", df_encoded.isnull().sum().sum())


# Save feature engineered dataset
df_encoded.to_csv("feature_engineered_train.csv", index=False)

print("\nFeature engineered dataset saved successfully!")
from sklearn.preprocessing import StandardScaler

# Scale numerical features
scaler = StandardScaler()

scale_columns = ["Sales", "Postal Code"]

df_encoded[scale_columns] = scaler.fit_transform(
    df_encoded[scale_columns]
)

print("\nNumerical Features Scaled:")
print(df_encoded[scale_columns].head())
df_encoded.to_csv("feature_engineered_train.csv", index=False)

print("\nFinal feature engineered dataset saved successfully!")