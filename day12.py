import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load feature engineered dataset
df = pd.read_csv("feature_engineered_train.csv")

print("Dataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# Target column
target = "Sales"

# Select numerical columns only
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=[target])
y = df[target]


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# Create Linear Regression model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")


# Generate predictions
y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])


# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Model coefficients
print("\nModel Coefficients:")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)


# Actual vs Predicted visualization
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()


# Save prediction results
predictions = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

predictions.to_csv("day12_predictions.csv", index=False)

print("\nPrediction output saved successfully!")