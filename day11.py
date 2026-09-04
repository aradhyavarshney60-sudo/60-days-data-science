import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load feature engineered dataset
df = pd.read_csv("feature_engineered_train.csv")

print("Dataset Shape:")
print(df.shape)


# Select numerical columns
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumerical Columns:")
print(numeric_columns)


# Target variable
target = "Sales"

# Features and target
X = df[numeric_columns].drop(columns=[target])
y = df[target]


print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


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


# Create baseline ML model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)

print("\nModel Training Completed!")


# Generate predictions
y_pred = model.predict(X_test)


print("\nFirst 10 Predictions:")
print(y_pred[:10])


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Save prediction results
predictions = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

predictions.to_csv("prediction_output.csv", index=False)


print("\nPrediction output saved successfully!")