import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


# Baseline Linear Regression model
baseline_model = LinearRegression()

baseline_model.fit(X_train, y_train)

baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)


# Ridge Regression model
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

ridge_train_pred = ridge_model.predict(X_train)
ridge_test_pred = ridge_model.predict(X_test)


# Lasso Regression model
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1, max_iter=10000))
])

lasso_model.fit(X_train, y_train)

lasso_train_pred = lasso_model.predict(X_train)
lasso_test_pred = lasso_model.predict(X_test)


# Function to calculate model performance
def evaluate_model(model_name, y_train, train_pred, y_test, test_pred):

    train_mae = mean_absolute_error(y_train, train_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    train_rmse = train_mse ** 0.5
    train_r2 = r2_score(y_train, train_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = test_mse ** 0.5
    test_r2 = r2_score(y_test, test_pred)

    return {
        "Model": model_name,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    }


# Evaluate all models
results = []

results.append(
    evaluate_model(
        "Linear Regression",
        y_train,
        baseline_train_pred,
        y_test,
        baseline_test_pred
    )
)

results.append(
    evaluate_model(
        "Ridge Regression",
        y_train,
        ridge_train_pred,
        y_test,
        ridge_test_pred
    )
)

results.append(
    evaluate_model(
        "Lasso Regression",
        y_train,
        lasso_train_pred,
        y_test,
        lasso_test_pred
    )
)


# Create comparison dataframe
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df)


# Check overfitting
print("\nOverfitting Analysis:")

for index, row in results_df.iterrows():

    r2_gap = row["Train R2"] - row["Test R2"]

    print(
        row["Model"],
        "R2 Gap:",
        r2_gap
    )

    if r2_gap > 0.10:
        print("Possible overfitting detected.")
    else:
        print("No major overfitting detected.")


# Visualize Train vs Test R2
plt.figure(figsize=(9, 6))

x = range(len(results_df))

plt.bar(
    [i - 0.2 for i in x],
    results_df["Train R2"],
    width=0.4,
    label="Train R2"
)

plt.bar(
    [i + 0.2 for i in x],
    results_df["Test R2"],
    width=0.4,
    label="Test R2"
)

plt.xticks(x, results_df["Model"])

plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.title("Train vs Test R2 Score")

plt.legend()

plt.show()


# Compare RMSE
plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["Test RMSE"]
)

plt.xlabel("Models")
plt.ylabel("Test RMSE")

plt.title("Test RMSE Comparison")

plt.xticks(rotation=15)

plt.show()


# Save model comparison results
results_df.to_csv(
    "day13_model_comparison.csv",
    index=False
)


# Save prediction results
predictions = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Linear Regression": baseline_test_pred,
    "Ridge Regression": ridge_test_pred,
    "Lasso Regression": lasso_test_pred
})

predictions.to_csv(
    "day13_predictions.csv",
    index=False
)


print("\nModel comparison saved successfully!")
print("Prediction output saved successfully!")
print("\nDay 13 Overfitting and Regularization completed successfully!")