import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# DAY 14: ADAPTING MODELS TO CHANGING CONSTRAINTS
# ============================================================


# ------------------------------------------------------------
# 1. Load feature engineered dataset
# ------------------------------------------------------------

df = pd.read_csv("feature_engineered_train.csv")

print("Dataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. Define target
# ------------------------------------------------------------

target = "Sales"


# ------------------------------------------------------------
# 3. Select numerical features only
# ------------------------------------------------------------

X = df.select_dtypes(include=["int64", "float64"]).drop(
    columns=[target]
)

y = df[target]


print("\nNumerical Features:")
print(X.columns.tolist())

print("\nTarget:")
print(target)


# ------------------------------------------------------------
# 4. Split dataset into training and testing sets
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5. Train best-performing model from Day 13
# ------------------------------------------------------------

# Day 13 comparison showed Linear Regression
# as the best-performing model.

baseline_model = LinearRegression()

baseline_model.fit(
    X_train,
    y_train
)

baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)


# ------------------------------------------------------------
# 6. Evaluate baseline model
# ------------------------------------------------------------

baseline_train_mae = mean_absolute_error(
    y_train,
    baseline_train_pred
)

baseline_test_mae = mean_absolute_error(
    y_test,
    baseline_test_pred
)

baseline_train_mse = mean_squared_error(
    y_train,
    baseline_train_pred
)

baseline_test_mse = mean_squared_error(
    y_test,
    baseline_test_pred
)

baseline_train_rmse = baseline_train_mse ** 0.5
baseline_test_rmse = baseline_test_mse ** 0.5

baseline_train_r2 = r2_score(
    y_train,
    baseline_train_pred
)

baseline_test_r2 = r2_score(
    y_test,
    baseline_test_pred
)


print("\n===================================")
print("BASELINE MODEL PERFORMANCE")
print("===================================")

print("Train MAE:", baseline_train_mae)
print("Test MAE:", baseline_test_mae)

print("Train RMSE:", baseline_train_rmse)
print("Test RMSE:", baseline_test_rmse)

print("Train R2:", baseline_train_r2)
print("Test R2:", baseline_test_r2)


# ------------------------------------------------------------
# 7. Analyze model coefficients
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": baseline_model.coef_
})

coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
)


print("\n===================================")
print("FEATURE IMPORTANCE")
print("===================================")

print(coefficients)


# ------------------------------------------------------------
# 8. Identify most important feature
# ------------------------------------------------------------

important_feature = coefficients.iloc[0]["Feature"]

print("\nMost Important Feature:")
print(important_feature)


# ------------------------------------------------------------
# 9. Remove the most important feature
# ------------------------------------------------------------

X_reduced = X.drop(
    columns=[important_feature]
)

print("\n===================================")
print("FEATURE REMOVAL")
print("===================================")

print("Removed Feature:", important_feature)

print("\nRemaining Features:")
print(X_reduced.columns.tolist())


# ------------------------------------------------------------
# 10. Split reduced dataset
# ------------------------------------------------------------

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------
# 11. Retrain Linear Regression model
# ------------------------------------------------------------

adapted_model = LinearRegression()

adapted_model.fit(
    X_train_reduced,
    y_train_reduced
)

adapted_train_pred = adapted_model.predict(
    X_train_reduced
)

adapted_test_pred = adapted_model.predict(
    X_test_reduced
)


# ------------------------------------------------------------
# 12. Evaluate adapted model
# ------------------------------------------------------------

adapted_train_mae = mean_absolute_error(
    y_train_reduced,
    adapted_train_pred
)

adapted_test_mae = mean_absolute_error(
    y_test_reduced,
    adapted_test_pred
)

adapted_train_mse = mean_squared_error(
    y_train_reduced,
    adapted_train_pred
)

adapted_test_mse = mean_squared_error(
    y_test_reduced,
    adapted_test_pred
)

adapted_train_rmse = adapted_train_mse ** 0.5
adapted_test_rmse = adapted_test_mse ** 0.5

adapted_train_r2 = r2_score(
    y_train_reduced,
    adapted_train_pred
)

adapted_test_r2 = r2_score(
    y_test_reduced,
    adapted_test_pred
)


print("\n===================================")
print("AFTER FEATURE REMOVAL")
print("===================================")

print("Train MAE:", adapted_train_mae)
print("Test MAE:", adapted_test_mae)

print("Train RMSE:", adapted_train_rmse)
print("Test RMSE:", adapted_test_rmse)

print("Train R2:", adapted_train_r2)
print("Test R2:", adapted_test_r2)


# ------------------------------------------------------------
# 13. Before vs After comparison
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],

    "Before Feature Removal": [
        baseline_test_mae,
        baseline_test_mse,
        baseline_test_rmse,
        baseline_test_r2
    ],

    "After Feature Removal": [
        adapted_test_mae,
        adapted_test_mse,
        adapted_test_rmse,
        adapted_test_r2
    ]
})


print("\n===================================")
print("PERFORMANCE COMPARISON")
print("===================================")

print(comparison)


# ------------------------------------------------------------
# 14. Calculate performance change
# ------------------------------------------------------------

mae_change = adapted_test_mae - baseline_test_mae
rmse_change = adapted_test_rmse - baseline_test_rmse
r2_change = adapted_test_r2 - baseline_test_r2


print("\n===================================")
print("PERFORMANCE CHANGE")
print("===================================")

print("MAE Change:", mae_change)
print("RMSE Change:", rmse_change)
print("R2 Change:", r2_change)


# ------------------------------------------------------------
# 15. Analysis
# ------------------------------------------------------------

print("\n===================================")
print("DAY 14 ANALYSIS")
print("===================================")

if adapted_test_r2 < baseline_test_r2:

    print(
        "Model performance decreased after removing "
        "the important feature."
    )

elif adapted_test_r2 > baseline_test_r2:

    print(
        "Model performance improved after removing "
        "the important feature."
    )

else:

    print(
        "Model performance remained almost unchanged."
    )


# ------------------------------------------------------------
# 16. Save model comparison
# ------------------------------------------------------------

comparison.to_csv(
    "day14_model_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 17. Save prediction results
# ------------------------------------------------------------

predictions = pd.DataFrame({
    "Actual Sales": y_test_reduced.values,
    "Predicted Sales Before Feature Removal": baseline_test_pred,
    "Predicted Sales After Feature Removal": adapted_test_pred
})


predictions.to_csv(
    "day14_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 18. Visualization: Before vs After R2
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    ["Before Feature Removal", "After Feature Removal"],
    [baseline_test_r2, adapted_test_r2]
)

plt.xlabel("Model Version")
plt.ylabel("Test R2 Score")
plt.title("Before vs After Feature Removal")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 19. Visualization: Before vs After RMSE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    ["Before Feature Removal", "After Feature Removal"],
    [baseline_test_rmse, adapted_test_rmse]
)

plt.xlabel("Model Version")
plt.ylabel("Test RMSE")
plt.title("Before vs After Feature Removal")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 20. Final message
# ------------------------------------------------------------

print("\nModel comparison saved successfully!")

print("Prediction output saved successfully!")

print(
    "\nDay 14 Adapting Models to Changing Constraints "
    "completed successfully!"
)