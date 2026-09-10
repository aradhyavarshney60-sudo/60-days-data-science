# ============================================================
# DAY 17 - LOAN APPROVAL PREDICTION USING DECISION TREE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. CREATE / LOAD LOAN DATASET
# ============================================================

print("=" * 60)
print("DAY 17 - LOAN APPROVAL PREDICTION")
print("=" * 60)

file_name = "loan_approval_dataset.csv"

if os.path.exists(file_name):

    print("\nLoading existing loan dataset...")
    df = pd.read_csv(file_name)

else:

    print("\nLoan dataset not found.")
    print("Creating sample loan approval dataset...")

    np.random.seed(42)

    n = 500

    data = {
        "Age": np.random.randint(21, 65, n),
        "Income": np.random.randint(20000, 150000, n),
        "Credit_Score": np.random.randint(300, 850, n),
        "Loan_Amount": np.random.randint(5000, 100000, n),
        "Employment_Years": np.random.randint(0, 30, n),
        "Existing_Debt": np.random.randint(0, 80000, n),
        "Savings": np.random.randint(1000, 100000, n)
    }

    df = pd.DataFrame(data)

    # Create realistic loan approval rule
    approval_score = (
        (df["Credit_Score"] >= 650).astype(int)
        + (df["Income"] >= 50000).astype(int)
        + (df["Employment_Years"] >= 3).astype(int)
        + (df["Savings"] >= 10000).astype(int)
        + (df["Existing_Debt"] < 40000).astype(int)
        + (df["Loan_Amount"] < df["Income"] * 1.2).astype(int)
    )

    df["Loan_Approved"] = (approval_score >= 4).astype(int)

    df.to_csv(file_name, index=False)

    print(f"Dataset created and saved as: {file_name}")


# ============================================================
# 2. BASIC DATA EXPLORATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())


# ============================================================
# 3. IDENTIFY FEATURES AND TARGET
# ============================================================

target_column = "Loan_Approved"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(target_column)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. TRAIN DECISION TREE CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DECISION TREE")
print("=" * 60)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nPredictions generated successfully!")


# ============================================================
# 7. MODEL ACCURACY
# ============================================================

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy : {test_accuracy:.4f}")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_test,
    y_test_pred,
    target_names=["Rejected", "Approved"]
))


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_test_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected", "Approved"]
)

disp.plot()

plt.title("Loan Approval - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "day17_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. VISUALIZE DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("CREATING DECISION TREE VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for Loan Approval Prediction")

plt.savefig(
    "day17_decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Decision tree visualization saved successfully!")


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature importance:")
print(feature_importance)


# ============================================================
# 12. FEATURE IMPORTANCE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Decision Tree Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "day17_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Feature importance visualization saved successfully!")


# ============================================================
# 13. OVERFITTING ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS")
print("=" * 60)

depths = range(1, 11)

train_scores = []
test_scores = []

for depth in depths:

    temp_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    temp_model.fit(X_train, y_train)

    train_pred = temp_model.predict(X_train)
    test_pred = temp_model.predict(X_test)

    train_scores.append(
        accuracy_score(y_train, train_pred)
    )

    test_scores.append(
        accuracy_score(y_test, test_pred)
    )


# Print accuracy for each depth

for depth, train_score, test_score in zip(
    depths,
    train_scores,
    test_scores
):

    print(
        f"Depth {depth}: "
        f"Train Accuracy = {train_score:.4f}, "
        f"Test Accuracy = {test_score:.4f}"
    )


# ============================================================
# 14. PLOT OVERFITTING ANALYSIS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(depths),
    train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    list(depths),
    test_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting Analysis")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "day17_overfitting_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. FIND BEST TREE DEPTH
# ============================================================

best_index = np.argmax(test_scores)
best_depth = list(depths)[best_index]
best_test_accuracy = test_scores[best_index]

print("\nBest Tree Depth:", best_depth)
print(f"Best Test Accuracy: {best_test_accuracy:.4f}")


# ============================================================
# 16. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)

print("""
Decision Tree can help financial institutions make loan
approval decisions using customer information.

Important features can indicate which factors have the
largest influence on loan approval.

For example:
- Higher credit score generally supports approval.
- Higher income can improve loan eligibility.
- Lower existing debt can reduce financial risk.
- Longer employment history can indicate stability.
- Higher savings can support repayment ability.

However, decision trees can overfit when the tree becomes
too deep. Limiting max_depth helps control model complexity.
""")


# ============================================================
# 17. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "day17_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")


# ============================================================
# 18. SAVE PREDICTION RESULTS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Loan_Approved"] = y_test.values
prediction_results["Predicted_Loan_Approved"] = y_test_pred

prediction_results["Prediction_Correct"] = (
    prediction_results["Actual_Loan_Approved"]
    ==
    prediction_results["Predicted_Loan_Approved"]
)

prediction_results.to_csv(
    "day17_predictions.csv",
    index=False
)

print("Prediction results saved successfully!")


# ============================================================
# 19. SAVE MODEL RESULTS
# ============================================================

model_results = pd.DataFrame({
    "Model": ["Decision Tree Classifier"],
    "Criterion": ["Gini"],
    "Max_Depth": [5],
    "Training_Accuracy": [train_accuracy],
    "Testing_Accuracy": [test_accuracy],
    "Best_Depth_From_Analysis": [best_depth],
    "Best_Test_Accuracy": [best_test_accuracy]
})

model_results.to_csv(
    "day17_model_results.csv",
    index=False
)

print("Model results saved successfully!")


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 17 SUMMARY")
print("=" * 60)

print("Loan approval dataset loaded successfully.")
print("Decision Tree classifier trained successfully.")
print("Predictions generated successfully.")
print("Decision tree visualization created successfully.")
print("Feature importance analysis completed.")
print("Overfitting analysis completed.")
print("Confusion matrix created successfully.")
print("Prediction results saved successfully.")
print("Model results saved successfully.")

print(
    "\nDay 17 Loan Approval Prediction "
    "using Decision Trees completed successfully! 🌳"
)

print("=" * 60)