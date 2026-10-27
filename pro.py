
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# 1. Load the Dataset

df = pd.read_csv("Customer Churn.csv")

df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# 2. Data Cleaning

df = df.drop_duplicates()

unwanted_columns = [
    "Customer ID",
    "CustomerID",
    "Anonymous Customer ID",
    "FN",
    "FP"
]

df = df.drop(
    columns=unwanted_columns,
    errors="ignore"
)

for column in df.columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(subset=["Churn"])

df["Churn"] = df["Churn"].astype(int)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:", df.shape)


# 3. Exploratory Data Analysis


# Graph 1: Customer Churn Distribution

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

churn_percentage = (
    df["Churn"].value_counts(normalize=True) * 100
)

print("\nChurn Percentage:")
print(churn_percentage.round(2))

plt.figure(figsize=(8, 5))

ax = df["Churn"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title(
    "Telecom Customer Churn Distribution",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Customer Churn Status")
plt.ylabel("Number of Customers")

plt.xticks(
    ticks=[0, 1],
    labels=["No Churn", "Churn"],
    rotation=0
)

plt.grid(axis="y", alpha=0.25)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3)

plt.tight_layout()
plt.show()


# Graph 2: Customer Churn by Complaints

complaint_churn = pd.crosstab(
    df["Complains"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Complaints:")
print(complaint_churn.round(2))

ax = complaint_churn.reindex(
    index=[0, 1],
    columns=[0, 1],
    fill_value=0
).plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(
    "Customer Churn by Complaints",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Customer Complaints")
plt.ylabel("Customers (%)")

plt.xticks(
    ticks=[0, 1],
    labels=["No Complaint", "Complaint"],
    rotation=0
)

plt.legend(
    ["No Churn", "Churn"],
    title="Customer Status"
)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# Graph 3: Customer Churn by Tariff Plan

tariff_churn = pd.crosstab(
    df["Tariff Plan"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Tariff Plan:")
print(tariff_churn.round(2))

ax = tariff_churn.reindex(
    index=[1, 2],
    columns=[0, 1],
    fill_value=0
).plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title(
    "Customer Churn by Tariff Plan",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Tariff Plan")
plt.ylabel("Customers (%)")

plt.xticks(
    ticks=[0, 1],
    labels=["Pay as You Go", "Contractual"],
    rotation=0
)

plt.legend(
    ["No Churn", "Churn"],
    title="Customer Status"
)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# Graph 4: Churn Rate by Subscription Length

subscription_churn = df.groupby(
    "Subscription Length"
)["Churn"].mean() * 100

plt.figure(figsize=(10, 5))

plt.plot(
    subscription_churn.index,
    subscription_churn.values,
    marker=".",
    linewidth=2
)

plt.title(
    "Customer Churn Rate by Subscription Length",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Subscription Length (Months)")
plt.ylabel("Churn Rate (%)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Graph 5: Customer Value — Churned vs Non-Churned

df.boxplot(
    column="Customer Value",
    by="Churn",
    grid=False,
    figsize=(8, 5)
)

plt.suptitle("")

plt.title(
    "Customer Value: Churned vs Retained Customers",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.xlabel("Customer Churn Status")
plt.ylabel("Customer Value")

plt.xticks(
    ticks=[1, 2],
    labels=["No Churn", "Churn"]
)

plt.tight_layout()
plt.show()


# 4. Prepare Features and Target

X = df.drop(columns=["Churn"])

y = df["Churn"]

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())


# 5. Split the Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])


# 6. Handle Missing Values Without Imputation

X_train = X_train.fillna(X_train.median())

X_test = X_test.fillna(X_train.median())


# 7. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# 8. Define Five Machine Learning Models

models = {

    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=3000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ),

    "SVM": SVC(
        C=1.0,
        kernel="rbf",
        random_state=42
    )
}


# 9. Train and Evaluate Models

results = []

predictions_by_model = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    predictions_by_model[name] = predictions

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Balanced Accuracy": balanced_accuracy

    })


# 10. Compare Model Performance

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
)

print("\n" + "=" * 100)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-score": "{:.4f}".format,
            "Balanced Accuracy": "{:.4f}".format
        }
    )
)


# Graph 6: Performance Comparison of Five ML Models

results_df.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Balanced Accuracy"
    ]
].plot(
    kind="bar",
    figsize=(13, 6)
)

plt.title(
    "Performance Comparison of Five ML Models",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()


# 11. Identify the Best Model Based on F1-score

best_model_name = results_df.iloc[0]["Model"]

best_f1 = results_df.iloc[0]["F1-score"]

best_predictions = predictions_by_model[
    best_model_name
]

print("\n" + "=" * 60)
print("FINAL MODEL SELECTION")
print("=" * 60)

print("Best Model:", best_model_name)

print(f"Best F1-score: {best_f1:.4f}")


# Graph 7: Confusion Matrix — Best Performing Model

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    labels=[0, 1],
    display_labels=["No Churn", "Churn"],
    values_format="d"
)

plt.title(
    f"Confusion Matrix — Best Performing Model\n{best_model_name}",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.tight_layout()
plt.show()


# 12. Classification Reports for All Models

for name, predictions in predictions_by_model.items():

    print("\n" + "=" * 60)
    print(f"CLASSIFICATION REPORT: {name}")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )


# Graph 8: Precision vs Recall — Five-Model Comparison

plt.figure(figsize=(9, 6))

plt.scatter(
    results_df["Recall"],
    results_df["Precision"],
    s=120
)

for _, row in results_df.iterrows():

    plt.annotate(
        row["Model"],
        (row["Recall"], row["Precision"]),
        xytext=(6, 5),
        textcoords="offset points",
        fontsize=9
    )

plt.title(
    "Precision vs Recall: Five-Model Comparison",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Recall (Churners Identified)")
plt.ylabel("Precision (Correct Churn Predictions)")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 13. Baseline Accuracy Comparison

baseline_accuracy = (y_test == 0).mean()

print("\n" + "=" * 60)
print("BASELINE ACCURACY COMPARISON")
print("=" * 60)

print(
    f"Majority Class Baseline Accuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Best Model Accuracy: "
    f"{results_df.iloc[0]['Accuracy']:.4f}"
)


# 14. Display Best Model Performance

best_result = results_df.iloc[0]

print("\nBEST MODEL PERFORMANCE")

print(f"Model:             {best_model_name}")
print(f"Accuracy:          {best_result['Accuracy']:.4f}")
print(f"Precision:         {best_result['Precision']:.4f}")
print(f"Recall:            {best_result['Recall']:.4f}")
print(f"F1-score:          {best_result['F1-score']:.4f}")
print(f"Balanced Accuracy: {best_result['Balanced Accuracy']:.4f}")


# 15. Final Conclusion

print("\nPROJECT CONCLUSION")

print(
    f"Among the five evaluated algorithms, {best_model_name} "
    f"achieved the highest F1-score on the test dataset."
)

print(
    "The models were evaluated using accuracy, precision, "
    "recall, F1-score and balanced accuracy."
)

print(
    "The project analyzes customer usage patterns, complaints, "
    "subscription details and customer value to predict churn."
)

print(
    "The baseline accuracy was also calculated to evaluate "
    "model performance fairly."
)
