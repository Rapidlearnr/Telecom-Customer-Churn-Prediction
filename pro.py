
import pandas as pd
import matplotlib.pyplot as plt

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
