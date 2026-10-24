
import pandas as pd

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
