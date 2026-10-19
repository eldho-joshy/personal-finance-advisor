from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(RAW_FILE)

print("\n========== DATASET OVERVIEW ==========")
print(f"Dataset shape: {df.shape}")
print("\nFirst five records:")
print(df.head())

# --------------------------------------------------
# 3. Data quality checks
# --------------------------------------------------

print("\n========== DATA QUALITY ==========")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate transaction IDs:")
print(df["transaction_id"].duplicated().sum())

print("\nDuplicate complete records:")
print(df.duplicated().sum())

# --------------------------------------------------
# 4. Data cleaning
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Remove duplicate records and transaction IDs.
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["transaction_id"])

# Remove records with missing critical values.
df = df.dropna(
    subset=[
        "transaction_id",
        "date",
        "type",
        "category",
        "amount",
    ]
)

# Standardize text columns.
for column in ["type", "category", "payment_method"]:
    df[column] = df[column].astype("string").str.strip()

# Keep valid transaction types and positive amounts.
df = df[
    df["type"].isin(["Income", "Expense"])
    & (df["amount"] > 0)
].copy()

# Add analytical columns.
df["month"] = df["date"].dt.to_period("M").astype(str)
df["year"] = df["date"].dt.year
df["month_number"] = df["date"].dt.month

# Save the cleaned dataset.
CLEAN_FILE = PROCESSED_DIR / "cleaned_transactions.csv"
df.to_csv(CLEAN_FILE, index=False)

print("\n========== CLEANING RESULTS ==========")
print(f"Cleaned dataset shape: {df.shape}")
print(f"Cleaned data saved to: {CLEAN_FILE}")

# --------------------------------------------------
# 5. Monthly financial summary
# --------------------------------------------------

monthly = (
    df.pivot_table(
        index="month",
        columns="type",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)

for column in ["Income", "Expense"]:
    if column not in monthly.columns:
        monthly[column] = 0.0

monthly["Net_Savings"] = monthly["Income"] - monthly["Expense"]

monthly["Savings_Rate_Percent"] = np.where(
    monthly["Income"] > 0,
    monthly["Net_Savings"] / monthly["Income"] * 100,
    np.nan,
)

monthly.to_csv(
    PROCESSED_DIR / "monthly_summary.csv",
    index=False,
)

print("\n========== MONTHLY FINANCIAL SUMMARY ==========")
print(monthly.round(2).to_string(index=False))

# --------------------------------------------------
# 6. Category-wise expense analysis
# --------------------------------------------------

expenses = df[df["type"] == "Expense"].copy()

category_summary = (
    expenses.groupby("category")["amount"]
    .agg(["sum", "mean", "count"])
    .reset_index()
    .rename(
        columns={
            "sum": "total_expense",
            "mean": "average_transaction",
            "count": "transaction_count",
        }
    )
    .sort_values("total_expense", ascending=False)
)

category_summary.to_csv(
    PROCESSED_DIR / "category_summary.csv",
    index=False,
)

print("\n========== CATEGORY-WISE EXPENSES ==========")
print(category_summary.round(2).to_string(index=False))

# --------------------------------------------------
# 7. Generate visualizations
# --------------------------------------------------

sns.set_theme(style="whitegrid")

# Chart 1: Monthly income vs expenses.
plt.figure(figsize=(12, 6))

plt.plot(
    monthly["month"],
    monthly["Income"],
    marker="o",
    label="Income",
)

plt.plot(
    monthly["month"],
    monthly["Expense"],
    marker="o",
    label="Expenses",
)

plt.title("Monthly Income vs Expenses")
plt.xlabel("Month")
plt.ylabel("Amount (INR)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_income_vs_expenses.png",
    dpi=150,
)
plt.close()

# Chart 2: Total expenses by category.
plt.figure(figsize=(10, 6))

sns.barplot(
    data=category_summary,
    x="total_expense",
    y="category",
)

plt.title("Total Expenses by Category")
plt.xlabel("Total Expenses (INR)")
plt.ylabel("Category")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "expenses_by_category.png",
    dpi=150,
)
plt.close()

# Chart 3: Monthly savings rate.
plt.figure(figsize=(12, 6))

plt.bar(
    monthly["month"],
    monthly["Savings_Rate_Percent"],
)

plt.title("Monthly Savings Rate")
plt.xlabel("Month")
plt.ylabel("Savings Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_savings_rate.png",
    dpi=150,
)
plt.close()

print("\n========== ANALYSIS COMPLETED ==========")
print(f"Charts saved to: {OUTPUT_DIR}")
print("All analysis outputs have been generated.")