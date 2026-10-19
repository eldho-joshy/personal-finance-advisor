from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# ==========================================
# 1. Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAD_THRESHOLD = 3.5

# ==========================================
# 2. Load transactions
# ==========================================

df = pd.read_csv(RAW_FILE)
df["date"] = pd.to_datetime(df["date"])

expenses = df.loc[
    df["type"].eq("Expense")
].copy()

expenses["is_injected_anomaly"] = (
    expenses["is_injected_anomaly"]
    .fillna(False)
    .astype(bool)
)

# Use earlier transactions as reference data and
# later transactions as test data.
expenses = expenses.sort_values("date").reset_index(drop=True)

split_index = int(len(expenses) * 0.70)

reference = expenses.iloc[:split_index].copy()
test = expenses.iloc[split_index:].copy()

print("Reference transactions:", len(reference))
print("Test transactions:", len(test))
print(
    "Injected anomalies in test:",
    int(test["is_injected_anomaly"].sum()),
)

# ==========================================
# 3. Calculate MAD from reference data only
# ==========================================

def calculate_mad(values):
    median = values.median()
    return (values - median).abs().median()


category_stats = reference.groupby("category")["amount"].agg(
    category_median="median",
    category_mad=calculate_mad,
)

global_median = reference["amount"].median()
global_mad = calculate_mad(reference["amount"])

if global_mad <= 0:
    global_mad = 1.0

# ==========================================
# 4. Score unseen test transactions
# ==========================================

test = test.join(category_stats, on="category")

test["category_median"] = (
    test["category_median"].fillna(global_median)
)

test["category_mad"] = (
    test["category_mad"]
    .where(test["category_mad"] > 0, global_mad)
    .fillna(global_mad)
)

test["mad_score"] = (
    0.6745
    * (test["amount"] - test["category_median"])
    / test["category_mad"]
)

test["mad_prediction"] = test["mad_score"] > MAD_THRESHOLD

# ==========================================
# 5. Evaluate
# ==========================================

y_true = test["is_injected_anomaly"].astype(bool)
y_pred = test["mad_prediction"].astype(bool)

print("\n========== TEMPORAL MAD VALIDATION ==========")
print("Transactions evaluated:", len(test))
print("Actual anomalies:", int(y_true.sum()))
print("Transactions flagged:", int(y_pred.sum()))

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    )
)

print(
    "\nPrecision:",
    round(precision_score(y_true, y_pred, zero_division=0), 3),
)

print(
    "Recall:",
    round(recall_score(y_true, y_pred, zero_division=0), 3),
)

print(
    "F1-score:",
    round(f1_score(y_true, y_pred, zero_division=0), 3),
)

print(
    "Accuracy:",
    round(accuracy_score(y_true, y_pred), 3),
)

# ==========================================
# 6. Save results
# ==========================================

output_file = OUTPUT_DIR / "mad_temporal_validation.csv"

test.to_csv(output_file, index=False)

print(f"\nValidation results saved to: {output_file}")