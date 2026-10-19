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

# -----------------------------------------------
# 1. File paths
# -----------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"

RESULT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"
)

# -----------------------------------------------
# 2. Load datasets
# -----------------------------------------------

raw = pd.read_csv(RAW_FILE)
results = pd.read_csv(RESULT_FILE)

# Use the original dataset for ground-truth labels.
actual = raw.loc[
    raw["type"].eq("Expense"),
    ["transaction_id", "is_injected_anomaly", "category", "amount"],
].copy()

actual["is_injected_anomaly"] = (
    actual["is_injected_anomaly"].fillna(False).astype(bool)
)

# Get Isolation Forest predictions.
if_predictions = results[
    ["transaction_id", "is_anomaly"]
].copy()

if_predictions = if_predictions.rename(
    columns={"is_anomaly": "isolation_forest"}
)

# Merge actual labels and Isolation Forest predictions.
evaluation = actual.merge(
    if_predictions,
    on="transaction_id",
    how="inner",
    validate="one_to_one",
)

# -----------------------------------------------
# 3. Detect anomalies using MAD
# -----------------------------------------------

# MAD = Median Absolute Deviation.
# It measures how far values typically deviate
# from the median and is less sensitive to outliers
# than standard deviation.

def calculate_mad(values):
    median = values.median()
    return (values - median).abs().median()


# Calculate median and MAD separately for each category.
category_stats = (
    evaluation.groupby("category")["amount"]
    .agg(
        category_median="median",
        category_mad=calculate_mad,
    )
)

evaluation = evaluation.join(
    category_stats,
    on="category",
)

# Use the global MAD when a category has zero MAD.
global_median = evaluation["amount"].median()

global_mad = calculate_mad(evaluation["amount"])

if global_mad == 0:
    global_mad = 1.0

evaluation["effective_mad"] = (
    evaluation["category_mad"]
    .where(evaluation["category_mad"] > 0, global_mad)
)

# Calculate category-relative robust z-scores.
evaluation["mad_score"] = (
    0.6745
    * (
        evaluation["amount"]
        - evaluation["category_median"]
    )
    / evaluation["effective_mad"]
)

# Flag unusually high expenses.
MAD_THRESHOLD = 3.5

evaluation["mad_anomaly"] = (
    evaluation["mad_score"] > MAD_THRESHOLD
)

# -----------------------------------------------
# 4. Evaluate both methods
# -----------------------------------------------

y_true = evaluation["is_injected_anomaly"].astype(bool)

models = {
    "Isolation Forest": evaluation["isolation_forest"].astype(bool),
    "Category-based MAD": evaluation["mad_anomaly"].astype(bool),
}

print("\n========== ANOMALY MODEL COMPARISON ==========")
print("Transactions evaluated:", len(evaluation))
print("Injected anomalies:", int(y_true.sum()))

comparison = []

for model_name, predictions in models.items():
    comparison.append(
        {
            "Model": model_name,
            "Flagged": int(predictions.sum()),
            "Precision": round(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                3,
            ),
            "Recall": round(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                3,
            ),
            "F1 Score": round(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                3,
            ),
            "Accuracy": round(
                accuracy_score(y_true, predictions),
                3,
            ),
        }
    )

    print(f"\n{model_name}")
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[False, True],
        )
    )

comparison_df = pd.DataFrame(comparison)

print("\n========== COMPARISON SUMMARY ==========")
print(comparison_df.to_string(index=False))

# -----------------------------------------------
# 5. Save comparison results
# -----------------------------------------------

output_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)

comparison_df.to_csv(output_file, index=False)

print(f"\nComparison saved to: {output_file}")