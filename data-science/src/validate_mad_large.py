from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# ==========================================
# 1. Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRANSACTIONS = 5000
TEST_FRACTION = 0.30
ANOMALY_FRACTION = 0.02
MAD_THRESHOLD = 3.5

rng = np.random.default_rng(RANDOM_STATE)

# ==========================================
# 2. Define spending categories
# ==========================================

categories = {
    "Food": (250, 100),
    "Transport": (120, 50),
    "Shopping": (400, 180),
    "Entertainment": (200, 100),
    "Bills": (1500, 350),
    "Rent": (10000, 700),
    "Healthcare": (500, 250),
    "Personal Care": (250, 100),
}

category_names = list(categories.keys())

probabilities = np.array(
    [0.25, 0.15, 0.15, 0.10, 0.12, 0.08, 0.05, 0.10]
)
probabilities = probabilities / probabilities.sum()

# ==========================================
# 3. Generate synthetic transactions
# ==========================================

def generate_transactions(n, seed):
    local_rng = np.random.default_rng(seed)

    dates = pd.date_range(
        start="2024-01-01",
        periods=n,
        freq="D",
    )

    selected_categories = local_rng.choice(
        category_names,
        size=n,
        p=probabilities,
    )

    amounts = []

    for category in selected_categories:
        mean, std = categories[category]
        amount = max(
            20.0,
            local_rng.normal(mean, std),
        )
        amounts.append(amount)

    df = pd.DataFrame({
        "date": dates,
        "category": selected_categories,
        "amount": amounts,
    })

    df["is_injected_anomaly"] = False

    # Select anomalies independently of the model.
    anomaly_count = max(
        1,
        round(n * ANOMALY_FRACTION),
    )

    anomaly_indices = local_rng.choice(
        df.index,
        size=anomaly_count,
        replace=False,
    )

    for idx in anomaly_indices:
        category = df.loc[idx, "category"]
        mean, std = categories[category]

        # Inject unusually high expenses.
        multiplier = local_rng.uniform(3.0, 6.0)

        df.loc[idx, "amount"] = mean + multiplier * std
        df.loc[idx, "is_injected_anomaly"] = True

    return df


df = generate_transactions(
    N_TRANSACTIONS,
    RANDOM_STATE,
)

# ==========================================
# 4. Split chronologically
# ==========================================

split_index = int(
    len(df) * (1 - TEST_FRACTION)
)

reference = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("Total transactions:", len(df))
print("Reference transactions:", len(reference))
print("Test transactions:", len(test))
print(
    "Anomalies in reference:",
    int(reference["is_injected_anomaly"].sum()),
)
print(
    "Anomalies in test:",
    int(test["is_injected_anomaly"].sum()),
)

# ==========================================
# 5. Calculate reference statistics
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
# 6. Evaluate the later test period
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

test["mad_prediction"] = (
    test["mad_score"] > MAD_THRESHOLD
)

y_true = test["is_injected_anomaly"].astype(bool)
y_pred = test["mad_prediction"].astype(bool)

print("\n========== LARGE TEMPORAL MAD VALIDATION ==========")
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
# 7. Save the validation results
# ==========================================

output_file = (
    OUTPUT_DIR / "mad_large_validation.csv"
)

test.to_csv(output_file, index=False)

print(f"\nValidation results saved to: {output_file}")