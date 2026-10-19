from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==================================================
# 1. File paths and configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAD_THRESHOLD = 3.5
CONTAMINATION = 0.02
RANDOM_STATE = 42

# ==================================================
# 2. Load and prepare transactions
# ==================================================

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])

expenses = df.loc[df["type"].eq("Expense")].copy()
expenses = expenses.reset_index(drop=True)

if expenses.empty:
    raise ValueError("No expense transactions were found.")

required_columns = {
    "transaction_id",
    "date",
    "category",
    "amount",
    "type",
}

missing_columns = required_columns.difference(expenses.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if (expenses["amount"] <= 0).any():
    raise ValueError("Expense amounts must be positive.")

# Preserve labels for evaluation only.
# Never use these labels as model features.
if "is_injected_anomaly" not in expenses.columns:
    raise ValueError(
        "Missing is_injected_anomaly. "
        "Check the data generation and cleaning scripts."
    )

expenses["is_injected_anomaly"] = (
    expenses["is_injected_anomaly"]
    .fillna(False)
    .astype(bool)
)

print("Total expense transactions:", len(expenses))

# ==================================================
# 3. Category-based MAD detection
# ==================================================

def calculate_mad(values):
    """Calculate median absolute deviation."""
    median = values.median()
    return (values - median).abs().median()


category_stats = expenses.groupby("category")["amount"].agg(
    category_median="median",
    category_mad=calculate_mad,
)

# Calculate a fallback MAD for categories with zero MAD.
global_median = expenses["amount"].median()
global_mad = calculate_mad(expenses["amount"])

if global_mad <= 0:
    global_mad = 1.0

expenses = expenses.join(category_stats, on="category")

expenses["category_median"] = (
    expenses["category_median"].fillna(global_median)
)

expenses["category_mad"] = (
    expenses["category_mad"]
    .where(expenses["category_mad"] > 0, global_mad)
    .fillna(global_mad)
)

expenses["mad_score"] = (
    0.6745
    * (expenses["amount"] - expenses["category_median"])
    / expenses["category_mad"]
)

expenses["mad_is_anomaly"] = (
    expenses["mad_score"] > MAD_THRESHOLD
)

# ==================================================
# 4. Isolation Forest detection
# ==================================================

numeric_features = expenses[
    [
        "amount",
        "category_median",
        "category_mad",
    ]
].copy()

numeric_features["category_amount_ratio"] = (
    expenses["amount"]
    / expenses["category_median"].replace(0, np.nan)
).fillna(1.0)

numeric_features["day_of_month"] = expenses["date"].dt.day
numeric_features["day_of_week"] = expenses["date"].dt.dayofweek

category_features = pd.get_dummies(
    expenses["category"],
    prefix="category",
    dtype=int,
)

features = pd.concat(
    [numeric_features, category_features],
    axis=1,
)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
)

isolation_forest.fit(features_scaled)

expenses["isolation_forest_label"] = (
    isolation_forest.predict(features_scaled)
)

expenses["isolation_forest_is_anomaly"] = (
    expenses["isolation_forest_label"] == -1
)

expenses["isolation_forest_score"] = (
    -isolation_forest.score_samples(features_scaled)
)

# ==================================================
# 5. Combined prediction
# ==================================================

# MAD is the primary detector because it performed
# better in the independent synthetic validation.
expenses["is_anomaly"] = expenses["mad_is_anomaly"]

expenses["anomaly_method"] = np.where(
    expenses["mad_is_anomaly"],
    "Category-based MAD",
    "Normal",
)

# ==================================================
# 6. Save results
# ==================================================

output_file = OUTPUT_DIR / "anomaly_results.csv"

expenses.to_csv(output_file, index=False)

mad_anomalies = expenses.loc[
    expenses["mad_is_anomaly"]
].sort_values("mad_score", ascending=False)

if_anomalies = expenses.loc[
    expenses["isolation_forest_is_anomaly"]
].sort_values("isolation_forest_score", ascending=False)

print("\n========== ANOMALY DETECTION RESULTS ==========")

print("Transactions analyzed:", len(expenses))

print("\nCategory-based MAD")
print("Potential anomalies:", len(mad_anomalies))
print(
    "Percentage flagged:",
    round(len(mad_anomalies) / len(expenses) * 100, 2),
    "%",
)

print("\nIsolation Forest")
print("Potential anomalies:", len(if_anomalies))
print(
    "Percentage flagged:",
    round(len(if_anomalies) / len(expenses) * 100, 2),
    "%",
)

print("\nTop MAD anomalies:")

if mad_anomalies.empty:
    print("No transactions exceeded the MAD threshold.")
else:
    print(
        mad_anomalies[
            [
                "transaction_id",
                "date",
                "category",
                "amount",
                "mad_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

print(f"\nResults saved to: {output_file}")

# ==================================================
# 7. Visualize both models
# ==================================================

plt.figure(figsize=(12, 6))

normal = expenses.loc[~expenses["mad_is_anomaly"]]

plt.scatter(
    normal["date"],
    normal["amount"],
    label="Normal (MAD)",
    alpha=0.6,
)

plt.scatter(
    mad_anomalies["date"],
    mad_anomalies["amount"],
    label="MAD anomaly",
    marker="x",
)

plt.title("Financial Transaction Anomaly Detection")
plt.xlabel("Date")
plt.ylabel("Transaction Amount (INR)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

chart_file = FIGURE_DIR / "transaction_anomalies.png"

plt.savefig(chart_file, dpi=150)
plt.close()

print(f"Chart saved to: {chart_file}")