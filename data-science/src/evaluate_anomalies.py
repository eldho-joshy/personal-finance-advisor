from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
RESULT_FILE = PROJECT_ROOT / "data" / "processed" / "anomaly_results.csv"

# Load the original dataset and model predictions.
raw = pd.read_csv(RAW_FILE)
results = pd.read_csv(RESULT_FILE)

# Get the actual labels from the original dataset.
actual_labels = raw.loc[
    raw["type"].eq("Expense"),
    ["transaction_id", "is_injected_anomaly"],
]

# Get predictions from the anomaly detection results.
predictions = results[["transaction_id", "is_anomaly"]]

# Merge actual labels with model predictions.
evaluation = actual_labels.merge(
    predictions,
    on="transaction_id",
    how="inner",
    validate="one_to_one",
)

# Remove rows with missing evaluation values.
evaluation = evaluation.dropna(
    subset=["is_injected_anomaly", "is_anomaly"]
)

# Convert labels to Boolean values.
y_true = evaluation["is_injected_anomaly"].astype(bool)
y_pred = evaluation["is_anomaly"].astype(bool)

# Display evaluation results.
print("\n========== ANOMALY MODEL EVALUATION ==========")

print(f"Transactions evaluated: {len(evaluation)}")
print(f"Injected anomalies: {y_true.sum()}")
print(f"Detected anomalies: {y_pred.sum()}")

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
    "F1 score:",
    round(f1_score(y_true, y_pred, zero_division=0), 3),
)

print("\nDetailed report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[False, True],
        target_names=["Normal", "Injected anomaly"],
        zero_division=0,
    )
)
# Show injected anomalies and whether the model detected them.
print("\n========== INJECTED ANOMALY DETAILS ==========")

injected_details = evaluation[
    evaluation["is_injected_anomaly"].astype(bool)
][
    ["transaction_id", "is_injected_anomaly", "is_anomaly"]
]

print(injected_details.to_string(index=False))

# Show false positives.
print("\n========== FALSE POSITIVES ==========")

false_positives = evaluation[
    (~evaluation["is_injected_anomaly"].astype(bool))
    & evaluation["is_anomaly"].astype(bool)
]

print(f"False positives: {len(false_positives)}")

# Show false negatives.
print("\n========== MISSED ANOMALIES ==========")

false_negatives = evaluation[
    evaluation["is_injected_anomaly"].astype(bool)
    & (~evaluation["is_anomaly"].astype(bool))
]

print(f"Missed anomalies: {len(false_negatives)}")
# Inspect missed anomalies and correctly detected anomalies.
print("\n========== ANOMALY INVESTIGATION ==========")

investigation = evaluation.merge(
    raw[
        [
            "transaction_id",
            "date",
            "category",
            "amount",
            "description",
        ]
    ],
    on="transaction_id",
    how="left",
    validate="one_to_one",
)

investigation["result"] = "Normal transaction"

investigation.loc[
    investigation["is_injected_anomaly"].astype(bool)
    & investigation["is_anomaly"].astype(bool),
    "result",
] = "Detected anomaly"

investigation.loc[
    investigation["is_injected_anomaly"].astype(bool)
    & ~investigation["is_anomaly"].astype(bool),
    "result",
] = "Missed anomaly"

investigation.loc[
    ~investigation["is_injected_anomaly"].astype(bool)
    & investigation["is_anomaly"].astype(bool),
    "result",
] = "False positive"

print(
    investigation[
        [
            "transaction_id",
            "date",
            "category",
            "amount",
            "description",
            "result",
        ]
    ]
    .loc[
        investigation["result"].isin(
            [
                "Detected anomaly",
                "Missed anomaly",
                "False positive",
            ]
        )
    ]
    .sort_values("amount", ascending=False)
    .to_string(index=False)
)