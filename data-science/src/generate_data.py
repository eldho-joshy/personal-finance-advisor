import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42

random.seed(SEED)
rng = np.random.default_rng(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "transactions.csv"

START_DATE = "2025-10-01"
END_DATE = "2026-09-30"

# Illustrative configuration, not real financial data.
MONTHLY_INCOME_MIN = 25000
MONTHLY_INCOME_MAX = 60000

PAYMENT_METHODS = [
    "UPI",
    "Cash",
    "Debit Card",
    "Credit Card",
]

# Typical illustrative expense ranges in INR.
VARIABLE_EXPENSES = {
    "Food": (100, 500),
    "Transport": (50, 250),
    "Shopping": (300, 1800),
    "Entertainment": (150, 800),
    "Personal Care": (100, 600),
}

OCCASIONAL_EXPENSES = {
    "Education": (300, 2000),
    "Healthcare": (300, 2500),
}

# Recurring monthly expenses.
RECURRING_EXPENSES = {
    "Rent": (5000, 12000),
    "Bills": (800, 2500),
    "Subscriptions": (100, 600),
}

# --------------------------------------------------
# 2. Generate monthly income
# --------------------------------------------------

dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)

monthly_income = {}

for month in pd.period_range(
    start=START_DATE,
    end=END_DATE,
    freq="M",
):
    monthly_income[str(month)] = round(
        random.uniform(
            MONTHLY_INCOME_MIN,
            MONTHLY_INCOME_MAX,
        ),
        2,
    )

transactions = []
transaction_id = 1


def add_transaction(
    date,
    transaction_type,
    category,
    amount,
    payment_method,
    description,
):
    global transaction_id

    transactions.append({
        "transaction_id": f"TXN{transaction_id:05d}",
        "date": pd.Timestamp(date),
        "type": transaction_type,
        "category": category,
        "amount": round(float(amount), 2),
        "payment_method": payment_method,
        "description": description,
    })

    transaction_id += 1


# --------------------------------------------------
# 3. Generate transactions
# --------------------------------------------------

for month in pd.period_range(
    start=START_DATE,
    end=END_DATE,
    freq="M",
):
    month_start = month.start_time
    month_end = month.end_time.normalize()

    income_date = month_start

    add_transaction(
        income_date,
        "Income",
        "Income",
        monthly_income[str(month)],
        "Bank Transfer",
        "Monthly income",
    )

    # Recurring expenses occur once per month.
    for category, (minimum, maximum) in (
        RECURRING_EXPENSES.items()
    ):
        day = min(5, month_end.day)

        add_transaction(
            month_start.replace(day=day),
            "Expense",
            category,
            rng.uniform(minimum, maximum),
            "UPI",
            f"Monthly {category.lower()}",
        )

    # Daily variable expenses.
    for date in pd.date_range(
        month_start,
        month_end,
        freq="D",
    ):
        # Zero, one, or two variable expenses per day.
        number_of_expenses = rng.choice(
            [0, 1, 2],
            p=[0.20, 0.60, 0.20],
        )

        for _ in range(number_of_expenses):
            category = random.choice(
                list(VARIABLE_EXPENSES.keys())
            )

            minimum, maximum = VARIABLE_EXPENSES[
                category
            ]

            amount = rng.uniform(
                minimum,
                maximum,
            )

            add_transaction(
                date,
                "Expense",
                category,
                amount,
                random.choice(PAYMENT_METHODS),
                f"{category} expense",
            )

        # Occasional expenses are less frequent.
        if rng.random() < 0.12:
            category = random.choice(
                list(OCCASIONAL_EXPENSES.keys())
            )

            minimum, maximum = OCCASIONAL_EXPENSES[
                category
            ]

            add_transaction(
                date,
                "Expense",
                category,
                rng.uniform(minimum, maximum),
                random.choice(PAYMENT_METHODS),
                f"{category} expense",
            )

# --------------------------------------------------
# 4. Inject synthetic spending spikes
# --------------------------------------------------

spike_count = 6

expense_indices = [
    index
    for index, transaction in enumerate(transactions)
    if transaction["type"] == "Expense"
]

spike_indices = random.sample(
    expense_indices,
    min(spike_count, len(expense_indices)),
)

for index in expense_indices:
    transactions[index]["is_injected_anomaly"] = False

for index in spike_indices:
    transactions[index]["amount"] = round(
        transactions[index]["amount"] * 5,
        2,
    )

    transactions[index]["is_injected_anomaly"] = True


# --------------------------------------------------
# 5. Create and save the dataset
# --------------------------------------------------

df = pd.DataFrame(transactions)

df = df.sort_values(
    by=["date", "transaction_id"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSynthetic dataset generated successfully.")
print(f"Transactions: {len(df):,}")
print(f"Income transactions: {(df['type'] == 'Income').sum()}")
print(f"Expense transactions: {(df['type'] == 'Expense').sum()}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total income: INR {df.loc[df['type'] == 'Income', 'amount'].sum():,.2f}")
print(f"Total expenses: INR {df.loc[df['type'] == 'Expense', 'amount'].sum():,.2f}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst five transactions:")
print(df.head().to_string(index=False))