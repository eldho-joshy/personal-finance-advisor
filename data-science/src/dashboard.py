from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ==================================================
# 1. Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTIONS_FILE = DATA_DIR / "cleaned_transactions.csv"
ANOMALIES_FILE = DATA_DIR / "anomaly_results.csv"
MONTHLY_FILE = DATA_DIR / "monthly_summary.csv"
CATEGORY_FILE = DATA_DIR / "category_summary.csv"
VALIDATION_FILE = DATA_DIR / "mad_large_validation.csv"

st.set_page_config(
    page_title="Personal Finance Advisor",
    page_icon="💰",
    layout="wide",
)

# ==================================================
# 2. Load data
# ==================================================

@st.cache_data
def load_data():
    transactions = pd.read_csv(TRANSACTIONS_FILE)
    anomalies = pd.read_csv(ANOMALIES_FILE)
    monthly = pd.read_csv(MONTHLY_FILE)
    categories = pd.read_csv(CATEGORY_FILE)

    validation = (
        pd.read_csv(VALIDATION_FILE)
        if VALIDATION_FILE.exists()
        else pd.DataFrame()
    )

    transactions["date"] = pd.to_datetime(
        transactions["date"], errors="coerce"
    )

    anomalies["date"] = pd.to_datetime(
        anomalies["date"], errors="coerce"
    )

    if "month" not in transactions.columns:
        transactions["month"] = (
            transactions["date"].dt.to_period("M").astype(str)
        )

    return transactions, anomalies, monthly, categories, validation


try:
    (
        transactions,
        anomalies,
        monthly,
        categories,
        validation,
    ) = load_data()
except FileNotFoundError as error:
    st.error(f"Required data file was not found: {error}")
    st.stop()

# ==================================================
# 3. Sidebar filters
# ==================================================

st.sidebar.title("Dashboard Filters")

if transactions.empty:
    st.warning("No transactions are available.")
    st.stop()

# Define date_range before any code uses it.
valid_dates = transactions["date"].dropna()
if valid_dates.empty:
    st.error("The transactions file does not contain valid dates.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

date_range = st.sidebar.date_input(
    "Transaction date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range_filter",
)

available_categories = sorted(
    transactions["category"].dropna().unique().tolist()
)

with st.sidebar.expander("Categories", expanded=False):
    selected_categories = st.multiselect(
        "Filter by category (optional)",
        options=available_categories,
        default=[],
        key="category_filter",
    )

available_types = sorted(
    transactions["type"].dropna().unique().tolist()
)

with st.sidebar.expander("Transaction Types", expanded=False):
    selected_types = st.multiselect(
        "Filter by transaction type (optional)",
        options=available_types,
        default=[],
        key="type_filter",
    )

# ==================================================
# Apply filters
# ==================================================
filtered = transactions.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered = filtered.loc[
        filtered["date"].dt.date.between(start_date, end_date)
    ]

if selected_categories:
    filtered = filtered.loc[
        filtered["category"].isin(selected_categories)
    ]

if selected_types:
    filtered = filtered.loc[
        filtered["type"].isin(selected_types)
    ]
# ==================================================
# 4. Dashboard header
# ==================================================

st.title("Personal Finance Advisor")
st.caption(
    "Financial analytics, spending insights, and potential anomaly detection"
)

if not filtered.empty:
    income = filtered.loc[
        filtered["type"].eq("Income"), "amount"
    ].sum()

    expenses_total = filtered.loc[
        filtered["type"].eq("Expense"), "amount"
    ].sum()
else:
    income = 0.0
    expenses_total = 0.0

net_savings = income - expenses_total
savings_rate = (
    net_savings / income * 100
    if income > 0
    else 0.0
)

# ==================================================
# 5. KPI cards
# ==================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Income",
    f"₹{income:,.2f}",
)

col2.metric(
    "Total Expenses",
    f"₹{expenses_total:,.2f}",
)

col3.metric(
    "Net Savings",
    f"₹{net_savings:,.2f}",
)

col4.metric(
    "Savings Rate",
    f"{savings_rate:.1f}%",
)

st.divider()

# ==================================================
# 6. Monthly financial trends
# ==================================================

st.subheader("Monthly Financial Trends")

monthly_filtered = filtered.copy()

if not monthly_filtered.empty:
    monthly_filtered["month"] = (
        monthly_filtered["date"].dt.to_period("M").astype(str)
    )

    monthly_trends = (
        monthly_filtered.pivot_table(
            index="month",
            columns="type",
            values="amount",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    if "Income" not in monthly_trends.columns:
        monthly_trends["Income"] = 0.0

    if "Expense" not in monthly_trends.columns:
        monthly_trends["Expense"] = 0.0

    monthly_trends["Savings"] = (
        monthly_trends["Income"] - monthly_trends["Expense"]
    )

    trend_data = monthly_trends.melt(
        id_vars="month",
        value_vars=["Income", "Expense", "Savings"],
        var_name="Metric",
        value_name="Amount",
    )

    fig = px.line(
        trend_data,
        x="month",
        y="Amount",
        color="Metric",
        markers=True,
        title="Income, Expenses, and Net Savings",
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount (INR)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No transactions match the selected filters.")

# ==================================================
# 7. Category analysis
# ==================================================

left, right = st.columns(2)

with left:
    st.subheader("Expenses by Category")

    expense_data = filtered.loc[
        filtered["type"].eq("Expense")
    ]

    category_totals = (
        expense_data.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )

    if not category_totals.empty:
        fig = px.bar(
            category_totals,
            x="category",
            y="amount",
            title="Total Spending by Category",
            labels={
                "category": "Category",
                "amount": "Amount (INR)",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        pie = px.pie(
            category_totals,
            names="category",
            values="amount",
            title="Spending Distribution",
            hole=0.4,
        )

        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No expense transactions match the selected filters.")

with right:
    st.subheader("Transaction Summary")

    if not filtered.empty:
        summary = (
            filtered.groupby(["type", "category"])
            .agg(
                Total_Amount=("amount", "sum"),
                Transactions=("amount", "count"),
            )
            .reset_index()
        )

        summary["Total_Amount"] = summary["Total_Amount"].round(2)

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transactions match the current filters.")
# ==================================================
# 8. Anomaly detection
# ==================================================

st.divider()
st.subheader("Potential Spending Anomalies")

# CSV files may contain boolean values or strings such as "True"/"False".
def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

if "mad_is_anomaly" in anomalies.columns:
    anomaly_flag_column = "mad_is_anomaly"
elif "is_anomaly" in anomalies.columns:
    anomaly_flag_column = "is_anomaly"
else:
    anomaly_flag_column = None

if anomaly_flag_column:
    anomaly_data = anomalies.loc[
        as_bool(anomalies[anomaly_flag_column])
    ].copy()

    if not anomaly_data.empty:
        anomaly_data["date"] = pd.to_datetime(
            anomaly_data["date"], errors="coerce"
        )

        # Apply the selected date range.
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            anomaly_data = anomaly_data.loc[
                anomaly_data["date"].dt.date.between(start_date, end_date)
            ]

        # Empty optional filters mean "show all", not "show none".
        if selected_categories:
            anomaly_data = anomaly_data.loc[
                anomaly_data["category"].isin(selected_categories)
            ]

        if selected_types and "type" in anomaly_data.columns:
            anomaly_data = anomaly_data.loc[
                anomaly_data["type"].isin(selected_types)
            ]

    st.metric(
        "Potential anomalies in selected period",
        len(anomaly_data),
    )

    if not anomaly_data.empty:
        display_columns = [
            column
            for column in [
                "transaction_id",
                "date",
                "category",
                "amount",
                "mad_score",
                "anomaly_score",
            ]
            if column in anomaly_data.columns
        ]

        st.dataframe(
            anomaly_data[display_columns].sort_values(
                "amount", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.scatter(
            anomaly_data,
            x="date",
            y="amount",
            color="category",
            hover_data=[
                column
                for column in ["transaction_id", "mad_score", "anomaly_score"]
                if column in anomaly_data.columns
            ],
            title="Potential Anomalies Over Time",
            labels={"amount": "Amount (INR)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No potential anomalies match the current filters.")
else:
    st.warning(
        "The anomaly results file does not contain a recognized "
        "anomaly prediction column."
    )

st.caption(
    "An anomaly is an unusually high or unusual transaction, "
    "not proof of fraud."
)

# ==================================================
# 9. Model validation results
# ==================================================

st.divider()
st.subheader("Model Validation")

if not validation.empty:
    required = {"is_injected_anomaly", "mad_prediction"}

    if required.issubset(validation.columns):

        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
            confusion_matrix,
        )

        y_true = (
            validation["is_injected_anomaly"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        y_pred = (
            validation["mad_prediction"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[False, True],
        )

        results = pd.DataFrame([{
            "Model": "Category-based MAD",
            "Test Transactions": len(validation),
            "Actual Anomalies": int(y_true.sum()),
            "Detected Anomalies": int(y_pred.sum()),
            "Precision": round(
                precision_score(y_true, y_pred, zero_division=0), 3
            ),
            "Recall": round(
                recall_score(y_true, y_pred, zero_division=0), 3
            ),
            "F1 Score": round(
                f1_score(y_true, y_pred, zero_division=0), 3
            ),
            "Accuracy": round(
                accuracy_score(y_true, y_pred), 3
            ),
        }])

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        st.write("Confusion Matrix")

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Normal", "Actual Anomaly"],
            columns=["Predicted Normal", "Predicted Anomaly"],
        )

        st.dataframe(
            matrix_df,
            use_container_width=True,
        )

    else:
        st.warning(
            "The validation file is missing the required columns."
        )
else:
    st.info("No validation results are available.")

st.caption(
    "Metrics are based on synthetic validation data "
    "and do not establish real-world detection accuracy."
)

# ==================================================
# 10. Transaction explorer
# ==================================================

st.divider()
st.subheader("Transaction Explorer")

search = st.text_input(
    "Search by description, category, or transaction ID"
)

explorer = filtered.copy()

if search:
    mask = pd.Series(False, index=explorer.index)

    for column in ["description", "category", "transaction_id"]:
        if column in explorer.columns:
            mask |= (
                explorer[column]
                .astype(str)
                .str.contains(search, case=False, na=False)
            )

    explorer = explorer.loc[mask]

st.dataframe(
    explorer.sort_values("date", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# ==================================================
# 11. Download filtered transactions
# ==================================================

csv_data = explorer.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered transactions (CSV)",
    data=csv_data,
    file_name="filtered_transactions.csv",
    mime="text/csv",
)

st.caption("Personal Finance Advisor | Synthetic demonstration dataset")