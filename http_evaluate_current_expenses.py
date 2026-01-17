import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols
from utils.ntfy_utils import ntfy_format, ntfy_send
import requests

# ------------------------------------------------------------------
# IMPORT YOUR EXISTING HELPERS
# ------------------------------------------------------------------
# from your_module import load_finance_dataframe
# from your_module import logger

budget_df = pd.read_json("results/monthly_budget.json", orient="index")
normalized_budget_df = pd.read_json("results/normalized_budget.json", orient="index")

# ------------------------------------------------------------------
# BUDGET DEFINITION (from your printed output)
# ------------------------------------------------------------------
BUDGET_DATA = budget_df
print("Budget data:")
print(BUDGET_DATA)

# ------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------
def get_current_month_expenses(spreadsheet_name: str, credentials_path: str | Path) -> pd.DataFrame:
    """
    Download expenses and return total expenses per category for the current month.

    Output: df with:
        index   = ["Current expenses"]
        columns = expense categories
    """

    df = load_finance_dataframe(spreadsheet_name, credentials_path, worksheet_name="Enero 2026")

    if df.empty:
        raise ValueError("No expense data was loaded.")

    now = datetime.now()
    current_month_df = df[
        (df["Fecha"].dt.year == now.year) &
        (df["Fecha"].dt.month == now.month)
    ]

    if current_month_df.empty:
        raise ValueError("No expenses found for the current month.")

    # Sum by category (columns)
    monthly_expenses = {
        col: current_month_df[col].sum()
        for col in numeric_cols
        if col in current_month_df.columns
    }

    monthly_expenses = pd.DataFrame([monthly_expenses], index=["Expenses"])

    # Ensure same column order & fill missing
    monthly_expenses = monthly_expenses.reindex(
        columns=BUDGET_DATA.columns,
        fill_value=0.0,
    )

    print("\nCurrent expenses:")
    print(monthly_expenses.head())

    return monthly_expenses

def normalize_expenses(monthly_expenses: pd.DataFrame, budget_df: pd.DataFrame) -> pd.DataFrame:
    """
    Express current monthly expenses as a fraction of the budget.
    """

    # Align columns explicitly
    cols = monthly_expenses.columns.intersection(budget_df.columns)

    current = monthly_expenses.loc["Expenses", cols]
    budget = budget_df.loc["Budget", cols]

    # Avoid division by zero
    budget = budget.replace(0, pd.NA)

    normalized = current / budget * 100

    # Restore DataFrame shape
    normalized_expenses = pd.DataFrame(
        [normalized],
        index=["Expenses"]
    )

    print("\nNormalized expenses (Expenses / Budget):")
    print(normalized_expenses)

    return normalized_expenses

def plot_expenses_vs_budget(monthly_expenses: pd.DataFrame, budget_df: pd.DataFrame) -> None:
    """
    Plot bar chart of actual expenses with
    expected expenses and budget as reference lines.
    """
    categories = budget_df.columns

    expected = budget_df.loc["Expected expenses", categories]
    budget = budget_df.loc["Budget", categories]
    current = monthly_expenses.loc["Expenses", categories]

    x = range(len(categories))
    width = 0.6

    plt.figure(figsize=(14, 6))

    # Expected
    plt.bar(x, expected.values, width=width*1.3, label="Expected expenses", alpha=0.35)

    # Budget
    plt.bar(x, budget.values, width=width*1.3, label="Budget", alpha=0.35)

    # Current
    plt.bar(x, current.values, width=width, label="Current expenses", alpha=1)

    plt.xticks(x, categories, rotation=45, ha="right")
    plt.ylabel("€")
    plt.title("Current Month Expenses vs Expected & Budget")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    spreadsheet_name = "Gastos Pablo"      # change if needed
    credentials = Path("config/service_account_keys.json")
    ntfy_url="https://ntfy.sh/<url>"

    monthly_expenses = get_current_month_expenses(spreadsheet_name, credentials)
    normalized_expenses = normalize_expenses(monthly_expenses, budget_df)
    message_monthly = ntfy_format(monthly_expenses, "€")
    message_normalized = ntfy_format(normalized_expenses, "%")
    ntfy_send(message_monthly, ntfy_url)
    ntfy_send(message_normalized, ntfy_url)
    #plot_expenses_vs_budget(monthly_expenses, BUDGET_DATA)
