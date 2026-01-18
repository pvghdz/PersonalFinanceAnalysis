import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from utils.finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols

def get_current_month_expenses(spreadsheet_name: str, credentials_path: str | Path, budget_data: pd.DataFrame) -> pd.DataFrame:
    """
    Download expenses and return total expenses per category for the current month.

    Output: df with:
        index   = ["Expenses"]
        columns = expense categories
    """

    df = load_finance_dataframe(spreadsheet_name, credentials_path, worksheet_name="Enero 2026")

    if df.empty:
        raise ValueError("No expense data was loaded.")

    now = datetime.now()
    current_month_df = df[(df["Fecha"].dt.year == now.year) & (df["Fecha"].dt.month == now.month)]

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
    monthly_expenses = monthly_expenses.reindex(columns=budget_data.columns, fill_value=0.0)

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

    print("\nNormalized expenses (Current / Budget):")
    print(normalized_expenses)

    return normalized_expenses

def plot_expenses_vs_budget(monthly_expenses: pd.DataFrame, budget_df: pd.DataFrame, return_figure: bool = True) -> plt.Figure | None:
    """
    Plot bar chart of actual expenses with
    expected expenses and budget as reference lines.
    
    Args:
        monthly_expenses: DataFrame with expenses
        budget_df: DataFrame with budget data
        return_figure: If True, return the figure instead of showing it. Default: True
    
    Returns:
        plt.Figure if return_figure=True, None otherwise
    """
    categories = budget_df.columns

    expected = budget_df.loc["Expected expenses", categories]
    budget = budget_df.loc["Budget", categories]
    current = monthly_expenses.loc["Expenses", categories]

    x = range(len(categories))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 6))

    # Expected
    ax.bar(x, expected.values, width=width*1.3, label="Expected expenses", alpha=0.35)

    # Budget
    ax.bar(x, budget.values, width=width*1.3, label="Budget", alpha=0.35)

    # Current
    ax.bar(x, current.values, width=width, label="Expenses", alpha=1)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("€")
    ax.set_title("Current Month Expenses vs Expected & Budget")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    
    if return_figure:
        return fig
    else:
        plt.show()
        return None