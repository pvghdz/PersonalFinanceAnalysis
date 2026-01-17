import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64

import paho.mqtt.publish as publish

from finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols

"""
This code evaluates the current expenses and sends the relevant data to a specific MQTT topic

Show the plots through MQTT by using Node-RED as an intermediary
"""

# ------------------------------------------------------------------
# MQTT CONFIG
# ------------------------------------------------------------------
MQTT_HOST = "<IP>"
MQTT_PORT = 1883
MQTT_USER = "<User>"
MQTT_PASSWORD = "<Pswd>"
MQTT_BASE_TOPIC = "finance"

# ------------------------------------------------------------------
# LOAD BUDGET DATA
# ------------------------------------------------------------------
budget_df = pd.read_json("results/monthly_budget.json", orient="index")
normalized_budget_df = pd.read_json("results/normalized_budget.json", orient="index")

BUDGET_DATA = budget_df

# ------------------------------------------------------------------
# HELPERS: MQTT PUBLISH
# ------------------------------------------------------------------
def publish_dataframe(df: pd.DataFrame, topic_suffix: str) -> None:
    payload = df.to_json(orient="split")
    publish.single(
        f"{MQTT_BASE_TOPIC}/{topic_suffix}",
        payload,
        hostname=MQTT_HOST,
        port=MQTT_PORT,
        auth={
            "username": MQTT_USER,
            "password": MQTT_PASSWORD,
        },
    )

def publish_plot(fig: plt.Figure, topic_suffix: str) -> None:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = base64.b64encode(buf.getvalue()).decode("utf-8")

    publish.single(
        f"{MQTT_BASE_TOPIC}/{topic_suffix}",
        payload,
        hostname=MQTT_HOST,
        port=MQTT_PORT,
        auth={
            "username": MQTT_USER,
            "password": MQTT_PASSWORD,
        },
    )

# ------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------
def get_current_month_expenses(spreadsheet_name: str, credentials_path: str | Path) -> pd.DataFrame:
    df = load_finance_dataframe(
        spreadsheet_name,
        credentials_path,
        worksheet_name="Enero 2026"
    )

    if df.empty:
        raise ValueError("No expense data was loaded.")

    now = datetime.now()
    current_month_df = df[
        (df["Fecha"].dt.year == now.year) &
        (df["Fecha"].dt.month == now.month)
    ]

    if current_month_df.empty:
        raise ValueError("No expenses found for the current month.")

    monthly_expenses = {
        col: current_month_df[col].sum()
        for col in numeric_cols
        if col in current_month_df.columns
    }

    monthly_expenses = pd.DataFrame(
        [monthly_expenses],
        index=["Current expenses"]
    ).reindex(
        columns=BUDGET_DATA.columns,
        fill_value=0.0,
    )

    return monthly_expenses

def normalize_expenses(
    monthly_expenses: pd.DataFrame,
    budget_df: pd.DataFrame
) -> pd.DataFrame:
    cols = monthly_expenses.columns.intersection(budget_df.columns)

    current = monthly_expenses.loc["Current expenses", cols]
    budget = budget_df.loc["Budget", cols].replace(0, pd.NA)

    normalized = current / budget

    return pd.DataFrame(
        [normalized],
        index=["Current expenses"]
    )

def plot_expenses_vs_budget(
    monthly_expenses: pd.DataFrame,
    budget_df: pd.DataFrame
) -> plt.Figure:
    categories = budget_df.columns

    expected = budget_df.loc["Expected expenses", categories]
    budget = budget_df.loc["Budget", categories]
    current = monthly_expenses.loc["Current expenses", categories]

    x = range(len(categories))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x, expected.values, width=width*1.3, label="Expected", alpha=0.35)
    ax.bar(x, budget.values, width=width*1.3, label="Budget", alpha=0.35)
    ax.bar(x, current.values, width=width, label="Current", alpha=1)

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("€")
    ax.set_title("Current Month Expenses vs Expected & Budget")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig

# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    SPREADSHEET_NAME = "Gastos Pablo"
    CREDENTIALS_PATH = Path("config/service_account_keys.json")

    monthly_expenses = get_current_month_expenses(
        SPREADSHEET_NAME,
        CREDENTIALS_PATH
    )

    normalized_expenses = normalize_expenses(
        monthly_expenses,
        budget_df
    )

    # Publish DataFrames
    publish_dataframe(monthly_expenses, "monthly_expenses")
    publish_dataframe(normalized_expenses, "normalized_expenses")

    # Publish plot
    fig = plot_expenses_vs_budget(monthly_expenses, BUDGET_DATA)
    publish_plot(fig, "plot/expenses_vs_budget")