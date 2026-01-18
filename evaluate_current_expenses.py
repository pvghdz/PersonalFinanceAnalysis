import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from utils.finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols
from utils.ntfy_utils import ntfy_format, ntfy_send
import requests
from utils.expense_analysis_utils import get_current_month_expenses, normalize_expenses, plot_expenses_vs_budget
from utils.mqtt_utils import publish_dataframe, publish_plot
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate this month's expenses against your defined budget."
    )

    parser.add_argument(
        "-m", "--method",
        choices=["n", "m", "none"],
        default="n",
        help="Method by which to communicate the results: 'n' to use ntfy, 'm' to use MQTT, 'none' to not transmit the results (default: n)"
    )

    parser.add_argument(
        "--url",
        default="https://ntfy.sh/pablo_budget",
        help="ntfy URL (required if --method is 'n'. Default: https://ntfy.sh/pablo_budget)"
    )

    parser.add_argument(
        "--mqtt-host",
        help="MQTT broker host (required if --method is 'm')"
    )

    parser.add_argument(
        "--spreadsheet",
        default="Gastos Pablo",
        help="Name of the spreadsheet in Google Drive (default: Gastos Pablo)"
    )

    parser.add_argument(
        "--credentials",
        default="config/service_account_keys.json",
        help="Path for the Google API credentials file (default: config/service_account_keys.json)"
    )

    return parser.parse_args()

def main() -> None:
    # Parse
    args = parse_arguments()
    
    if args.method == "none":
        args.method = None

    if args.method == "n" and not args.url:
        parser.error("--url is required when --method is 'n'")

    if args.method == "m" and not args.mqtt_host:
        parser.error("--mqtt-host is required when --method is 'm'")
    
    # Load budget
    print("Loading budget ...")
    try:
        budget_df = pd.read_json("results/monthly_budget.json", orient="index")
        normalized_budget_df = pd.read_json("results/normalized_budget.json", orient="index")
        budget_data = budget_df
        print("\nBudget found!")
        print("\nBudget:")
        print(budget_df)
        print("\nNormalized budget:")
        print(normalized_budget_df)
    except Exception as e:
        print("\n", e)

    spreadsheet_name = args.spreadsheet      # change if needed
    credentials = Path(args.credentials)

    try:
        print("\nDownloading current month expenses ...")
        monthly_expenses = get_current_month_expenses(spreadsheet_name, credentials, budget_data)
        normalized_expenses = normalize_expenses(monthly_expenses, budget_df)
        print("\nDownload successful!")
    except Exception as e:
        print("\n", e)

    if args.method == "n":
        print(f"\nExpense breakdown will be sent to {args.url}")
        message_monthly = ntfy_format(monthly_expenses, "€")
        message_normalized = ntfy_format(normalized_expenses, "%")
        ntfy_send(message_monthly, args.url)
        ntfy_send(message_normalized, args.url)
        print("\nDone.")
    
    elif args.method == "m":
        print(f"\nExpense breakdown will be sent to an MQTT broker at {args.mqtt-host}")
        # Publish DataFrames
        publish_dataframe(monthly_expenses, "monthly_expenses")
        publish_dataframe(normalized_expenses, "normalized_expenses")

        # Publish plot (using return_figure=True to get figure object for MQTT)
        fig = plot_expenses_vs_budget(monthly_expenses, budget_data, return_figure=True)
        publish_plot(fig, "plot/expenses_vs_budget")

    else:
        print("\nDone. Expense breakdown will not be transmitted anywhere.")

if __name__ == "__main__":
    main()