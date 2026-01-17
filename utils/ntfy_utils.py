import requests
import pandas as pd
from datetime import datetime

def ntfy_format(df: pd.DataFrame, units: str = "€") -> str:
    """
    Format expenses DataFrame as a Markdown message for ntfy.
    """
    row = df.loc["Expenses"]

    lines = [
        "**Current Month Expenses**",
        "",
        "| Category | Amount |",
    ]

    for category, value in row.items():
        lines.append(f"| {category} | {value:,.2f} {units} |")

    return "\n".join(lines)

def ntfy_send(message: str, url: str):
    """
    Sends ntfy message to a specified url
    """
    print(f"\nAttempting to send ntfy message to \n{url}\n")

    if not url:                     # guard if URL not provided
        print("No url found\n")
        print(message)
        return
    try:       
        print("url found!")                               # avoid crashing on network errors
        requests.post(url, data=message.encode(), timeout=5)
        print("ntfy message sent.")
    except Exception:
        pass