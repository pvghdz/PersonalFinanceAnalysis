import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import re
from config.numeric_cols import numeric_cols
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def parse_euro_number(series: pd.Series):
    """
    Convert European formatted numbers to float64
    '1.097,96 €' -> 1097.96
    """
    cleaned = (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce") # ‘coerce’ =  invalid parsing will be set as NaN.

def extract_income(values, header):
    found_income = None

    for row in values:
        if not row or not row[0]:
            continue

        if row[0].strip().lower() == "ingreso":
            for col_name in ["Concepto", "Vendedor"]:
                if col_name in header:
                    col_idx = header.index(col_name)
                    raw_val = row[col_idx]
                    val = parse_euro_number(pd.Series([raw_val])).iloc[0]
                    val_notparsed = pd.Series([raw_val]).iloc[0]
                    if pd.isna(val):
                        continue

                    if found_income is None:
                        found_income = val_notparsed
                        print(f"Found income: {val_notparsed}")
                    elif val != found_income:
                        raise ValueError(
                            f"Conflicting income values found: {found_income} vs {val}"
                        )

    return found_income if found_income is not None else np.nan

def clean_single_worksheet(ws):
    """
    Load and clean a single worksheet.
    Output: Dataframe without blank rows or whitespaces. Fecha column is filled forward.
    """
    values = ws.get_all_values()
    if len(values) < 2:
        return pd.DataFrame()

    header = [h.strip() for h in values[0]] # .strip() removes leading and trailing whitespace
    rows = values[1:]

    income = extract_income(values, header) # extract income before truncation
    if income == 0:
        raise ValueError(f"Income is zero in worksheet '{ws.title}'")

    df = pd.DataFrame(rows, columns=header)
    df = df.dropna(how="all")

    if "Fecha" in df.columns:
        df["Fecha"] = df["Fecha"].replace("", np.nan).ffill()

        # Drop everything after summary
        # Future improvement: check that all three summary numbers are the same to validate the integrity of the data.
        summary_idx = df[df["Fecha"].str.contains("Total mensual", na=False)].index
        if len(summary_idx) > 0:
            df = df.loc[: summary_idx[0] - 1]

    # Drop blank "spacer" rows. Spacer rows are rows with empty "Cantidad" column.
    if "Cantidad" in df.columns:
        df = df[df["Cantidad"].notna() & (df["Cantidad"].astype(str) != "")]

    df["sheet_name"] = ws.title

    df["Ingreso"] = pd.Series(dtype="object")
    if len(df) > 0:
        df.loc[df.index[0], "Ingreso"] = income

    # DEBUG CHECK
    incomes = df["Ingreso"].dropna().unique()
    if len(incomes) != 1:
        raise ValueError(
            f"Worksheet '{ws.title}' has {len(incomes)} income values: {incomes}"
        )

    return df
