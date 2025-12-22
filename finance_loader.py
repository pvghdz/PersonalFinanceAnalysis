import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import re
from config.numeric_cols import numeric_cols
from pathlib import Path

# ----------------------------
# HELPERS
# ----------------------------
def parse_euro_number(series: pd.Series) -> pd.Series:
    """
    Convert European formatted numbers like:
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
    return pd.to_numeric(cleaned, errors="coerce")

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

def _clean_single_worksheet(ws) -> pd.DataFrame:
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

class _LocalWorksheet:
    def __init__(self, name, df):
        self.title = name
        self._df = df

    def get_all_values(self):
        return [list(self._df.columns)] + self._df.astype(str).fillna("").values.tolist()


class _LocalSpreadsheet:
    def __init__(self, path):
        self._sheets = pd.read_excel(path, sheet_name=None)

    def worksheets(self):
        return [
            _LocalWorksheet(name, df)
            for name, df in self._sheets.items()
        ]

# ----------------------------
# PUBLIC API
# ----------------------------
def load_finance_dataframe(
    spreadsheet_name: str,
    credentials_path: str | Path,
) -> pd.DataFrame:
    """
    Load all worksheets from a Google Spreadsheet and return a clean DataFrame.
    Output: DataFrame with clean data, containing all worksheets in a single object.
    """
    # ----------------------------
    # AUTHENTICATION & DATA DOWNLOAD
    # ----------------------------
    
    scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

    creds = Credentials.from_service_account_file(
        credentials_path,
        scopes=scopes,
    )
    client = gspread.authorize(creds)
    spreadsheet = client.open(spreadsheet_name)
    
# DEBUG: load local Excel instead of Google Sheets
    #spreadsheet = _LocalSpreadsheet("data/Gastos Pablo.xlsx")
    #print("Successfully read local spreadsheet.\n")

    # ----------------------------
    # CLEAN DATA
    # ----------------------------
    dfs = []

    for ws in spreadsheet.worksheets():
        df_ws = _clean_single_worksheet(ws)
        if not df_ws.empty:
            dfs.append(df_ws)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    """
    numeric_cols = [
            "Cantidad",
            "Suma",
            "Alimentos",
            "Recurrentes",
            "Extraordinarios",
            "Salidas",
            "Viajes",
            "Otros/Únicos",
            "Inversiones",
            "Ingreso",
        ]
    """

    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_euro_number(df[col])

    # Parse dates
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

    print("Data was parsed successfully!")
    return df