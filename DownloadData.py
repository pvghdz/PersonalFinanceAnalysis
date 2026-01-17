import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
import numpy as np

# ----------------------------
# AUTHENTICATION
# ----------------------------
"""
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_file(
    "ServiceAccountKeys/analisisfinanzaspersonales-50c60b68412d.json",
    scopes=scope
)
client = gspread.authorize(creds)

spreadsheet = client.open("Gastos Pablo")

print([ws.title for ws in spreadsheet.worksheets()])
"""
spreadsheet = pd.read_excel("/data/Gastos Pablo")
print("Successfully read Spreadsheet.\n")
# ----------------------------
# HELPERS
# ----------------------------
def looks_like_date(value: str) -> bool:
    """Keep rows where Fecha looks like DD/MM/YYYY."""
    if not value:
        return False
    return bool(re.match(r"\d{1,2}/\d{1,2}/\d{4}", value))

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
    )
    # Replace empty strings with NaN, then convert to numeric
    cleaned = cleaned.replace("", pd.NA)
    
    return pd.to_numeric(cleaned, errors='coerce')

# ----------------------------
# LOAD ALL SHEETS
# ----------------------------
dfs = []

for ws in spreadsheet.worksheets():
    values = ws.get_all_values()
    if len(values) < 2:
        continue

    header = values[0]
    rows = values[1:]

    df = pd.DataFrame(rows, columns=header)

    # Drop completely empty rows
    df = df.dropna(how="all")

    if "Fecha" in df.columns:
        # 1. Replace empty strings with NaN
        df["Fecha"] = df["Fecha"].replace("", np.nan)

        # 2. Forward-fill merged cells
        df["Fecha"] = df["Fecha"].ffill()

        # 3. Find the index of the "Total mensual" row in Fecha column to drop end-of-month summary
        total_index = df[df["Fecha"].str.contains("Total mensual", na=False)].index
        if len(total_index) > 0:
            df = df.loc[:total_index[0]-1]  # keep all rows *before* summary

    # Drop rows where Cantidad is empty or """
    df = df[df["Cantidad"].notna() & (df["Cantidad"].astype(str) != "")]

    df["sheet_name"] = ws.title  # optional but very useful
    dfs.append(df)

# ----------------------------
# COMBINE EVERYTHING
# ----------------------------
df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# NUMERIC CLEANUP
# ----------------------------
for col in ["Cantidad", "Suma", "Alimentos", "Recurrentes", "Extraordinarios", "Salidas", "Viajes", "Otros/Únicos", "Inversiones"]:
    if col in df.columns:
        df[col] = parse_euro_number(df[col])

# Optional: parse Fecha as datetime
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

print(df.head())
print(df.info())

#df.to_excel("data/finanzas_personales.xlsx", index=False)

