import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import re
from config.numeric_cols import numeric_cols
from pathlib import Path
import logging
from utils.finance_loader_utils import clean_single_worksheet, extract_income, parse_euro_number
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# HELPERS FOR DEGUBBING - USEFUL TO WORK WITH DOWNLOADED EXCEL SHEETS
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

def load_finance_dataframe(spreadsheet_name: str, credentials_path: str | Path):
    """
    Load all worksheets from a Google Spreadsheet and return a clean DataFrame.
    Input: Spreadsheet name and credentials to open it
    Output: DataFrame with clean data, containing all worksheets in a single object.
    """

    # AUTHENTICATION & DATA DOWNLOAD

    logger.info("=" * 60)
    logger.info("Starting data download")
    logger.info("=" * 60)
    logger.info("\nAttempting to download data from Google Spreadsheets API")

    try:
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
    
    except Exception as e:
        print(e)
        logger.warning(e)

    # DEBUG: load local Excel instead of Google Sheets
    #spreadsheet = _LocalSpreadsheet("data/Gastos Pablo.xlsx")
    #print("Successfully read local spreadsheet.\n")

    # CLEAN DATA
    dfs = []

    for ws in spreadsheet.worksheets():
        df_ws = clean_single_worksheet(ws)
        if not df_ws.empty:
            dfs.append(df_ws)

    logger.info("\nSpreadsheets downloaded and cleaned!\n")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = parse_euro_number(df[col])

    # Parse dates
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

    print("Data was parsed successfully!")
    logger.info("\nData was parsed successfully!\n")
    return df