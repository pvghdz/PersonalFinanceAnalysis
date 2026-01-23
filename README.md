# Personal Finance Analysis

A personal finance analysis toolkit that downloads expense data from Google Sheets, performs statistical analysis, detects outliers, and generates monthly budgets. The system supports HTTP/ntfy and MQTT notification methods for expense monitoring.

MQTT + NodeRED can be used to support transmitting plots. 

## Overview

This project provides tools to:
- Download and clean financial data from Google Sheets (see sample in the Examples folder)
- Detect outliers in expense categories using statistical methods (functions are to be added to support Excel files)
- Generate monthly budgets based on historical data using a modified z-score
- Monitor current expenses against budgets
- Send notifications via HTTP (ntfy.sh) or MQTT

## Scripts

### `utils/finance_loader.py`

**Purpose**: Data loading module that connects to Google Sheets and retrieves financial data.

**Functionality**:
- Authenticates with Google Sheets API using service account credentials
- Downloads data from specified worksheets (or all worksheets)
- Cleans and parses European-formatted numbers (e.g., "1.097,96 €" → 1097.96)
- Handles date parsing and data normalization
- Returns a unified DataFrame containing all expense data

**Key Function**: `load_finance_dataframe(spreadsheet_name, credentials_path, worksheet_name=None)`

**Dependencies**: `gspread`, `google-auth`, `pandas`

**Import**: `from utils.finance_loader import load_finance_dataframe`

---

### `utils/detect_outliers.py`

**Purpose**: Detects outliers in expense categories using modified Z-scores based on Median Absolute Deviation (MAD).

**Functionality**:
- Loads financial data and organizes it into monthly category matrices (absolute and normalized)
- Computes statistics (median, MAD, mean, std, min, max) for each expense category
- Calculates modified Z-scores for each category-month combination
- Identifies outliers based on a configurable threshold (default: |modified Z-score| > 2.0)
- Generates cleaned matrices with outliers removed (set to NaN)
- Creates visualizations showing expenses with outliers highlighted and outlier masks
- Logs detailed analysis results

**Key Function**: `no_outlier_dataframe(truncate, truncationDateDown, truncationDateUp, outlier_threshold, setup_logs, save_dir, debug)`

**Outputs**:
- Cleaned absolute expense matrix
- Cleaned normalized expense matrix
- Outlier mask (boolean DataFrame)
- Visualization plots saved to `results/` directory

**Dependencies**: `pandas`, `numpy`, `scipy`, `matplotlib`

**Import**: `from utils.detect_outliers import no_outlier_dataframe`

---

### `statistical_analysis.py`

**Purpose**: Main analysis script that orchestrates outlier detection, computes statistics, and generates monthly budgets.

**Functionality**:
- Runs outlier detection on historical expense data
- Computes statistical summaries (mean, std, median, MAD, min, max) on cleaned data
- Generates monthly budgets based on statistical analysis:
  - **Expected expenses**: Median value for each category
  - **Budget**: Median + MAD (Median Absolute Deviation) for each category
- Supports date truncation to analyze specific time periods
- Saves budgets as JSON files (`monthly_budget.json`, `normalized_budget.json`)
- Provides command-line interface with configurable parameters

**Command-line Arguments**:
- `--tdown`: Lower truncation date (default: "2024-10-30")
- `--tup`: Upper truncation date (default: "2026-01-01")
- `--outlier-threshold`: Modified Z-score threshold (default: 2.0, recommended: 3.5)
- `--save-dir`: Output directory (default: "results")
- `-l, --log-mode`: Log file mode - 'w' to overwrite, 'a' to append (default: 'w')

**Usage Example**:
```bash
python statistical_analysis.py -l w --tdown 2024-11-01 --tup 2025-12-31 --outlier-threshold 2.0
```

**Dependencies**: `pandas`, `numpy`, `utils.detect_outliers`

---

### `evaluate_current_expenses.py`

**Purpose**: Evaluates current month expenses against the generated budget and provides comparison analysis.

**Functionality**:
- Downloads current month's expense data from Google Sheets
- Loads budget data from `results/monthly_budget.json` and `results/normalized_budget.json`
- Calculates total expenses per category for the current month
- Normalizes expenses as a fraction of the budget (Current / Budget)
- Optionally plots a bar chart comparing current expenses vs expected expenses vs budget
- Prints detailed expense summaries

**Key Functions**:
- `get_current_month_expenses(spreadsheet_name, credentials_path)`: Retrieves current month expenses
- `normalize_expenses(monthly_expenses, budget_df)`: Normalizes expenses against budget
- `plot_expenses_vs_budget(monthly_expenses, budget_df)`: Creates visualization

**Dependencies**: `pandas`, `matplotlib`, `utils.finance_loader`, `utils.expense_analysis_utils`

**Command-line Arguments**:
- `-m, --method`: How to transmit results — `"n"` for ntfy, `"m"` for MQTT, `"none"` to only print locally (default: `"n"`)
- `--url`: ntfy URL when using `--method n` (default: `https://ntfy.sh/pablo_budget`)
- `--mqtt-host`: MQTT broker host when using `--method m`
- `--spreadsheet`: Google Sheets spreadsheet name (default: `"Gastos Pablo"`)
- `--credentials`: Path to the Google API credentials JSON file (default: `config/service_account_keys.json`)

**Usage Examples**:

Send the current month expense breakdown via **ntfy** (default method):

```bash
python evaluate_current_expenses.py -m n --url https://ntfy.sh/pablo_budget --spreadsheet "Gastos Pablo" --credentials config/service_account_keys.json
```

Send the current month expense breakdown via **MQTT**:

```bash
python evaluate_current_expenses.py -m m --mqtt-host <IP> --spreadsheet "Gastos Pablo" --credentials config/service_account_keys.json
```

Run the evaluation **without transmitting** results anywhere (just prints to the console):

```bash
python evaluate_current_expenses.py -m none
```

---

### `utils/mqtt_utils.py`

**Purpose**: Evaluates current expenses and publishes data via MQTT for integration with home automation systems (e.g., Node-RED).

**Functionality**:
- Performs expense evaluation using functions from `utils.expense_analysis_utils`
- Publishes expense data as JSON to MQTT topics:
  - `finance/monthly_expenses`: Current month expenses
  - `finance/normalized_expenses`: Normalized expenses
  - `finance/plot/expenses_vs_budget`: Base64-encoded plot image
- Enables integration with MQTT-based dashboards and automation systems

**Key Functions**:
- `publish_dataframe(df, topic_suffix)`: Publishes DataFrame as JSON to MQTT
- `publish_plot(fig, topic_suffix)`: Publishes matplotlib figure as base64-encoded image

**Configuration**:
- MQTT connection parameters are loaded from `config/mqtt_config.py`
- Edit `config/mqtt_config.py` to set: `mqtt_host`, `mqtt_port`, `mqtt_user`, `mqtt_password`, `mqtt_base_topic`

**Dependencies**: `pandas`, `matplotlib`, `paho-mqtt`, `utils.finance_loader`, `utils.expense_analysis_utils`, `config.mqtt_config`

**Usage**: Can be run as a script (if `__main__` block is present) or imported as a module

---

### Utility Modules (`utils/`)

The `utils/` folder contains reusable modules:

- **`finance_loader.py`**: Core data loading functionality (see above)
- **`detect_outliers.py`**: Outlier detection functionality (see above)
- **`expense_analysis_utils.py`**: Shared functions for expense analysis (`get_current_month_expenses`, `normalize_expenses`, `plot_expenses_vs_budget`)
- **`mqtt_utils.py`**: MQTT publishing utilities (see above)
- **`ntfy_utils.py`**: HTTP notification utilities for ntfy.sh
- **`logging_utils.py`**: Logging configuration and formatting utilities
- **`plotting_utils.py`**: Plotting utilities for expense visualizations
- **`finance_loader_utils.py`**: Helper functions for data cleaning and parsing

---

## Configuration

### Required Files

- `config/service_account_keys.json`: Google service account credentials for accessing Google Sheets
- `config/numeric_cols.py`: Defines which columns contain numeric data
- `config/mqtt_config.py`: MQTT connection configuration (if using MQTT features)
- `data/Gastos Pablo.xlsx`: Local backup of expense data (optional)

### Output Files

All results are saved to the `results/` directory:
- `monthly_budget.json`: Monthly budget in absolute values
- `normalized_budget.json`: Monthly budget normalized by income
- `expenses_with_outliers.png`: Visualization of expenses with outliers highlighted
- `outlier_mask.png`: Visualization of detected outliers
- `results.log`: Detailed analysis logs

## Installation

1. Install dependencies:
```bash
pip install -r config/requirements.txt
```

2. Set up Google Sheets API credentials:
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Place it in `config/service_account_keys.json`
   - Share your Google Sheet with the service account email

3. Configure notification endpoints (if using MQTT):
   - Edit `config/mqtt_config.py` to set your MQTT connection parameters:
     - `mqtt_host`: MQTT broker hostname or IP
     - `mqtt_port`: MQTT broker port (default: 1883)
     - `mqtt_user`: MQTT username
     - `mqtt_password`: MQTT password
     - `mqtt_base_topic`: Base topic for publishing (default: "finance")

## How to use / Workflow

1. **Generate Budgets**: Run `statistical_analysis.py` to analyze historical data and generate budgets
2. **Monitor Expenses**: Run `evaluate_current_expenses.py` to check current month spending
3. **Review Outliers**: The outlier detection is automatically performed by `statistical_analysis.py`, or you can use `utils/detect_outliers.py` directly
4. **MQTT Publishing**: Run `utils/mqtt_utils.py` (if configured) to publish expense data to MQTT topics

## Dependencies

See `requirements.txt` for the complete list. Key dependencies include:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `matplotlib`: Plotting and visualization
- `gspread`: Google Sheets API client
- `paho-mqtt`: MQTT client (for MQTT variant)
- `requests`: HTTP client (for HTTP variant)
