import pandas as pd
from finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols
import scipy.stats as stats
import numpy as np

debug=False

def _compute_mad(month_df):
    """
    Calculate MAD (Median Absolute Deviation) for a single month's transactions.
    
    Args:
        month_df: DataFrame containing transactions for a single month
        
    Returns:
        dict with median, mad, outliers, normal_transactions, and budget_baseline
    """
    # Get individual Cantidad values for this month (not monthly totals)
    amounts = month_df["Cantidad"].astype(float).dropna()
    
    if len(amounts) == 0:
        return {
            'median': None,
            'mad': None,
            'outliers': pd.Series(dtype=float),
            'normal_transactions': pd.Series(dtype=float),
            'budget_baseline': None
        }
    
    median = amounts.median()
    mad = (amounts - median).abs().median()
    
    # Identify outliers using modified Z-score (2.5 * MAD threshold)
    outliers = amounts[(amounts - median).abs() > 2.5 * mad]
    normal_transactions = amounts.drop(outliers.index)
    
    # Calculate budget baseline from non-outlier transactions
    budget_baseline = normal_transactions.mean() if len(normal_transactions) > 0 else median
    
    return {
        'median': median,
        'mad': mad,
        'outliers': outliers,
        'normal_transactions': normal_transactions,
        'budget_baseline': budget_baseline
    }

def compute_category_statistics(absolute_matrix):
    """
    Calculate statistics (median, MAD, modified Z-score) for each category across months.
    
    Args:
        absolute_matrix: DataFrame with categories as rows, months as columns
        
    Returns:
        DataFrame with statistics for each category
    """
    stats_dict = {}
    
    for category in absolute_matrix.index:
        values = absolute_matrix.loc[category].dropna()
        
        if len(values) == 0:
            stats_dict[category] = {
                'median': np.nan,
                'mad': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan
            }
            continue
        
        median = values.median()
        mad = (values - median).abs().median()
        mean = values.mean()
        std = values.std()
        
        # Calculate modified Z-score for each month
        modified_z_scores = []
        for month in absolute_matrix.columns:
            val = absolute_matrix.loc[category, month]
            if pd.isna(val) or mad == 0:
                modified_z_scores.append(np.nan)
            else:
                # Modified Z-score = 0.6745 * (value - median) / MAD
                mod_z = 0.6745 * (val - median) / mad
                modified_z_scores.append(mod_z)
        
        stats_dict[category] = {
            'median': median,
            'mad': mad,
            'mean': mean,
            'std': std,
            'min': values.min(),
            'max': values.max(),
            'modified_z_scores': pd.Series(modified_z_scores, index=absolute_matrix.columns)
        }
    
    # Create summary DataFrame
    stats_df = pd.DataFrame({
        'median': [stats_dict[cat]['median'] for cat in absolute_matrix.index],
        'mad': [stats_dict[cat]['mad'] for cat in absolute_matrix.index],
        'mean': [stats_dict[cat]['mean'] for cat in absolute_matrix.index],
        'std': [stats_dict[cat]['std'] for cat in absolute_matrix.index],
        'min': [stats_dict[cat]['min'] for cat in absolute_matrix.index],
        'max': [stats_dict[cat]['max'] for cat in absolute_matrix.index]
    }, index=absolute_matrix.index)
    
    # Add modified Z-scores as separate columns (one per month)
    for category in absolute_matrix.index:
        mod_z_series = stats_dict[category]['modified_z_scores']
        for month in mod_z_series.index:
            col_name = f'mod_z_{month}'
            stats_df.loc[category, col_name] = mod_z_series[month]
    
    return stats_df, stats_dict 

truncation = True
truncationDate = '2024-10-30'
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

df = load_finance_dataframe(
    spreadsheet_name="Gastos Pablo",
    credentials_path="ServiceAccountKeys/analisisfinanzaspersonales-50c60b68412d.json",
)

print('------------------------')

print('\n',df.head(),'\n')
print(df.info(),'\n')

# Only analyze months after October 2024
if truncation:
    df = df[df['Fecha'] > truncationDate] # filter dataframe

months = df["sheet_name"].unique()
print(f'Dataframe truncated. Analyzing months: {months}\n')

#expense_matrix = {}
#norm_expense_matrix = {}
#expense_matrix['Order'] = numeric_cols
#norm_expense_matrix['Order'] = numeric_cols
#month_df = df[df["sheet_name"] == months]
expenses = [c for c in numeric_cols if c != "Ingreso"]
monthly_results = {}

for month in months:
    month_df = df[df["sheet_name"] == month]

    print(f'{month} df = \n{month_df.head()}')

    if month_df.empty:
        continue

    # Debug: Check what we're summing
    if debug:
        print(f"\n=== Debug for {month} ===")
        print(f"Number of rows: {len(month_df)}")
        print(f"Sample Cantidad values (first 10):")
        print(month_df["Cantidad"].head(10).tolist())
        print(f"Cantidad sum: {month_df['Cantidad'].sum():.2f}")
        print(f"Cantidad mean: {month_df['Cantidad'].mean():.2f}")
        print(f"Are there duplicates? {month_df.duplicated().sum()} duplicate rows")
    
    # Calculate sums for expense categories
    monthly_sum = month_df[expenses].sum()
    
    # Get income (should be single value in first row)
    monthly_income = month_df["Ingreso"].iloc[0] if len(month_df) > 0 else np.nan
    
    # Add Ingreso to the monthly_sum Series
    monthly_sum_with_income = monthly_sum.copy()
    monthly_sum_with_income["Ingreso"] = monthly_income

    # Calculate normalized values
    if pd.isna(monthly_income) or monthly_income == 0:
        monthly_norm = monthly_sum * np.nan
        monthly_norm["Ingreso"] = 1.0  # Ingreso normalized to itself
    else:
        monthly_norm = monthly_sum / monthly_income
        monthly_norm["Ingreso"] = 1.0  # Ingreso normalized to itself

    summary_df = pd.DataFrame({
        "absolute": monthly_sum_with_income,
        "normalized": monthly_norm
    })

    monthly_results[month] = summary_df

    print(f"Month {month} analyzed.")

absolute_matrix = pd.concat(
    {m: df["absolute"] for m, df in monthly_results.items()},
    axis=1
)

normalized_matrix = pd.concat(
    {m: df["normalized"] for m, df in monthly_results.items()},
    axis=1
)

# Calculate statistics for each category across months
category_stats_df, category_stats_dict = compute_category_statistics(absolute_matrix)

print(f'\n{"="*60}')
print(f'Expense matrix (absolute):\n{absolute_matrix}')
print(f'\n{"="*60}')
print(f'Normalized expense matrix:\n{normalized_matrix}')
print(f'\n{"="*60}')
print(f'Category Statistics:\n{category_stats_df}')
print(f'\n{"="*60}')

# Identify outliers using modified Z-score threshold (e.g., |mod_z| > 2.5)
print('\nOutliers (|modified Z-score| > 2.5):')
outlier_threshold = 2.5
for category in absolute_matrix.index:
    mod_z_cols = [col for col in category_stats_df.columns if col.startswith('mod_z_')]
    for col in mod_z_cols:
        mod_z = category_stats_df.loc[category, col]
        if not pd.isna(mod_z) and abs(mod_z) > outlier_threshold:
            month = col.replace('mod_z_', '')
            value = absolute_matrix.loc[category, month]
            print(f'  {category} in {month}: value={value:.2f}, mod_z={mod_z:.2f}')