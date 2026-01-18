import pandas as pd
try:
    from .finance_loader import load_finance_dataframe
except ImportError:
    # Fallback for when running as __main__
    from utils.finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols
import scipy.stats as stats
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils.logging_utils import setup_logging, format_dataframe_for_logging
from utils.plotting_utils import plot_expenses_with_outliers, plot_outlier_mask

logger = logging.getLogger(__name__)

def compute_outlier_statistics(absolute_matrix):
    """
    Calculate statistics (median, MAD, modified Z-score) for each category across months.
    
    Input:
        absolute_matrix: DataFrame with categories as rows, months as columns
        
    Output:
        Different DataFrame with statistics for each category
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

def remove_outliers_from_matrix(absolute_matrix, category_stats_df, outlier_threshold=2.0):
    """
    Remove outliers from expense matrix based on modified Z-score.
    
    Args:
        absolute_matrix: DataFrame with categories as rows, months as columns
        category_stats_df: DataFrame with statistics including modified Z-scores
        outlier_threshold: Threshold for modified Z-score (default 2.0)
        
    Returns:
        cleaned_matrix: DataFrame with outliers set to NaN
        outlier_mask: DataFrame indicating which values are outliers (True = outlier)
    """
    cleaned_matrix = absolute_matrix.copy()
    outlier_mask = pd.DataFrame(False, index=absolute_matrix.index, columns=absolute_matrix.columns)
    
    mod_z_cols = [col for col in category_stats_df.columns if col.startswith('mod_z_')]
    
    for category in absolute_matrix.index:
        for col in mod_z_cols:
            mod_z = category_stats_df.loc[category, col]
            if not pd.isna(mod_z) and abs(mod_z) > outlier_threshold:
                month = col.replace('mod_z_', '')
                # Mark as outlier
                outlier_mask.loc[category, month] = True
                # Set to NaN in cleaned matrix
                cleaned_matrix.loc[category, month] = np.nan
    
    return cleaned_matrix, outlier_mask

def no_outlier_dataframe(
    truncate: bool, 
    truncationDateDown: str,
    truncationDateUp: str,
    outlier_threshold: float = 2.0,
    setup_logs: bool = True,
    save_dir: str = 'results',
    debug: bool = True,
):
    """
    Load finance data, compute monthly category matrices, detect outliers
    using modified Z-scores, and return a cleaned matrix and outlier mask.

    Returns
    -------
    cleaned_absolute_matrix : pd.DataFrame
        Absolute expense matrix with outliers replaced by NaN
        (categories x months).

    outlier_mask : pd.DataFrame
        Boolean mask of detected outliers (True = outlier).
    """

    df = load_finance_dataframe(spreadsheet_name="Gastos Pablo", credentials_path="config/service_account_keys.json", worksheet_name=None) # None = examine all worksheets

    logger.info("=" * 60)
    logger.info("Starting outlier detection")
    logger.info("=" * 60)

    # Ensure Fecha is datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    if truncate:
        df = df[
            (df["Fecha"] > pd.to_datetime(truncationDateDown))
            & (df["Fecha"] < pd.to_datetime(truncationDateUp))
        ]

    months = df["sheet_name"].unique()
    logger.info(f"Dataframe truncated. Analyzing months: {months}")

    income_subtracted_expenses = 'Cantidad menos inversiones'
    expenses = [c for c in numeric_cols if c != "Ingreso"]
    expenses.insert(1,income_subtracted_expenses)

    monthly_results = {}

    for month in months:
        month_df = df[df["sheet_name"] == month]
        month_df.insert(month_df.columns.get_loc("Cantidad") + 1, income_subtracted_expenses, 0.0*len(df.index))
        logger.debug(f'{month} df head:\n{month_df.head()}')

        if month_df.empty:
            logger.warning(f'Month {month} has no data, skipping')
            continue

        # Debug: Check what we're summing
        if debug:
            logger.debug(f"\n=== Debug for {month} ===")
            logger.debug(f"Number of rows: {len(month_df)}")
            logger.debug(f"Sample Cantidad values (first 10): {month_df['Cantidad'].head(10).tolist()}")
            logger.debug(f"Cantidad sum: {month_df['Cantidad'].sum():.2f}")
            logger.debug(f"Cantidad mean: {month_df['Cantidad'].mean():.2f}")
            logger.debug(f"Are there duplicates? {month_df.duplicated().sum()} duplicate rows")
        
        # Calculate sums for expense categories
        monthly_sum = month_df[expenses].sum()
        monthly_sum[income_subtracted_expenses] = monthly_sum["Cantidad"] - monthly_sum["Inversiones"]
        
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

        logger.info(f"Month {month} analyzed - Income: {monthly_income:.2f}, Total expenses: {monthly_sum.sum():.2f}")

    absolute_matrix = pd.concat(
        {m: df["absolute"] for m, df in monthly_results.items()},
        axis=1
    )

    normalized_matrix = pd.concat(
        {m: df["normalized"] for m, df in monthly_results.items()},
        axis=1
    )

    # Calculate statistics for each category across months
    category_stats_df, category_stats_dict = compute_outlier_statistics(absolute_matrix)

    logger.info('\n'+'='*60)
    logger.info('ANALYSIS RESULTS')
    logger.info('='*60)

    # Transpose matrices for better readability (months as rows, categories as columns)
    # Round to 2 decimal places for cleaner output
    absolute_matrix_T = absolute_matrix.T.round(2)
    normalized_matrix_T = normalized_matrix.T.round(2)

    logger.info(f'\nExpense matrix (absolute) - Transposed (months x categories):')
    logger.info(f'{format_dataframe_for_logging(absolute_matrix_T, max_cols=15)}')
    logger.info(f'\nNormalized expense matrix - Transposed (months x categories):')
    logger.info(f'{format_dataframe_for_logging(normalized_matrix_T, max_cols=15)}')

    # For category statistics, separate the main stats from modified Z-scores
    main_stats = category_stats_df[['median', 'mad', 'mean', 'std', 'min', 'max']].round(2)
    mod_z_cols = [col for col in category_stats_df.columns if col.startswith('mod_z_')]

    logger.info(f'\nCategory Statistics - Main metrics:')
    logger.info(f'{main_stats.to_string()}')

    if mod_z_cols:
        mod_z_df = category_stats_df[mod_z_cols].round(2)
        logger.info(f'\nCategory Statistics - Modified Z-scores:')
        # mod_z_display = mod_z_df.iloc[:, :10] if len(mod_z_df.columns) > 10 else mod_z_df # Show only first 10 modified Z-score columns to avoid overwhelming output
        mod_z_display = mod_z_df
        logger.info(f'{format_dataframe_for_logging(mod_z_display.T, max_cols=20)}')
        #if len(mod_z_df.columns) > 10:
        #    logger.info(f'  ... (showing 10 of {len(mod_z_df.columns)} months)')

    # Identify and remove outliers
    logger.info('\n' + '='*60)
    logger.info(f'OUTLIER DETECTION (|modified Z-score| > {outlier_threshold}):')
    logger.info('='*60)

    outlier_count = 0
    outlier_details = []
    for category in absolute_matrix.index:
        mod_z_cols = [col for col in category_stats_df.columns if col.startswith('mod_z_')]
        for col in mod_z_cols:
            mod_z = category_stats_df.loc[category, col]
            if not pd.isna(mod_z) and abs(mod_z) > outlier_threshold:
                month = col.replace('mod_z_', '')
                value = absolute_matrix.loc[category, month]
                outlier_details.append((category, month, value, mod_z))
                logger.warning(f'  {category} in {month}: value={value:.2f}, mod_z={mod_z:.2f}')
                outlier_count += 1

    if outlier_count == 0:
        logger.info('  No outliers detected')
    else:
        logger.info(f'  Total outliers found: {outlier_count}')

    # Remove outliers and create clean matrices
    cleaned_absolute_matrix, outlier_mask = remove_outliers_from_matrix(
        absolute_matrix, category_stats_df, outlier_threshold
    )

    cleaned_normalized_matrix, outlier_mask = remove_outliers_from_matrix(
        normalized_matrix, category_stats_df, outlier_threshold
    )

    logger.info('\n' + '='*60)
    logger.info('CLEAN DATA (outliers removed):')
    logger.info('='*60)

    # Show cleaned matrix
    cleaned_matrix_T = cleaned_absolute_matrix.T.round(2)
    logger.info(f'\nClean expense matrix (absolute) - Transposed:')
    logger.info(f'{format_dataframe_for_logging(cleaned_matrix_T, max_cols=15)}')

    cleaned_norm_matrix_T = cleaned_normalized_matrix.T.round(2)
    logger.info(f'\nClean normalized matrix - Transposed:')
    logger.info(f'{format_dataframe_for_logging(cleaned_norm_matrix_T, max_cols=15)}')

    # Show outlier mask
    if outlier_count > 0:
        logger.info(f'\nOutlier mask (True = outlier, False = normal):')
        outlier_mask_T = outlier_mask.T
        logger.info(f'{format_dataframe_for_logging(outlier_mask_T, max_cols=15)}')

    logger.info('\n' + '='*60)
    logger.info('Analysis complete')
    logger.info('='*60)

    # Export cleaned matrix for further analysis
    logger.info(f'\Clean matrices available for statistical analysis:')
    logger.info(f'  - Use "cleaned_absolute_matrix" for analysis without outliers')
    logger.info(f'  - Use "cleaned_normalized_matrix" for a ratio-based analysis without outliers')
    logger.info(f'  - Use "cleaned_stats_df" for summary statistics on cleaned data\n')

    # Plot outlier mask
    plot_outlier_mask(outlier_mask_T, outlier_threshold, save_dir)
    plot_expenses_with_outliers(absolute_matrix_T, outlier_mask_T, save_dir)

    return cleaned_absolute_matrix, cleaned_normalized_matrix, outlier_mask 

if __name__ == "__main__":

    truncationDateDown = '2024-10-30'
    truncationDateUp = '2026-01-01'
    outlier_threshold = 2.0
    debug=False
    save_dir = 'results'

    from utils.logging_utils import setup_logging

    setup_logging(
        log_file="results/results.log",
        level=logging.INFO,
        mode="a",
    )

    cleaned_matrix, outlier_mask = no_outlier_dataframe(
        setup_logs=False
    )