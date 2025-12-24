import pandas as pd
from finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols
import scipy.stats as stats
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from detect_outliers import no_outlier_dataframe
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def compute_cleaned_statistics(cleaned_matrix):
    """
    Compute statistics on cleaned matrix (without outliers).
    
    Args:
        cleaned_matrix: DataFrame with outliers removed (set to NaN)
        
    Returns:
        DataFrame with statistics for cleaned data
    """
    stats_dict = {}
    
    for category in cleaned_matrix.index:
        values = cleaned_matrix.loc[category].dropna()
        
        if len(values) == 0:
            stats_dict[category] = {
                'mean': np.nan,
                'stdev': np.nan,
                'median': np.nan,
                'mad': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
            continue

        median = values.median()
        
        stats_dict[category] = {
            'mean': values.mean(),
            'stdev': values.std(),
            'median': median,
            'mad': (values - median).abs().median(),
            'min': values.min(),
            'max': values.max(),
            'count': len(values)
        }
    
    return pd.DataFrame(stats_dict).T

def produce_budget(statistics_dataframe):
    budget = {}

    for category, row in statistics_dataframe.iterrows():
        if category != "Ingreso":
            budget[category] = {
                'Expected expenses': row['median'],
                'Budget': row['median'] + row['mad'],
            }
        else:
            continue

    return pd.DataFrame(budget)

def main(produce_log='w'):
    setup_logging(
        log_file="results/results.log",
        level=logging.INFO,
        mode=produce_log,   # append to same log
    )

    logger.info('\n'+"#" * 60)
    logger.info(f'{datetime.now().replace(microsecond=0)}: LOGGING STARTED')
    if produce_log == 'w':
        logger.info('Current setup: re-write previous log files')
    else:
        logger.info('Current setup: append to previous log files')
    logger.info("#" * 60)
    #logger.info('\n'+"=" * 60)
    #logger.info("STARTING STATISTICAL ANALYSIS")
    #logger.info("=" * 60)

    truncationDateDown = '2024-10-30'
    truncationDateUp = '2026-01-01'
    outlier_threshold = 2.0
    save_dir = 'results'

    clean_matrix, clean_norm_matrix, outlier_mask = no_outlier_dataframe(
        True,                       # truncate analysis dates? True = yes
        truncationDateDown,
        truncationDateUp,
        outlier_threshold,
        setup_logs=False,   # VERY IMPORTANT
        save_dir=save_dir,
        debug = True,
    )

    statistics_matrix = compute_cleaned_statistics(clean_matrix)
    norm_statistics_matrix = compute_cleaned_statistics(clean_norm_matrix)

    logger.info("\nCleaned absolute statistics:")
    logger.info(statistics_matrix.round(2).to_string())

    logger.info("\nCleaned normalized statistics:")
    logger.info(norm_statistics_matrix.round(2).to_string())

    # Calculate budget
    monthly_budget = produce_budget(statistics_matrix)

    logger.info('\n'+'='*60)
    logger.info('BUDGET')
    logger.info('='*60)
    logger.info('Calculated monthly budget:')
    logger.info(monthly_budget.round(2).to_string())
    

if __name__ == "__main__":
    main('w')