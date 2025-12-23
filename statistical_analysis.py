import pandas as pd
from finance_loader import load_finance_dataframe
from config.numeric_cols import numeric_cols
import scipy.stats as stats
import numpy as np
import logging
import matplotlib.pyplot as plt
from detect_outliers import no_outlier_dataframe

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
                'median': np.nan,
                'std': np.nan,
                'mad': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
            continue

        median = values.median()
        
        stats_dict[category] = {
            'mean': values.mean(),
            'median': median,
            'std': values.std(),
            'mad': (values - median).abs().median(),
            'min': values.min(),
            'max': values.max(),
            'count': len(values)
        }
    
    return pd.DataFrame(stats_dict).T

def main(setup_logs:bool = True):
    #if setup_logs:
        #setup_logging(level=logging.INFO, mode='a') # mode = a -> append to the log file produced by the detect_outliers.py script

    truncationDateDown = '2024-10-30'
    truncationDateUp = '2026-01-01' 
    outlier_threshold = 2.0

    # clean_matrix = matrix without outliers
    clean_matrix, outlier_mask = no_outlier_dataframe(truncationDateDown,  truncationDateUp, outlier_threshold, setup_logs = True)
    statistics_matrix = compute_cleaned_statistics(clean_matrix)
    print(statistics_matrix.head())

if __name__ == "__main__":
    main()