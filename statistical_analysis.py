import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from config.numeric_cols import numeric_cols
from utils.detect_outliers import no_outlier_dataframe
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Core logic (unchanged)
# ---------------------------------------------------------------------

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
        if category == "Ingreso":
            continue

        budget[category] = {
            'Expected expenses': row['median'],
            'Budget': row['median'] + row['mad'],
        }

    return pd.DataFrame(budget)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run statistical analysis with optional truncation and outlier removal."
    )

    parser.add_argument(
        "-l", "--log-mode",
        choices=["w", "a"],
        default="w",
        help="Log file mode: 'w' to overwrite, 'a' to append (default: w)"
    )

    parser.add_argument(
        "--tdown",
        default="2024-10-30",
        help="Lower truncation date (YYYY-MM-DD). Default: 2024-10-30"
    )

    parser.add_argument(
        "--tup",
        default="2026-01-01",
        help="Upper truncation date (YYYY-MM-DD). Default: 2026-01-01"
    )

    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=2.0,
        help="Outlier threshold for the modified z-score (MAD-based). Default: 2.0. Recommended: 3.5."
    )

    parser.add_argument(
        "--save-dir",
        default="results",
        help="Directory where results and plots will be saved (default: results)"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    setup_logging(log_file=f"{args.save_dir}/results.log", level=logging.INFO, mode=args.log_mode)

    logger.info("\n" + "#" * 60)
    logger.info(f"{datetime.now().replace(microsecond=0)}: LOGGING STARTED")
    logger.info(
        "Current setup: %s log files",
        "re-write previous" if args.log_mode == "w" else "append to previous"
    )
    logger.info("#" * 60)

    clean_matrix, clean_norm_matrix, outlier_mask = no_outlier_dataframe(
        True,                      # truncate analysis dates
        args.tdown,
        args.tup,
        args.outlier_threshold,
        setup_logs=False,          # VERY IMPORTANT
        save_dir=args.save_dir,
        debug=True,
    )

    statistics_matrix = compute_cleaned_statistics(clean_matrix)
    norm_statistics_matrix = compute_cleaned_statistics(clean_norm_matrix)

    logger.info("\nCleaned absolute statistics:")
    logger.info(statistics_matrix.round(2).to_string())

    logger.info("\nCleaned normalized statistics:")
    logger.info(norm_statistics_matrix.round(2).to_string())

    # Budgets
    monthly_budget = produce_budget(statistics_matrix)
    normalized_budget = produce_budget(norm_statistics_matrix)

    logger.info("\n" + "=" * 60)
    logger.info("BUDGET")
    logger.info("=" * 60)
    logger.info("Calculated monthly budget:")
    logger.info(monthly_budget.round(2).to_string())
    logger.info("\nCalculated normalized budget:")
    logger.info(normalized_budget.round(2).to_string())

    # Save outputs
    monthly_budget.to_json(f"{args.save_dir}/monthly_budget.json", orient="index", indent=4, force_ascii=False)
    normalized_budget.to_json(f"{args.save_dir}/normalized_budget.json", orient="index", indent=4, force_ascii=False)


if __name__ == "__main__":
    main()