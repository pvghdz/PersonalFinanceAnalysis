import logging

def setup_logging(
    log_file: str = "outlier_detection.log",
    level=logging.INFO,
    mode: str = "a",   # append by default
):
    """
    Configure root logger once for the entire application.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        # Logging already configured → do nothing
        return

    formatter = logging.Formatter(
        "%(message)s"
    )

    fh = logging.FileHandler(log_file, mode=mode)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

def format_dataframe_for_logging(df, max_cols=None):
    """
    Used in detect_outliers.py
    Format dataframe for logging with better readability.
    Optionally limit number of columns shown if too wide.
    """
    if max_cols and len(df.columns) > max_cols:
        # Show first few and last few columns
        first_cols = df.iloc[:, :max_cols//2]
        last_cols = df.iloc[:, -max_cols//2:]
        # Create ellipsis column
        ellipsis_series = pd.Series(['...'] * len(df), index=df.index, name='...')
        formatted = pd.concat([first_cols, ellipsis_series, last_cols], axis=1)
        return formatted.to_string()
    return df.to_string()