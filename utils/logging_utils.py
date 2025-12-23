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
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    fh = logging.FileHandler(log_file, mode=mode)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)