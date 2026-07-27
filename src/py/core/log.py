# log.py
import logging
import sys
from pathlib import Path


def configure_logging(log_path: Path) -> logging.Logger:
    """
    Configure the root logger to write to both stdout and a log file.

    Args:
        log_path: Path to the log file. Parent directories are created if needed.

    Returns:
        The configured root logger.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
