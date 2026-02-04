import logging
from typing import Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """
    Setup global logging configuration.

    Args:
        log_level: Logging level (default: INFO)
        log_format: Format string for log messages
        log_file: Optional file path to write logs to
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]  # Console handler

    if log_file:
        handlers.append(logging.FileHandler(log_file))  # File handler if specified

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)


def logger_factory(
    class_name: str, logging_level: int = logging.INFO
) -> logging.Logger:
    """
    Create a logger instance for a specific class.

    Args:
        class_name: Name of the class (usually __name__ or self.__class__.__name__)

    Returns:
        Configured logger instance
    """
    logger_instance = logging.getLogger(class_name)
    logger_instance.setLevel(logging_level)
    return logger_instance
