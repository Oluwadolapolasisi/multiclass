"""
logger_setup.py

This module sets up the logging configuration for the application using the Loguru library. 
It defines how logs are formatted, where they are stored, and the log levels to be used throughout the application. 

Usage:
- Import the module: `from logger_setup import setup_logger
                      setup_logger()`

- Log messages with different levels:
  - `logger.debug("Debug message")`
  - `logger.info("Informational message")`
  - `logger.warning("Warning message")`
  - `logger.error("Error message")`
  - `logger.critical("Critical message")`
  - `logger.exception("Exception message")`
  - `logger.success("Success message")`


Functions:
        setup_logger(): Configures the Loguru logger to write logs to files with specific formatting and retention policies.
Log Files:
- `logs/app_{time:YYYY-MM-DD}.log`: Contains all logs with level INFO and above, rotated daily and retained for 7 days.
- `logs/errors.log`: Contains all logs with level ERROR and above, including backtrace and
diagnostic information.

Version: 1.0
Date: 02-05-2026

"""
import os
from loguru import logger


def setup_logger():
    """
    Configures the Loguru logger to write logs to files with specific formatting and retention policies.
    The logger is set up to write INFO level and above logs to a daily rotated file in the 'logs' directory,
    and ERROR level and above logs to a separate file with backtrace and diagnostic information.
    The function ensures that the 'logs' directory exists and handles any exceptions that may occur during setup,
    logging them as errors.

    Returns:
        None
    
    """
    try:
        os.makedirs(name="logs", exist_ok=True)
        FORMAT_STYLE = "{time} | {level} | {name}:{function}:{line} | {message}"
        #logger.remove()

        logger.add("logs/app_{time:YYYY-MM-DD}.log", rotation="00:00",
                   retention="7 days", level="INFO", format=FORMAT_STYLE)
        logger.add("logs/errors.log", level="ERROR",
                   backtrace=True, diagnose=True)
        logger.info("Logger setup completed successfully.")

    except Exception as error:
        logger.error(f"Error occurred while setting up logger: {error}")
