"""Centralized logging configuration for PyTAAA.

This module provides a consistent logging setup across all PyTAAA modules.
Following PEP 8 and project logging requirements.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    logger_name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and optional file output.

    Args:
        logger_name: Name of the logger to create
        log_file: Optional file path for log output
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # File handler with rotation (only if log_file is provided)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(
    module_name: str,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger for a module with optional file output.

    Args:
        module_name: Name of the module requesting the logger
        log_file: Optional file path for log output

    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(
        f"PyTAAA.{module_name}",
        log_file=log_file
    )


# Create main system logger (with default log file)
system_logger = setup_logger('PyTAAA', log_file='pytaaa_system.log')

# Create Monte Carlo specific logger
monte_carlo_logger = setup_logger(
    'PyTAAA.MonteCarloBacktest',
    log_file='monte_carlo_backtest.log'
)


def prompt_continue_iteration() -> bool:
    """Prompt user to continue iteration.

    Returns:
        bool: True if user wants to continue, False otherwise.
    """
    while True:
        response = input("Continue to iterate? [y/n]: ").lower().strip()
        if response in ["y", "yes"]:
            logging.info("User chose to continue iteration")
            return True
        elif response in ["n", "no"]:
            logging.info("User chose to stop iteration")
            return False
        print("Please enter 'y' or 'n'")