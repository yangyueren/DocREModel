import os
import sys
import logging

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    rf_handler = logging.FileHandler('./log/output_prompt2.log')
    # rf_handler = logging.FileHandler('./log/output_opt_loss.log')
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    rf_handler.setLevel(logging.DEBUG)
    logger.addHandler(rf_handler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    return logger

logger = get_logger()
