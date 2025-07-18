# log_util.py
import logging

def init_logger(filename='phase-trans-experiment.log'):
    """basic logging configuration."""
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='a'
    )