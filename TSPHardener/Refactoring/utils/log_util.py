# log_util.py
import logging
import json

def init_logger(filename='experiment.log'):
    """basic logging configuration."""
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode='a'
    )

def log_experiment_info(config, error_msg, logger=None):
    """
    Logs detailed information about failed experiment configurations.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    config_json = json.dumps(config, indent=4)
    logger.error(f"Experiment failed with configuration:\n{config_json}\nError: {error_msg}")