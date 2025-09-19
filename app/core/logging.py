"""
Logging configuration for the application.
"""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging():
    """Setup application logging configuration."""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Check if logging config file exists
    config_path = Path("config/logging.conf")
    
    if config_path.exists():
        # Use logging configuration file
        logging.config.fileConfig(config_path, disable_existing_loggers=False)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/app.log')
            ]
        )
    
    # Set third-party library log levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
