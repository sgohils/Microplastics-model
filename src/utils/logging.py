"""Logging utilities."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "microplastic_gnn", 
                 log_file: str | None = None,
                 level: str = "INFO") -> logging.Logger:
    """Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, 
                           output_dir: str) -> logging.Logger:
    """Get a logger configured for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"{experiment_name}_{timestamp}.log"
    return setup_logger(experiment_name, str(log_file))
