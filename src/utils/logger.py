"""
Logging Utilities
Setup and manage logging for the IoT Botnet Detection project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "iot_botnet",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (if None, uses timestamp)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_dir_path / log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for tracking experiment metrics and results"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs/experiments"):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.txt"
        
        self.logger = setup_logger(
            name=f"experiment_{experiment_name}",
            log_file=self.log_file.name,
            log_dir=str(self.log_dir)
        )
    
    def log_parameters(self, params: dict):
        """Log experiment parameters"""
        self.logger.info("=" * 50)
        self.logger.info("EXPERIMENT PARAMETERS")
        self.logger.info("=" * 50)
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 50)
    
    def log_metrics(self, metrics: dict, phase: str = "training"):
        """Log experiment metrics"""
        self.logger.info(f"\n{phase.upper()} METRICS:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.4f}")
            else:
                self.logger.info(f"  {metric_name}: {value}")
    
    def log_model_info(self, model_name: str, model_params: dict):
        """Log model information"""
        self.logger.info(f"\nMODEL: {model_name}")
        self.logger.info("Parameters:")
        for param, value in model_params.items():
            self.logger.info(f"  {param}: {value}")
    
    def log_step(self, message: str):
        """Log a processing step"""
        self.logger.info(f"\n>>> {message}")
    
    def log_error(self, error: Exception):
        """Log an error"""
        self.logger.error(f"ERROR: {str(error)}", exc_info=True)


if __name__ == "__main__":
    # Test logger
    logger = setup_logger()
    logger.info("Testing logger setup")
    
    # Test experiment logger
    exp_logger = ExperimentLogger("test_experiment")
    exp_logger.log_parameters({"learning_rate": 0.001, "batch_size": 32})
    exp_logger.log_metrics({"accuracy": 0.95, "loss": 0.05})
