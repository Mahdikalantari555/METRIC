"""
Logger utilities for METRIC ETa pipeline.

Provides a Logger class for consistent logging across the project
with Loguru-based logging, colored output, and progress bar integration.
"""

from loguru import logger
import sys
from typing import Optional
from contextlib import contextmanager
from pathlib import Path


class Logger:
    """
    Logger class for METRIC ETa pipeline.
    
    Features:
    - Loguru-based logging with colored output
    - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - File rotation and retention
    - Context manager support
    - Progress bar integration
    """
    
    _loggers: dict = {}
    _default_config: dict = {
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                  "<cyan>{message}</cyan>",
        "level": "INFO",
        "colorize": True,
        "serialize": False,
        "backtrace": True,
        "diagnose": True,
        "rotation": "10 MB",
        "retention": "10 files",
        "compression": "gz"
    }
    
    @staticmethod
    def setup(
        name: str = "metric_et",
        log_file: Optional[str] = None,
        level: str = "INFO",
        console: bool = True,
        rotation: str = "10 MB",
        retention: str = "10 files"
    ) -> None:
        """
        Initialize logger with specified configuration.
        
        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console: Whether to output to console
            rotation: Log file rotation size
            retention: Log file retention policy
        """
        # Remove default handler
        logger.remove()
        
        # Add console handler
        if console:
            logger.add(
                sys.stdout,
                format=Logger._default_config["format"],
                level=level,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # Add file handler if log_file specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                str(log_path),
                format=Logger._default_config["format"],
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
                serialize=False,
                backtrace=True,
                diagnose=True
            )
        
        Logger._loggers[name] = logger
        Logger._default_config["level"] = level
    
    @staticmethod
    def get_logger(name: str):
        """
        Get logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in Logger._loggers:
            Logger.setup(name=name)
        return Logger._loggers[name]
    
    @staticmethod
    def debug(message: str, **kwargs) -> None:
        """Log debug message"""
        logger.opt(depth=1).debug(message, **kwargs)
    
    @staticmethod
    def info(message: str, **kwargs) -> None:
        """Log info message"""
        logger.opt(depth=1).info(message, **kwargs)
    
    @staticmethod
    def warning(message: str, **kwargs) -> None:
        """Log warning message"""
        logger.opt(depth=1).warning(message, **kwargs)
    
    @staticmethod
    def error(message: str, **kwargs) -> None:
        """Log error message"""
        logger.opt(depth=1).error(message, **kwargs)
    
    @staticmethod
    def critical(message: str, **kwargs) -> None:
        """Log critical message"""
        logger.opt(depth=1).critical(message, **kwargs)
    
    @staticmethod
    def exception(message: str, **kwargs) -> None:
        """Log exception with traceback"""
        logger.opt(depth=1).exception(message, **kwargs)
    
    @staticmethod
    def log_progress(current: int, total: int, message: str = "Processing") -> None:
        """Log progress information"""
        percent = (current / total) * 100 if total > 0 else 0
        logger.info(f"{message}: {current}/{total} ({percent:.1f}%)")
    
    @staticmethod
    def log_step(step: str, status: str = "COMPLETED") -> None:
        """Log pipeline step"""
        logger.info(f"[{status}] {step}")
    
    @staticmethod
    def configure_for_testing() -> None:
        """Configure logger for testing (quiet mode)"""
        Logger.setup(name="metric_et", level="DEBUG", console=False)
    
    @staticmethod
    def configure_for_production(log_file: str = "logs/metric_et.log") -> None:
        """Configure logger for production"""
        Logger.setup(name="metric_et", log_file=log_file, level="INFO")


@contextmanager
def log_step(name: str):
    """
    Context manager for logging pipeline steps.
    
    Usage:
        with log_step("Processing data"):
            # do work
    """
    Logger.info(f"Starting: {name}")
    try:
        yield
        Logger.log_step(name, "COMPLETED")
    except Exception as e:
        Logger.log_step(name, "FAILED")
        Logger.exception(f"Error in {name}: {e}")
        raise


def get_progress_bar(total: int, desc: str = "Processing"):
    """
    Create a progress bar for loops.
    
    Args:
        total: Total number of items
        desc: Description of the progress
        
    Returns:
        tqdm progress bar instance
    """
    from tqdm import tqdm
    return tqdm(total=total, desc=desc, unit="items")


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time
        def my_function():
            pass
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        Logger.debug(f"{func.__name__} took {execution_time:.4f} seconds")
        return result
    return wrapper
