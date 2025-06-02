"""
Logging configuration using loguru
This module provides a comprehensive logging setup that can be used across your application.
"""

import sys
import os
from pathlib import Path
from loguru import logger
from datetime import datetime

# Try to import LOGS_DIR, fallback to local path if running directly
try:
    from .setup_path import LOGS_DIR
except ImportError:
    # When running directly, use relative import or fallback
    try:
        from setup_path import LOGS_DIR
    except ImportError:
        # Fallback to hardcoded path if setup_path not available
        LOGS_DIR = Path(".logs")


class LoguruConfig:
    """Configuration class for loguru setup"""
    
    def __init__(self):
        # Remove default logger
        logger.remove()
        
        # Configuration settings
        self.log_dir = LOGS_DIR
        self.log_dir.mkdir(exist_ok=True)
        
        # Log file settings
        self.log_file = self.log_dir / "app_{time}.log"
        self.error_log_file = self.log_dir / "errors_{time}.log"
        
        # Rotation settings
        self.rotation = "500 MB"  # Rotate when file reaches 500MB
        self.retention = "10 days"  # Keep logs for 10 days
        self.compression = "zip"  # Compress rotated logs
        
        # Format settings
        self.format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        self.simple_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        
        # Detailed format for file logging
        self.detailed_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "context: {extra}"
        )
    
    def setup_logging(self, 
                     console_level="INFO",
                     file_level="DEBUG",
                     enable_console=True,
                     enable_file=True,
                     enable_error_file=True,
                     colorize=True,
                     backtrace=True,
                     diagnose=True,
                     enqueue=True):
        """
        Setup logging with various handlers and configurations
        
        Args:
            console_level: Minimum level for console output
            file_level: Minimum level for file output
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_error_file: Enable separate error file
            colorize: Enable colored output in console
            backtrace: Enable detailed traceback
            diagnose: Enable variable values in traceback
            enqueue: Enable thread-safe logging
        """
        
        # Console handler
        if enable_console:
            logger.add(
                sys.stderr,
                format=self.format,
                level=console_level,
                colorize=colorize,
                backtrace=backtrace,
                diagnose=diagnose,
                enqueue=enqueue
            )
        
        # File handler for all logs
        if enable_file:
            logger.add(
                self.log_file,
                format=self.detailed_format,
                level=file_level,
                rotation=self.rotation,
                retention=self.retention,
                compression=self.compression,
                backtrace=backtrace,
                diagnose=diagnose,
                enqueue=enqueue
            )
        
        # Separate file handler for errors only
        if enable_error_file:
            logger.add(
                self.error_log_file,
                format=self.detailed_format,
                level="ERROR",
                rotation=self.rotation,
                retention=self.retention,
                compression=self.compression,
                backtrace=backtrace,
                diagnose=diagnose,
                enqueue=enqueue
            )
        
        # Add custom log level
        # Check if level exists before creating it
        try:
            logger.level("TRACE", no=5, color="<white>")
        except ValueError:
            # Level already exists, that's fine
            pass
        
        try:
            logger.level("SUCCESS", no=25, color="<green><bold>")
        except ValueError:
            # Level already exists, that's fine
            pass
        
        return logger


# Utility functions for common logging patterns

def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def setup_module_logger(module_name):
    """Create a logger with module context"""
    return logger.bind(module=module_name)


def log_performance(func):
    """Decorator to log function performance"""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper


# Example configuration presets

def setup_development_logging():
    """Setup logging for development environment"""
    config = LoguruConfig()
    config.setup_logging(
        console_level="DEBUG",
        file_level="DEBUG",
        enable_console=True,
        enable_file=True,
        enable_error_file=True,
        colorize=True
    )
    logger.info("Logging configured for DEVELOPMENT environment")
    return logger


def setup_production_logging():
    """Setup logging for production environment"""
    config = LoguruConfig()
    config.setup_logging(
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        enable_error_file=True,
        colorize=False,  # No colors in production
        enqueue=True  # Thread-safe for production
    )
    logger.info("Logging configured for PRODUCTION environment")
    return logger


def setup_testing_logging():
    """Setup minimal logging for testing"""
    config = LoguruConfig()
    config.setup_logging(
        console_level="ERROR",
        file_level="DEBUG",
        enable_console=True,
        enable_file=False,
        enable_error_file=False
    )
    logger.info("Logging configured for TESTING environment")
    return logger


# Example usage and demonstration

if __name__ == "__main__":
    # Choose environment
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        log = setup_production_logging()
    elif env == "testing":
        log = setup_testing_logging()
    else:
        log = setup_development_logging()
    
    # Basic logging examples
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Context binding example
    request_logger = logger.bind(request_id="12345", user_id="user_001")
    request_logger.info("Processing user request")
    
    # Structured logging
    logger.info("User logged in", extra={
        "user_id": "user_001",
        "ip_address": "192.168.1.1",
        "timestamp": datetime.now().isoformat()
    })
    
    # Exception handling
    try:
        1 / 0
    except Exception as e:
        logger.exception("An error occurred")
    
    # Using decorators
    @log_function_call
    @log_performance
    def example_function(x, y):
        """Example function with logging decorators"""
        import time
        time.sleep(0.1)  # Simulate some work
        return x + y
    
    result = example_function(5, 3)
    
    # Module-specific logger
    module_logger = setup_module_logger("my_module")
    module_logger.info("This is a module-specific log")
    
    # Conditional logging
    debug_mode = True
    if debug_mode:
        logger.debug("Debug mode is enabled")
    
    # Lazy evaluation
    expensive_value = lambda: sum(range(1000000))
    logger.debug("Expensive calculation: {}", expensive_value)
    
    # Custom formatting for specific message
    logger.opt(colors=True).info("This is a <red>colored</red> message")
    
    # Logging with extra fields
    logger.bind(server="web-01").info("Server started")
    
    # Example of filtering
    logger.add(
        sys.stderr,
        filter=lambda record: "special" in record["message"],
        level="DEBUG"
    )
    
    logger.debug("This message will be filtered out")
    logger.debug("This special message will be shown")