import sys
import os
import re
import json
import functools
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing import Optional, Dict, Any, Union, Callable

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


class SensitiveDataFilter:
    """Filter to remove sensitive information from logs"""
    
    def __init__(self):
        # Patterns for sensitive data detection
        self.sensitive_patterns = [
            r'\b(?:password|passwd|pwd)\s*[=:]\s*[^\s]+',
            r'\b(?:token|key|secret)\s*[=:]\s*[^\s]+',
            r'\b(?:api_key|apikey)\s*[=:]\s*[^\s]+',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Email
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sensitive_patterns]
    
    def sanitize_message(self, message: str) -> str:
        """Remove sensitive data from log message"""
        sanitized = message
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('***REDACTED***', sanitized)
        return sanitized
    
    def __call__(self, record):
        """Filter function for loguru"""
        record["message"] = self.sanitize_message(record["message"])
        return record


class StructuredLoggingFormatter:
    """Custom formatter for structured logging"""
    
    def __init__(self, include_extra: bool = True, service_name: str = "app"):
        self.include_extra = include_extra
        self.service_name = service_name
    
    def format(self, record):
        """Format log record as structured JSON"""
        formatted = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "service": self.service_name,
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        
        if self.include_extra and record.get("extra"):
            formatted["context"] = record["extra"]
        
        # Add exception info if present
        if record["exception"]:
            formatted["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback.format() if record["exception"].traceback else None
            }
        
        return json.dumps(formatted, default=str)


class LoguruConfig:
    """Production-ready configuration class for loguru setup"""
    
    def __init__(self, service_name: str = "app", environment: str = None):
        # Remove default logger
        logger.remove()
        
        # Configuration settings
        self.service_name = service_name
        self.environment = environment or os.getenv("ENVIRONMENT", "development").lower()
        self.log_dir = LOGS_DIR
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize security filter
        self.security_filter = SensitiveDataFilter()
        
        # Environment-specific settings
        self._configure_environment_settings()
        
        # Log file settings
        self.log_file = self.log_dir / f"{service_name}_{{time}}.log"
        self.error_log_file = self.log_dir / f"{service_name}_errors_{{time}}.log"
        self.security_log_file = self.log_dir / f"{service_name}_security_{{time}}.log"
        self.performance_log_file = self.log_dir / f"{service_name}_performance_{{time}}.log"
        
        # Rotation settings - environment specific
        self.rotation_size = os.getenv("LOG_ROTATION_SIZE", "100 MB")
        self.retention_days = os.getenv("LOG_RETENTION_DAYS", "30 days")
        self.compression = os.getenv("LOG_COMPRESSION", "gz")
        
        # Format settings
        self._setup_formats()
    
    def _configure_environment_settings(self):
        """Configure settings based on environment"""
        if self.environment == "production":
            self.default_console_level = "WARNING"
            self.default_file_level = "INFO"
            self.enable_backtrace = False
            self.enable_diagnose = False
            self.enable_colorize = False
            self.enable_sensitive_filter = True
        elif self.environment == "staging":
            self.default_console_level = "INFO"
            self.default_file_level = "DEBUG"
            self.enable_backtrace = False
            self.enable_diagnose = False
            self.enable_colorize = False
            self.enable_sensitive_filter = True
        else:  # development/testing
            self.default_console_level = "DEBUG"
            self.default_file_level = "DEBUG"
            self.enable_backtrace = True
            self.enable_diagnose = True
            self.enable_colorize = True
            self.enable_sensitive_filter = False
    
    def _setup_formats(self):
        """Setup logging formats"""
        # Console format (with optional colors)
        if self.enable_colorize:
            self.console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        else:
            self.console_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
        
        # File format with context
        self.file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "extra: {extra}"
        )
        
        # Structured JSON formatter
        self.structured_formatter = StructuredLoggingFormatter(
            include_extra=True,
            service_name=self.service_name
        )
    
    def setup_logging(self,
                     console_level: Optional[str] = None,
                     file_level: Optional[str] = None,
                     enable_console: bool = True,
                     enable_file: bool = True,
                     enable_error_file: bool = True,
                     enable_structured_logging: bool = False,
                     enable_performance_logging: bool = False,
                     enable_security_logging: bool = False,
                     **kwargs) -> logger:
        """
        Setup production-ready logging with various handlers and configurations
        
        Args:
            console_level: Minimum level for console output
            file_level: Minimum level for file output
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_error_file: Enable separate error file
            enable_structured_logging: Enable JSON structured logging
            enable_performance_logging: Enable performance logging
            enable_security_logging: Enable security event logging
            **kwargs: Additional configuration options
        """
        
        # Use environment defaults if not specified
        console_level = console_level or self.default_console_level
        file_level = file_level or self.default_file_level
        
        # Console handler
        if enable_console:
            logger.add(
                sys.stderr,
                format=self.console_format,
                level=console_level,
                colorize=self.enable_colorize,
                backtrace=self.enable_backtrace,
                diagnose=self.enable_diagnose,
                enqueue=True,
                filter=self.security_filter if self.enable_sensitive_filter else None,
                catch=True
            )
        
        # Main file handler
        if enable_file:
            logger.add(
                str(self.log_file),
                format=self.file_format,
                level=file_level,
                rotation=self.rotation_size,
                retention=self.retention_days,
                compression=self.compression,
                backtrace=self.enable_backtrace,
                diagnose=self.enable_diagnose,
                enqueue=True,
                filter=self.security_filter if self.enable_sensitive_filter else None,
                catch=True
            )
        
        # Error-only file handler
        if enable_error_file:
            logger.add(
                str(self.error_log_file),
                format=self.file_format,
                level="ERROR",
                rotation=self.rotation_size,
                retention=self.retention_days,
                compression=self.compression,
                backtrace=self.enable_backtrace,
                diagnose=self.enable_diagnose,
                enqueue=True,
                filter=self.security_filter if self.enable_sensitive_filter else None,
                catch=True
            )
        
        # Structured JSON logging
        if enable_structured_logging:
            logger.add(
                str(self.log_dir / f"{self.service_name}_structured_{{time}}.json"),
                format=lambda record: self.structured_formatter.format(record),
                level=file_level,
                rotation=self.rotation_size,
                retention=self.retention_days,
                compression=self.compression,
                enqueue=True,
                filter=self.security_filter if self.enable_sensitive_filter else None,
                catch=True
            )
        
        # Performance logging
        if enable_performance_logging:
            logger.add(
                str(self.performance_log_file),
                format=self.file_format,
                level="INFO",
                rotation=self.rotation_size,
                retention=self.retention_days,
                compression=self.compression,
                enqueue=True,
                filter=lambda record: "performance" in record.get("extra", {}),
                catch=True
            )
        
        # Security logging
        if enable_security_logging:
            logger.add(
                str(self.security_log_file),
                format=self.file_format,
                level="WARNING",
                rotation=self.rotation_size,
                retention=self.retention_days,
                compression=self.compression,
                enqueue=True,
                filter=lambda record: "security" in record.get("extra", {}),
                catch=True
            )
        
        # Add custom log levels
        self._setup_custom_levels()
        
        # Log configuration info
        logger.info(
            f"Logging configured for {self.environment.upper()} environment",
            extra={"service": self.service_name, "environment": self.environment}
        )
        
        return logger
    
    def _setup_custom_levels(self):
        """Setup custom log levels"""
        custom_levels = [
            ("TRACE", 5, "<white>"),
            ("SUCCESS", 25, "<green><bold>"),
            ("SECURITY", 35, "<red><bold>"),
            ("PERFORMANCE", 15, "<blue>")
        ]
        
        for name, level, color in custom_levels:
            try:
                logger.level(name, no=level, color=color)
            except ValueError:
                # Level already exists
                pass


# Enhanced utility functions for production

def sanitize_args(*args, **kwargs):
    """Sanitize function arguments for logging"""
    filter_obj = SensitiveDataFilter()
    sanitized_args = tuple(filter_obj.sanitize_message(str(arg)) for arg in args)
    sanitized_kwargs = {k: filter_obj.sanitize_message(str(v)) for k, v in kwargs.items()}
    return sanitized_args, sanitized_kwargs


def log_function_call(include_args: bool = False, include_result: bool = False, sanitize: bool = True):
    """Enhanced decorator to log function calls with security options"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if include_args:
                if sanitize:
                    safe_args, safe_kwargs = sanitize_args(*args, **kwargs)
                    logger.debug(f"Calling {func_name}", extra={
                        "function": func_name,
                        "args": safe_args,
                        "kwargs": safe_kwargs
                    })
                else:
                    logger.debug(f"Calling {func_name}", extra={
                        "function": func_name,
                        "args": args,
                        "kwargs": kwargs
                    })
            else:
                logger.debug(f"Calling {func_name}", extra={"function": func_name})
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    if sanitize:
                        safe_result = SensitiveDataFilter().sanitize_message(str(result))
                        logger.debug(f"{func_name} completed", extra={
                            "function": func_name,
                            "result": safe_result
                        })
                    else:
                        logger.debug(f"{func_name} completed", extra={
                            "function": func_name,
                            "result": result
                        })
                else:
                    logger.debug(f"{func_name} completed", extra={"function": func_name})
                
                return result
            except Exception as e:
                logger.exception(f"Error in {func_name}", extra={
                    "function": func_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                raise
        return wrapper
    return decorator


def log_performance(threshold_ms: float = 1000.0):
    """Enhanced performance logging decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                extra_data = {
                    "performance": True,
                    "function": func.__name__,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "threshold_ms": threshold_ms
                }
                
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"Slow execution: {func.__name__} took {elapsed_ms:.2f}ms",
                        extra=extra_data
                    )
                else:
                    logger.debug(
                        f"Performance: {func.__name__} took {elapsed_ms:.2f}ms",
                        extra=extra_data
                    )
                
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Failed execution: {func.__name__} failed after {elapsed_ms:.2f}ms",
                    extra={
                        "performance": True,
                        "function": func.__name__,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "error": str(e)
                    }
                )
                raise
        return wrapper
    return decorator


def log_security_event(event_type: str, details: Dict[str, Any] = None):
    """Log security-related events"""
    logger.warning(
        f"Security event: {event_type}",
        extra={
            "security": True,
            "event_type": event_type,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
    )


def setup_module_logger(module_name: str):
    """Create a logger with module context"""
    return logger.bind(module=module_name)


def get_logger_for_service(service_name: str):
    """Get a logger bound to a specific service"""
    return logger.bind(service=service_name)


# Production-ready configuration presets

def setup_production_logging(service_name: str = "app") -> logger:
    """Setup logging for production environment"""
    config = LoguruConfig(service_name=service_name, environment="production")
    return config.setup_logging(
        console_level="WARNING",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        enable_error_file=True,
        enable_structured_logging=True,
        enable_performance_logging=True,
        enable_security_logging=True
    )


def setup_staging_logging(service_name: str = "app") -> logger:
    """Setup logging for staging environment"""
    config = LoguruConfig(service_name=service_name, environment="staging")
    return config.setup_logging(
        console_level="INFO",
        file_level="DEBUG",
        enable_console=True,
        enable_file=True,
        enable_error_file=True,
        enable_structured_logging=True,
        enable_performance_logging=True,
        enable_security_logging=True
    )


def setup_development_logging(service_name: str = "app") -> logger:
    """Setup logging for development environment"""
    config = LoguruConfig(service_name=service_name, environment="development")
    return config.setup_logging(
        console_level="DEBUG",
        file_level="DEBUG",
        enable_console=True,
        enable_file=True,
        enable_error_file=True,
        enable_structured_logging=False,
        enable_performance_logging=False,
        enable_security_logging=False
    )


def setup_testing_logging(service_name: str = "app") -> logger:
    """Setup minimal logging for testing"""
    config = LoguruConfig(service_name=service_name, environment="testing")
    return config.setup_logging(
        console_level="ERROR",
        file_level="WARNING",
        enable_console=True,
        enable_file=False,
        enable_error_file=False,
        enable_structured_logging=False,
        enable_performance_logging=False,
        enable_security_logging=False
    )


# Health check and monitoring functions

def log_health_check(component: str, status: str, details: Dict[str, Any] = None):
    """Log health check results"""
    level = "INFO" if status == "healthy" else "ERROR"
    getattr(logger, level.lower())(
        f"Health check - {component}: {status}",
        extra={
            "health_check": True,
            "component": component,
            "status": status,
            "details": details or {}
        }
    )


def log_system_metrics(metrics: Dict[str, Any]):
    """Log system metrics"""
    logger.info("System metrics", extra={
        "metrics": True,
        "data": metrics,
        "timestamp": datetime.now().isoformat()
    })


# Example usage and demonstration
if __name__ == "__main__":
    # Get service name from environment or use default
    service_name = os.getenv("SERVICE_NAME", "bitcoin-app")
    
    # Setup logging based on environment
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        log = setup_production_logging(service_name)
    elif env == "staging":
        log = setup_staging_logging(service_name)
    elif env == "testing":
        log = setup_testing_logging(service_name)
    else:
        log = setup_development_logging(service_name)
    
    # Example logging with different levels
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Security event example
    log_security_event("login_attempt", {
        "user_id": "user123",
        "ip_address": "192.168.1.100",
        "success": True
    })
    
    # Performance logging example
    @log_performance(threshold_ms=100)
    @log_function_call(include_args=True, include_result=True)
    def example_function(x: int, y: int, password: str = "secret123") -> int:
        """Example function with logging decorators"""
        import time
        time.sleep(0.05)  # Simulate work
        return x + y
    
    result = example_function(5, 3, password="secret123")
    
    # Health check example
    log_health_check("database", "healthy", {"response_time_ms": 45})
    
    # System metrics example
    log_system_metrics({
        "cpu_usage": 25.5,
        "memory_usage": 67.8,
        "disk_usage": 45.2
    })
    
    # Structured logging example
    logger.info("User action completed", extra={
        "user_id": "user123",
        "action": "file_upload",
        "file_size": 1024000,
        "duration_ms": 250
    })