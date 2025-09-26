"""
Logging Utilities

This module provides centralized logging configuration and utilities
for the Climate-Adaptive Seed AI Bank system.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records"""
    
    def __init__(self, context: Dict[str, Any]):
        super().__init__()
        self.context = context
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to log record"""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['ENDC']}"
            )
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: str = None,
    file_path: Optional[str] = None,
    file_max_bytes: int = 10 * 1024 * 1024,
    file_backup_count: int = 5,
    console_output: bool = True,
    json_format: bool = False,
    colored_output: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Set up logging configuration for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        file_path: Path to log file (if None, no file logging)
        file_max_bytes: Maximum size of log file before rotation
        file_backup_count: Number of backup files to keep
        console_output: Whether to output to console
        json_format: Whether to use JSON formatting
        colored_output: Whether to use colored console output
        context: Additional context to include in all log messages
    """
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Default format
    if log_format is None:
        if json_format:
            log_format = None  # Will use JSONFormatter
        else:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if json_format:
            console_formatter = JSONFormatter()
        elif colored_output and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            console_formatter = ColoredFormatter(log_format)
        else:
            console_formatter = logging.Formatter(log_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_path:
        # Ensure log directory exists
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=file_max_bytes,
            backupCount=file_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(log_format)
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        root_logger.addFilter(context_filter)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, File: {file_path or 'None'}")


def get_logger(
    name: str,
    level: Optional[str] = None,
    extra_context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (typically __name__)
        level: Override logging level for this logger
        extra_context: Additional context for this logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    if extra_context:
        context_filter = ContextFilter(extra_context)
        logger.addFilter(context_filter)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Enhanced logger adapter with additional methods"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with additional context"""
        extra_fields = {**self.extra, **kwargs}
        self.logger.log(level, message, extra={'extra_fields': extra_fields})
    
    def debug_context(self, message: str, **kwargs) -> None:
        """Log debug message with context"""
        self.log_with_context(logging.DEBUG, message, **kwargs)
    
    def info_context(self, message: str, **kwargs) -> None:
        """Log info message with context"""
        self.log_with_context(logging.INFO, message, **kwargs)
    
    def warning_context(self, message: str, **kwargs) -> None:
        """Log warning message with context"""
        self.log_with_context(logging.WARNING, message, **kwargs)
    
    def error_context(self, message: str, **kwargs) -> None:
        """Log error message with context"""
        self.log_with_context(logging.ERROR, message, **kwargs)
    
    def critical_context(self, message: str, **kwargs) -> None:
        """Log critical message with context"""
        self.log_with_context(logging.CRITICAL, message, **kwargs)


def create_logger_adapter(
    name: str,
    context: Dict[str, Any],
    level: Optional[str] = None
) -> LoggerAdapter:
    """
    Create a logger adapter with context
    
    Args:
        name: Logger name
        context: Context to include with all log messages
        level: Optional logging level override
        
    Returns:
        LoggerAdapter instance
    """
    logger = get_logger(name, level)
    return LoggerAdapter(logger, context)


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        import time
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str, **context) -> float:
        """End timing an operation and log duration"""
        import time
        if operation not in self.start_times:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'duration_seconds': duration,
                    **context
                }
            }
        )
        
        return duration
    
    def log_metrics(self, metrics: Dict[str, Any], operation: str = "metrics") -> None:
        """Log performance metrics"""
        self.logger.info(
            f"Performance metrics: {operation}",
            extra={'extra_fields': {'operation': operation, **metrics}}
        )


class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.security")
    
    def log_authentication_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: str = None,
        user_agent: str = None
    ) -> None:
        """Log authentication attempt"""
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'} for user: {user_id}",
            extra={
                'extra_fields': {
                    'event_type': 'authentication',
                    'user_id': user_id,
                    'success': success,
                    'ip_address': ip_address,
                    'user_agent': user_agent
                }
            }
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: str = None
    ) -> None:
        """Log authorization failure"""
        self.logger.warning(
            f"Authorization failed - User: {user_id}, Resource: {resource}, Action: {action}",
            extra={
                'extra_fields': {
                    'event_type': 'authorization_failure',
                    'user_id': user_id,
                    'resource': resource,
                    'action': action,
                    'ip_address': ip_address
                }
            }
        )
    
    def log_suspicious_activity(
        self,
        description: str,
        user_id: str = None,
        ip_address: str = None,
        severity: str = 'medium'
    ) -> None:
        """Log suspicious activity"""
        self.logger.warning(
            f"Suspicious activity detected: {description}",
            extra={
                'extra_fields': {
                    'event_type': 'suspicious_activity',
                    'description': description,
                    'user_id': user_id,
                    'ip_address': ip_address,
                    'severity': severity
                }
            }
        )


def configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries"""
    # Reduce verbosity of common third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('pika').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


# Default logging configuration
def setup_default_logging() -> None:
    """Set up default logging configuration"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    setup_logging(
        level="INFO",
        file_path="./logs/seed_ai_bank.log",
        console_output=True,
        json_format=False,
        colored_output=True,
        context={
            'service': 'seed_ai_bank',
            'version': '1.0.0'
        }
    )
    
    configure_third_party_loggers()


# Example usage
if __name__ == "__main__":
    # Test logging setup
    setup_default_logging()
    
    logger = get_logger(__name__)
    logger.info("Test log message")
    
    # Test performance logger
    perf_logger = PerformanceLogger(__name__)
    perf_logger.start_timer("test_operation")
    import time
    time.sleep(0.1)
    perf_logger.end_timer("test_operation", user_id="test_user")
    
    # Test security logger
    sec_logger = SecurityLogger(__name__)
    sec_logger.log_authentication_attempt("test_user", True, "127.0.0.1")