import logging
import os
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from config import settings

class CustomFormatter(logging.Formatter):
    """
    Custom formatter with color coding for different log levels
    """
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and configuration
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it to prevent duplicate handlers
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(settings.LOG_LEVEL)

    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create a rotating file handler with a delay
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'bot.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        delay=True  # Don't open the file until first write
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # File handler for errors with a delay
    error_file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        delay=True  # Don't open the file until first write
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(error_file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # Timed rotating file handler for daily logs with a delay
    daily_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, 'daily.log'),
        when="midnight",
        interval=1,
        backupCount=30,
        delay=True  # Don't open the file until first write
    )
    daily_handler.namer = lambda name: name + ".log"
    daily_handler.rotator = safe_rotate
    daily_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(daily_handler)

    return logger

def safe_rotate(source, dest):
    """Safely rotate log files with proper error handling and cleanup"""
    for _ in range(5):  # Try up to 5 times
        try:
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except (OSError, PermissionError):
                    # If we can't remove the old file, append a timestamp to make a unique name
                    new_dest = f"{dest}.{int(time.time())}"
                    if os.path.exists(source):
                        os.rename(source, new_dest)
                    return
            if os.path.exists(source):
                os.rename(source, dest)
            return
        except PermissionError:
            time.sleep(1)  # Wait for 1 second before retrying
    print(f"Failed to rotate log file after 5 attempts: {source} -> {dest}")

def cleanup_logger(logger):
    """Clean up logger handlers and close file handles"""
    for handler in logger.handlers[:]:
        try:
            handler.close()
            logger.removeHandler(handler)
        except Exception as e:
            print(f"Error closing logger handler: {e}")

def log_function_call(func):
    """
    Decorator to log function calls
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Function {func.__name__} completed")
        return result
    return wrapper

# Usage example
"""
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    @log_function_call
    def example_function():
        print("This is an example function")

    example_function()
"""