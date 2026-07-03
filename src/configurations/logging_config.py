import sys
import structlog
import logging

def uppercase_level(logger, method_name, event_dict):
    """Processor to uppercase log levels."""
    if "level" in event_dict:
        event_dict["level"] = event_dict["level"].upper()
    return event_dict

def custom_dev_renderer(logger, method_name, event_dict):
    """Custom renderer to match the specific format with emojis."""
    level = event_dict.pop("level", method_name.upper())
    
    # Emoji based on level
    if level == "INFO":
        emoji = "🟢"
    elif level == "WARNING":
        emoji = "🟡"
    elif level in ("ERROR", "CRITICAL"):
        emoji = "🔴"
    elif level == "DEBUG":
        emoji = "🔵"
    else:
        emoji = "⚪"
        
    timestamp = event_dict.pop("timestamp", "")
    logger_name = event_dict.pop("logger", "app")
    event = event_dict.pop("event", "")
    
    # Extract exceptions if present
    exc_info = event_dict.pop("exc_info", None)
    stack_info = event_dict.pop("stack_info", None)
    
    # Format any remaining kwargs
    kwargs = " ".join(f"{k}={v}" for k, v in event_dict.items())
    if kwargs:
        event = f"{event} {kwargs}"
        
    # Use cyan/blue color for timestamp
    colored_time = f"\033[36m{timestamp}\033[0m"
    
    log_line = f"{colored_time} {emoji} {level}:{logger_name}:{event}"
    
    if stack_info:
        log_line += f"\n{stack_info}"
    if exc_info:
        log_line += f"\n{exc_info}"
        
    return log_line

def setup_structured_logging():
    """Configure structlog globally with uppercase levels."""
    
    # Check if running in a terminal (local development)
    if sys.stderr.isatty():
        # Local Development: Custom renderer
        renderer = custom_dev_renderer
        time_fmt = "%d/%m/%Y %H:%M:%S"
    else:
        # Production: JSON Renderer
        renderer = structlog.processors.JSONRenderer()
        time_fmt = "iso"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt=time_fmt),
            uppercase_level,  # Enforce uppercase levels (INFO, ERROR, etc.)
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also intercept standard python logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
