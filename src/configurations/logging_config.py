import sys
import structlog
import logging

def uppercase_level(logger, method_name, event_dict):
    """Processor to uppercase log levels."""
    if "level" in event_dict:
        event_dict["level"] = event_dict["level"].upper()
    return event_dict

def setup_structured_logging():
    """Configure structlog globally with uppercase levels."""
    
    # Check if running in a terminal (local development)
    if sys.stderr.isatty():
        # Local Development: Pretty console renderer
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Production: JSON Renderer
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
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
