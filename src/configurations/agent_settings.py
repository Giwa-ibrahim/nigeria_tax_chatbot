"""
Centralized LLM configuration and token management settings
"""

# Unified LLM Configuration Constants
# We use the same limits across all providers for simplicity and predictability
CONTEXT_WINDOW = 8000
SYSTEM_PROMPT_TOKENS = 500
MAX_OUTPUT_TOKENS = 2000

# Tokens available for conversation history + user data
AVAILABLE_CONTEXT = CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS - MAX_OUTPUT_TOKENS

# Context Management Settings
CONTEXT_THRESHOLD = 0.80  # Summarize when reaching 80% of available context
MIN_MESSAGES_BEFORE_SUMMARY = 15  # Don't summarize too early (prevents summarizing long first messages)
SUMMARY_KEEP_RECENT = 5  # Keep last 5 messages after summarization
