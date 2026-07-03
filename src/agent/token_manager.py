"""
Smart token counting and context management
"""
import tiktoken
from typing import List, Dict, Tuple, Optional
from src.configurations.agent_settings import (
    AVAILABLE_CONTEXT,
    MIN_MESSAGES_BEFORE_SUMMARY,
    CONTEXT_THRESHOLD,
    SUMMARY_KEEP_RECENT,
)
import structlog
from src.services.llm import LLMManager
from src.agent.prompt_library.system_prompts import CONVERSATION_SUMMARY_PROMPT

logger = structlog.get_logger("token_manager")
class TokenManager:
    def __init__(self, provider: str = "groq"):
        # Provider is kept for interface compatibility, but we use unified constants
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Similar tokenization
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in message list"""
        total = 0
        for msg in messages:
            total += self.count_tokens(str(msg.get("content", "")))
            total += 4  # Message overhead
        return total
    
    def should_summarize(self, current_tokens: int, message_count: int) -> bool:
        """Check if context should be summarized"""
        threshold_tokens = AVAILABLE_CONTEXT * CONTEXT_THRESHOLD
        return (current_tokens > threshold_tokens and 
                message_count >= MIN_MESSAGES_BEFORE_SUMMARY)
    
    async def prepare_context(
        self, 
        messages: List[Dict],
        user_profile: Dict,
        user_preferences: Dict,
        llm_provider: str,
        thread_id: Optional[str] = None
    ) -> Tuple[List[Dict], bool]:
        """
        Prepare context for LLM, summarize if needed.
        If thread_id is provided, persists the summary to chat_summaries table.

        Returns: (prepared_messages, was_summarized)
        """
        # Count current tokens
        message_tokens = self.count_messages_tokens(messages)
        profile_tokens = self.count_tokens(str(user_profile))
        pref_tokens = self.count_tokens(str(user_preferences))
        total_tokens = message_tokens + profile_tokens + pref_tokens

        # Check if summarization needed
        if self.should_summarize(total_tokens, len(messages)):
            old_messages = messages[:-SUMMARY_KEEP_RECENT]
            recent_messages = messages[-SUMMARY_KEEP_RECENT:]

            # Create summary using the externalized prompt
            summary_prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation=old_messages)
            llm_manager = LLMManager()
            llm = llm_manager.get_llm()
            summary = llm.invoke(summary_prompt)

            # Replace old messages with summary
            summary_message = {
                "role": "system",
                "content": f"[Previous conversation summary]: {summary.content}"
            }

            prepared_messages = [summary_message] + recent_messages

            # Persist summary to DB if thread_id is available
            if thread_id:
                try:
                    from src.database.repository import ChatSummaryRepository
                    msg_range = f"1-{len(old_messages)}"
                    await ChatSummaryRepository.create_summary(
                        session_id=thread_id,
                        summary_text=summary.content,
                        message_range=msg_range
                    )
                except Exception as e:
                    logger.error("Could not persist summary", error=str(e))

            return prepared_messages, True

        return messages, False
