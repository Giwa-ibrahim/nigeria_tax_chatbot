"""
Smart token counting and context management
"""
import tiktoken
from typing import List, Dict, Tuple
from src.configurations.agent_settings import (
    AVAILABLE_CONTEXT,
    MIN_MESSAGES_BEFORE_SUMMARY,
    CONTEXT_THRESHOLD,
    SUMMARY_KEEP_RECENT,
)
from src.services.llm import LLMManager
from src.agent.prompt_library.system_prompts import CONVERSATION_SUMMARY_PROMPT

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
        llm_provider: str
    ) -> Tuple[List[Dict], bool]:
        """
        Prepare context for LLM, summarize if needed
        
        Returns: (prepared_messages, was_summarized)
        """
        # Count current tokens
        message_tokens = self.count_messages_tokens(messages)
        profile_tokens = self.count_tokens(str(user_profile))
        pref_tokens = self.count_tokens(str(user_preferences))
        total_tokens = message_tokens + profile_tokens + pref_tokens
        
        # Check if summarization needed
        if self.should_summarize(total_tokens, len(messages)):
            # Summarize old messages
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
            return prepared_messages, True
        
        return messages, False
