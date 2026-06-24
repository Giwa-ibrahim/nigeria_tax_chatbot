"""
Central context preparation - loads everything before router
"""
from typing import Dict, List, Optional
from src.database.chat_manager import ChatManager
from src.database.repository import ChatUserRepository
from src.agent.token_manager import TokenManager
from src.database.connection import get_async_engine
from sqlalchemy import text
import logging
import asyncio

logger = logging.getLogger("context_preparator")

class ContextPreparator:
    """Prepares all context before passing to router"""
    
    def __init__(self, provider: str = "groq"):
        self.token_manager = TokenManager(provider)
    
    async def prepare_full_context(
        self,
        user_id: str,
        thread_id: str,
        current_query: str,
        provider: str = "groq"
    ) -> Dict:
        """
        Load and prepare everything the agent needs
        
        Returns complete context package for router
        """
        # 1. Load context data concurrently
        messages, main_app_data, user_preferences = await asyncio.gather(
            ChatManager.get_session_history(thread_id),
            self._load_user_data(user_id),
            self._load_user_preferences(user_id)
        )
        
        # 4. Count tokens and summarize if needed
        prepared_messages, was_summarized = await self.token_manager.prepare_context(
            messages=messages,
            user_profile=main_app_data,
            user_preferences=user_preferences,
            llm_provider=provider
        )
        
        # 5. Build context package
        context_package = {
            "messages": prepared_messages,
            "user_profile": main_app_data,
            "user_preferences": user_preferences,
            "current_query": current_query,
            "metadata": {
                "was_summarized": was_summarized,
                "message_count": len(prepared_messages),
                "total_tokens": self.token_manager.count_messages_tokens(prepared_messages)
            }
        }
        
        return context_package
    
    async def _load_user_data(self, user_id: str) -> Dict:
        """Load user data from main app tables"""
        engine = get_async_engine()
        
        try:
            async with engine.begin() as conn:
                # Get latest tax calculation
                result = await conn.execute(
                    text("""
                        SELECT input_payload, result_payload 
                        FROM tax_calculations 
                        WHERE user_id = :user_id 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """),
                    {"user_id": user_id}
                )
                row = result.fetchone()
                
                if row:
                    return {
                        "has_tax_data": True,
                        "inputs": row[0],  # JSON: grossIncome, pensionContribution, etc.
                        "last_calculation": row[1]
                    }
        except Exception as e:
            logger.warning(f"Could not load tax_calculations for user {user_id}. Table might not exist yet: {e}")
            
        return {"has_tax_data": False}
    
    async def _load_user_preferences(self, user_id: str) -> Dict:
        """Load learned preferences from user_preferences table"""
        engine = get_async_engine()
        
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("""
                        SELECT preferred_communication_style, topic_interests, calculation_defaults
                        FROM user_preferences 
                        WHERE user_id = :user_id
                    """),
                    {"user_id": user_id}
                )
                row = result.fetchone()
                
                if row:
                    return {
                        "communication_style": row[0],
                        "topic_interests": row[1],
                        "calculation_defaults": row[2]
                    }
        except Exception as e:
            logger.warning(f"Could not load user_preferences for user {user_id}. Table might not exist yet: {e}")
            
        return {}  # First time user
