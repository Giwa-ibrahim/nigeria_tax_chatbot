"""
Central context preparation - loads everything before router
"""
from typing import Dict, List, Optional, Tuple
from src.database.chat_manager import ChatManager
from src.agent.token_manager import TokenManager
from src.services.user_data import UserDataService
from src.database.connection import get_async_engine
from sqlalchemy import text
import logging
import asyncio

logger = logging.getLogger("context_preparator")


def _format_currency(amount: float) -> str:
    return f"₦{amount:,.2f}"


def _build_global_user_context(profile: Optional[Dict], tax_data: Optional[Dict], income_sources: List, expenses: Dict) -> Optional[str]:
    """
    Build a single formatted context string from all user data.
    Returns None if no meaningful data exists.
    """
    parts = []

    # User name / greeting
    if profile and profile.get("display_name"):
        parts.append(f"User's name: {profile['display_name']}")
        if profile.get("email"):
            parts.append(f"User's email: {profile['email']}")

    # PAYE / tax calculation context
    if tax_data and tax_data.get("has_tax_data"):
        inp = tax_data.get("input_payload", {})
        res = tax_data.get("result_payload", {})
        gross = inp.get("grossIncome", 0)
        freq = inp.get("frequency", "monthly").capitalize()
        pension = inp.get("pensionContribution", 0)
        nhf = inp.get("nhfContribution", 0)
        nhis = inp.get("nhisContribution", 0)
        other = inp.get("otherDeductions", 0)
        monthly_tax = res.get("monthlyTax", 0)
        effective_rate = res.get("effectiveRate", 0)

        parts.append(f"""
TAX PROFILE (from calculator):
  Gross {freq} Income: {_format_currency(gross)}
  Pension: {_format_currency(pension)} | NHF: {_format_currency(nhf)} | NHIS: {_format_currency(nhis)} | Other: {_format_currency(other)}
  Last Calculated Tax: {_format_currency(monthly_tax)}/month | Effective Rate: {effective_rate:.2f}%""")

    # Income sources
    if income_sources:
        total = sum(
            s["amount"] if s.get("frequency") == "monthly" else s["amount"] / 12
            for s in income_sources
        )
        src_lines = "  " + "\n  ".join(
            f"- {s.get('source','?').capitalize()}: {_format_currency(s['amount'])} ({s.get('frequency','')})"
            for s in income_sources
        )
        parts.append(f"\nINCOME SOURCES:\n{src_lines}\n  Total Monthly: {_format_currency(total)}")

    # Expenses
    if expenses:
        total_exp = sum(expenses.values())
        exp_lines = "  " + "\n  ".join(
            f"- {cat.capitalize()}: {_format_currency(amt)}"
            for cat, amt in sorted(expenses.items(), key=lambda x: x[1], reverse=True)
        )
        parts.append(f"\nEXPENSES:\n{exp_lines}\n  Total: {_format_currency(total_exp)}")

    if not parts:
        return None

    return "===== USER PROFILE (pre-loaded) =====\n" + "\n".join(parts) + "\n======================================"


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
        Load and prepare everything the agent needs concurrently.

        Returns complete context package for router including global_user_context.
        """
        # Fetch ALL data in parallel — session history + all user profile tables
        (
            messages,
            profile,
            tax_data,
            income_sources,
            expenses,
            user_preferences
        ) = await asyncio.gather(
            ChatManager.get_session_history(thread_id),
            UserDataService.get_user_profile(user_id),
            UserDataService.get_latest_tax_calculation(user_id),
            UserDataService.get_user_income_sources(user_id),
            UserDataService.get_user_expenses_by_category(user_id),
            self._load_user_preferences(user_id)
        )

        # Build unified user profile string (injected dynamically by agents)
        global_user_context = _build_global_user_context(profile, tax_data, income_sources, expenses)

        # Summarize context if token budget exceeded
        user_profile_dict = {"has_tax_data": bool(tax_data and tax_data.get("has_tax_data")), "profile": profile}
        prepared_messages, was_summarized = await self.token_manager.prepare_context(
            messages=messages,
            user_profile=user_profile_dict,
            user_preferences=user_preferences,
            llm_provider=provider
        )

        return {
            "messages": prepared_messages,
            "user_profile": user_profile_dict,
            "user_preferences": user_preferences,
            "global_user_context": global_user_context,
            "current_query": current_query,
            "metadata": {
                "was_summarized": was_summarized,
                "message_count": len(prepared_messages),
                "has_user_data": global_user_context is not None,
            }
        }

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
            logger.warning(f"Could not load user_preferences for user {user_id}: {e}")
        return {}


