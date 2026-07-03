"""
Background task to learn user preferences.
"""
from typing import List, Dict, Any
from fastapi import BackgroundTasks
from src.database.connection import get_async_engine
from sqlalchemy import text
import logging
import json
import uuid

logger = logging.getLogger("pref_learner")

class PrefLearner:
    """Learns user preferences."""
    
    async def learn_prefs(
        self,
        user_id: str,
        messages: List[Dict],
        agent_types: List[str],
        user_profile: Dict = None
    ):
        """Update preferences and persist calculation_defaults if available."""
        if not user_id:
            return
            
        logger.info(f"Learning prefs for user: {user_id}")
        engine = get_async_engine()
        
        new_interests = self._get_interests(agent_types)
        style = self._get_style(messages)
        calculation_defaults = self._extract_calculation_defaults(user_profile)
        
        try:
            async with engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT topic_interests, total_sessions FROM user_preferences WHERE user_id = :uid"),
                    {"uid": user_id}
                )
                row = result.fetchone()
                
                if not row:
                    await conn.execute(
                        text("""
                            INSERT INTO user_preferences 
                            (id, user_id, preferred_communication_style, topic_interests, total_sessions, calculation_defaults)
                            VALUES (:id, :uid, :style, :interests, 1, CAST(:calc_defaults AS jsonb))
                        """),
                        {
                            "id": str(uuid.uuid4()),
                            "uid": user_id,
                            "style": style,
                            "interests": json.dumps(new_interests),
                            "calc_defaults": json.dumps(calculation_defaults)
                        }
                    )
                else:
                    existing_interests = row[0] or {}
                    total_sessions = (row[1] or 0) + 1
                    
                    for topic, count in new_interests.items():
                        existing_interests[topic] = existing_interests.get(topic, 0) + count

                    # Build update: always update style/interests/sessions;
                    # only overwrite calculation_defaults if we extracted new data
                    if calculation_defaults:
                        await conn.execute(
                            text("""
                                UPDATE user_preferences 
                                SET topic_interests = :interests,
                                    total_sessions = :sessions,
                                    preferred_communication_style = :style,
                                    calculation_defaults = CAST(:calc_defaults AS jsonb),
                                    last_updated = NOW()
                                WHERE user_id = :uid
                            """),
                            {
                                "uid": user_id,
                                "interests": json.dumps(existing_interests),
                                "sessions": total_sessions,
                                "style": style,
                                "calc_defaults": json.dumps(calculation_defaults)
                            }
                        )
                    else:
                        await conn.execute(
                            text("""
                                UPDATE user_preferences 
                                SET topic_interests = :interests,
                                    total_sessions = :sessions,
                                    preferred_communication_style = :style,
                                    last_updated = NOW()
                                WHERE user_id = :uid
                            """),
                            {
                                "uid": user_id,
                                "interests": json.dumps(existing_interests),
                                "sessions": total_sessions,
                                "style": style
                            }
                        )
        except Exception as e:
            logger.error(f"Failed to update prefs: {e}")
    
    def _get_style(self, messages: List[Dict]) -> str:
        """Detect concise vs detailed."""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return "balanced"
            
        avg_len = sum(len(m.get("content", "")) for m in user_msgs) / len(user_msgs)
        return "concise" if avg_len < 50 else "detailed" if avg_len > 150 else "balanced"
    
    def _get_interests(self, types: List[str]) -> Dict[str, int]:
        """Count agent topics."""
        from collections import Counter
        return dict(Counter([t for t in types if t and t != "general"]))

    def _extract_calculation_defaults(self, user_profile: Dict = None) -> Dict:
        """
        Extract PAYE calculation inputs from the pre-loaded user profile.
        Persisting these means the bot can pre-fill PAYE without asking every time.
        """
        if not user_profile:
            return {}

        tax_data = user_profile.get("tax_data") or {}
        if not tax_data.get("has_tax_data"):
            return {}

        inp = tax_data.get("input_payload") or {}
        defaults = {}

        for key in ["grossIncome", "pensionContribution", "nhfContribution", "nhisContribution", "otherDeductions", "frequency"]:
            if key in inp:
                defaults[key] = inp[key]

        return defaults


def schedule_learning(bg_tasks: BackgroundTasks, user_id: str, result: Dict[str, Any]):
    """Schedule learning."""
    learner = PrefLearner()
    msgs = result.get("messages", [])
    
    fmt_msgs = []
    for m in msgs:
        if hasattr(m, "content") and hasattr(m, "type"):
            role = "user" if m.type == "human" else "assistant" if m.type == "ai" else "system"
            fmt_msgs.append({"role": role, "content": m.content})
        elif isinstance(m, dict):
            fmt_msgs.append(m)

    bg_tasks.add_task(
        learner.learn_prefs,
        user_id=user_id,
        messages=fmt_msgs,
        agent_types=[result.get("route_used", "general")],
        user_profile=result.get("user_profile")
    )
