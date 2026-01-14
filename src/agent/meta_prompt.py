import logging, json, re
from typing import Dict, Optional
from src.services.llm import LLMManager
from src.agent.utils import format_chat_history

logger = logging.getLogger("meta_prompt")


def analyze_query_for_tax_calculation(query: str, chat_history: str = "") -> Dict:
    """
    LLM-powered meta-analysis: Intelligently determines what's needed for tax calculations
    and the best interaction strategy.
    """
    
    meta_analysis_prompt = f"""Analyze this tax-related query to determine the best response approach.

CONVERSATION HISTORY:
{chat_history if chat_history else "No previous conversation"}

CURRENT USER QUERY:
{query}

PAYE TAX CALCULATION REQUIREMENTS (Nigeria Tax Act 2025/2026):
- **Mandatory**: Gross monthly/annual income
- **Deductions (reduce tax)**: Pension (8%), NHF (2.5%), NHIS, life insurance, mortgage interest, rent (20% max ₦500k)
- **Calculation**: Annual Income - Deductions - ₦800k (tax-free) → Apply progressive rates (15%-25%) → Monthly PAYE

RESPOND IN JSON FORMAT:
{{
    "is_calculation_request": true/false,
    "needs_clarification": true/false,
    "missing_info": ["field1", "field2"],
    "user_mood": "engaged/impatient/neutral",
    "approach": "collect/conditional/direct",
    "reasoning": "brief explanation"
}}

GUIDANCE:
- **is_calculation_request**: true if user wants actual tax amount calculated
- **needs_clarification**: true if calculation requested but missing salary OR deductions
- **missing_info**: List specific missing fields (salary, pension, nhf, nhis, rent, insurance, mortgage)
- **user_mood**: Detect from conversation - engaged (cooperative), impatient (frustrated), neutral (normal)
- **approach**: 
  - "collect" → Politely ask for missing info (if user engaged and haven't asked repeatedly)
  - "conditional" → Give estimates/ranges (if user impatient or asked many times)
  - "direct" → Answer directly (if not calculation request or has complete info)

Return ONLY valid JSON:"""

    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(meta_analysis_prompt)

        # Extract JSON from response (in case LLM adds extra text)
        content = response.content.strip()
        # Try to find JSON block
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        analysis = json.loads(content)
        logger.info(f"✅ Meta-analysis: {analysis['approach']} approach, Calc request: {analysis.get('is_calculation_request', False)}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in meta-analysis: {e}")
        # fallback: Basic parsing to detect salary presence
        has_salary = bool(re.search(r'\d+[k\s]|₦\s*\d+|\d+,\d+', query.lower()))
        
        return {
            "is_calculation_request": has_salary,
            "needs_clarification": has_salary,
            "missing_info": ["pension", "nhf", "rent"] if has_salary else [],
            "user_mood": "neutral",
            "approach": "collect" if has_salary else "direct",
            "reasoning": "Fallback due to parsing error"
        }


def generate_clarification_request(missing_info: list, user_mood: str, user_query: str = "") -> str:
    """
    LLM generates personalized, friendly clarification requests.
    """
    if user_mood == "impatient":
        return None
    
    clarification_prompt = f"""Generate a friendly, persuasive request for missing tax information.

USER'S QUERY: {user_query}
USER MOOD: {user_mood}
MISSING INFORMATION: {', '.join(missing_info)}

FIELD EXPLANATIONS:
- **Pension**: Retirement Savings Account (RSA) contribution (usually 8% of basic + housing + transport)
- **NHF**: National Housing Fund - 2.5% of gross salary contribution
- **NHIS**: National Health Insurance Scheme - monthly health insurance payment
- **Insurance**: Annual life insurance premium
- **Mortgage**: Annual mortgage interest payment (primary residence only)
- **Rent**: Annual rent amount (gets 20% relief, max ₦500,000/year)

LANGUAGE DETECTION:
- If query has Pidgin markers (wetin, dey, abeg, oga, etc.) → Respond in Nigerian Pidgin
- Otherwise → Use friendly Standard English

YOUR TASK:
1. Friendly intro explaining WHY you need this info
2. Ask for 2-3 most important missing fields (with clear explanations)
3. Persuasive message: Deductions REDUCE tax - helping them pay LESS
4. Note: "Skip any that don't apply"

Keep it conversational, warm, and emphasize tax savings!

CLARIFICATION REQUEST:"""
    
    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(clarification_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating clarification: {e}")
        return None


def generate_conditional_answer(query: str, missing_info: list, partial_answer: str) -> str:
    """
    LLM generates conditional answer with examples when user is impatient.
    """
    conditional_prompt = f"""User wants tax calculation but won't provide complete info. Give helpful conditional answer.

USER QUERY: {query}
MISSING INFO: {', '.join(missing_info)}
CONTEXT: {partial_answer}

LANGUAGE: Match user's language (Pidgin if they use e.g "wetin/dey/abeg", otherwise Standard English)

TASK:
1. Provide general PAYE calculation examples (use realistic Nigerian salaries, e.g. ₦150k, ₦300k, ₦500k, etc.)
2. Show how deductions reduce tax:
   - Pension (8%): Saves ₦X in tax
   - NHF (2.5%): Saves ₦Y
   - Rent relief: Up to ₦100k/year savings
3. Use Nigeria Tax Act 2025/2026 rates: 0% (first ₦800k), then 15%-25% progressive
4. Emphasize: "Without deductions, you might overpay!"
5. Invite them to share details for exact calculation

Be friendly and persuasive!

RESPONSE:"""

    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(conditional_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating conditional answer: {e}")
        return partial_answer


def create_engagement_response(query: str, context: str, chat_history: str = "") -> str:
    """
    LLM creates engaging educational response for interested users.
    """
    engagement_prompt = f"""You're a friendly Nigerian tax guide explaining taxes in simple terms.

CONVERSATION: {chat_history if chat_history else "First message"}
USER QUESTION: {query}
CONTEXT: {context}

LANGUAGE: Match user's language (Pidgin if they use "wetin/dey/abeg", otherwise Standard English)

APPROACH:
1. Break down complex tax concepts into simple steps
2. Use relatable Nigerian examples (realistic salaries, scenarios)
3. Explain WHY things work this way
4. Make taxes feel approachable and empowering
5. End with invitation for follow-up questions

Tone: Like explaining to a friend over coffee - warm, clear, engaging!

RESPONSE:"""

    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(engagement_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating engagement response: {e}")
        return context
