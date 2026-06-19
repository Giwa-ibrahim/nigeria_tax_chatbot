"""
Session Context Enrichment - Format user data for agent consumption

This module takes raw data from UserDataService and formats it into
structured context strings that agents can easily understand and use.
"""

from typing import Dict, Optional
from src.services.user_data import get_user_context


def format_currency(amount: float) -> str:
    """Format number as Nigerian Naira"""
    return f"₦{amount:,.2f}"


def build_paye_context(user_context: Dict) -> Optional[str]:
    """
    Build formatted context for PAYE agent with pre-loaded tax data
    
    Args:
        user_context: Output from UserDataService.get_complete_user_context()
        
    Returns:
        Formatted context string or None if no tax data available
    """
    tax_data = user_context.get("tax_calculation", {})
    
    if not tax_data.get("has_tax_data"):
        return None
    
    input_payload = tax_data.get("input_payload", {})
    result_payload = tax_data.get("result_payload", {})
    
    # Extract key values
    gross_income = input_payload.get("grossIncome", 0)
    frequency = input_payload.get("frequency", "monthly")
    income_type = input_payload.get("incomeType", "salary")
    pension = input_payload.get("pensionContribution", 0)
    nhf = input_payload.get("nhfContribution", 0)
    nhis = input_payload.get("nhisContribution", 0)
    other_deductions = input_payload.get("otherDeductions", 0)
    
    # Extract result values
    monthly_tax = result_payload.get("monthlyTax", 0)
    effective_rate = result_payload.get("effectiveRate", 0)
    
    # Build formatted context
    context = f"""
═══════════════════════════════════════════════════
📋 USER TAX PROFILE (PRE-LOADED FROM MAIN APP)
═══════════════════════════════════════════════════

💰 INCOME:
   • Gross {frequency.capitalize()}: {format_currency(gross_income)}
   • Type: {income_type.capitalize()}

🏦 DEDUCTIONS (On Record):
   • Pension: {format_currency(pension)}
   • NHF: {format_currency(nhf)}
   • NHIS: {format_currency(nhis)}
   • Other: {format_currency(other_deductions)}

📊 LAST TAX CALCULATION:
   • Monthly Tax: {format_currency(monthly_tax)}
   • Effective Rate: {effective_rate:.2f}%

⚠️  IMPORTANT INSTRUCTIONS:
   • USE THIS DATA for PAYE calculations by default
   • If user says "calculate my PAYE" → use these values directly
   • Only ask questions if:
     - User explicitly mentions changes ("I got a raise")
     - User provides new numbers
     - User asks about different scenarios

═══════════════════════════════════════════════════
"""
    
    return context


def build_income_summary(user_context: Dict) -> Optional[str]:
    """
    Build formatted income summary for financial advice agent
    
    Args:
        user_context: Output from UserDataService.get_complete_user_context()
        
    Returns:
        Formatted income summary or None if no income data
    """
    income_sources = user_context.get("income_sources", [])
    total_monthly = user_context.get("total_monthly_income", 0)
    
    if not income_sources:
        return None
    
    # Build income breakdown
    sources_text = []
    for source in income_sources:
        amount = source["amount"]
        freq = source["frequency"]
        source_type = source["source"]
        sources_text.append(f"   • {source_type.capitalize()}: {format_currency(amount)} ({freq})")
    
    context = f"""
═══════════════════════════════════════════════════
💵 USER INCOME PROFILE (PRE-LOADED)
═══════════════════════════════════════════════════

📊 INCOME SOURCES ({len(income_sources)}):
{chr(10).join(sources_text)}

💰 TOTAL MONTHLY INCOME: {format_currency(total_monthly)}

⚠️  Use this data when giving financial advice
═══════════════════════════════════════════════════
"""
    
    return context


def build_expense_summary(user_context: Dict) -> Optional[str]:
    """
    Build formatted expense summary for financial advice agent
    
    Args:
        user_context: Output from UserDataService.get_complete_user_context()
        
    Returns:
        Formatted expense summary or None if no expense data
    """
    expenses = user_context.get("expenses", {})
    
    if not expenses:
        return None
    
    # Calculate total expenses
    total_expenses = sum(expenses.values())
    
    # Build expense breakdown
    expense_lines = []
    for category, amount in sorted(expenses.items(), key=lambda x: x[1], reverse=True):
        percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
        expense_lines.append(f"   • {category.capitalize()}: {format_currency(amount)} ({percentage:.1f}%)")
    
    # Calculate disposable income
    total_monthly_income = user_context.get("total_monthly_income", 0)
    disposable = total_monthly_income - total_expenses
    
    context = f"""
═══════════════════════════════════════════════════
💸 USER EXPENSE PROFILE (PRE-LOADED)
═══════════════════════════════════════════════════

📊 MONTHLY EXPENSES:
{chr(10).join(expense_lines)}

💰 TOTAL EXPENSES: {format_currency(total_expenses)}
💵 DISPOSABLE INCOME: {format_currency(disposable)}

⚠️  Use this when advising on savings and budgeting
═══════════════════════════════════════════════════
"""
    
    return context


def build_greeting_context(user_context: Dict) -> Optional[str]:
    """
    Build personalized greeting context with user name
    
    Args:
        user_context: Output from UserDataService.get_complete_user_context()
        
    Returns:
        Greeting context with user's display name
    """
    profile = user_context.get("profile")
    
    if not profile:
        return None
    
    display_name = profile.get("display_name", "User")
    
    return f"User's name: {display_name} (use for personalized greetings)"


async def enrich_agent_state(state: Dict, user_id: str) -> Dict:
    """
    Main enrichment function - fetches and formats ALL user data
    
    This is the primary function to call. It:
    1. Fetches complete user context from main app
    2. Formats data for different agent types
    3. Adds formatted contexts to agent state
    
    Args:
        state: Current agent state dict
        user_id: User UUID as string
        
    Returns:
        Enhanced state dict with added context fields:
        - paye_user_context: Formatted tax data for PAYE agent
        - income_summary_context: Income summary for financial advice
        - expense_summary_context: Expense summary for financial advice
        - greeting_context: User name for personalization
        - raw_user_context: Raw data for advanced use cases
    """
    # Fetch complete user context
    user_context = await get_user_context(user_id)
    
    # Build formatted contexts for different purposes
    paye_context = build_paye_context(user_context)
    income_context = build_income_summary(user_context)
    expense_context = build_expense_summary(user_context)
    greeting_context = build_greeting_context(user_context)
    
    # Add all contexts to state
    enhanced_state = state.copy()
    
    if paye_context:
        enhanced_state["paye_user_context"] = paye_context
    
    if income_context:
        enhanced_state["income_summary_context"] = income_context
    
    if expense_context:
        enhanced_state["expense_summary_context"] = expense_context
    
    if greeting_context:
        enhanced_state["greeting_context"] = greeting_context
    
    # Store raw context for advanced use cases
    enhanced_state["raw_user_context"] = user_context
    enhanced_state["has_user_data"] = user_context.get("has_data", False)
    
    return enhanced_state


def get_paye_instructions(has_context: bool) -> str:
    """
    Get instructions for PAYE agent based on whether user data exists
    
    Args:
        has_context: Whether paye_user_context exists in state
        
    Returns:
        Instruction string for system prompt
    """
    if has_context:
        return """
✅ USER DATA IS PRE-LOADED (see above)

CALCULATION APPROACH:
1. If user says "calculate my PAYE" or "what's my tax":
   → Use the pre-loaded data DIRECTLY, don't ask questions
2. If user mentions changes ("I got a raise to 100k"):
   → Use the NEW values provided
3. If user asks "what if" scenarios:
   → Use the provided hypothetical values

DO NOT ask for basic info that's already loaded!
"""
    else:
        return """
⚠️ NO TAX DATA ON RECORD FOR THIS USER

CALCULATION APPROACH:
You MUST ask the user for:
1. Monthly or annual gross salary
2. Pension contribution (if any)
3. NHF contribution (if any)
4. NHIS contribution (if any)
5. Other deductions (if any)

Then perform the PAYE calculation using Nigerian tax bands.
"""


# Convenience function for testing
async def preview_enrichment(user_id: str) -> None:
    """
    Preview formatted contexts for a user (for testing)
    
    Args:
        user_id: User UUID as string
    """
    state = {"query": "test"}
    enriched = await enrich_agent_state(state, user_id)
    
    print("\n" + "="*60)
    print("ENRICHED STATE PREVIEW")
    print("="*60)
    
    if "paye_user_context" in enriched:
        print("\n📋 PAYE CONTEXT:")
        print(enriched["paye_user_context"])
    
    if "income_summary_context" in enriched:
        print("\n💵 INCOME CONTEXT:")
        print(enriched["income_summary_context"])
    
    if "expense_summary_context" in enriched:
        print("\n💸 EXPENSE CONTEXT:")
        print(enriched["expense_summary_context"])
    
    if "greeting_context" in enriched:
        print("\n👋 GREETING CONTEXT:")
        print(enriched["greeting_context"])
    
    print("\n" + "="*60)
