import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history
from src.tools.web_search import search_web
from src.agent.meta_prompt import (
    generate_clarification_request,
    generate_conditional_answer,
    create_engagement_response
)
from src.agent.context_injector import build_user_context_block

logger = logging.getLogger("paye_agent")


async def paye_calculation_agent(state: AgentState) -> AgentState:
    """
    PAYE Calculation Agent - Handles PAYE-specific questions with intelligent meta-prompting.
    
    🆕 PERSONALIZATION: Now checks for pre-loaded user data from main app!
    - If user has tax calculation history → uses it automatically
    - If no data → asks questions like before
    
    Uses meta-analysis to determine the best interaction approach:
    - Friendly information collection for engaged users
    - Conditional answers for impatient users
    - Step-by-step guidance for learning-focused users
    """
    logger.info("💰 PAYE Calculation Agent processing...")
    
    query = state["query"]
    chat_history = format_chat_history(state.get("messages", []))
    user_preferences = state.get("user_preferences", {})

    # Read meta_analysis from router (already computed — no extra LLM call needed)
    meta_analysis = state.get("meta_analysis") or {}
    approach = meta_analysis.get("approach", "direct")
    user_mood = meta_analysis.get("user_mood", "neutral")
    needs_clarification = meta_analysis.get("needs_clarification", False)
    missing_info = meta_analysis.get("missing_info", [])
    is_calculation_request = meta_analysis.get("is_calculation_request", False)

    # If meta_analysis was not pre-computed (fallback), use defaults
    if not meta_analysis.get("route"):
        logger.info("⚠️ No pre-computed meta_analysis found — using fallback defaults")
        approach = "direct"
        user_mood = "neutral"
        needs_clarification = False
        missing_info = []
        is_calculation_request = False

    # Dynamic user context injection (LLM-driven via router's needs_user_context flag)
    user_context_block = build_user_context_block(state)
    has_user_data = bool(user_context_block)

    if has_user_data:
        logger.info("✅ User profile context will be injected into PAYE prompt")
    else:
        logger.info("ℹ️ No personal context needed for this query")

    logger.info(f"📊 Calc request: {is_calculation_request}, Approach: {approach}, Mood: {user_mood}, Needs info: {needs_clarification}")

    # OVERRIDE CLARIFICATION IF USER DATA EXISTS
    if is_calculation_request and has_user_data:
        logger.info("🎯 Calculation requested + user data exists → Using pre-loaded data!")
        needs_clarification = False
        missing_info = []
    
    # If user wants calculation but missing deductions - DON'T search RAG yet, just ask for info
    if is_calculation_request and needs_clarification and approach == "collect":
        # FRIENDLY INFORMATION COLLECTION - Don't waste RAG/web search
        logger.info("💬 User wants calculation but missing deductions - asking for info...")
        clarification = generate_clarification_request(missing_info, user_mood, query, user_preferences)
        
        if clarification:
            state["paye_answer"] = clarification
            state["model_used"] = "meta_prompt"
            logger.info("✅ PAYE Agent: Requested missing deduction info")
            return state
    
    # Proceed with RAG search
    logger.info("📖 Querying knowledge base...")

    # Inject user context into RAG query if personal data is warranted
    rag_context = chat_history
    if user_context_block:
        rag_context = f"{user_context_block}\n\n{chat_history}"
        logger.info("📋 Injected user profile into RAG context")

    result = query_rag(
        user_query=query,
        collection_type="paye",
        top_k=3,
        return_sources=True,
        chat_history=rag_context,
        user_preferences=user_preferences
    )

    combined_context = result['answer']

    # Prepend user data block to LLM context if available
    if user_context_block:
        combined_context = f"{user_context_block}\n\n{combined_context}"
    
    # Continue with decision tree
    logger.info(f"📊 Approach: {approach}, Mood: {user_mood}")
    
    # Continue with decision tree for other cases
    if approach == "collect" and needs_clarification and missing_info:
        # This shouldn't trigger (handled above) but just in case
        logger.info("💬 Generating clarification request...")
        clarification = generate_clarification_request(missing_info, user_mood, query, user_preferences)
        
        if clarification:
            state["paye_answer"] = clarification
        else:
            state["paye_answer"] = create_engagement_response(query, combined_context, chat_history, user_preferences)
            
    elif approach == "conditional" and missing_info:
        # CONDITIONAL ANSWER FOR IMPATIENT USERS
        logger.info("⚡ Generating conditional answer for impatient user...")
        state["paye_answer"] = generate_conditional_answer(query, missing_info, combined_context, user_preferences)
        
    elif approach == "collect" and user_mood == "engaged":
        # ENGAGEMENT MODE: Step-by-step educational response
        logger.info("📚 Creating engaging educational response...")
        state["paye_answer"] = create_engagement_response(query, combined_context, chat_history, user_preferences)
        
    else:
        # DIRECT ANSWER: User has provided enough info or asking general question
        logger.info("✅ Using direct RAG answer...")
        state["paye_answer"] = result["answer"]
    
    state["model_used"] = result["model_used"]
    
    # Add sources if not already present
    if "sources" not in state or not state["sources"]:
        state["sources"] = result.get("sources", [])
    else:
        state["sources"].extend(result.get("sources", []))
    
    logger.info("✅ PAYE Calculation Agent completed")
    return state

