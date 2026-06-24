import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history
from src.tools.web_search import search_web
from src.agent.meta_prompt import (
    analyze_query_for_tax_calculation,
    generate_clarification_request,
    generate_conditional_answer,
    create_engagement_response
)
from src.agent.context_enrichment import get_paye_instructions

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
    
    # 🆕 CHECK FOR PRE-LOADED USER DATA
    paye_context = state.get("paye_user_context")
    has_user_data = paye_context is not None
    
    if has_user_data:
        logger.info("✅ User has pre-loaded tax data from main app!")
    else:
        logger.info("⚠️  No tax data found - will ask questions")
    
    # META-PROMPTING: Analyze the query first
    logger.info("🤔 Analyzing query to determine approach...")
    meta_analysis = analyze_query_for_tax_calculation(query, chat_history)
    
    # DECISION TREE based on meta-analysis
    approach = meta_analysis.get("approach", "direct")
    user_mood = meta_analysis.get("user_mood", "neutral")
    needs_clarification = meta_analysis.get("needs_clarification", False)
    missing_info = meta_analysis.get("missing_info", [])
    is_calculation_request = meta_analysis.get("is_calculation_request", False)
    
    logger.info(f"📊 Calc request: {is_calculation_request}, Approach: {approach}, Mood: {user_mood}, Needs info: {needs_clarification}")
    
    # 🆕 OVERRIDE CLARIFICATION IF USER DATA EXISTS
    # If user wants calculation AND we have their data → skip questions!
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
    
    # Otherwise, proceed with RAG search (no web search for PAYE - faster)
    logger.info("📖 Querying knowledge base...")
    
    # 🆕 INCLUDE USER CONTEXT IN RAG QUERY if available
    rag_context = chat_history
    if paye_context:
        rag_context = f"{paye_context}\n\n{chat_history}"
        logger.info("📋 Added pre-loaded tax data to RAG context")
    
    result = query_rag(
        user_query=query,
        collection_type="paye",
        top_k=3,
        return_sources=True,
        chat_history=rag_context,  # 🆕 Enhanced with user data
        user_preferences=user_preferences
    )
    
    combined_context = result['answer']
    
    # 🆕 PREPEND USER INSTRUCTIONS if data exists
    if paye_context:
        instructions = get_paye_instructions(has_context=True)
        combined_context = f"{instructions}\n\n{combined_context}"
    
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

