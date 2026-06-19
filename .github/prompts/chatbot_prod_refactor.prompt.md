# Plan: Production Readiness & Personalization for Nigerian Tax Chatbot

## 📊 **PROGRESS TRACKER**

### **Status Legend**
- ✅ **Completed** - Done and tested
- 🔄 **In Progress** - Currently working on
- ⏳ **Pending** - Not started yet
- ❌ **Blocked** - Waiting on dependencies

### **Phase Status Overview**
| Phase | Name | Status | Timeline | Blockers |
|-------|------|--------|----------|----------|
| **Phase 0** | Database & User Profiles | 🔄 **80% Complete** | Days 0-2 | ✅ None - Ready to implement |
| **Phase 1** | Critical Security | ⏳ Pending | Days 2-4 | Phase 0 complete |
| **Phase 2** | Reliability & Errors | ⏳ Pending | Days 5-7 | Phase 1 complete |
| **Phase 3** | PostgreSQL Optimization | ⏳ Pending | Days 8-10 | Phase 0 complete |
| **Phase 4** | Multi-LLM Providers | ⏳ Pending | Days 11-13 | Phase 2 complete |
| **Phase 5** | Monitoring & Observability | ⏳ Pending | Days 14-16 | Phase 4 complete |
| **Phase 6** | RAG - BM25 Keyword | ⏳ Pending | Days 17-19 | Phase 5 complete |
| **Phase 7** | RAG - Hybrid Search | ⏳ Pending | Days 20-22 | Phase 6 complete |
| **Phase 8** | Final Polish & Testing | ⏳ Pending | Days 23-25 | All phases complete |

---

## TL;DR
The chatbot has solid architecture (LangGraph multi-agent, RAG, memory) but needs production hardening AND user personalization. **Custom database complete + main app schema discovered** - ready to implement personalization!

**🎯 COMPLETED**: 
- Custom PostgreSQL database (4 tables, UUID format, pure SQLAlchemy)
- ✅ Main app schema analyzed (5 tables: profiles, financial_income, financial_expenses, tax_calculations)

**🎯 NEXT**: 
- Update API endpoints to use chat tables (ChatManager integration)
- Build data service to query main app tables
- Pre-load user tax data into session state
- Update PAYE agent to skip repetitive questions

**🚀 QUICK WINS**: Cerebras LLM (unlimited free) + Hybrid RAG (BM25 + Semantic) + User Personalization = **99%+ uptime, +40% accuracy, instant PAYE calculations, $0/month**

---

## 🎁 **FREE LLM API Keys Setup** (Do This First!)

| Provider | Model | Speed | Limits | Truly Free? | Setup Link |
|----------|-------|-------|--------|-------------|------------|
| **Cerebras** | llama-3.3-70b | ⚡ 1,800 tok/s | ✅ UNLIMITED | YES - Forever | [inference.cerebras.ai](https://inference.cerebras.ai/) |
| **Groq** *(current)* | llama-3.3-70b | ⚡ 500+ tok/s | 30 req/min | YES - Free tier | [console.groq.com](https://console.groq.com/) |
| **Cohere** *(current)* | command-r7b | ⚡ Medium | Trial + API | YES - Free tier exists | [cohere.com](https://cohere.com/) |

**Cascade Strategy**: Groq (primary) → Cerebras (secondary) → Cohere (fallback)  
**Expected uptime**: 99%+ | **Total cost**: $0/month forever

---

## ✅ **PHASE 0: Database & User Profile Integration** (Days 0-2)

**Goal**: Build foundation for personalized tax advice + integrate APIs with chat database  
**Status**: 🔄 **80% Complete**  
**Priority**: 🔴 **CRITICAL** - Foundation for all phases

### **Completed Tasks** ✅

#### 1. **Custom Database Setup** ✅ (DONE!)
- ✅ Created 4-table schema with `chat_` prefix
  - `chat_sessions` - Conversation tracking with UUID
  - `chat_messages` - Message history
  - `chat_summaries` - Context management (Phase 8)
  - `chat_users` - User tracking & rate limiting
- ✅ Pure SQLAlchemy implementation (no LangGraph dependency)
- ✅ UUID format for all IDs (PostgreSQL native)
- ✅ Connection pooling configured (pool_size=10, max_overflow=10)
- ✅ SessionStore for agent state management
- ✅ Repository pattern (ChatSessionRepository, ChatMessageRepository, ChatUserRepository)
- ✅ High-level ChatManager for easy API integration
- ✅ Main app schema analyzed (5 tables discovered)

**Files Created/Modified**:
- ✅ `src/database/models.py` - 4 models with UUID
- ✅ `src/database/connection.py` - Connection pooling
- ✅ `src/database/session_store.py` - Pure SQLAlchemy state storage
- ✅ `src/database/repository.py` - Data access layer
- ✅ `src/database/utils.py` - ChatManager utilities
- ✅ `src/database/__init__.py` - Clean exports
- ✅ `inspect_table.py` - Schema inspection tool (created for analysis)

---

### **Pending Tasks** ⏳

**✅ BLOCKER REMOVED**: Main app schema analyzed - 5 tables discovered!

#### 2. **Main App Data Service** ⏳ (3 hours)
**Approach**: Query existing main app tables directly (NO need to modify chat_users!)

**📊 Discovered Main App Schema**:

**Table 1: `profiles` (7 fields)** - Basic user info
```sql
id              UUID
user_id         UUID (Primary identifier)
email           TEXT
display_name    TEXT
avatar_url      TEXT (nullable)
created_at      TIMESTAMP
updated_at      TIMESTAMP
```
*Chatbot usage*: Greet user by name, not email confirmations

---

<!-- **Table 2: `financial_profiles` (4 fields)** - ⚠️ Currently EMPTY, skip for now
```sql
user_id            UUID
preferred_currency TEXT
created_at         TIMESTAMP
updated_at         TIMESTAMP
```
*Chatbot usage*: Can use later when populated -->

---

**Table 3: `financial_income` (8 fields)** - ⭐ **CRITICAL FOR PAYE**
```sql
id          UUID
user_id     UUID
amount      DECIMAL   -- e.g., 120000
frequency   TEXT      -- "monthly", "annual"
source      TEXT      -- "freelance", "salary", "business"
start_date  DATE (nullable)
notes       TEXT (nullable)
created_at  TIMESTAMP
```
*Chatbot usage*: 
- Auto-fill salary without asking
- Sum multiple income sources
- Know if freelance vs employed (different tax rules)

**Sample data**:
```json
{
  "amount": 120000,
  "frequency": "monthly",
  "source": "freelance"
}
```

---

**Table 4: `financial_expenses` (8 fields)** - Expense tracking
```sql
id          UUID
user_id     UUID
amount      DECIMAL   -- e.g., 15000
frequency   TEXT      -- "monthly", "annual"
category    TEXT      -- "data", "rent", "transport", "food"
start_date  DATE (nullable)
notes       TEXT (nullable)
created_at  TIMESTAMP
```
*Chatbot usage*:
- Calculate disposable income (income - expenses)
- Identify tax-deductible expenses (rent, pension)
- Budget optimization suggestions

---

**Table 5: `tax_calculations` (6 fields)** - ⭐ **MOST VALUABLE**
```sql
id              UUID
user_id         UUID
input_payload   JSON   -- Contains ALL deduction data
result_payload  JSON   -- Contains tax breakdown
rules_version   TEXT   -- "tax_rules_2024_v1"
created_at      TIMESTAMP
```

**`input_payload` structure** (ALL deductions here!):
```json
{
  "grossIncome": 77000,
  "frequency": "monthly",
  "incomeType": "freelance",
  "pensionContribution": 0,
  "nhfContribution": 0,
  "nhisContribution": 2,
  "otherDeductions": 11
}
```

**`result_payload` structure**:
```json
{
  "monthlyTax": 3941.24,
  "annualTax": 47294.84,
  "effectiveRate": 5.12,
  "taxableIncome": 539044,
  "consolidatedReliefAllowance": 384800,
  "breakdown": [
    {"band": "First ₦300,000", "rate": 0.07, "taxAmount": 21000},
    {"band": "Next ₦300,000", "rate": 0.11, "taxAmount": 26294.84}
  ],
  "explanation": "Based on your freelance income of ₦924,000..."
}
```

*Chatbot usage*:
- **Pre-fill ALL deductions** (pension, NHF, NHIS)
- Show tax history and trends
- Compare current vs previous calculations
- Reuse explanation format

---

**Implementation Plan**:

Create `src/services/main_app_data.py`:
```python
async def get_user_income_sources(user_id: str) -> List[Dict]
async def get_latest_tax_calculation(user_id: str) -> Optional[Dict]
async def get_user_expenses_by_category(user_id: str) -> Dict[str, float]
async def get_complete_user_context(user_id: str) -> Dict
```

**SQL Queries Needed**:
```sql
-- Get all income sources
SELECT amount, frequency, source 
FROM financial_income 
WHERE user_id = :user_id;

-- Get latest tax calculation (has ALL deduction data!)
SELECT input_payload, result_payload, created_at
FROM tax_calculations
WHERE user_id = :user_id
ORDER BY created_at DESC
LIMIT 1;

-- Get expenses by category
SELECT category, SUM(amount) as total
FROM financial_expenses
WHERE user_id = :user_id
GROUP BY category;
```

**Deliverable**: Service that queries 3 main app tables directly

---

#### 3. **Session Context Enrichment** ⏳ (2 hours)

Create `src/agent/context_enrichment.py`:

**Functions**:
```python
def build_paye_context(user_context: Dict) -> Optional[str]
    # Formats tax_calculations data for PAYE agent
    # Extracts: grossIncome, pensionContribution, nhfContribution, nhisContribution
    
def build_income_summary(user_context: Dict) -> Optional[str]
    # Formats financial_income data for financial advice agent
    # Shows: total monthly income, number of sources
    
async def enrich_agent_state(state: Dict, user_id: str) -> Dict
    # Main function - fetches data and adds to agent state
    # Returns enhanced state with "paye_user_context" and "income_summary_context"
```

**Example formatted context**:
```
═══════════════════════════════════════════════════
📋 USER TAX PROFILE (PRE-LOADED)
═══════════════════════════════════════════════════

💰 INCOME:
   • Gross Monthly: ₦77,000.00
   • Type: Freelance

🏦 DEDUCTIONS (On Record):
   • Pension: ₦0.00
   • NHF: ₦0.00
   • NHIS: ₦2.00
   • Other: ₦11.00

📊 LAST TAX RESULT:
   • Monthly Tax: ₦3,941.24
   • Effective Rate: 5.12%

⚠️ USE THIS DATA for calculations. Only ask if user mentions changes.
```

**Deliverable**: Context enrichment that pre-loads user data into agent state

---

#### 4. **Update PAYE Agent** ⏳ (2 hours)

Modify `src/agent/sub_agents/paye.py`:

**Add if/else logic**:
```python
async def paye_calculation_agent(state: AgentState) -> AgentState:
    query = state["query"]
    
    # CHECK IF USER DATA IS PRE-LOADED
    paye_context = state.get("paye_user_context")
    
    if paye_context:
        # ✅ USER DATA EXISTS - Use it!
        system_prompt = f"""
You are a Nigerian PAYE tax expert.

{paye_context}

INSTRUCTIONS:
- Use the pre-loaded data for calculations
- If user says "calculate my PAYE" → use data directly
- Only ask questions if user mentions changes

USER QUERY: {query}
"""
    else:
        # ❌ NO DATA - Ask questions
        system_prompt = f"""
⚠️ NO SALARY DATA ON RECORD

Ask user for:
- Monthly/annual gross salary
- Pension, NHF, NHIS contributions
- Other deductions

USER QUERY: {query}
"""
    
    # Run LLM with appropriate prompt...
```

**Deliverable**: PAYE agent with personalization logic

---

#### 5. **Update Main Agent Entry Point** ⏳ (1 hour)

Modify `src/agent/main_agent.py`:

```python
from src.agent.context_enrichment import enrich_agent_state

async def run_agent(query: str, user_id: str, thread_id: str):
    """Main agent with context enrichment."""
    
    # Initialize state
    initial_state = {
        "query": query,
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": []
    }
    
    # 🆕 ENRICH STATE WITH MAIN APP DATA
    enriched_state = await enrich_agent_state(initial_state, user_id)
    
    # Run agent with enriched state
    agent = get_compiled_agent()
    result = await agent.ainvoke(enriched_state)
    
    return result
    
    if user_profile:
        # 🆕 PERSONALIZED FINANCIAL ADVICE PROMPT
        income = user_profile.get("monthly_salary", 0)
        goal = user_profile.get("financial_goal", "general savings")
        risk = user_profile.get("risk_tolerance", "medium")
        
        personalized_prompt = f"""
USER FINANCIAL PROFILE:
- Monthly Income: ₦{income:,}
- Financial Goal: {goal}
- Risk Tolerance: {risk}
- Current Investments: {"Yes" if user_profile.get('has_investments') else "No"}

Provide personalized advice considering:
1. Income level (suggest realistic savings amounts)
2. Financial goal (tailor recommendations)
3. Risk tolerance (appropriate investment options)
"""
        state["financial_profile_context"] = personalized_prompt
    
    # Continue with existing financial advice logic...
```

**Deliverable**: PAYE agent with personalization logic

---

### **Phase 0 Deliverables Summary** (UPDATED)

#### ✅ **Completed** (80%)
1. Custom 4-table database schema (chat_sessions, chat_messages, chat_summaries, chat_users)
2. Pure SQLAlchemy implementation (no LangGraph dependency)
3. UUID format for all IDs (PostgreSQL native)
4. Repository pattern & ChatManager utilities
5. Connection pooling configured (pool_size=10, max_overflow=10)
6. SessionStore for agent state management
7. ✅ **Main app schema analyzed** (5 tables discovered via inspect_table.py)

#### ⏳ **Pending** (20%)
8. **API Integration** - Update endpoints to use ChatManager - 2 hours
9. Main app data service (`src/services/main_app_data.py`) - 3 hours
10. Session context enrichment (`src/agent/context_enrichment.py`) - 2 hours
11. Update PAYE agent with if/else logic - 2 hours
12. Update main agent entry point - 1 hour

**Total Remaining**: 10 hours (2 hours API + 8 hours personalization)

#### ⏳ **Pending** (20%)
8. Main app data service (`src/services/main_app_data.py`) - 3 hours
9. Session context enrichment (`src/agent/context_enrichment.py`) - 2 hours
10. Update PAYE agent with if/else logic - 2 hours
11. Update main agent entry point - 1 hour
12. **Update API endpoints to use chat tables** - 2 hours

#### 6. **API Integration with Chat Tables** ⏳ (NEW - 2 hours)

**Goal**: Update all API routes to use the new custom chat database tables

**Files to Modify**:
```
src/api/routes/chat_agent.py        # Main chat endpoint
src/api/routes/prompts.py           # Prompt management
src/api/routes/webhook.py           # WhatsApp integration (if active)
```

**Changes Needed**:

1. **Update Chat Endpoint** - `src/api/routes/chat_agent.py`:
```python
from src.database import ChatManager

@router.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id  # or extract from auth
    
    # 🆕 START NEW SESSION using ChatManager
    thread_id = await ChatManager.start_session(
        user_id=user_id,
        extra_metadata={"source": "api", "ip": request.client.host}
    )
    
    # 🆕 SAVE USER MESSAGE
    await ChatManager.add_user_message(thread_id, request.message)
    
    # Run agent (already enriched in main_agent.py)
    response = await run_agent(
        query=request.message,
        user_id=user_id,
        thread_id=thread_id
    )
    
    # 🆕 SAVE ASSISTANT RESPONSE
    await ChatManager.add_assistant_message(
        thread_id=thread_id,
        content=response["output"],
        agent_type=response.get("agent_type", "general"),
        tokens_used=response.get("tokens_used", 0)
    )
    
    return {
        "thread_id": thread_id,
        "response": response["output"]
    }
```

2. **Add Conversation History Endpoint**:
```python
@router.get("/chat/history/{thread_id}")
async def get_history(thread_id: str, limit: int = 50):
    """Get conversation history."""
    messages = await ChatManager.get_session_history(thread_id, limit)
    return {"thread_id": thread_id, "messages": messages}
```

3. **Add User Sessions Endpoint**:
```python
@router.get("/chat/sessions")
async def get_user_sessions(user_id: str, limit: int = 10):
    """Get user's recent chat sessions."""
    from src.database.repository import ChatSessionRepository
    
    sessions = await ChatSessionRepository.get_user_sessions(user_id, limit)
    return {
        "user_id": user_id,
        "sessions": [
            {
                "thread_id": s.id,
                "created_at": s.created_at,
                "message_count": s.message_count,
                "status": s.status
            }
            for s in sessions
        ]
    }
```

4. **Update Rate Limiting** (uses chat_users table):
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    user_id = request.user_id
    
    # 🆕 CHECK RATE LIMIT using ChatManager
    allowed = await ChatManager.check_user_rate_limit(
        user_id=user_id,
        max_requests=20,  # 20 requests per hour
        window_minutes=60
    )
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )
    
    # ... rest of chat logic
```

**Benefits**:
- ✅ All conversations stored in PostgreSQL
- ✅ Conversation history retrieval
- ✅ Rate limiting per user (from chat_users table)
- ✅ Token usage tracking
- ✅ Session management (start/end/archive)

**Deliverable**: APIs fully integrated with custom chat database tables

---

#### 📋 **What Questions the Chatbot Will Skip After Implementation**

**❌ BEFORE Personalization** (user must answer):
- "What's your monthly salary?"
- "What's your pension contribution?"
- "Are you enrolled in NHF?"
- "Are you enrolled in NHIS?"
- "Any other deductions?"
- "Is this freelance or salary income?"

**✅ AFTER Personalization** (auto-loaded from `tax_calculations` table):
- Salary: From `input_payload.grossIncome`
- Pension: From `input_payload.pensionContribution`
- NHF: From `input_payload.nhfContribution`
- NHIS: From `input_payload.nhisContribution`
- Other deductions: From `input_payload.otherDeductions`
- Income type: From `input_payload.incomeType` or `financial_income.source`

**User Experience**:
- User: "calculate my PAYE"
- Chatbot: *instantly shows calculation using pre-loaded data* ✅
- Only asks questions if user mentions changes ("my salary increased")

**Time Estimate**: 10 hours remaining (8 hours personalization + 2 hours API integration)  
**Blocker**: ✅ **REMOVED** - Schema discovered, ready to implement!

---

## 🔒 **PHASE 1: Critical Security Fixes** (Days 2-4)

**Goal**: Prevent abuse, protect API  
**Status**: ⏳ Pending  
**Priority**: 🔴 **CRITICAL** - Must complete before launch  
**Dependencies**: Phase 0 complete

### **Tasks**

1. **Add rate limiting** (2 hours)
   - Install: `pip install slowapi`
   - Add to `src/main.py`: 10 req/min per IP on `/chat`
   - Cost guard: max 20 LLM calls per user per hour

2. **Fix CORS** (30 mins)
   - Replace `"*"` with `ALLOWED_ORIGINS` env variable
   - Default: `["http://localhost:3000", "https://yourdomain.com"]`

3. **Input validation** (2 hours)
   - Max query length: 500 chars
   - Sanitize HTML/script tags
   - Add Pydantic validators to all schemas

4. **Basic authentication header** (1 hour)
   - Enforce `api-key-header` on all routes except health checks
   - Remove WhatsApp webhook from auth requirement (dormant)

**Deliverables**: Protected API, no abuse vectors  
**Time**: 1.5 days  
**Test**: Try 15 requests in 1 min → should get 429 after 10th

---

## 🔄 **PHASE 2: Reliability & Error Handling** (Days 5-7)

**Goal**: Handle failures gracefully  
**Status**: ⏳ Pending  
**Dependencies**: Phase 1 complete

### **Tasks**

1. **Retry logic** (3 hours)
   - Install: `pip install tenacity`
   - Wrap `LLMManager.get_llm()` with `@retry(wait=wait_exponential(multiplier=2, min=2, max=8))`
   - Wrap Tavily, Cohere API calls

2. **Circuit breakers** (2 hours)
   - Install: `pip install pybreaker`
   - Add circuit breaker to LLM services
   - Open after 5 failures, half-open after 60s

3. **Structured logging** (2 hours)
   - Install: `pip install structlog`
   - Replace all `logging.info` with JSON logs
   - Add user_id, thread_id, latency tracking

**Deliverables**: System handles LLM failures, logs structured  
**Time**: 1.5 days  
**Test**: Disable Groq key → should fallback to Cohere without crash

**Note**: ✅ Global state cleanup already done in Phase 0 (SessionStore, ChatManager)

---

## 💾 **PHASE 3: PostgreSQL Indexes** (Days 8) - **QUICK WIN**

**Goal**: Add critical indexes (30 mins)  
**Status**: ⏳ Pending  
**Dependencies**: Phase 0 complete

### **Tasks**

1. **Add PostgreSQL indexes** (30 mins)
   - Add indexes to chat tables for faster queries:
     ```sql
     CREATE INDEX idx_chat_sessions_user_updated ON chat_sessions(user_id, updated_at DESC);
     CREATE INDEX idx_chat_messages_session_created ON chat_messages(session_id, created_at);
     CREATE INDEX idx_chat_users_last_activity ON chat_users(last_activity DESC);
     ```
   - Test: Conversation load should be <200ms

**Deliverables**: Database indexed, queries fast  
**Time**: 30 mins (quick win!)  
**Test**: Load 50-message conversation → <200ms

**Note**: ⏳ Connection pool tuning and query optimization can be deferred to post-launch based on actual load patterns

---

## 🤖 **PHASE 4: Multi-LLM Providers** (Days 11-13)

**Goal**: 99%+ uptime with free LLMs  
**Status**: ⏳ Pending  
**Dependencies**: Phase 2 complete (retry/circuit breakers needed)

### **Tasks**

1. **Add Cerebras as primary** (2 hours)
   - Sign up: https://inference.cerebras.ai (free forever, unlimited)
   - Install: `pip install langchain-cerebras`
   - Update `LLMManager._initialize_cerebras()`
   - Make it primary (fastest, truly free)

2. **Update cascade logic** (1 hour)
   - Simplify to 3 providers: Cerebras → Groq → Cohere
   - Test cascade: disable Cerebras → should use Groq

3. **Test full cascade** (1 hour)
   - Disable Cerebras → verify Groq pickup
   - Disable Groq → verify Cohere pickup
   - Log which provider handled each request

**Deliverables**: 3-provider cascade (all free), uptime 99%+  
**Time**: 1 day  
**Test**: Disable primary → verify seamless failover

---

## 📊 **PHASE 5: Monitoring with LangSmith** (Days 14-15) - **SIMPLIFIED**

**Goal**: See what's happening in production (NO local servers!)  
**Status**: ⏳ Pending  
**Dependencies**: Phase 4 complete

### **Tasks**

1. **Set up LangSmith** (2 hours) - **FREE & EASY**
   - Sign up: https://smith.langchain.com (free tier: 5,000 traces/month)
   - Install: `pip install langsmith`
   - Add to `.env`: `LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2=true`
   - Automatic tracing of all LangGraph agent runs
   - **Benefits**: See entire agent execution, token counts, latency, errors

2. **Simple health checks** (1 hour)
   - `/health` - liveness check
   - `/health/deep` - verify DB + LLM connectivity
   - No Prometheus, no Grafana, no local servers!

3. **PostgreSQL logging table** (2 hours)
   - Create `system_logs` table:
     ```sql
     CREATE TABLE system_logs (
       id UUID PRIMARY KEY,
       timestamp TIMESTAMP,
       level TEXT,  -- info/warning/error
       message TEXT,
       user_id UUID,
       thread_id UUID,
       tokens_used INT,
       latency_ms INT,
       llm_provider TEXT
     );
     ```
   - Log all requests to PostgreSQL (simple query to see stats)

4. **Uptime monitoring** (30 mins)
   - Uptime Robot free tier: https://uptimerobot.com
   - Monitor `/health` endpoint every 5 minutes
   - Email alert if down

**Deliverables**: LangSmith tracing + basic health checks + PostgreSQL logs  
**Time**: 1.5 days  
**Cost**: $0 (LangSmith free tier is generous!)

**Why LangSmith?**
- ✅ Zero infrastructure (cloud-hosted)
- ✅ Automatic LangGraph integration
- ✅ See token usage, costs, latency
- ✅ Debug agent decisions visually
- ✅ No Prometheus, Grafana, or local servers

---

## 🔍 **PHASE 6: Hybrid RAG (BM25 + Semantic)** (Days 17-19) - **COMBINED**

**Goal**: Better retrieval accuracy with hybrid search  
**Status**: ⏳ Pending  
**Dependencies**: Phase 5 complete

**Why Hybrid?** Semantic search misses exact phrases ("Section 24"), BM25 finds keywords. Combine both = best of both worlds!

### **Tasks**

1. **Implement Hybrid Retrieval** (4 hours)
   - Install: `pip install rank-bm25`
   - Create `src/tools/retrieval/hybrid_retriever.py`
   - Build BM25 index at startup (in-memory, ~5MB)
   - Implement Reciprocal Rank Fusion (RRF):
     ```python
     # Get top 10 from semantic search
     semantic_results = vector_search(query, k=10)
     
     # Get top 10 from BM25
     bm25_results = bm25_search(query, k=10)
     
     # Merge with RRF (simple weighted average)
     final_results = reciprocal_rank_fusion(semantic_results, bm25_results)
     ```

2. **Update RAG pipeline** (2 hours)
   - Modify `src/tools/rag.py` to use hybrid retriever
   - Test with queries like:
     - "PAYE calculation" (should use semantic)
     - "Section 24 of tax law" (should use BM25)
     - "How do I calculate my tax?" (hybrid combines both)

3. **Simple evaluation** (2 hours)
   - Create 10 test queries
   - Compare old vs new retrieval
   - Log which performs better
   - No complex metrics needed yet!

**Deliverables**: Hybrid RAG working, better accuracy  
**Time**: 2 days  
**Expected improvement**: +30-40% for keyword queries

**Note**: ⏳ Advanced features (contextual embeddings, flashrank reranking) can be deferred to post-launch

---

## 🎨 **PHASE 7: Context Intelligence & Production Polish** (Days 20-23) - **UPGRADED**

**Goal**: Smart context management + learning from user behavior  
**Status**: ⏳ Pending  
**Dependencies**: Phase 6 complete

**Your Vision**: Agent that learns, manages context intelligently, and gets smarter over time!

**✅ Key Features:**
1. **Agent Settings** - Centralized LLM config (context window, token limits)
2. **Smart Token Management** - Auto-summarize when reaching 80% capacity
3. **User Preferences Learning** - Save & load learned preferences across sessions
4. **Context Preparation Layer** - Clean, ready-to-use context before router
5. **Intelligence Validation** - Ensure everything works

**📁 New Files to Create**:
- `src/configurations/agent_settings.py` - LLM configs (context windows, token limits)
- `src/agent/token_manager.py` - Token counting & smart summarization
- `src/database/models.py` - Add UserPreference table
- `src/agent/context_preparation.py` - Central context orchestration
- `src/agent/preference_learner.py` - Background learning task
- `tests/test_intelligence.py` - Intelligence validation tests
- `tests/load_test.py` - Load testing with Locust

### **Architecture Flow**
```
User Request → FastAPI Endpoint
    ↓
Context Preparation Layer (NEW! 🧠)
    │
    ├─→ Load main app profile data (tax_calculations table)
    │   └─→ grossIncome, pensionContribution, nhfContribution, etc.
    │
    ├─→ Load learned user preferences (user_preferences table)
    │   └─→ communication_style, common_questions, topic_interests
    │
    ├─→ Load conversation history (chat_messages table)
    │   └─→ Last N messages from current session
    │
    ├─→ Count tokens (TokenManager)
    │   ├─→ System prompt tokens
    │   ├─→ Profile data tokens
    │   ├─→ Preferences tokens
    │   ├─→ Conversation history tokens
    │   └─→ Total < 80% of context window?
    │       ├─→ YES: Pass all messages
    │       └─→ NO: Summarize old messages, keep recent 5
    │
    └─→ Build context package
        └─→ {messages, user_profile, user_preferences, metadata}
    ↓
Router Agent (receives ready-to-use context)
    ├─→ Routing decision based on query + preferences
    └─→ Route to: tax_policy | paye | financial_advice | combined
    ↓
Sub-agent executes (uses profile + preferences)
    ↓
Response generated
    ↓
Background Task (async, non-blocking)
    └─→ Update user_preferences table
        ├─→ Track common questions
        ├─→ Update topic interests
        └─→ Increment session count
    ↓
Return response to user
```

**Key Benefits**:
1. ✅ **Clean separation**: Context prep separate from agent logic
2. ✅ **Pre-loaded data**: Router gets everything it needs
3. ✅ **Smart truncation**: Only summarizes when truly needed
4. ✅ **Learning loop**: Background task updates preferences
5. ✅ **No blocking**: Preference learning doesn't slow response

### **Tasks**

#### **1. Agent Settings Configuration** (2 hours) - **FOUNDATION**
Create `src/configurations/agent_settings.py`:
```python
"""
Centralized LLM configuration and token management settings
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class LLMConfig:
    """Configuration for each LLM provider"""
    name: str
    context_window: int  # Max tokens
    system_prompt_tokens: int  # Estimated
    max_output_tokens: int
    
    @property
    def available_context(self) -> int:
        """Tokens available for conversation history + user data"""
        return self.context_window - self.system_prompt_tokens - self.max_output_tokens

# LLM Configurations
LLM_CONFIGS: Dict[str, LLMConfig] = {
    "groq": LLMConfig(
        name="llama-3.3-70b-versatile",
        context_window=8000,
        system_prompt_tokens=500,  # Measure actual
        max_output_tokens=2000
    ),
    "cerebras": LLMConfig(
        name="llama-3.3-70b",
        context_window=8000,
        system_prompt_tokens=500,
        max_output_tokens=2000
    ),
    "cohere": LLMConfig(
        name="command-r7b-12-2024",
        context_window=4096,
        system_prompt_tokens=400,
        max_output_tokens=1500
    )
}

# Context Management Settings
CONTEXT_THRESHOLD = 0.80  # Summarize when reaching 80% of available context
MIN_MESSAGES_BEFORE_SUMMARY = 15  # Don't summarize too early
SUMMARY_KEEP_RECENT = 5  # Keep last 5 messages after summarization
```

**Test**: Import and verify configurations

---

#### **2. Token Management Utility** (3 hours) - **SMART TRUNCATION**
Create `src/agent/token_manager.py`:
```python
"""
Smart token counting and context management
"""
import tiktoken
from typing import List, Dict, Tuple
from src.configurations.agent_settings import LLM_CONFIGS, CONTEXT_THRESHOLD

class TokenManager:
    def __init__(self, provider: str = "groq"):
        self.config = LLM_CONFIGS[provider]
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Similar tokenization
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
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
        from src.configurations.agent_settings import MIN_MESSAGES_BEFORE_SUMMARY
        
        threshold_tokens = self.config.available_context * CONTEXT_THRESHOLD
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
            from src.configurations.agent_settings import SUMMARY_KEEP_RECENT
            from src.services.llm import LLMManager
            
            # Summarize old messages
            old_messages = messages[:-SUMMARY_KEEP_RECENT]
            recent_messages = messages[-SUMMARY_KEEP_RECENT:]
            
            # Create summary
            summary_prompt = f"Summarize this conversation concisely:\n{old_messages}"
            llm = LLMManager.get_llm(provider=llm_provider)
            summary = await llm.ainvoke(summary_prompt)
            
            # Replace old messages with summary
            summary_message = {
                "role": "system",
                "content": f"[Previous conversation summary]: {summary.content}"
            }
            
            prepared_messages = [summary_message] + recent_messages
            return prepared_messages, True
        
        return messages, False
```

**Test**: 30-message conversation → should trigger summarization at message 20

---

#### **3. User Preferences Table** (2 hours) - **LEARNING MECHANISM**
Update `src/database/models.py` - add new table:
```python
class UserPreference(Base):
    """
    Learned user preferences across sessions
    Helps agent get smarter over time
    """
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)  # From main app
    
    # Learned preferences
    preferred_communication_style = Column(String)  # "concise" | "detailed" | "technical"
    common_questions = Column(JSON, default=list)  # Track frequent queries
    topic_interests = Column(JSON, default=dict)  # {"paye": 15, "tax_policy": 8}
    calculation_defaults = Column(JSON, default=dict)  # Preferred PAYE inputs
    
    # Metadata
    total_sessions = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_user_pref_user_updated", "user_id", "last_updated"),
    )
```

**Migration**:
```bash
alembic revision -m "add_user_preferences_table"
alembic upgrade head
```

---

#### **4. Context Preparation Layer** (4 hours) - **CENTRAL ORCHESTRATION**
Create `src/agent/context_preparation.py`:
```python
"""
Central context preparation - loads everything before router
"""
from typing import Dict, List, Optional
from src.database.utils import ChatManager
from src.database.repository import ChatUserRepository
from src.agent.token_manager import TokenManager
from src.database.connection import get_async_engine
from sqlalchemy import text

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
        # 1. Load conversation history
        messages = await ChatManager.get_session_history(thread_id)
        
        # 2. Load main app profile data (from tax_calculations)
        main_app_data = await self._load_main_app_data(user_id)
        
        # 3. Load learned user preferences
        user_preferences = await self._load_user_preferences(user_id)
        
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
    
    async def _load_main_app_data(self, user_id: str) -> Dict:
        """Load user data from main app tables"""
        engine = get_async_engine()
        
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
            
            return {"has_tax_data": False}
    
    async def _load_user_preferences(self, user_id: str) -> Dict:
        """Load learned preferences from user_preferences table"""
        engine = get_async_engine()
        
        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT preferred_communication_style, common_questions, 
                           topic_interests, calculation_defaults
                    FROM user_preferences 
                    WHERE user_id = :user_id
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "communication_style": row[0],
                    "common_questions": row[1],
                    "topic_interests": row[2],
                    "calculation_defaults": row[3]
                }
            
            return {}  # First time user
```

**Test**: Load context for existing user → verify all data present

---

#### **5. Background Preference Learning** (3 hours) - **GET SMARTER**
Create `src/agent/preference_learner.py`:
```python
"""
Background task to learn user preferences after session ends
"""
from typing import List, Dict
from src.database.connection import get_async_engine
from sqlalchemy import text
import asyncio

class PreferenceLearner:
    """Learns from user behavior and updates preferences"""
    
    async def update_preferences_after_session(
        self,
        user_id: str,
        session_messages: List[Dict],
        agent_types_used: List[str]
    ):
        """
        Background task: analyze session and update user preferences
        Call this after user session ends (no new messages for 30+ mins)
        """
        engine = get_async_engine()
        
        async with engine.begin() as conn:
            # Check if preferences exist
            result = await conn.execute(
                text("SELECT id FROM user_preferences WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            exists = result.fetchone()
            
            if not exists:
                # Create new preferences
                await conn.execute(
                    text("""
                        INSERT INTO user_preferences 
                        (user_id, preferred_communication_style, common_questions, 
                         topic_interests, total_sessions)
                        VALUES (:user_id, :style, :questions, :interests, 1)
                    """),
                    {
                        "user_id": user_id,
                        "style": self._detect_communication_style(session_messages),
                        "questions": self._extract_questions(session_messages),
                        "interests": self._count_topics(agent_types_used)
                    }
                )
            else:
                # Update existing preferences
                await conn.execute(
                    text("""
                        UPDATE user_preferences 
                        SET 
                            common_questions = common_questions || :new_questions::jsonb,
                            topic_interests = topic_interests || :new_interests::jsonb,
                            total_sessions = total_sessions + 1,
                            last_updated = NOW()
                        WHERE user_id = :user_id
                    """),
                    {
                        "user_id": user_id,
                        "new_questions": self._extract_questions(session_messages),
                        "new_interests": self._count_topics(agent_types_used)
                    }
                )
    
    def _detect_communication_style(self, messages: List[Dict]) -> str:
        """Detect if user prefers concise or detailed responses"""
        user_messages = [m for m in messages if m.get("role") == "user"]
        avg_length = sum(len(m.get("content", "")) for m in user_messages) / len(user_messages)
        
        if avg_length < 50:
            return "concise"
        elif avg_length > 150:
            return "detailed"
        return "balanced"
    
    def _extract_questions(self, messages: List[Dict]) -> List[str]:
        """Extract common question patterns"""
        questions = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                if "calculate" in content and "paye" in content:
                    questions.append("paye_calculation")
                elif "tax reform" in content or "2026" in content:
                    questions.append("tax_policy")
        return questions[:5]  # Keep top 5
    
    def _count_topics(self, agent_types: List[str]) -> Dict[str, int]:
        """Count which topics user asks about most"""
        from collections import Counter
        return dict(Counter(agent_types))
```

**Background Task Integration** in `src/api/routes/chat_agent.py`:
```python
# After sending response, schedule background task
background_tasks.add_task(
    preference_learner.update_preferences_after_session,
    user_id=user_id,
    session_messages=messages,
    agent_types_used=["paye", "tax_policy"]  # Track from agent state
)
```

**When to Trigger Background Learning?**
- **Option 1 (Simple)**: After EVERY message (small overhead, always up-to-date)
- **Option 2 (Efficient)**: Only when session is "complete":
  - User hasn't sent message in 30+ minutes
  - User explicitly ends chat
  - Session has 5+ messages (minimum for learning)
  
**Recommendation**: Use Option 1 for simplicity at launch, optimize later if needed

---

#### **6. Update Main Agent Entry Point** (2 hours)
Update `src/agent/main_agent.py` to use context preparation:
```python
from src.agent.context_preparation import ContextPreparator

async def run_agent(user_id: str, thread_id: str, query: str, provider: str = "groq"):
    """Main agent entry with context preparation"""
    
    # Prepare full context BEFORE router
    preparator = ContextPreparator(provider)
    context = await preparator.prepare_full_context(
        user_id=user_id,
        thread_id=thread_id,
        current_query=query,
        provider=provider
    )
    
    # Pass clean context to router
    from src.agent.compiled_agent import compiled_agent
    result = await compiled_agent.ainvoke({
        "messages": context["messages"],
        "user_profile": context["user_profile"],
        "user_preferences": context["user_preferences"],
        "query": query
    })
    
    return result
```

---

#### **7. Intelligence Validation** (4 hours)
Create `tests/test_intelligence.py`:
```python
"""
Test chatbot intelligence and context management
"""
import pytest
from src.agent.main_agent import run_agent

@pytest.mark.asyncio
async def test_context_summarization():
    """Test that long conversations get summarized"""
    # Simulate 25-message conversation
    for i in range(25):
        await run_agent(
            user_id="test_user",
            thread_id="test_thread",
            query=f"Message {i}"
        )
    
    # Should have triggered summarization
    # Verify by checking message count in state

@pytest.mark.asyncio
async def test_preference_learning():
    """Test that user preferences are learned"""
    # User asks PAYE questions multiple times
    for _ in range(3):
        await run_agent(
            user_id="test_user_2",
            thread_id="test_thread_2",
            query="Calculate my PAYE"
        )
    
    # Check user_preferences table
    # Should have "paye_calculation" in common_questions

@pytest.mark.asyncio
async def test_profile_preload():
    """Test that main app data is pre-loaded"""
    response = await run_agent(
        user_id="user_with_tax_data",
        thread_id="new_thread",
        query="Calculate my PAYE"
    )
    
    # Should NOT ask for salary (already has it from tax_calculations)
    assert "what is your salary" not in response.lower()
```

Run tests:
```bash
pytest tests/test_intelligence.py -v
```

---

#### **8. Load Testing** (3 hours)
Create `tests/load_test.py`:
```python
from locust import HttpUser, task, between

class TaxChatbotUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def ask_paye_question(self):
        self.client.post("/chat", json={
            "user_id": "test_user",
            "query": "Calculate my PAYE tax"
        })
```

Run load test:
```bash
locust -f tests/load_test.py --headless -u 50 -r 10 --run-time 5m
```

Target: p95 latency <3s, 0% errors

---

**Deliverables**: 
- ✅ Agent settings configuration
- ✅ Smart token management (auto-summarization)
- ✅ User preferences learning (gets smarter over time!)
- ✅ Context preparation layer (clean architecture)
- ✅ Intelligence validated (tests pass)
- ✅ Load tested (50 users, <3s latency)

**Time**: 3 days  
**Complexity**: Medium-High (but worth it for intelligence!)

**Why This Makes Chatbot SMARTER**:
1. ✅ **Never forgets**: Loads preferences from past sessions
2. ✅ **Learns behavior**: Tracks common questions, preferred topics
3. ✅ **Smart context**: Auto-summarizes when needed, not before
4. ✅ **Pre-loaded data**: Uses main app profile (no repeated questions)
5. ✅ **Clean architecture**: Context prep happens BEFORE routing

**The chatbot will be MUCH smarter!** 🧠🚀

---

## 📊 **Phase Dependencies & Timeline** (UPDATED - WITH INTELLIGENCE)

```
Days 0-2:   Phase 0 (Database & Personalization) ← 🔄 80% COMPLETE
         ↓
Days 2-4:   Phase 1 (Security) ← 🔴 MUST DO BEFORE LAUNCH
         ↓
Days 5-6:   Phase 2 (Reliability) ← Simplified (removed redundant tasks)
         ↓
Day 8:      Phase 3 (PostgreSQL Indexes) ← 30 mins only! Quick win
         ↓
Days 11-12: Phase 4 (Multi-LLM) ← 3 free providers
         ↓
Days 14-15: Phase 5 (LangSmith Monitoring) ← NO local servers!
         ↓
Days 17-19: Phase 6 (Hybrid RAG) ← Combined BM25 + Semantic
         ↓
Days 20-23: Phase 7 (Context Intelligence + Learning) ← 🧠 SMART FEATURES
```

**Critical Path**: Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5  
**Launch Minimum**: Phases 0-4 COMPLETE (2 weeks)  
**Recommended**: Phases 0-5 COMPLETE (2.5 weeks - includes monitoring)  
**Ideal**: All phases COMPLETE (3 weeks - includes intelligence features!)

---

## 🎯 **Launch Criteria** (UPDATED)

### **Minimum Viable Launch** (2 weeks)
- ✅ Phase 0 COMPLETE (Database + API + Personalization)
- ✅ Phase 1 COMPLETE (Security - rate limiting, CORS, validation)
- ✅ Phase 2 COMPLETE (Reliability - retry logic, circuit breakers)
- ✅ Phase 3 COMPLETE (Indexes - 30 mins)
- ✅ Phase 4 COMPLETE (3 LLM providers with fallback)

### **Recommended Launch** (2.5 weeks)
- ✅ All above + Phase 5 COMPLETE (LangSmith monitoring)

### **Ideal Launch** (3 weeks) - **WITH INTELLIGENCE** 🧠
- ✅ All above + Phase 6 COMPLETE (Hybrid RAG)
- ✅ Phase 7 COMPLETE (Context intelligence + preference learning)
  - Smart token management (auto-summarization at 80%)
  - User preference learning (gets smarter over time)
  - Context preparation layer (clean architecture)
  - Intelligence validated (tests pass)
  - Load tested (50 users, <3s latency)

---

## 🆚 **Old vs New Plan Comparison**

| Aspect | Old Plan | New Plan | Why Changed? |
|--------|----------|----------|--------------|
| **Phase 2** | 4 tasks, 2 days | 3 tasks, 1.5 days | Removed redundant checkpointer cleanup (done in Phase 0) |
| **Phase 3** | 3 tasks, 1.5 days | 1 task, 30 mins | Just indexes; defer tuning to post-launch |
| **Phase 5** | Prometheus + Grafana + local servers | LangSmith only | No infrastructure! Cloud-hosted, easier |
| **Phase 6 & 7** | Separate BM25 + Hybrid (5 days) | Combined (2 days) | Build hybrid directly, skip intermediate |
| **Phase 7** | Generic polish | Intelligence features | Agent settings, token mgmt, preference learning, context prep |
| **Total Time** | 25 days | ~21 days | More realistic, with intelligence! |

---

## ✅ **Success Metrics** (UPDATED)
        ↓
Days 14-16: Phase 5 (Monitoring) ← Depends on Phase 4
        ↓
Days 17-19: Phase 6 (BM25) ← Depends on Phase 5
        ↓
Days 20-22: Phase 7 (Hybrid RAG) ← Depends on Phase 6
        ↓
Days 23-25: Phase 8 (Final Polish) ← Depends on ALL previous
```

**Critical Path**: Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5  
**Launch Minimum**: Phases 0-4 COMPLETE  
**Recommended**: Phases 0-5 COMPLETE (includes monitoring)  
**Optimization**: Phases 6-8 (defer if time-constrained)

---

## 🎯 **Launch Criteria**

### **Minimum Viable (2 weeks)**
- ✅ Phase 0 COMPLETE (Database + User Profiles)
- ✅ Phase 1 COMPLETE (Security)
- ✅ Phase 2 COMPLETE (Reliability)
- ✅ Phase 3 COMPLETE (PostgreSQL)
- ✅ Phase 4 COMPLETE (Multi-LLM)

### **Recommended (3 weeks)**
- ✅ All above + Phase 5 COMPLETE (Monitoring)

### **Ideal (4 weeks)**
---

## ✅ **Success Metrics** (UPDATED)

### **Phase 0 - Personalization**
- User profiles syncing successfully
- Agents using profile data in responses
- Reduced "please provide salary" questions
## ✅ **Success Metrics** (UPDATED)

### **Phase 0 - Personalization**
- User profiles syncing successfully from main app
- Agents using profile data in responses
- PAYE calculations use pre-loaded salary data
- No repeated questions about basic info

### **Technical Performance**
- Uptime: >99%
- Latency p95: <3s
- Error rate: <1%
- Cost: $0/month (free LLMs + LangSmith free tier)

### **RAG Quality** (after Phase 6)
- Retrieval Precision@3: >80%
- Hybrid better than semantic-only for keyword queries
- Exact phrase queries work (BM25 component)

### **Security**
- No successful abuse attempts
- Rate limiting working (20 req/hour/user)
- Input validation active
- Auth protecting endpoints

### **Intelligence Validation** (Phase 7) - **NEW!** 🧠
- ✅ Context management: 30+ message conversations work without overflow
- ✅ Auto-summarization: Triggers at 80% of context window
- ✅ Preference learning: User preferences saved after session
- ✅ Smart preloading: Next session loads previous preferences
- ✅ Profile integration: Uses main app tax data automatically
- ✅ 8/10 intelligence test scenarios pass
- ✅ Token counting accurate (±5% variance)
- ✅ Context preparation layer: <100ms overhead

---

## 📝 **Next Steps** (UPDATED WITH INTELLIGENCE)

### **Immediate Actions** (Today)
1. ✅ **Review and approve upgraded plan** (just done!)
2. **Complete Phase 0 remaining tasks** (10 hours):
   - API integration (2h)
   - Main app data service (3h)
   - Context enrichment (2h)
   - PAYE agent updates (2h)
   - Main agent entry point (1h)

### **This Week** (Days 1-4)
- **Complete Phase 0** (finish personalization)
- **Complete Phase 1** (security - CRITICAL!)
- **Start Phase 2** (reliability)

### **Next Week** (Days 5-12)
- **Complete Phase 2** (reliability)
- **Complete Phase 3** (indexes - 30 mins!)
- **Complete Phase 4** (multi-LLM)
- **Sign up for Cerebras API** (free unlimited)

### **Week 3** (Days 13-20)
- **Complete Phase 5** (LangSmith monitoring)
- **Sign up for LangSmith** (free 5k traces/month)
- **Complete Phase 6** (Hybrid RAG)

### **Final Days** (Days 21-23) - **INTELLIGENCE PHASE** 🧠
- **Implement agent settings** (LLM configs, token limits)
- **Build token manager** (smart summarization)
- **Create user_preferences table** (learning mechanism)
- **Build context preparation layer** (orchestration)
- **Implement preference learner** (background task)
- **Intelligence testing** (validate smartness)
- **Load testing** (50 users, <3s latency)
- **🚀 GO LIVE!**

---

**Total Timeline**: ~3 weeks (21-23 days)  
**Budget**: $0/month (truly free!)  
**Risk Level**: Low (simplified, tested approach)  
**Launch Confidence**: High (realistic plan + intelligence features)  
**Complexity**: Medium (removed local servers, added smart features)

---

## 🎉 **Key Improvements in This Revision**

1. ✅ **Removed redundant Phase 2 Task 3** (_checkpointer already cleaned up)
2. ✅ **Simplified Phase 3** (just indexes, 30 mins instead of 1.5 days)
3. ✅ **Replaced Prometheus/Grafana with LangSmith** (no local servers!)
4. ✅ **Combined BM25 + Hybrid RAG** (one phase instead of two)
5. ✅ **Added Agent Settings** (centralized LLM configuration)
6. ✅ **Added Token Manager** (smart context summarization at 80%)
7. ✅ **Added User Preferences** (learns from user behavior over time)
8. ✅ **Added Context Preparation Layer** (clean architecture, loads everything before router)
9. ✅ **Reduced total time** (21 days vs 25 days original)
10. ✅ **Addressed intelligence concerns** (preference learning + validation tests)

**You're right to question complexity AND add intelligence!** This plan is now both practical AND smart. 🚀🧠
