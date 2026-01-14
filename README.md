# Nigerian Smart AI Tax Assistant 🇳🇬

An intelligent, multi-agent tax advisory chatbot leveraging RAG (Retrieval-Augmented Generation) and LangGraph to provide accurate information about Nigerian tax policies, PAYE calculations, and personalized financial advice.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)

## ✨ Features

- **🤖 Multi-Agent System**: Intelligent routing between specialized agents (Tax Policy, PAYE, Financial Advice)
- **📚 RAG-Powered Responses**: Vector database retrieval for accurate, context-aware answers
- **💰 PAYE Calculator**: Interactive salary tax calculation with detailed breakdowns
- **🔍 Web Search Integration**: Real-time tax information via Tavily API
- **💾 Conversation Memory**: PostgreSQL-backed chat history with thread persistence
- **🎯 Personalized Advice**: Tailored financial recommendations based on user context
- **🌐 Dual Interface**: Chainlit UI and FastAPI REST endpoints
- **🐳 Docker Ready**: Containerized deployment with docker-compose

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Chainlit UI]
        API[FastAPI REST API]
    end
    
    subgraph "Agent Orchestration"
        MainAgent[Main Agent]
        Router[Intelligent Router]
        
        subgraph "Specialized Agents"
            TaxAgent[Tax Policy Agent]
            PayeAgent[PAYE Calculator Agent]
            FinAgent[Financial Advice Agent]
            CombAgent[Combined Agent]
        end
        
        ResponseGen[Response Generator]
    end
    
    subgraph "Tools & Services"
        RAG[RAG Tool]
        WebSearch[Web Search Tool]
        LLM[LLM Manager<br/>Groq/Gemini/Cohere]
    end
    
    subgraph "Data Layer"
        VectorDB[(ChromaDB<br/>Vector Store)]
        PostgreSQL[(PostgreSQL<br/>Chat History)]
        Embeddings[HuggingFace<br/>Embeddings]
    end
    
    UI --> MainAgent
    API --> MainAgent
    MainAgent --> Router
    Router --> TaxAgent
    Router --> PayeAgent
    Router --> FinAgent
    Router --> CombAgent
    
    TaxAgent --> RAG
    PayeAgent --> RAG
    FinAgent --> WebSearch
    CombAgent --> RAG
    
    TaxAgent --> ResponseGen
    PayeAgent --> ResponseGen
    FinAgent --> ResponseGen
    CombAgent --> ResponseGen
    
    RAG --> VectorDB
    RAG --> Embeddings
    WebSearch --> LLM
    
    MainAgent --> PostgreSQL
    ResponseGen --> LLM
    
    style MainAgent fill:#4A90E2
    style Router fill:#50C878
    style VectorDB fill:#FF6B6B
    style PostgreSQL fill:#FF6B6B
```

## 🔄 Workflow

```mermaid
sequenceDiagram
    participant User
    participant MainAgent
    participant Router
    participant Agent
    participant RAG
    participant LLM
    participant VectorDB
    participant Memory
    
    User->>MainAgent: Send Query
    MainAgent->>Memory: Load Chat History
    Memory-->>MainAgent: Historical Context
    
    MainAgent->>Router: Route Query + Context
    Router->>LLM: Analyze Intent
    LLM-->>Router: Determine Route
    Router-->>MainAgent: Route Decision
    
    alt Tax Policy Query
        MainAgent->>Agent: Tax Policy Agent
        Agent->>RAG: Retrieve Tax Documents
        RAG->>VectorDB: Semantic Search
        VectorDB-->>RAG: Relevant Chunks
        RAG-->>Agent: Context
    else PAYE Calculation
        MainAgent->>Agent: PAYE Agent
        Agent->>RAG: Retrieve PAYE Rules
        RAG->>VectorDB: Get Calculation Logic
        VectorDB-->>RAG: Tax Tables
        RAG-->>Agent: Calculation Context
    else Financial Advice
        MainAgent->>Agent: Financial Agent
        Agent->>RAG: Web Search
        RAG-->>Agent: Latest Info
    end
    
    Agent->>LLM: Generate Response
    LLM-->>Agent: AI Response
    Agent->>MainAgent: Formatted Answer
    
    MainAgent->>Memory: Save Conversation
    MainAgent-->>User: Final Response
```

## 🛠️ Tech Stack

### Core Framework
- **LangChain** - LLM orchestration and chain management
- **LangGraph** - Multi-agent workflow orchestration
- **Chainlit** - Interactive chat UI
- **FastAPI** - REST API endpoints

### LLM Providers
- **Groq** (llama-3.3-70b-versatile) - Primary reasoning model
- **Google Gemini** (gemini-flash) - Alternative LLM
- **Cohere** - Reranking and embeddings

### Vector Store & Embeddings
- **ChromaDB** - Vector database for document retrieval
- **HuggingFace Transformers** - Sentence embeddings

### Data Storage
- **PostgreSQL** - Conversation memory and checkpointing
- **Azure PostgreSQL** - Production database

### Tools & Integrations
- **Tavily API** - Web search for real-time information
- **Docker** - Containerization

## 📦 Prerequisites

- Python 3.10 or higher
- PostgreSQL 13+ (local or cloud)
- Docker & Docker Compose (optional)
- API Keys:
  - Groq API Key
  - Google API Key (Gemini)
  - Cohere API Key
  - Tavily API Key

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd tax_chatbot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/database

# LLM API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
COHERE_API_KEY=your_cohere_api_key

# Web Search
TAVILY_API_KEY=your_tavily_api_key

# LLM Configuration
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-flash-latest
TEMPERATURE=0.3
MAX_TOKENS=2048

# API Security
ENDPOINT_AUTH_KEY=your_secure_key
```

### 5. Initialize Vector Database

```bash
# Place your tax documents in dataset/raw_data/
# Run preprocessing script
python src/script/data_preprocessing.py
```

### 6. Set Up PostgreSQL Database

```bash
# Run database initialization
python src/database/create_chainlit_db.py
```

## ⚙️ Configuration

The application uses a centralized configuration system located at `src/configurations/config.py`. Key settings include:

- **LLM Models**: Choose between Groq, Gemini, or Cohere
- **Temperature**: Control response creativity (0.0-1.0)
- **Max Tokens**: Response length limit
- **Vector Store**: ChromaDB configuration
- **Database**: PostgreSQL connection settings

## 🎮 Usage

### Chainlit Interface

Run the interactive chat interface:

```bash
chainlit run app.py -w
```

Access the UI at: `http://localhost:8000`

### FastAPI Interface

Start the REST API server:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

API docs available at: `http://localhost:8080/docs`

### Sample API Request

```bash
curl -X POST "http://localhost:8080/api/chat" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the new PAYE tax rates for 2026?",
    "user_id": "user123",
    "thread_id": "thread456"
  }'
```

## 📁 Project Structure

```
tax_chatbot/
├── src/
│   ├── agent/                    # Multi-agent system
│   │   ├── graph_builder/        # LangGraph workflow
│   │   ├── sub_agents/           # Specialized agents
│   │   └── main_agent.py         # Main orchestrator
│   ├── api/                      # FastAPI routes
│   │   └── routes/               # API endpoints
│   ├── tools/                    # RAG & web search tools
│   │   └── retrieval/            # Document retrieval
│   ├── services/                 # LLM & prompt services
│   ├── vector_db/                # Vector store management
│   ├── database/                 # PostgreSQL setup
│   └── configurations/           # App configuration
├── dataset/                      # Training & processed data
│   ├── raw_data/                 # Original documents
│   └── processed_data/           # Cleaned documents
├── chroma_db/                    # ChromaDB storage
├── app.py                        # Chainlit entry point
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker orchestration
└── dockerfile                    # Container configuration
```

## 🔌 API Endpoints

### Chat Endpoint

**POST** `/api/chat`

Request body:
```json
{
  "message": "User query",
  "user_id": "unique_user_id",
  "thread_id": "conversation_thread_id"
}
```

Response:
```json
{
  "response": "AI-generated response",
  "route": "paye|tax|financial|both",
  "sources": ["doc1.txt", "doc2.txt"],
  "thread_id": "conversation_thread_id"
}
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t tax_chatbot-api:latest .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f
```

### Environment Configuration

Update `docker-compose.yml` with your environment variables or use an `.env` file.

### Push to Registry

```bash
# Tag image
docker tag tax_chatbot-api:latest yourusername/tax_chatbot-api:latest

# Push to Docker Hub
docker push yourusername/tax_chatbot-api:latest
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Squod Tech Team** - Initial work

## 🙏 Acknowledgments

- Nigerian Federal Inland Revenue Service (FIRS) for tax documentation
- LangChain and LangGraph communities
- All contributors and users of this project

---

**Note**: This chatbot provides general tax information and calculations. Always consult with a certified tax professional for specific tax advice.
