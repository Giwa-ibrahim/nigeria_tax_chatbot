from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    GROQ_API_KEY:str
    GROQ_MODEL:str
    COHERE_API_KEY:str
    COHERE_MODEL:str
    CEREBRAS_API_KEY:str = ""
    CEREBRAS_MODEL:str = ""
    TEMPERATURE:float
    MAX_TOKENS:int
    DATABASE_URL:str
    TAVILY_API_KEY:str
    ACCESS_TOKEN:str = ""
    APP_ID:str = ""
    APP_SECRET:str = ""
    RECIPIENT_WAID:str = ""
    VERSION:str = ""
    PHONE_NUMBER_ID:str = ""
    WHATSAPP_VERIFY_TOKEN:str = ""
    ENDPOINT_AUTH_KEY:str = ""
    ALLOWED_ORIGINS: str = "http://localhost:8000"
    
    # LangSmith Monitoring
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: str
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "taxbot"

settings = Settings()
