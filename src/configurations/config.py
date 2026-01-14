from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    GROQ_API_KEY:str
    GOOGLE_API_KEY:str
    GROQ_MODEL:str
    GEMINI_MODEL:str
    TEMPERATURE:float
    MAX_TOKENS:int
    COHERE_API_KEY:str
    DATABASE_URL:str
    TAVILY_API_KEY:str
    ENDPOINT_AUTH_KEY:str

settings = Settings()
