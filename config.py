from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    secret_key: str
    database_url: str
    algorithm: str
    access_token_expire_minutes: int
    google_api_key: str
    deepseek_api_key: str
    grok_api_key: str
    openrouter_api_key: str
    tavily_api_key: str
    mail_username: str
    mail_password: str
    mail_from: str
    mail_port: int
    mail_server: str
    mail_starttls: bool
    mail_ssl_tls: bool
    # Qdrant Cloud — required for RAG functionality
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()