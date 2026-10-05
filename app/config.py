
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    google_api_key: str
    primary_model: str = "models/gemini-3.6-flash"
    fallback_model: str = "models/gemini-3.5-flash"

    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "production-api"

    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {"env_file": ".env","extra": "ignore"}

    @property
    def is_production(self)-> bool:
        return self.app_env == "production"

@lru_cache
def get_settings()-> Settings:
    '''Cached setting instance -loaded once, reused everywhere'''
    return Settings()