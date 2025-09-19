"""
Application configuration management using Pydantic settings.
"""

import os
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    api_version: str = Field(default="v1", env="API_VERSION")
    
    # Model Configuration
    pegasus_model_path: str = Field(
        default="google/pegasus-newsroom", 
        env="PEGASUS_MODEL_PATH"
    )
    roberta_model_path: str = Field(
        default="deepset/roberta-base-squad2", 
        env="ROBERTA_MODEL_PATH"
    )
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2", 
        env="SENTENCE_TRANSFORMER_MODEL"
    )
    
    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(
        default="us-west1-gcp", 
        env="PINECONE_ENVIRONMENT"
    )
    pinecone_index_name: str = Field(
        default="news-embeddings", 
        env="PINECONE_INDEX_NAME"
    )
    pinecone_dimension: int = Field(default=384, env="PINECONE_DIMENSION")
    
    # RSS Configuration
    rss_feeds: List[str] = Field(
        default=[
            "https://feeds.reuters.com/reuters/topNews",
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.bbci.co.uk/news/world/rss.xml"
        ],
        env="RSS_FEEDS"
    )
    
    # Summarization Settings
    max_summary_length: int = Field(default=150, env="MAX_SUMMARY_LENGTH")
    min_summary_length: int = Field(default=50, env="MIN_SUMMARY_LENGTH")
    
    # QA Settings
    qa_confidence_threshold: float = Field(default=0.5, env="QA_CONFIDENCE_THRESHOLD")
    max_context_length: int = Field(default=512, env="MAX_CONTEXT_LENGTH")
    
    # Performance Settings
    batch_size: int = Field(default=8, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            """Parse environment variables, especially for lists."""
            if field_name == "rss_feeds":
                return [url.strip() for url in raw_val.split(",")]
            return cls.json_loads(raw_val)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
