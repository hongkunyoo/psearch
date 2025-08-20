from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    notes_directory: Path = Field(
        default_factory=lambda: Path.home() / "notes",
        description="Directory containing notes to index"
    )
    
    index_directory: Path = Field(
        default_factory=lambda: Path.home() / ".personal_search" / "index",
        description="Directory to store the search index"
    )
    
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model to use"
    )
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for using OpenAI embeddings"
    )
    
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for indexing"
    )
    
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between text chunks"
    )
    
    top_k: int = Field(
        default=5,
        description="Number of results to return"
    )
    
    use_local_embeddings: bool = Field(
        default=True,
        description="Use local embeddings instead of OpenAI"
    )


settings = Settings()