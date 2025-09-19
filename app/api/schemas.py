"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# Request schemas
class SummarizeRequest(BaseModel):
    """Request schema for text summarization."""
    text: str = Field(..., description="Text to summarize", min_length=50)
    max_length: Optional[int] = Field(None, description="Maximum summary length", ge=20, le=500)
    min_length: Optional[int] = Field(None, description="Minimum summary length", ge=10, le=100)


class QARequest(BaseModel):
    """Request schema for question answering."""
    question: str = Field(..., description="Question to answer", min_length=5)
    context: Optional[str] = Field(None, description="Context for answering (optional for RAG)")
    confidence_threshold: Optional[float] = Field(None, description="Minimum confidence score", ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    """Request schema for semantic search."""
    query: str = Field(..., description="Search query", min_length=3)
    limit: Optional[int] = Field(10, description="Number of results to return", ge=1, le=50)
    filter_source: Optional[str] = Field(None, description="Filter by news source")
    filter_date_from: Optional[datetime] = Field(None, description="Filter articles from this date")
    filter_date_to: Optional[datetime] = Field(None, description="Filter articles to this date")


# Response schemas
class SummarizeResponse(BaseModel):
    """Response schema for text summarization."""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text length")
    summary_length: int = Field(..., description="Summary text length")
    compression_ratio: float = Field(..., description="Compression ratio (summary/original)")


class QAResponse(BaseModel):
    """Response schema for question answering."""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score")
    has_answer: bool = Field(..., description="Whether an answer was found")
    context_used: Optional[str] = Field(None, description="Context used for answering")
    source_articles: Optional[List[str]] = Field(None, description="Source article URLs for RAG")


class NewsArticleSchema(BaseModel):
    """Schema for news article data."""
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    url: str = Field(..., description="Article URL")
    source: Optional[str] = Field(None, description="News source")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    summary: Optional[str] = Field(None, description="Article summary")
    authors: List[str] = Field(default_factory=list, description="Article authors")
    similarity_score: Optional[float] = Field(None, description="Similarity score for search results")


class SearchResponse(BaseModel):
    """Response schema for semantic search."""
    query: str = Field(..., description="Original search query")
    results: List[NewsArticleSchema] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search time in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    pinecone_connected: bool = Field(..., description="Whether Pinecone is connected")
    total_articles: Optional[int] = Field(None, description="Total articles in database")


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Internal schemas
class ArticleEmbedding(BaseModel):
    """Schema for article embeddings."""
    article_id: str = Field(..., description="Unique article identifier")
    embedding: List[float] = Field(..., description="Article embedding vector")
    metadata: dict = Field(..., description="Article metadata")


class ModelStatus(BaseModel):
    """Schema for model status information."""
    model_name: str = Field(..., description="Model name")
    is_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device model is running on")
    memory_usage: Optional[str] = Field(None, description="Memory usage information")
