"""
Main API routes for the news briefing chatbot.
"""

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QARequest,
    QAResponse,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
    NewsArticleSchema
)
from app.services.model_manager import ModelManager
from app.services.pinecone_service import PineconeService
from app.services.rss_service import RSSService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global services (will be initialized in main.py)
rss_service = RSSService()
pinecone_service = PineconeService()


def get_model_manager(request: Request) -> ModelManager:
    """Dependency to get model manager from app state."""
    if not hasattr(request.app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return request.app.state.model_manager


@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Health check endpoint."""
    try:
        # Check model status
        models_loaded = model_manager.is_initialized
        
        # Check Pinecone connection
        pinecone_connected = pinecone_service.is_initialized
        if not pinecone_connected:
            try:
                await pinecone_service.initialize()
                pinecone_connected = True
            except Exception:
                pinecone_connected = False
        
        # Get total articles count
        total_articles = None
        if pinecone_connected:
            try:
                stats = await pinecone_service.get_index_stats()
                total_articles = stats.get("total_vector_count", 0)
            except Exception as e:
                logger.warning(f"Could not get article count: {e}")
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=models_loaded,
            pinecone_connected=pinecone_connected,
            total_articles=total_articles
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(
    request: SummarizeRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Summarize text using the Pegasus model."""
    try:
        # Get summarization model
        model = model_manager.get_summarization_model()
        
        # Generate summary
        summary = await model.predict(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        # Calculate metrics
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        return SummarizeResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/qa", response_model=QAResponse)
async def answer_question(
    request: QARequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Answer questions using RoBERTa with optional RAG."""
    try:
        qa_model = model_manager.get_qa_model()
        
        # If context is provided, use direct QA
        if request.context:
            result = await qa_model.predict(
                question=request.question,
                context=request.context,
                confidence_threshold=request.confidence_threshold
            )
            
            return QAResponse(
                answer=result["answer"],
                confidence=result["confidence"],
                has_answer=result["has_answer"],
                context_used=request.context[:200] + "..." if len(request.context) > 200 else request.context
            )
        
        # Use RAG (Retrieval-Augmented Generation)
        return await _answer_with_rag(request, model_manager)
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


async def _answer_with_rag(request: QARequest, model_manager: ModelManager) -> QAResponse:
    """Answer question using RAG (Retrieval-Augmented Generation)."""
    try:
        # Initialize Pinecone if needed
        if not pinecone_service.is_initialized:
            await pinecone_service.initialize()
        
        # Get embedding model
        embedding_model = model_manager.get_embedding_model()
        
        # Generate query embedding
        query_embedding = await embedding_model.predict(request.question)
        
        # Search for relevant articles
        search_results = await pinecone_service.query_vectors(
            query_vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        if not search_results:
            return QAResponse(
                answer="I couldn't find relevant information to answer your question.",
                confidence=0.0,
                has_answer=False
            )
        
        # Get QA model
        qa_model = model_manager.get_qa_model()
        
        # Try to answer using each retrieved context
        best_answer = None
        source_articles = []
        
        for result in search_results:
            if "metadata" in result and "content" in result["metadata"]:
                context = result["metadata"]["content"]
                source_url = result["metadata"].get("url", "")
                
                # Try to answer with this context
                qa_result = await qa_model.predict(
                    question=request.question,
                    context=context,
                    confidence_threshold=request.confidence_threshold
                )
                
                if qa_result["has_answer"]:
                    if not best_answer or qa_result["confidence"] > best_answer["confidence"]:
                        best_answer = qa_result
                        best_answer["context_used"] = context[:300] + "..." if len(context) > 300 else context
                        source_articles = [source_url] if source_url else []
        
        if best_answer:
            return QAResponse(
                answer=best_answer["answer"],
                confidence=best_answer["confidence"],
                has_answer=True,
                context_used=best_answer.get("context_used"),
                source_articles=source_articles
            )
        
        # No good answer found
        return QAResponse(
            answer="I found some relevant articles but couldn't extract a confident answer to your question.",
            confidence=0.0,
            has_answer=False,
            source_articles=[r["metadata"].get("url") for r in search_results if "metadata" in r]
        )
        
    except Exception as e:
        logger.error(f"RAG question answering failed: {e}")
        raise


@router.get("/search", response_model=SearchResponse)
async def search_articles(
    query: str,
    limit: int = 10,
    filter_source: str = None,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Search articles using semantic similarity."""
    start_time = time.time()
    
    try:
        # Validate parameters
        if limit > 50:
            limit = 50
        
        # Initialize Pinecone if needed
        if not pinecone_service.is_initialized:
            await pinecone_service.initialize()
        
        # Get embedding model
        embedding_model = model_manager.get_embedding_model()
        
        # Generate query embedding
        query_embedding = await embedding_model.predict(query)
        
        # Build filter
        filter_dict = {}
        if filter_source:
            filter_dict["source"] = filter_source
        
        # Search for similar articles
        search_results = await pinecone_service.query_vectors(
            query_vector=query_embedding,
            top_k=limit,
            filter_dict=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Convert to response format
        articles = []
        for result in search_results:
            if "metadata" in result:
                metadata = result["metadata"]
                
                article = NewsArticleSchema(
                    title=metadata.get("title", ""),
                    content=metadata.get("content", "")[:500] + "...",  # Truncate for response
                    url=metadata.get("url", ""),
                    source=metadata.get("source"),
                    published_date=metadata.get("published_date"),
                    summary=metadata.get("summary"),
                    authors=metadata.get("authors", []),
                    similarity_score=result["score"]
                )
                articles.append(article)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=query,
            results=articles,
            total_results=len(articles),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Article search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Article search failed: {str(e)}")


@router.post("/ingest")
async def ingest_articles(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Ingest articles from RSS feeds and store in Pinecone."""
    try:
        # Initialize Pinecone if needed
        if not pinecone_service.is_initialized:
            await pinecone_service.initialize()
        
        # Fetch articles from RSS feeds
        articles = await rss_service.fetch_all_feeds()
        
        if not articles:
            return {"message": "No articles found", "ingested_count": 0}
        
        # Get embedding model
        embedding_model = model_manager.get_embedding_model()
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, article in enumerate(articles):
            # Generate embedding
            embedding = await embedding_model.predict(article.text_content)
            
            # Create vector data
            vector_id = f"article_{hash(article.url)}_{i}"
            metadata = {
                "title": article.title,
                "content": article.content,
                "url": article.url,
                "source": article.source,
                "published_date": article.published_date.isoformat() if article.published_date else None,
                "summary": article.summary,
                "authors": article.authors
            }
            
            vectors.append((vector_id, embedding, metadata))
        
        # Upsert to Pinecone
        await pinecone_service.upsert_vectors(vectors)
        
        return {
            "message": f"Successfully ingested {len(articles)} articles",
            "ingested_count": len(articles),
            "sources": list(set(article.source for article in articles if article.source))
        }
        
    except Exception as e:
        logger.error(f"Article ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Article ingestion failed: {str(e)}")


# Error handler
@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_type="HTTPException"
        ).dict()
    )
