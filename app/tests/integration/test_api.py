"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import create_app
from app.services.model_manager import ModelManager


@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = MagicMock(spec=ModelManager)
    manager.is_initialized = True
    
    # Mock summarization model
    mock_sum_model = AsyncMock()
    mock_sum_model.predict.return_value = "This is a test summary."
    manager.get_summarization_model.return_value = mock_sum_model
    
    # Mock QA model
    mock_qa_model = AsyncMock()
    mock_qa_model.predict.return_value = {
        "answer": "Test answer",
        "confidence": 0.8,
        "has_answer": True,
        "start_char": 0,
        "end_char": 10
    }
    manager.get_qa_model.return_value = mock_qa_model
    
    # Mock embedding model
    mock_emb_model = AsyncMock()
    mock_emb_model.predict.return_value = [0.1, 0.2, 0.3, 0.4]
    manager.get_embedding_model.return_value = mock_emb_model
    
    return manager


@pytest.fixture
def test_app(mock_model_manager):
    """Create test FastAPI app with mocked dependencies."""
    app = create_app()
    
    # Override the model manager dependency
    def override_get_model_manager():
        return mock_model_manager
    
    # We need to mock the lifespan to avoid model loading
    with patch('app.main.ModelManager') as mock_manager_class:
        mock_manager_class.return_value = mock_model_manager
        app.state.model_manager = mock_model_manager
        
        yield app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Test cases for health endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        with patch('app.api.routes.pinecone_service') as mock_pinecone:
            mock_pinecone.is_initialized = True
            mock_pinecone.get_index_stats.return_value = {"total_vector_count": 100}
            
            response = client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "1.0.0"
            assert data["models_loaded"] is True


class TestSummarizationEndpoint:
    """Test cases for summarization endpoint."""
    
    def test_summarize_success(self, client):
        """Test successful text summarization."""
        request_data = {
            "text": "This is a long article that needs to be summarized. " * 10,
            "max_length": 100,
            "min_length": 20
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "original_length" in data
        assert "summary_length" in data
        assert "compression_ratio" in data
        assert data["summary"] == "This is a test summary."
    
    def test_summarize_invalid_input(self, client):
        """Test summarization with invalid input."""
        request_data = {
            "text": "Short",  # Too short
            "max_length": 100
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_summarize_missing_text(self, client):
        """Test summarization with missing text field."""
        request_data = {
            "max_length": 100
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        assert response.status_code == 422


class TestQAEndpoint:
    """Test cases for QA endpoint."""
    
    def test_qa_with_context_success(self, client):
        """Test QA with provided context."""
        request_data = {
            "question": "What is artificial intelligence?",
            "context": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "confidence_threshold": 0.5
        }
        
        response = client.post("/api/v1/qa", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "has_answer" in data
        assert data["answer"] == "Test answer"
        assert data["confidence"] == 0.8
        assert data["has_answer"] is True
    
    def test_qa_with_rag(self, client):
        """Test QA with RAG (no context provided)."""
        with patch('app.api.routes.pinecone_service') as mock_pinecone, \
             patch('app.api.routes._answer_with_rag') as mock_rag:
            
            mock_rag.return_value = {
                "answer": "RAG answer",
                "confidence": 0.7,
                "has_answer": True,
                "source_articles": ["http://example.com/article1"]
            }
            
            request_data = {
                "question": "What happened in the news today?",
            }
            
            response = client.post("/api/v1/qa", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "RAG answer"
    
    def test_qa_invalid_question(self, client):
        """Test QA with invalid question."""
        request_data = {
            "question": "Hi",  # Too short
            "context": "Some context here."
        }
        
        response = client.post("/api/v1/qa", json=request_data)
        assert response.status_code == 422


class TestSearchEndpoint:
    """Test cases for search endpoint."""
    
    def test_search_success(self, client):
        """Test successful article search."""
        with patch('app.api.routes.pinecone_service') as mock_pinecone:
            mock_pinecone.is_initialized = True
            mock_pinecone.query_vectors.return_value = [
                {
                    "id": "article_1",
                    "score": 0.95,
                    "metadata": {
                        "title": "Test Article",
                        "content": "This is test content.",
                        "url": "http://example.com/article1",
                        "source": "Test Source",
                        "published_date": "2023-01-01T00:00:00",
                        "summary": "Test summary",
                        "authors": ["Test Author"]
                    }
                }
            ]
            
            response = client.get("/api/v1/search?query=artificial intelligence&limit=5")
            
            assert response.status_code == 200
            data = response.json()
            assert "query" in data
            assert "results" in data
            assert "total_results" in data
            assert "search_time_ms" in data
            assert data["query"] == "artificial intelligence"
            assert len(data["results"]) == 1
            assert data["results"][0]["title"] == "Test Article"
    
    def test_search_with_filters(self, client):
        """Test search with source filter."""
        with patch('app.api.routes.pinecone_service') as mock_pinecone:
            mock_pinecone.is_initialized = True
            mock_pinecone.query_vectors.return_value = []
            
            response = client.get("/api/v1/search?query=news&filter_source=CNN&limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_results"] == 0
    
    def test_search_limit_validation(self, client):
        """Test search limit validation."""
        with patch('app.api.routes.pinecone_service') as mock_pinecone:
            mock_pinecone.is_initialized = True
            mock_pinecone.query_vectors.return_value = []
            
            # Test with limit over maximum
            response = client.get("/api/v1/search?query=test&limit=100")
            
            assert response.status_code == 200
            # Should be capped at 50


class TestIngestEndpoint:
    """Test cases for ingest endpoint."""
    
    def test_ingest_success(self, client):
        """Test successful article ingestion."""
        with patch('app.api.routes.rss_service') as mock_rss, \
             patch('app.api.routes.pinecone_service') as mock_pinecone:
            
            # Mock RSS service
            mock_article = MagicMock()
            mock_article.text_content = "Test article content"
            mock_article.title = "Test Article"
            mock_article.content = "Content"
            mock_article.url = "http://example.com"
            mock_article.source = "Test Source"
            mock_article.published_date = None
            mock_article.summary = None
            mock_article.authors = []
            
            mock_rss.fetch_all_feeds.return_value = [mock_article]
            
            # Mock Pinecone service
            mock_pinecone.is_initialized = True
            mock_pinecone.upsert_vectors.return_value = None
            
            response = client.post("/api/v1/ingest")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "ingested_count" in data
            assert data["ingested_count"] == 1
    
    def test_ingest_no_articles(self, client):
        """Test ingestion when no articles are found."""
        with patch('app.api.routes.rss_service') as mock_rss:
            mock_rss.fetch_all_feeds.return_value = []
            
            response = client.post("/api/v1/ingest")
            
            assert response.status_code == 200
            data = response.json()
            assert data["ingested_count"] == 0
