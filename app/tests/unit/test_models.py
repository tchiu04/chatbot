"""
Unit tests for NLP models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.base import BaseModel
from app.models.summarization import SummarizationModel
from app.models.qa import QAModel
from app.models.embeddings import EmbeddingModel


class TestBaseModel:
    """Test cases for BaseModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        
        class TestModel(BaseModel):
            async def load(self):
                pass
            
            async def predict(self, *args, **kwargs):
                pass
        
        model = TestModel("test-model")
        assert model.model_name == "test-model"
        assert not model.is_loaded
        assert model.device in ["cpu", "cuda"]


@pytest.mark.asyncio
class TestSummarizationModel:
    """Test cases for SummarizationModel."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock summarization model."""
        with patch('app.models.summarization.PegasusTokenizer'), \
             patch('app.models.summarization.PegasusForConditionalGeneration'):
            model = SummarizationModel("test-pegasus")
            model.tokenizer = MagicMock()
            model.model = MagicMock()
            model._is_loaded = True
            return model
    
    async def test_predict(self, mock_model):
        """Test text summarization prediction."""
        # Mock tokenizer and model outputs
        mock_model.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_model.model.generate.return_value = torch.tensor([[4, 5, 6]])
        mock_model.tokenizer.decode.return_value = "Test summary"
        
        # Test prediction
        result = await mock_model.predict("This is a test article.")
        
        assert result == "Test summary"
        mock_model.model.generate.assert_called_once()
    
    async def test_batch_predict(self, mock_model):
        """Test batch summarization."""
        # Mock outputs for batch processing
        mock_model.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        mock_model.model.generate.return_value = torch.tensor([[7, 8, 9], [10, 11, 12]])
        mock_model.tokenizer.decode.side_effect = ["Summary 1", "Summary 2"]
        
        texts = ["Article 1", "Article 2"]
        results = await mock_model.batch_predict(texts)
        
        assert len(results) == 2
        assert results == ["Summary 1", "Summary 2"]


@pytest.mark.asyncio
class TestQAModel:
    """Test cases for QAModel."""
    
    @pytest.fixture
    def mock_qa_model(self):
        """Create a mock QA model."""
        with patch('app.models.qa.AutoTokenizer'), \
             patch('app.models.qa.AutoModelForQuestionAnswering'):
            model = QAModel("test-roberta")
            model.tokenizer = MagicMock()
            model.model = MagicMock()
            model._is_loaded = True
            return model
    
    async def test_predict_with_answer(self, mock_qa_model):
        """Test QA prediction with confident answer."""
        # Mock tokenizer output
        mock_model_output = MagicMock()
        mock_model_output.start_logits = torch.tensor([0.1, 0.9, 0.2])
        mock_model_output.end_logits = torch.tensor([0.2, 0.1, 0.8])
        
        mock_qa_model.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 4], [5, 10], [11, 15]]])
        }
        mock_qa_model.model.return_value = mock_model_output
        
        question = "What is AI?"
        context = "AI is artificial intelligence."
        
        result = await mock_qa_model.predict(question, context)
        
        assert "answer" in result
        assert "confidence" in result
        assert "has_answer" in result
        assert isinstance(result["confidence"], float)
    
    async def test_batch_predict(self, mock_qa_model):
        """Test batch QA prediction."""
        # Mock individual predictions
        with patch.object(mock_qa_model, 'predict', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = {
                "answer": "Test answer",
                "confidence": 0.8,
                "has_answer": True
            }
            
            questions = ["Question 1", "Question 2"]
            contexts = ["Context 1", "Context 2"]
            
            results = await mock_qa_model.batch_predict(questions, contexts)
            
            assert len(results) == 2
            assert mock_predict.call_count == 2


@pytest.mark.asyncio
class TestEmbeddingModel:
    """Test cases for EmbeddingModel."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        with patch('app.models.embeddings.SentenceTransformer'):
            model = EmbeddingModel("test-sentence-transformer")
            model.model = MagicMock()
            model._is_loaded = True
            return model
    
    async def test_predict(self, mock_embedding_model):
        """Test single text embedding."""
        # Mock embedding output
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_embedding_model.model.encode.return_value = mock_embedding
        
        result = await mock_embedding_model.predict("Test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, mock_embedding)
    
    async def test_batch_predict(self, mock_embedding_model):
        """Test batch text embedding."""
        # Mock batch embedding output
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_embedding_model.model.encode.return_value = mock_embeddings
        
        texts = ["Text 1", "Text 2"]
        results = await mock_embedding_model.batch_predict(texts)
        
        assert isinstance(results, np.ndarray)
        assert results.shape == (2, 2)
        np.testing.assert_array_equal(results, mock_embeddings)
    
    async def test_similarity(self, mock_embedding_model):
        """Test similarity calculation."""
        # Mock embedding outputs
        mock_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        with patch.object(mock_embedding_model, 'batch_predict', new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = mock_embeddings
            
            similarity = await mock_embedding_model.similarity("Text 1", "Text 2")
            
            assert isinstance(similarity, float)
            assert 0.0 <= similarity <= 1.0
    
    async def test_find_most_similar(self, mock_embedding_model):
        """Test finding most similar texts."""
        # Mock embeddings
        query_embedding = np.array([1.0, 0.0])
        candidate_embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        
        with patch.object(mock_embedding_model, 'predict', new_callable=AsyncMock) as mock_predict, \
             patch.object(mock_embedding_model, 'batch_predict', new_callable=AsyncMock) as mock_batch:
            
            mock_predict.return_value = query_embedding
            mock_batch.return_value = candidate_embeddings
            
            candidates = ["Text 1", "Text 2", "Text 3"]
            results = await mock_embedding_model.find_most_similar("Query", candidates, top_k=2)
            
            assert len(results) == 2
            assert all(isinstance(item, tuple) for item in results)
            assert all(len(item) == 2 for item in results)
            assert all(isinstance(item[0], int) and isinstance(item[1], float) for item in results)
