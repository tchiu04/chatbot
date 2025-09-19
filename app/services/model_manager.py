"""
Model manager for loading and managing all NLP models.
"""

import logging
from typing import Optional

from app.core.config import get_settings
from app.models.embeddings import EmbeddingModel
from app.models.qa import QAModel
from app.models.summarization import SummarizationModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all NLP models for the application."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Model instances
        self.summarization_model: Optional[SummarizationModel] = None
        self.qa_model: Optional[QAModel] = None
        self.embedding_model: Optional[EmbeddingModel] = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize and load all models."""
        if self._initialized:
            logger.warning("Model manager already initialized")
            return
        
        try:
            logger.info("Initializing model manager...")
            
            # Initialize summarization model
            logger.info("Loading summarization model...")
            self.summarization_model = SummarizationModel(
                model_name=self.settings.pegasus_model_path,
                max_length=self.settings.max_summary_length,
                min_length=self.settings.min_summary_length
            )
            await self.summarization_model.load()
            
            # Initialize QA model
            logger.info("Loading QA model...")
            self.qa_model = QAModel(
                model_name=self.settings.roberta_model_path,
                confidence_threshold=self.settings.qa_confidence_threshold,
                max_context_length=self.settings.max_context_length
            )
            await self.qa_model.load()
            
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = EmbeddingModel(
                model_name=self.settings.sentence_transformer_model,
                batch_size=self.settings.batch_size
            )
            await self.embedding_model.load()
            
            self._initialized = True
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self) -> None:
        """Cleanup and unload all models."""
        logger.info("Cleaning up model manager...")
        
        try:
            if self.summarization_model:
                await self.summarization_model.unload()
                self.summarization_model = None
            
            if self.qa_model:
                await self.qa_model.unload()
                self.qa_model = None
            
            if self.embedding_model:
                await self.embedding_model.unload()
                self.embedding_model = None
            
            self._initialized = False
            logger.info("Model manager cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during model manager cleanup: {e}")
    
    def get_summarization_model(self) -> SummarizationModel:
        """Get the summarization model."""
        if not self._initialized or not self.summarization_model:
            raise RuntimeError("Summarization model not initialized")
        return self.summarization_model
    
    def get_qa_model(self) -> QAModel:
        """Get the QA model."""
        if not self._initialized or not self.qa_model:
            raise RuntimeError("QA model not initialized")
        return self.qa_model
    
    def get_embedding_model(self) -> EmbeddingModel:
        """Get the embedding model."""
        if not self._initialized or not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        return self.embedding_model
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model manager is initialized."""
        return self._initialized
    
    def get_model_status(self) -> dict:
        """Get the status of all models."""
        return {
            "initialized": self._initialized,
            "summarization_model": {
                "loaded": self.summarization_model.is_loaded if self.summarization_model else False,
                "info": self.summarization_model.get_model_info() if self.summarization_model else None
            },
            "qa_model": {
                "loaded": self.qa_model.is_loaded if self.qa_model else False,
                "info": self.qa_model.get_model_info() if self.qa_model else None
            },
            "embedding_model": {
                "loaded": self.embedding_model.is_loaded if self.embedding_model else False,
                "info": self.embedding_model.get_model_info() if self.embedding_model else None
            }
        }
