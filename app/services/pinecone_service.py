"""
Pinecone vector database service for semantic search.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinecone
from pinecone import Index

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for interacting with Pinecone vector database."""
    
    def __init__(self):
        self.settings = get_settings()
        self.index: Optional[Index] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Pinecone connection."""
        try:
            logger.info("Initializing Pinecone service...")
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment
            )
            
            # Check if index exists
            index_name = self.settings.pinecone_index_name
            
            if index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {index_name}")
                
                # Create index
                pinecone.create_index(
                    name=index_name,
                    dimension=self.settings.pinecone_dimension,
                    metric="cosine",
                    pods=1,
                    pod_type="p1.x1"
                )
                
                logger.info(f"Pinecone index '{index_name}' created successfully")
            
            # Connect to index
            self.index = pinecone.Index(index_name)
            
            self._initialized = True
            logger.info("Pinecone service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone service: {e}")
            raise
    
    async def upsert_vectors(
        self,
        vectors: List[Tuple[str, np.ndarray, Dict]],
        batch_size: int = 100
    ) -> None:
        """
        Upsert vectors to Pinecone index.
        
        Args:
            vectors: List of (id, vector, metadata) tuples
            batch_size: Batch size for upsert operations
        """
        if not self._initialized:
            raise RuntimeError("Pinecone service not initialized")
        
        try:
            logger.info(f"Upserting {len(vectors)} vectors to Pinecone...")
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Convert to Pinecone format
                upsert_data = [
                    {
                        "id": vector_id,
                        "values": vector.tolist(),
                        "metadata": metadata
                    }
                    for vector_id, vector, metadata in batch
                ]
                
                # Upsert batch
                self.index.upsert(vectors=upsert_data)
                
                logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            logger.info("Vector upsert completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise
    
    async def query_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query vectors from Pinecone index.
        
        Args:
            query_vector: Query vector
            top_k: Number of top results to return
            filter_dict: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching results with scores and metadata
        """
        if not self._initialized:
            raise RuntimeError("Pinecone service not initialized")
        
        try:
            # Query index
            response = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )
            
            # Format results
            results = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score)
                }
                
                if include_metadata and hasattr(match, "metadata"):
                    result["metadata"] = match.metadata
                
                results.append(result)
            
            logger.debug(f"Found {len(results)} matching vectors")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            raise
    
    async def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from Pinecone index.
        
        Args:
            ids: List of vector IDs to delete
        """
        if not self._initialized:
            raise RuntimeError("Pinecone service not initialized")
        
        try:
            logger.info(f"Deleting {len(ids)} vectors from Pinecone...")
            
            # Delete vectors
            self.index.delete(ids=ids)
            
            logger.info("Vector deletion completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise
    
    async def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        if not self._initialized:
            raise RuntimeError("Pinecone service not initialized")
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup Pinecone connection."""
        try:
            if self.index:
                self.index = None
            
            self._initialized = False
            logger.info("Pinecone service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Pinecone cleanup: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if Pinecone service is initialized."""
        return self._initialized
