"""
Embedding Generation Module
Handles text-to-vector conversion for semantic search
Uses fastembed (ONNX Runtime) — no PyTorch dependency, ~50MB RAM
"""
from typing import List, Union
import logging

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

import os
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks.
    Supports:
    1. Local embeddings (fastembed / ONNX) - 100% FREE, no PyTorch
    2. OpenAI embeddings (paid but high quality)
    """
    
    def __init__(self, model: str = None):
        self.model = model or settings.embedding_model
        
        # Auto-detect embedding type
        if settings.use_local_embeddings or not OPENAI_AVAILABLE or not settings.openai_api_key:
            # Use local embeddings (FREE)
            self._init_local_embeddings()
        elif "text-embedding" in self.model and OPENAI_AVAILABLE:
            # Use OpenAI
            self._init_openai_embeddings()
        else:
            # Fallback to local
            self._init_local_embeddings()
        
        logger.info(f"Initialized EmbeddingGenerator: type={self.embedding_type}, model={self.model}, dim={self.dimension}")
    
    def _init_local_embeddings(self):
        """Initialize fastembed ONNX model (FREE, no PyTorch, ~50MB RAM)"""
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed not installed. Install with: pip install fastembed"
            )

        self.embedding_type = "local"
        local_model = settings.local_embedding_model or "BAAI/bge-small-en-v1.5"

        logger.info(f"Loading fastembed model: {local_model}")
        self.local_model = TextEmbedding(local_model)
        # Detect dimension by embedding a dummy string
        self.dimension = len(next(self.local_model.embed(["dim_probe"])))
        self.model = local_model

        logger.info(f"fastembed ready! Model: {local_model}, Dimension: {self.dimension}")
    
    def _init_openai_embeddings(self):
        """Initialize OpenAI embeddings"""
        self.embedding_type = "openai"
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.dimension = settings.embedding_dimension
        
        logger.info(f" OpenAI embeddings ready! Model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        try:
            if self.embedding_type == "local":
                # Use fastembed (ONNX, no PyTorch)
                text = self._truncate_text(text, max_tokens=512)
                embedding = next(self.local_model.embed([text]))
                return embedding.tolist()
            
            else:
                # Use OpenAI
                text = self._truncate_text(text, max_tokens=8000)
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                
                if len(embedding) != self.dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self.dimension}, "
                        f"got {len(embedding)}"
                    )
                
                return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        More efficient than calling generate_embedding multiple times.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if self.embedding_type == "local":
                # Use fastembed (ONNX, no PyTorch)
                truncated_texts = [
                    self._truncate_text(text, max_tokens=512)
                    for text in valid_texts
                ]

                embeddings = list(self.local_model.embed(truncated_texts))

                logger.info(f"Generated {len(embeddings)} embeddings via fastembed (FREE)")

                return [emb.tolist() for emb in embeddings]
            
            else:
                # Use OpenAI
                truncated_texts = [
                    self._truncate_text(text, max_tokens=8000)
                    for text in valid_texts
                ]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=truncated_texts
                )

                embeddings = [item.embedding for item in response.data]
                logger.info(f"Generated {len(embeddings)} embeddings via OpenAI")
                return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise
    
    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        Truncate text to fit within token limit.
        Rough estimate: 1 token ≈ 4 characters
        """
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate and add indicator
        truncated = text[:max_chars - 20] + " [truncated...]"
        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
        
        return truncated
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator."""
        return self.dimension





# Example usage
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Single embedding
    sample_text = "Patient diagnosed with Type 2 Diabetes Mellitus"
    embedding = generator.generate_embedding(sample_text)
    print(f"Generated embedding with dimension: {len(embedding)}")
    
    # Batch embeddings
    sample_texts = [
        "Medications: Metformin 500mg twice daily",
        "Blood pressure: 120/80 mmHg",
        "Lab results show elevated glucose levels"
    ]
    embeddings = generator.generate_embeddings_batch(sample_texts)
    print(f"Generated {len(embeddings)} embeddings in batch")
