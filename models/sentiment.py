"""
Sentiment Analysis Model Wrapper
Supports multiple model versions for A/B testing
"""
from typing import Dict, Optional
from transformers import pipeline
import logging
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModel:
    """
    Sentiment analysis model using HuggingFace Transformers.
    Supports multiple versions for A/B testing.
    Uses DistilBERT fine-tuned on SST-2 for sentiment classification.
    """
    _instances: Dict[str, 'SentimentModel'] = {}
    
    def __init__(self, version: str = "v1"):
        """
        Initialize model for specific version.
        
        Args:
            version: Model version identifier (v1, v2, etc.)
        """
        self.version = version
        self._pipeline = None
        self._initialize_model()
    
    @classmethod
    def get_model(cls, version: str = "v1") -> 'SentimentModel':
        """
        Get or create model instance for specified version.
        Implements singleton pattern per version.
        
        Args:
            version: Model version identifier
            
        Returns:
            SentimentModel instance for the version
        """
        if version not in cls._instances:
            logger.info(f"Creating new model instance for version: {version}")
            cls._instances[version] = cls(version)
        return cls._instances[version]
    
    def _initialize_model(self) -> None:
        """
        Load the sentiment analysis model.
        For Phase 4, both v1 and v2 use same DistilBERT (infrastructure focus).
        In production, these would be different models or configurations.
        """
        if self._pipeline is None:
            try:
                logger.info(f"Loading sentiment analysis model (version: {self.version})...")
                # Force CPU device to avoid CUDA/GPU issues in Docker
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # -1 forces CPU, avoids CUDA/GPU segfaults
                )
                logger.info(f"Model {self.version} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {self.version}: {e}")
                raise RuntimeError(f"Model initialization failed: {e}")
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If text is invalid
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Use torch.no_grad() for inference to save memory and prevent crashes
            with torch.no_grad():
                # Pipeline returns list with single result for single input
                result = self._pipeline(text)[0]
            
            # Explicit garbage collection to prevent memory leaks in Docker
            gc.collect()
            
            return {
                "label": result["label"],
                "score": result["score"]
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._pipeline is not None

