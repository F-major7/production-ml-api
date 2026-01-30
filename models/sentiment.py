"""
Sentiment Analysis Model Wrapper
Singleton pattern for efficient model loading and reuse
"""
from typing import Dict, Optional
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModel:
    """
    Singleton sentiment analysis model using HuggingFace Transformers.
    Uses DistilBERT fine-tuned on SST-2 for sentiment classification.
    """
    _instance: Optional['SentimentModel'] = None
    _pipeline = None
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self) -> None:
        """
        Load the sentiment analysis model.
        Uses distilbert-base-uncased-finetuned-sst-2-english by default.
        """
        if self._pipeline is None:
            try:
                logger.info("Loading sentiment analysis model...")
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
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
            # Pipeline returns list with single result for single input
            result = self._pipeline(text)[0]
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

