"""
Dependency injection for FastAPI
"""
from models.sentiment import SentimentModel
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def get_model() -> SentimentModel:
    """
    Dependency injection for sentiment model.
    Returns singleton instance of SentimentModel.
    
    Returns:
        SentimentModel: Loaded sentiment analysis model
        
    Raises:
        HTTPException: If model fails to load
    """
    try:
        model = SentimentModel()
        if not model.is_loaded:
            logger.error("Model is not loaded")
            raise HTTPException(
                status_code=503,
                detail="Model not available"
            )
        return model
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Model initialization failed: {str(e)}"
        )

