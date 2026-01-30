"""
Dependency injection for FastAPI
"""
from typing import Optional
from models.sentiment import SentimentModel
from cache.redis_client import RedisCache
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
        # Use get_model() class method to get singleton instance
        model = SentimentModel.get_model("v1")
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


def get_redis() -> Optional[RedisCache]:
    """
    Dependency injection for Redis cache.
    Returns singleton instance of RedisCache or None if unavailable.
    
    Returns:
        RedisCache or None: Redis cache client if available
    """
    try:
        redis_client = RedisCache()
        if not redis_client.is_available:
            logger.warning("Redis client not available - running without cache")
            return None
        return redis_client
    except Exception as e:
        logger.warning(f"Failed to get Redis client: {e} - running without cache")
        return None

