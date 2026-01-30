"""
Analytics endpoints for prediction data
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

from db.database import get_db
from db.models import Prediction
from api.schemas import (
    AnalyticsSummaryResponse,
    SentimentDistributionResponse,
    RecentPredictionResponse
)

logger = logging.getLogger(__name__)

# Initialize rate limiter for analytics
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)


@router.get(
    "/summary",
    response_model=AnalyticsSummaryResponse,
    summary="Get overall prediction statistics"
)
@limiter.limit("60/minute")
async def get_analytics_summary(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary statistics for all predictions.
    
    Returns:
        - total_predictions: Total number of predictions
        - avg_confidence: Average confidence score
        - avg_latency_ms: Average latency in milliseconds
        - date_range: Earliest and latest prediction timestamps
    """
    try:
        # Query for summary statistics
        stmt = select(
            func.count(Prediction.id).label('total'),
            func.avg(Prediction.confidence_score).label('avg_confidence'),
            func.avg(Prediction.latency_ms).label('avg_latency'),
            func.min(Prediction.timestamp).label('earliest'),
            func.max(Prediction.timestamp).label('latest')
        )
        
        result = await db.execute(stmt)
        row = result.first()
        
        if row is None or row.total == 0:
            return AnalyticsSummaryResponse(
                total_predictions=0,
                avg_confidence=0.0,
                avg_latency_ms=0.0,
                date_range={"earliest": None, "latest": None}
            )
        
        return AnalyticsSummaryResponse(
            total_predictions=row.total,
            avg_confidence=round(float(row.avg_confidence), 4) if row.avg_confidence else 0.0,
            avg_latency_ms=round(float(row.avg_latency), 2) if row.avg_latency else 0.0,
            date_range={
                "earliest": row.earliest.isoformat() if row.earliest else None,
                "latest": row.latest.isoformat() if row.latest else None
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics summary: {str(e)}"
        )


@router.get(
    "/sentiment-distribution",
    response_model=SentimentDistributionResponse,
    summary="Get sentiment distribution counts"
)
@limiter.limit("60/minute")
async def get_sentiment_distribution(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Get count of predictions by sentiment.
    
    Returns:
        - positive: Count of positive predictions
        - negative: Count of negative predictions
        - neutral: Count of neutral predictions
    """
    try:
        # Query for sentiment counts
        stmt = select(
            Prediction.predicted_sentiment,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.predicted_sentiment)
        
        result = await db.execute(stmt)
        rows = result.all()
        
        # Initialize counts
        distribution = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        # Fill in actual counts
        for row in rows:
            sentiment = row.predicted_sentiment.lower()
            if sentiment in distribution:
                distribution[sentiment] = row.count
        
        return SentimentDistributionResponse(**distribution)
        
    except Exception as e:
        logger.error(f"Error getting sentiment distribution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sentiment distribution: {str(e)}"
        )


@router.get(
    "/recent",
    response_model=List[RecentPredictionResponse],
    summary="Get recent predictions"
)
@limiter.limit("60/minute")
async def get_recent_predictions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000, description="Number of recent predictions to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent predictions ordered by timestamp.
    
    Args:
        limit: Number of predictions to return (default 100, max 1000)
    
    Returns:
        List of recent predictions (excluding input text for privacy)
    """
    try:
        # Query for recent predictions
        stmt = select(Prediction).order_by(
            Prediction.timestamp.desc()
        ).limit(limit)
        
        result = await db.execute(stmt)
        predictions = result.scalars().all()
        
        # Convert to response schema (exclude input_text)
        return [
            RecentPredictionResponse(
                id=p.id,
                timestamp=p.timestamp,
                sentiment=p.predicted_sentiment,
                confidence=p.confidence_score,
                latency_ms=p.latency_ms,
                model_version=p.model_version
            )
            for p in predictions
        ]
        
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent predictions: {str(e)}"
        )

