"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime
from uuid import UUID


class PredictRequest(BaseModel):
    """Request schema for sentiment prediction"""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to analyze for sentiment"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty after stripping whitespace"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Text cannot be empty or whitespace only")
        return stripped


class PredictResponse(BaseModel):
    """Response schema for sentiment prediction"""

    sentiment: str = Field(
        ..., description="Predicted sentiment: positive, negative, or neutral"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    latency_ms: float = Field(
        ..., gt=0.0, description="Inference latency in milliseconds"
    )
    cache_hit: bool = Field(
        default=False, description="Whether result was served from cache"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9998,
                "latency_ms": 45.23,
                "cache_hit": False,
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""

    status: str = Field(..., description="Health status: healthy or unhealthy")
    model_loaded: bool = Field(
        ..., description="Whether the ML model is loaded and ready"
    )

    class Config:
        json_schema_extra = {"example": {"status": "healthy", "model_loaded": True}}


class ErrorResponse(BaseModel):
    """Response schema for error responses"""

    detail: str = Field(..., description="Error message describing what went wrong")
    error_type: str = Field(..., description="Type of error that occurred")

    class Config:
        json_schema_extra = {
            "example": {"detail": "Text cannot be empty", "error_type": "ValueError"}
        }


class BatchPredictRequest(BaseModel):
    """Request schema for batch sentiment prediction"""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze (max 100)",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate each text in the list"""
        if not v:
            raise ValueError("Texts list cannot be empty")

        validated = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")

            stripped = text.strip()
            if not stripped:
                raise ValueError(
                    f"Text at index {i} cannot be empty or whitespace only"
                )

            if len(stripped) > 5000:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of 5000 characters"
                )

            validated.append(stripped)

        return validated

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "I love this product!",
                    "This is terrible",
                    "It's okay, nothing special",
                ]
            }
        }


class BatchPredictResponse(BaseModel):
    """Response schema for batch sentiment prediction"""

    predictions: List[PredictResponse] = Field(
        ..., description="List of prediction results"
    )
    total: int = Field(..., description="Total number of predictions")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "sentiment": "positive",
                        "confidence": 0.9998,
                        "latency_ms": 45.23,
                    },
                    {
                        "sentiment": "negative",
                        "confidence": 0.9995,
                        "latency_ms": 38.17,
                    },
                ],
                "total": 2,
            }
        }


class AnalyticsSummaryResponse(BaseModel):
    """Response schema for analytics summary"""

    total_predictions: int = Field(..., description="Total number of predictions")
    avg_confidence: float = Field(..., description="Average confidence score")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    date_range: dict = Field(
        ..., description="Earliest and latest prediction timestamps"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 1000,
                "avg_confidence": 0.9234,
                "avg_latency_ms": 42.15,
                "date_range": {
                    "earliest": "2026-01-01T00:00:00",
                    "latest": "2026-01-30T12:00:00",
                },
            }
        }


class SentimentDistributionResponse(BaseModel):
    """Response schema for sentiment distribution"""

    positive: int = Field(..., description="Count of positive predictions")
    negative: int = Field(..., description="Count of negative predictions")
    neutral: int = Field(..., description="Count of neutral predictions")

    class Config:
        json_schema_extra = {
            "example": {"positive": 450, "negative": 350, "neutral": 200}
        }


class RecentPredictionResponse(BaseModel):
    """Response schema for a single recent prediction (without input text)"""

    id: UUID = Field(..., description="Prediction ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    sentiment: str = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., description="Confidence score")
    latency_ms: float = Field(..., description="Inference latency")
    model_version: str = Field(..., description="Model version used")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2026-01-30T12:00:00",
                "sentiment": "positive",
                "confidence": 0.9998,
                "latency_ms": 45.23,
                "model_version": "distilbert-v1",
            }
        }


class CacheStatsResponse(BaseModel):
    """Response schema for cache statistics"""

    hits: int = Field(..., description="Total cache hits")
    misses: int = Field(..., description="Total cache misses")
    hit_rate: float = Field(..., description="Cache hit rate percentage (0-100)")
    cache_size: int = Field(..., description="Current number of keys in cache")

    class Config:
        json_schema_extra = {
            "example": {"hits": 150, "misses": 50, "hit_rate": 75.0, "cache_size": 45}
        }


class RateLimitStatusResponse(BaseModel):
    """Response schema for rate limit status"""

    ip: str = Field(..., description="Client IP address")
    limit: str = Field(..., description="Rate limit (e.g., '100/minute')")
    remaining: int = Field(
        ..., ge=0, description="Requests remaining in current window"
    )
    reset_in_seconds: int = Field(
        ..., ge=0, description="Seconds until rate limit resets"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ip": "127.0.0.1",
                "limit": "100/minute",
                "remaining": 75,
                "reset_in_seconds": 45,
            }
        }


class ABPredictResponse(PredictResponse):
    """Response schema for A/B test predictions"""

    model_version: str = Field(..., description="Model version used (v1/v2)")

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9998,
                "latency_ms": 45.23,
                "cache_hit": False,
                "model_version": "v1",
            }
        }


class ModelStats(BaseModel):
    """Statistics for a single model version"""

    total_predictions: int = Field(..., description="Total predictions made")
    avg_confidence: float = Field(..., description="Average confidence score")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    sentiment_distribution: dict = Field(..., description="Count by sentiment type")


class ComparisonStats(BaseModel):
    """Comparison metrics between model versions"""

    confidence_diff: float = Field(..., description="Confidence difference (v2 - v1)")
    latency_diff: float = Field(..., description="Latency difference (v2 - v1)")
    sample_size_sufficient: bool = Field(
        ..., description="Both versions have >100 predictions"
    )
    traffic_distribution: dict = Field(
        ..., description="Traffic percentage per version"
    )


class ABComparisonResponse(BaseModel):
    """Response schema for A/B test comparison"""

    model_v1: ModelStats = Field(..., description="Statistics for model v1")
    model_v2: ModelStats = Field(..., description="Statistics for model v2")
    comparison: ComparisonStats = Field(..., description="Comparison metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "model_v1": {
                    "total_predictions": 523,
                    "avg_confidence": 0.9456,
                    "avg_latency_ms": 48.32,
                    "sentiment_distribution": {
                        "positive": 312,
                        "negative": 211,
                        "neutral": 0,
                    },
                },
                "model_v2": {
                    "total_predictions": 507,
                    "avg_confidence": 0.9445,
                    "avg_latency_ms": 47.89,
                    "sentiment_distribution": {
                        "positive": 298,
                        "negative": 209,
                        "neutral": 0,
                    },
                },
                "comparison": {
                    "confidence_diff": -0.0011,
                    "latency_diff": -0.43,
                    "sample_size_sufficient": True,
                    "traffic_distribution": {"v1": 50.8, "v2": 49.2},
                },
            }
        }
