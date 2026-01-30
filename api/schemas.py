"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request schema for sentiment prediction"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment"
    )
    
    @field_validator('text')
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
        ...,
        description="Predicted sentiment: positive, negative, or neutral"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    latency_ms: float = Field(
        ...,
        gt=0.0,
        description="Inference latency in milliseconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": 0.9998,
                "latency_ms": 45.23
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str = Field(
        ...,
        description="Health status: healthy or unhealthy"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error responses"""
    detail: str = Field(
        ...,
        description="Error message describing what went wrong"
    )
    error_type: str = Field(
        ...,
        description="Type of error that occurred"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Text cannot be empty",
                "error_type": "ValueError"
            }
        }

