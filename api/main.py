"""
Production ML API - Main FastAPI Application
Sentiment analysis API with full observability
"""
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse
)
from api.dependencies import get_model
from models.sentiment import SentimentModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Production ML API",
    version="1.0.0",
    description="Sentiment analysis with full observability",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Production ML API",
        "docs": "/docs"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check API and model health status.
    Returns 200 if healthy, 503 if model not loaded.
    """
    try:
        model = SentimentModel()
        model_loaded = model.is_loaded
        
        if model_loaded:
            return HealthResponse(
                status="healthy",
                model_loaded=True
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "model_loaded": False
                }
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False
            }
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict sentiment of text"
)
async def predict_sentiment(
    request: PredictRequest,
    model: SentimentModel = Depends(get_model)
):
    """
    Analyze sentiment of provided text.
    
    Returns:
        - sentiment: positive, negative, or neutral
        - confidence: score between 0 and 1
        - latency_ms: inference time in milliseconds
    """
    try:
        # Time the prediction
        start_time = time.perf_counter()
        
        # Get prediction from model
        result = model.predict(request.text)
        
        # Calculate latency
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        # Map HuggingFace labels to our schema
        sentiment_map = {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral"
        }
        
        sentiment = sentiment_map.get(
            result["label"].upper(),
            "neutral"
        )
        
        # Round confidence to 4 decimals
        confidence = round(result["score"], 4)
        
        return PredictResponse(
            sentiment=sentiment,
            confidence=confidence,
            latency_ms=latency_ms
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "error_type": "ValueError"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Production ML API...")
    try:
        model = SentimentModel()
        logger.info(f"Model loaded: {model.is_loaded}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Production ML API...")

