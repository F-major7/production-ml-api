"""
Production ML API - Main FastAPI Application
Sentiment analysis API with full observability
"""
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import time
import logging
from typing import Optional

from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse,
    BatchPredictRequest,
    BatchPredictResponse
)
from api.dependencies import get_model
from api import analytics
from models.sentiment import SentimentModel
from db.database import get_db, init_db, close_db
from db.models import Prediction

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

# Include analytics router
app.include_router(analytics.router)


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
    model: SentimentModel = Depends(get_model),
    db: Optional[AsyncSession] = Depends(get_db)
):
    """
    Analyze sentiment of provided text.
    
    Returns:
        - sentiment: positive, negative, or neutral
        - confidence: score between 0 and 1
        - latency_ms: inference time in milliseconds
    
    Also logs prediction to database for analytics.
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
        
        # Log prediction to database (gracefully handle DB errors)
        if db is not None:
            try:
                prediction = Prediction(
                    input_text=request.text,
                    predicted_sentiment=sentiment,
                    confidence_score=confidence,
                    latency_ms=latency_ms,
                    model_version="distilbert-v1",
                    cache_hit=False
                )
                db.add(prediction)
                await db.commit()
                logger.debug(f"Prediction logged to database: {prediction.id}")
            except Exception as db_error:
                logger.error(f"Failed to log prediction to database: {db_error}")
                # Don't fail the request if database logging fails
                await db.rollback()
        
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


@app.post(
    "/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Batch predict sentiment for multiple texts"
)
async def batch_predict_sentiment(
    request: BatchPredictRequest,
    model: SentimentModel = Depends(get_model),
    db: Optional[AsyncSession] = Depends(get_db)
):
    """
    Analyze sentiment for multiple texts in a single request.
    
    Args:
        request: BatchPredictRequest with list of texts (max 100)
    
    Returns:
        BatchPredictResponse with list of predictions and total count
    
    All predictions are logged to database for analytics.
    """
    try:
        predictions = []
        db_predictions = []
        
        for text in request.texts:
            # Time each prediction
            start_time = time.perf_counter()
            
            # Get prediction from model
            result = model.predict(text)
            
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
            
            # Add to response list
            predictions.append(
                PredictResponse(
                    sentiment=sentiment,
                    confidence=confidence,
                    latency_ms=latency_ms
                )
            )
            
            # Prepare database record
            if db is not None:
                db_predictions.append(
                    Prediction(
                        input_text=text,
                        predicted_sentiment=sentiment,
                        confidence_score=confidence,
                        latency_ms=latency_ms,
                        model_version="distilbert-v1",
                        cache_hit=False
                    )
                )
        
        # Bulk insert to database (gracefully handle DB errors)
        if db is not None and db_predictions:
            try:
                db.add_all(db_predictions)
                await db.commit()
                logger.info(f"Batch of {len(db_predictions)} predictions logged to database")
            except Exception as db_error:
                logger.error(f"Failed to log batch predictions to database: {db_error}")
                await db.rollback()
        
        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions)
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
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
    """Initialize model and database on startup"""
    logger.info("Starting Production ML API...")
    try:
        # Load ML model
        model = SentimentModel()
        logger.info(f"Model loaded: {model.is_loaded}")
        
        # Initialize database tables
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as db_error:
            logger.warning(f"Database initialization failed (will continue without DB): {db_error}")
    except Exception as e:
        logger.error(f"Failed to initialize on startup: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Production ML API...")
    try:
        await close_db()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

