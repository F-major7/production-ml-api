"""
Production ML API - Main FastAPI Application
Sentiment analysis API with full observability
"""
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from prometheus_client import make_asgi_app
import time
import logging
import json
from typing import Optional

from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    CacheStatsResponse,
    ABPredictResponse,
    ABComparisonResponse,
    ModelStats,
    ComparisonStats
)
from api.dependencies import get_model, get_redis
from api import analytics
from api.ab_testing import ab_router
from models.sentiment import SentimentModel
from cache.redis_client import RedisCache
from db.database import get_db, init_db, close_db
from db.models import Prediction
from monitoring.metrics import (
    track_request,
    track_cache_hit,
    track_cache_miss,
    track_prediction,
    get_cache_stats
)
from sqlalchemy import func, select

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

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

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
    redis: Optional[RedisCache] = Depends(get_redis),
    db: Optional[AsyncSession] = Depends(get_db)
):
    """
    Analyze sentiment of provided text with caching.
    
    Returns:
        - sentiment: positive, negative, or neutral
        - confidence: score between 0 and 1
        - latency_ms: inference time in milliseconds
        - cache_hit: whether result was served from cache
    
    Also logs prediction to database for analytics.
    """
    request_start = time.perf_counter()
    cache_hit = False
    sentiment = None
    confidence = None
    latency_ms = None
    
    try:
        # Try cache first if Redis is available
        if redis is not None:
            cache_key = redis.generate_cache_key(request.text)
            cached_result = await redis.get(cache_key)
            
            if cached_result is not None:
                # Cache hit!
                try:
                    cached_data = json.loads(cached_result)
                    sentiment = cached_data["sentiment"]
                    confidence = cached_data["confidence"]
                    # Latency for cache hit is just the cache lookup time
                    latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
                    # Ensure minimum latency of 0.01ms to satisfy validation
                    latency_ms = max(latency_ms, 0.01)
                    cache_hit = True
                    
                    # Track metrics
                    track_cache_hit()
                    track_prediction(sentiment)
                    
                    logger.debug(f"Cache hit for key: {cache_key}")
                except Exception as cache_error:
                    logger.error(f"Error parsing cached result: {cache_error}")
                    # Continue to model prediction if cache parse fails
                    cached_result = None
        
        # Cache miss or no cache - run model prediction
        if not cache_hit:
            # Time the model prediction
            model_start = time.perf_counter()
            
            # Get prediction from model
            result = model.predict(request.text)
            
            # Calculate latency
            model_end = time.perf_counter()
            latency_ms = round((model_end - model_start) * 1000, 2)
            
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
            
            # Cache the result if Redis is available
            if redis is not None:
                try:
                    cache_data = {
                        "sentiment": sentiment,
                        "confidence": confidence
                    }
                    cache_key = redis.generate_cache_key(request.text)
                    await redis.set(cache_key, json.dumps(cache_data), ttl=3600)
                    logger.debug(f"Cached result for key: {cache_key}")
                except Exception as cache_error:
                    logger.error(f"Failed to cache result: {cache_error}")
            
            # Track metrics
            track_cache_miss()
            track_prediction(sentiment)
        
        # Log prediction to database (gracefully handle DB errors)
        if db is not None:
            try:
                prediction = Prediction(
                    input_text=request.text,
                    predicted_sentiment=sentiment,
                    confidence_score=confidence,
                    latency_ms=latency_ms,
                    model_version="distilbert-v1",
                    cache_hit=cache_hit
                )
                db.add(prediction)
                await db.commit()
                logger.debug(f"Prediction logged to database: {prediction.id}")
            except Exception as db_error:
                logger.error(f"Failed to log prediction to database: {db_error}")
                # Don't fail the request if database logging fails
                await db.rollback()
        
        # Track request metrics
        request_latency = time.perf_counter() - request_start
        track_request("/predict", 200, request_latency)
        
        return PredictResponse(
            sentiment=sentiment,
            confidence=confidence,
            latency_ms=latency_ms,
            cache_hit=cache_hit
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
    redis: Optional[RedisCache] = Depends(get_redis),
    db: Optional[AsyncSession] = Depends(get_db)
):
    """
    Analyze sentiment for multiple texts in a single request with caching.
    
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
            cache_hit = False
            sentiment = None
            confidence = None
            latency_ms = None
            
            # Try cache first if Redis is available
            if redis is not None:
                cache_key = redis.generate_cache_key(text)
                cached_result = await redis.get(cache_key)
                
                if cached_result is not None:
                    try:
                        start_time = time.perf_counter()
                        cached_data = json.loads(cached_result)
                        sentiment = cached_data["sentiment"]
                        confidence = cached_data["confidence"]
                        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                        # Ensure minimum latency of 0.01ms to satisfy validation
                        latency_ms = max(latency_ms, 0.01)
                        cache_hit = True
                        track_cache_hit()
                    except Exception:
                        cached_result = None
            
            # Cache miss - run model prediction
            if not cache_hit:
                start_time = time.perf_counter()
                result = model.predict(text)
                end_time = time.perf_counter()
                latency_ms = round((end_time - start_time) * 1000, 2)
                
                sentiment_map = {
                    "POSITIVE": "positive",
                    "NEGATIVE": "negative",
                    "NEUTRAL": "neutral"
                }
                
                sentiment = sentiment_map.get(result["label"].upper(), "neutral")
                confidence = round(result["score"], 4)
                
                # Cache the result
                if redis is not None:
                    try:
                        cache_data = {"sentiment": sentiment, "confidence": confidence}
                        cache_key = redis.generate_cache_key(text)
                        await redis.set(cache_key, json.dumps(cache_data), ttl=3600)
                    except Exception as cache_error:
                        logger.error(f"Failed to cache result: {cache_error}")
                
                track_cache_miss()
            
            # Track metrics
            track_prediction(sentiment)
            
            # Add to response list
            predictions.append(
                PredictResponse(
                    sentiment=sentiment,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    cache_hit=cache_hit
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
                        cache_hit=cache_hit
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


@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Cache"],
    summary="Get cache statistics"
)
async def get_cache_statistics(
    redis: Optional[RedisCache] = Depends(get_redis)
):
    """
    Get cache hit/miss statistics and current cache size.
    
    Returns:
        - hits: Total cache hits
        - misses: Total cache misses
        - hit_rate: Cache hit rate percentage (0-100)
        - cache_size: Current number of keys in cache
    """
    try:
        # Get stats from metrics
        stats = get_cache_stats()
        
        # Get cache size from Redis
        cache_size = 0
        if redis is not None:
            cache_size = await redis.get_cache_size()
        
        return CacheStatsResponse(
            hits=stats["hits"],
            misses=stats["misses"],
            hit_rate=stats["hit_rate"],
            cache_size=cache_size
        )
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@app.post(
    "/predict/ab",
    response_model=ABPredictResponse,
    tags=["Prediction", "A/B Testing"],
    summary="Predict with A/B testing"
)
async def predict_sentiment_ab(
    request: PredictRequest,
    redis: Optional[RedisCache] = Depends(get_redis),
    db: Optional[AsyncSession] = Depends(get_db)
):
    """
    Analyze sentiment with A/B testing (50/50 split between model versions).
    
    Returns prediction plus which model version was used.
    """
    request_start = time.perf_counter()
    
    # Select model version for A/B test
    model_version = ab_router.select_model_version()
    model = ab_router.get_model_for_version(model_version)
    
    cache_hit = False
    sentiment = None
    confidence = None
    latency_ms = None
    
    try:
        # Try cache first (include model version in cache key)
        if redis is not None:
            cache_key = f"sentiment:{model_version}:{redis.generate_cache_key(request.text).split(':', 1)[1]}"
            cached_result = await redis.get(cache_key)
            
            if cached_result is not None:
                try:
                    cached_data = json.loads(cached_result)
                    sentiment = cached_data["sentiment"]
                    confidence = cached_data["confidence"]
                    latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
                    latency_ms = max(latency_ms, 0.01)
                    cache_hit = True
                    track_cache_hit()
                    track_prediction(sentiment, model_version)
                except Exception as cache_error:
                    logger.error(f"Error parsing cached result: {cache_error}")
                    cached_result = None
        
        # Cache miss - run model prediction
        if not cache_hit:
            model_start = time.perf_counter()
            result = model.predict(request.text)
            model_end = time.perf_counter()
            latency_ms = round((model_end - model_start) * 1000, 2)
            
            sentiment_map = {
                "POSITIVE": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral"
            }
            
            sentiment = sentiment_map.get(result["label"].upper(), "neutral")
            confidence = round(result["score"], 4)
            
            # Cache the result
            if redis is not None:
                try:
                    cache_data = {"sentiment": sentiment, "confidence": confidence}
                    cache_key = f"sentiment:{model_version}:{redis.generate_cache_key(request.text).split(':', 1)[1]}"
                    await redis.set(cache_key, json.dumps(cache_data), ttl=3600)
                except Exception as cache_error:
                    logger.error(f"Failed to cache result: {cache_error}")
            
            track_cache_miss()
            track_prediction(sentiment, model_version)
        
        # Log to database with model version
        if db is not None:
            try:
                prediction = Prediction(
                    input_text=request.text,
                    predicted_sentiment=sentiment,
                    confidence_score=confidence,
                    latency_ms=latency_ms,
                    model_version=model_version,
                    cache_hit=cache_hit
                )
                db.add(prediction)
                await db.commit()
            except Exception as db_error:
                logger.error(f"Failed to log prediction: {db_error}")
                await db.rollback()
        
        # Track request metrics with model version
        request_latency = time.perf_counter() - request_start
        track_request("/predict/ab", 200, request_latency, model_version)
        
        return ABPredictResponse(
            sentiment=sentiment,
            confidence=confidence,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            model_version=model_version
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get(
    "/ab/comparison",
    response_model=ABComparisonResponse,
    tags=["A/B Testing"],
    summary="Compare A/B test results"
)
async def get_ab_comparison(
    db: AsyncSession = Depends(get_db)
):
    """
    Get comparison statistics between model versions.
    Analyzes predictions from A/B testing to compare performance.
    """
    try:
        # Query predictions for v1
        v1_query = select(
            func.count(Prediction.id).label('total'),
            func.avg(Prediction.confidence_score).label('avg_confidence'),
            func.avg(Prediction.latency_ms).label('avg_latency'),
            Prediction.predicted_sentiment,
            func.count(Prediction.predicted_sentiment).label('sentiment_count')
        ).where(
            Prediction.model_version == 'v1'
        ).group_by(Prediction.predicted_sentiment)
        
        v1_results = await db.execute(v1_query)
        v1_data = v1_results.all()
        
        # Query predictions for v2
        v2_query = select(
            func.count(Prediction.id).label('total'),
            func.avg(Prediction.confidence_score).label('avg_confidence'),
            func.avg(Prediction.latency_ms).label('avg_latency'),
            Prediction.predicted_sentiment,
            func.count(Prediction.predicted_sentiment).label('sentiment_count')
        ).where(
            Prediction.model_version == 'v2'
        ).group_by(Prediction.predicted_sentiment)
        
        v2_results = await db.execute(v2_query)
        v2_data = v2_results.all()
        
        # Process v1 stats
        v1_total = sum(row.sentiment_count for row in v1_data)
        v1_sentiment_dist = {row.predicted_sentiment: row.sentiment_count for row in v1_data}
        v1_avg_conf = sum(row.avg_confidence * row.sentiment_count for row in v1_data) / v1_total if v1_total > 0 else 0
        v1_avg_lat = sum(row.avg_latency * row.sentiment_count for row in v1_data) / v1_total if v1_total > 0 else 0
        
        # Process v2 stats
        v2_total = sum(row.sentiment_count for row in v2_data)
        v2_sentiment_dist = {row.predicted_sentiment: row.sentiment_count for row in v2_data}
        v2_avg_conf = sum(row.avg_confidence * row.sentiment_count for row in v2_data) / v2_total if v2_total > 0 else 0
        v2_avg_lat = sum(row.avg_latency * row.sentiment_count for row in v2_data) / v2_total if v2_total > 0 else 0
        
        # Ensure all sentiments are present
        for sent in ["positive", "negative", "neutral"]:
            v1_sentiment_dist.setdefault(sent, 0)
            v2_sentiment_dist.setdefault(sent, 0)
        
        # Calculate comparison metrics
        total_predictions = v1_total + v2_total
        traffic_dist = {
            "v1": round((v1_total / total_predictions * 100), 2) if total_predictions > 0 else 0,
            "v2": round((v2_total / total_predictions * 100), 2) if total_predictions > 0 else 0
        }
        
        return ABComparisonResponse(
            model_v1=ModelStats(
                total_predictions=v1_total,
                avg_confidence=round(v1_avg_conf, 4),
                avg_latency_ms=round(v1_avg_lat, 2),
                sentiment_distribution=v1_sentiment_dist
            ),
            model_v2=ModelStats(
                total_predictions=v2_total,
                avg_confidence=round(v2_avg_conf, 4),
                avg_latency_ms=round(v2_avg_lat, 2),
                sentiment_distribution=v2_sentiment_dist
            ),
            comparison=ComparisonStats(
                confidence_diff=round(v2_avg_conf - v1_avg_conf, 4),
                latency_diff=round(v2_avg_lat - v1_avg_lat, 2),
                sample_size_sufficient=(v1_total >= 100 and v2_total >= 100),
                traffic_distribution=traffic_dist
            )
        )
    except Exception as e:
        logger.error(f"Error getting A/B comparison: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get A/B comparison: {str(e)}"
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

