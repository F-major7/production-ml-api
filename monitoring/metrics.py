"""
Prometheus metrics instrumentation for API monitoring
"""
import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Define Prometheus metrics

# API Request Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests by endpoint and status code',
    ['endpoint', 'status_code']
)

api_request_latency_seconds = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Cache Metrics
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Current cache hit rate (percentage)'
)

cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits'
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses'
)

# Prediction Metrics
predictions_by_sentiment = Counter(
    'predictions_by_sentiment',
    'Total predictions by sentiment type',
    ['sentiment']
)

# Rate Limiting Metrics
rate_limit_exceeded_total = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit exceeded (429) responses',
    ['endpoint']
)


def track_request(endpoint: str, status_code: int, latency: float, model_version: str = None) -> None:
    """
    Track API request metrics.
    
    Args:
        endpoint: API endpoint path
        status_code: HTTP status code
        latency: Request latency in seconds
        model_version: Optional model version for A/B testing
    """
    try:
        # Track requests with model version if provided
        if model_version:
            # For A/B endpoints, include model version in labels
            api_requests_total.labels(
                endpoint=f"{endpoint}_{model_version}", 
                status_code=status_code
            ).inc()
            api_request_latency_seconds.labels(
                endpoint=f"{endpoint}_{model_version}"
            ).observe(latency)
        else:
            # Standard tracking without model version
            api_requests_total.labels(endpoint=endpoint, status_code=status_code).inc()
            api_request_latency_seconds.labels(endpoint=endpoint).observe(latency)
    except Exception as e:
        logger.error(f"Error tracking request metrics: {e}")


def track_cache_hit() -> None:
    """Track a cache hit and update hit rate"""
    try:
        cache_hits_total.inc()
        _update_hit_rate()
    except Exception as e:
        logger.error(f"Error tracking cache hit: {e}")


def track_cache_miss() -> None:
    """Track a cache miss and update hit rate"""
    try:
        cache_misses_total.inc()
        _update_hit_rate()
    except Exception as e:
        logger.error(f"Error tracking cache miss: {e}")


def track_prediction(sentiment: str, model_version: str = None) -> None:
    """
    Track prediction by sentiment type.
    
    Args:
        sentiment: Sentiment label (positive/negative/neutral)
        model_version: Optional model version for A/B testing
    """
    try:
        if model_version:
            # Track with model version label for A/B testing
            predictions_by_sentiment.labels(
                sentiment=f"{sentiment}_{model_version}"
            ).inc()
        else:
            # Standard tracking
            predictions_by_sentiment.labels(sentiment=sentiment).inc()
    except Exception as e:
        logger.error(f"Error tracking prediction: {e}")


def _update_hit_rate() -> None:
    """Update cache hit rate gauge"""
    try:
        hit_rate = calculate_hit_rate()
        cache_hit_rate.set(hit_rate)
    except Exception as e:
        logger.error(f"Error updating hit rate: {e}")


def calculate_hit_rate() -> float:
    """
    Calculate cache hit rate percentage.
    
    Returns:
        Hit rate as percentage (0-100), or 0 if no cache activity
    """
    try:
        # Get current counter values using _value._value to access the actual count
        hits = cache_hits_total._value._value
        misses = cache_misses_total._value._value
        
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
    except Exception as e:
        logger.error(f"Error calculating hit rate: {e}")
        return 0.0


def get_cache_stats() -> dict:
    """
    Get current cache statistics.
    
    Returns:
        Dictionary with hits, misses, and hit_rate
    """
    try:
        hits = cache_hits_total._value._value
        misses = cache_misses_total._value._value
        hit_rate = calculate_hit_rate()
        
        return {
            "hits": int(hits),
            "misses": int(misses),
            "hit_rate": hit_rate
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0
        }


def track_rate_limit_exceeded(endpoint: str) -> None:
    """
    Track rate limit exceeded events.
    
    Args:
        endpoint: API endpoint that exceeded rate limit
    """
    try:
        rate_limit_exceeded_total.labels(endpoint=endpoint).inc()
    except Exception as e:
        logger.error(f"Error tracking rate limit exceeded: {e}")

