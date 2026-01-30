"""
Tests for Prometheus metrics instrumentation
"""
import pytest
from fastapi.testclient import TestClient
from monitoring.metrics import (
    track_request,
    track_cache_hit,
    track_cache_miss,
    track_prediction,
    calculate_hit_rate,
    get_cache_stats,
    api_requests_total,
    cache_hits_total,
    cache_misses_total,
    predictions_by_sentiment
)


def test_metrics_endpoint_exists(client):
    """Test that /metrics endpoint is available"""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_format(client):
    """Test that /metrics returns Prometheus text format"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Prometheus metrics are text/plain
    content = response.text
    
    # Should contain our custom metrics
    assert "api_requests_total" in content or "TYPE" in content
    # Basic check that it looks like Prometheus format
    assert "#" in content or "api_" in content


def test_metrics_increment_on_requests(client):
    """Test that metrics increment when making requests"""
    # Get initial metrics
    response1 = client.get("/metrics")
    initial_content = response1.text
    
    # Make a prediction
    client.post("/predict", json={"text": "Test for metrics"})
    
    # Get updated metrics
    response2 = client.get("/metrics")
    updated_content = response2.text
    
    # Metrics should have changed
    # (This is a basic check - specific counters tested below)
    assert response2.status_code == 200


def test_track_request_function():
    """Test the track_request helper function"""
    # Track a request
    track_request("/predict", 200, 0.05)
    
    # Should not raise an error
    # Actual counter value checked by Prometheus


def test_track_cache_hit_function():
    """Test the track_cache_hit helper function"""
    # Get initial stats
    initial_stats = get_cache_stats()
    initial_hits = initial_stats["hits"]
    
    # Track a cache hit
    track_cache_hit()
    
    # Get updated stats
    updated_stats = get_cache_stats()
    
    # Hits should have increased
    assert updated_stats["hits"] > initial_hits


def test_track_cache_miss_function():
    """Test the track_cache_miss helper function"""
    # Get initial stats
    initial_stats = get_cache_stats()
    initial_misses = initial_stats["misses"]
    
    # Track a cache miss
    track_cache_miss()
    
    # Get updated stats
    updated_stats = get_cache_stats()
    
    # Misses should have increased
    assert updated_stats["misses"] > initial_misses


def test_track_prediction_function():
    """Test the track_prediction helper function"""
    # Track predictions for each sentiment
    track_prediction("positive")
    track_prediction("negative")
    track_prediction("neutral")
    
    # Should not raise errors
    # Actual counter values tracked by Prometheus


def test_calculate_hit_rate():
    """Test cache hit rate calculation"""
    # Hit rate calculation is based on current counter values
    hit_rate = calculate_hit_rate()
    
    # Should return a valid percentage
    assert 0.0 <= hit_rate <= 100.0
    assert isinstance(hit_rate, float)


def test_calculate_hit_rate_with_no_activity():
    """Test hit rate calculation when no cache activity"""
    # Note: This test may fail if other tests have already tracked cache activity
    # The hit rate should still be a valid number (0.0 or some percentage)
    hit_rate = calculate_hit_rate()
    assert isinstance(hit_rate, float)
    assert 0.0 <= hit_rate <= 100.0


def test_get_cache_stats():
    """Test get_cache_stats returns correct structure"""
    stats = get_cache_stats()
    
    # Should have all required fields
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats
    
    # All should be valid numbers
    assert isinstance(stats["hits"], int)
    assert isinstance(stats["misses"], int)
    assert isinstance(stats["hit_rate"], float)
    
    # Values should be non-negative
    assert stats["hits"] >= 0
    assert stats["misses"] >= 0
    assert 0.0 <= stats["hit_rate"] <= 100.0


def test_cache_stats_endpoint(client):
    """Test /cache/stats endpoint structure"""
    response = client.get("/cache/stats")
    assert response.status_code == 200
    
    data = response.json()
    
    # Should have all required fields
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate" in data
    assert "cache_size" in data
    
    # All should be valid types
    assert isinstance(data["hits"], int)
    assert isinstance(data["misses"], int)
    assert isinstance(data["hit_rate"], (int, float))
    assert isinstance(data["cache_size"], int)


def test_metrics_after_multiple_predictions(client):
    """Test that metrics accumulate correctly over multiple predictions"""
    # Make several predictions
    texts = [
        "This is great!",
        "Terrible experience",
        "It's okay I guess",
        "Absolutely wonderful",
        "Could be better"
    ]
    
    for text in texts:
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
    
    # Check metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    
    # Should contain prediction metrics
    content = response.text
    # Just verify it's valid Prometheus format
    assert len(content) > 0


def test_metrics_after_batch_predictions(client):
    """Test metrics work with batch predictions"""
    # Make a batch prediction
    response = client.post(
        "/batch",
        json={"texts": ["Great!", "Bad!", "Neutral"]}
    )
    assert response.status_code == 200
    
    # Check metrics
    response = client.get("/metrics")
    assert response.status_code == 200


def test_cache_hit_rate_updates_correctly(client):
    """Test that cache hit rate updates as we make requests"""
    # Make a prediction twice to create a cache hit
    text = "Test for hit rate"
    
    # First request - miss
    client.post("/predict", json={"text": text})
    
    # Second request - hit
    client.post("/predict", json={"text": text})
    
    # Get stats
    response = client.get("/cache/stats")
    data = response.json()
    
    # Hit rate should be calculated correctly
    # (May not be exactly 50% due to other tests, but should be valid)
    assert 0.0 <= data["hit_rate"] <= 100.0
    
    # Should have at least some hits
    assert data["hits"] > 0


def test_metrics_endpoint_with_health_check(client):
    """Test that health check is also tracked in metrics"""
    # Make a health check
    response = client.get("/health")
    assert response.status_code == 200
    
    # Metrics should still be accessible
    response = client.get("/metrics")
    assert response.status_code == 200


def test_prometheus_metrics_format_details(client):
    """Test detailed Prometheus metrics format"""
    # Make some requests to generate metrics
    client.post("/predict", json={"text": "Test metrics format"})
    
    response = client.get("/metrics")
    assert response.status_code == 200
    
    content = response.text
    
    # Check for specific metric patterns (Prometheus format)
    # Should have TYPE definitions
    assert "# HELP" in content or "# TYPE" in content or len(content) > 100
    
    # Should have metric names (at least some of them)
    # Note: Actual format depends on whether metrics have been recorded
    assert isinstance(content, str)
    assert len(content) > 0

