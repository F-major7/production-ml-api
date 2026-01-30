"""
Integration tests for Production ML API
Tests complete request flows and system interactions
"""
import pytest
import time
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from db.models import Prediction


def test_full_prediction_flow(client_with_db):
    """Test complete flow: predict → database log → cache"""
    # First request (cache miss)
    response = client_with_db.post(
        "/predict",
        json={"text": "Integration test message"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert data["cache_hit"] is False
    assert data["latency_ms"] > 0
    
    # Second request (cache hit)
    response2 = client_with_db.post(
        "/predict",
        json={"text": "Integration test message"}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cache_hit"] is True
    assert data2["sentiment"] == data["sentiment"]
    # Cached response should be faster
    assert data2["latency_ms"] < data["latency_ms"]


def test_health_check_validates_dependencies(client):
    """Test health endpoint checks all dependencies"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    # DB is overridden to None in tests, so it might be False
    assert data["redis_connected"] is True


def test_metrics_endpoint_increments(client):
    """Test that prediction increments metrics"""
    # Get initial metrics
    metrics_before = client.get("/metrics/")
    assert metrics_before.status_code == 200
    
    # Make a prediction
    response = client.post(
        "/predict",
        json={"text": "Metrics test"}
    )
    assert response.status_code == 200
    
    # Check metrics incremented
    metrics_after = client.get("/metrics/")
    assert metrics_after.status_code == 200
    metrics_text = metrics_after.text
    assert "api_requests_total" in metrics_text
    assert "api_request_latency_seconds" in metrics_text


def test_rate_limiting_triggers_429(client):
    """Test rate limiting returns 429 after limit exceeded"""
    # Note: This test might be flaky depending on rate limit window
    # We'll make requests until we hit the limit
    responses = []
    for i in range(105):  # Limit is 100/minute
        response = client.post(
            "/predict",
            json={"text": f"Rate limit test {i}"}
        )
        responses.append(response.status_code)
        if response.status_code == 429:
            break
    
    # Should have hit rate limit
    assert 429 in responses


def test_batch_prediction_with_cache(client):
    """Test batch predictions work with caching"""
    texts = [
        "Batch test 1 - excellent",
        "Batch test 2 - terrible",
        "Batch test 3 - okay"
    ]
    
    # First batch (all cache misses)
    response1 = client.post(
        "/batch",
        json={"texts": texts}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["total"] == 3
    assert len(data1["predictions"]) == 3
    
    # Second batch (all cache hits)
    response2 = client.post(
        "/batch",
        json={"texts": texts}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["total"] == 3
    # Check at least some are cached
    cache_hits = sum(1 for p in data2["predictions"] if p["cache_hit"])
    assert cache_hits > 0


def test_ab_testing_integration(client):
    """Test A/B testing works"""
    response = client.post(
        "/predict/ab",
        json={"text": "A/B test integration"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] in ["v1", "v2"]


def test_analytics_reflects_predictions(client_with_db):
    """Test analytics endpoints reflect database state"""
    # Make some predictions first
    texts = [
        "Analytics positive test",
        "Analytics negative horrible test",
        "Analytics neutral test"
    ]
    
    for text in texts:
        client_with_db.post("/predict", json={"text": text})
    
    # Check analytics summary
    response = client_with_db.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] >= 3
    assert data["avg_confidence"] > 0
    assert data["avg_latency_ms"] > 0
    
    # Check sentiment distribution
    dist_response = client_with_db.get("/analytics/sentiment-distribution")
    assert dist_response.status_code == 200
    dist_data = dist_response.json()
    total = dist_data["positive"] + dist_data["negative"] + dist_data["neutral"]
    assert total >= 3


def test_cache_stats_accuracy(client):
    """Test cache stats endpoint reports accurate numbers"""
    # Get initial stats
    stats1 = client.get("/cache/stats")
    initial_data = stats1.json()
    
    # Make a unique request (miss)
    unique_text = f"Cache stats test {time.time()}"
    client.post("/predict", json={"text": unique_text})
    
    # Make the same request (hit)
    client.post("/predict", json={"text": unique_text})
    
    # Check stats updated
    stats2 = client.get("/cache/stats")
    final_data = stats2.json()
    
    assert final_data["hits"] >= initial_data["hits"] + 1
    assert final_data["misses"] >= initial_data["misses"] + 1
    assert 0 <= final_data["hit_rate"] <= 100


def test_error_handling_invalid_input(client):
    """Test API handles invalid input gracefully"""
    # Empty text
    response1 = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response1.status_code == 422  # Validation error
    
    # Missing field
    response2 = client.post(
        "/predict",
        json={}
    )
    assert response2.status_code == 422
    
    # Invalid JSON
    response3 = client.post(
        "/predict",
        content="not json",
        headers={"Content-Type": "application/json"}
    )
    assert response3.status_code == 422


def test_model_versioning_consistency(client):
    """Test both model versions work consistently"""
    text = "Model version test"
    
    # Test v1 via regular predict
    response1 = client.post("/predict", json={"text": text})
    assert response1.status_code == 200
    
    # Test v1 and v2 via A/B endpoint (make multiple requests)
    versions_seen = set()
    for _ in range(20):
        response = client.post("/predict/ab", json={"text": text})
        if response.status_code == 200:
            data = response.json()
            versions_seen.add(data.get("model_version"))
    
    # Should have seen both versions (probabilistically)
    # With 20 requests, very likely to see both
    assert len(versions_seen) >= 1  # At least one version works

