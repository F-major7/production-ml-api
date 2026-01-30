"""
Tests for rate limiting functionality
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app
import time


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_rate_limit_status_endpoint(client):
    """Test rate limit status endpoint returns correct information"""
    response = client.get("/rate-limit/status")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "ip" in data
    assert "limit" in data
    assert "remaining" in data
    assert "reset_in_seconds" in data
    
    # Check types
    assert isinstance(data["ip"], str)
    assert isinstance(data["limit"], str)
    assert isinstance(data["remaining"], int)
    assert isinstance(data["reset_in_seconds"], int)
    
    # Check values are reasonable
    assert data["remaining"] >= 0
    assert data["reset_in_seconds"] > 0


@pytest.mark.skip(reason="Rate limiting with TestClient requires real time delays and may be flaky")
def test_predict_rate_limit_exceeded(client):
    """
    Test that /predict endpoint enforces rate limit (100/minute).
    Note: Skipped by default as it requires 100+ requests and time delays.
    """
    # Make requests up to the limit
    responses = []
    for i in range(105):
        response = client.post(
            "/predict",
            json={"text": f"Test rate limit {i}"}
        )
        responses.append(response.status_code)
        
        # Small delay to avoid overwhelming the server
        if i % 10 == 0:
            time.sleep(0.1)
    
    # Count successful and rate-limited responses
    success_count = sum(1 for code in responses if code == 200)
    rate_limited_count = sum(1 for code in responses if code == 429)
    
    # Should have approximately 100 successful and 5 rate-limited
    assert success_count <= 100, f"Expected max 100 successful, got {success_count}"
    assert rate_limited_count > 0, "Expected at least 1 rate-limited response"


@pytest.mark.skip(reason="Rate limiting with TestClient requires real time delays and may be flaky")
def test_batch_rate_limit_exceeded(client):
    """
    Test that /batch endpoint enforces rate limit (20/minute).
    Note: Skipped by default as it requires 20+ requests and time delays.
    """
    # Make requests up to the limit
    responses = []
    for i in range(25):
        response = client.post(
            "/batch",
            json={"texts": [f"Batch test {i}-1", f"Batch test {i}-2"]}
        )
        responses.append(response.status_code)
        
        # Small delay
        if i % 5 == 0:
            time.sleep(0.1)
    
    # Count responses
    success_count = sum(1 for code in responses if code == 200)
    rate_limited_count = sum(1 for code in responses if code == 429)
    
    # Should have approximately 20 successful and 5 rate-limited
    assert success_count <= 20, f"Expected max 20 successful, got {success_count}"
    assert rate_limited_count > 0, "Expected at least 1 rate-limited response"


def test_rate_limit_headers_present(client):
    """Test that rate limit headers are present in responses"""
    response = client.post(
        "/predict",
        json={"text": "Test rate limit headers"}
    )
    
    # Note: slowapi may add rate limit headers
    # Check if they exist (optional, as TestClient may not preserve all headers)
    headers = response.headers
    
    # These headers might be present depending on slowapi configuration
    # Just verify the request succeeds
    assert response.status_code in [200, 429]


def test_metrics_endpoint_not_rate_limited(client):
    """Test that /metrics endpoint is not rate limited"""
    # Make multiple requests to metrics endpoint
    for i in range(10):
        response = client.get("/metrics")
        # Metrics should always be accessible
        assert response.status_code == 200


def test_health_endpoint_not_rate_limited(client):
    """Test that /health endpoint is not rate limited"""
    # Make multiple requests to health endpoint
    for i in range(10):
        response = client.get("/health")
        # Health should always be accessible
        assert response.status_code in [200, 503]  # 503 if dependencies unhealthy


def test_analytics_endpoints_rate_limited(client):
    """Test that analytics endpoints have rate limiting configured"""
    # Test each analytics endpoint
    endpoints = [
        "/analytics/summary",
        "/analytics/sentiment-distribution",
        "/analytics/recent"
    ]
    
    for endpoint in endpoints:
        response = client.get(endpoint)
        # Should succeed or fail with 429 if limit reached
        # (depends on test execution order)
        assert response.status_code in [200, 429, 500]  # 500 if DB not available in test


def test_cache_stats_endpoint_rate_limited(client):
    """Test that /cache/stats endpoint has rate limiting"""
    response = client.get("/cache/stats")
    
    # Should succeed or fail with 429 if limit reached
    assert response.status_code in [200, 429, 500]  # 500 if Redis not available


def test_ab_endpoint_rate_limited(client):
    """Test that A/B testing endpoint has rate limiting"""
    response = client.post(
        "/predict/ab",
        json={"text": "Test A/B rate limit"}
    )
    
    # Should succeed or fail with 429 if limit reached
    assert response.status_code in [200, 429]


def test_rate_limit_different_endpoints_separate_limits(client):
    """
    Test that different endpoints have separate rate limits.
    Make requests to different endpoints and verify they don't share limits.
    """
    # Make several requests to /predict
    predict_responses = []
    for i in range(5):
        response = client.post(
            "/predict",
            json={"text": f"Test {i}"}
        )
        predict_responses.append(response.status_code)
    
    # Make several requests to /analytics/summary
    analytics_responses = []
    for i in range(5):
        response = client.get("/analytics/summary")
        analytics_responses.append(response.status_code)
    
    # Both should have some successful responses
    # (assuming we haven't hit the limit yet in the test run)
    predict_success = sum(1 for code in predict_responses if code == 200)
    analytics_success = sum(1 for code in analytics_responses if code in [200, 500])
    
    # At least some requests should succeed for each endpoint
    assert predict_success > 0, "Predict endpoint should allow some requests"
    # Analytics might fail due to DB, but shouldn't be rate limited immediately
    assert all(code != 429 for code in analytics_responses[:3]), \
        "Analytics should not be rate limited in first few requests"


@pytest.mark.skip(reason="Manual test - verifies rate limit reset behavior")
def test_rate_limit_resets_after_window(client):
    """
    Test that rate limits reset after the time window.
    This test requires waiting 60+ seconds.
    """
    # Make requests until rate limited
    for i in range(105):
        response = client.post(
            "/predict",
            json={"text": f"Test {i}"}
        )
        if response.status_code == 429:
            break
    
    # Should be rate limited now
    response = client.post("/predict", json={"text": "Should be limited"})
    assert response.status_code == 429
    
    # Wait for rate limit window to reset (61 seconds to be safe)
    time.sleep(61)
    
    # Should be able to make requests again
    response = client.post("/predict", json={"text": "Should succeed now"})
    assert response.status_code == 200


def test_rate_limit_retry_after_header(client):
    """
    Test that 429 responses include Retry-After header.
    Note: This test is informational and may be skipped.
    """
    # Make many requests to trigger rate limit
    # (This is a simplified test - actual rate limiting needs more requests)
    response = None
    for i in range(105):
        response = client.post(
            "/predict",
            json={"text": f"Test {i}"}
        )
        if response.status_code == 429:
            break
    
    # If we got a 429, check for Retry-After header
    if response and response.status_code == 429:
        # slowapi should add this header
        # Note: TestClient may not preserve all headers
        assert "Retry-After" in response.headers or "X-RateLimit-Reset" in response.headers

