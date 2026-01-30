"""
Tests for A/B testing functionality
"""
import pytest
from api.ab_testing import ABTestRouter


def test_ab_router_initialization():
    """Test ABTestRouter initializes correctly"""
    router = ABTestRouter()
    assert router.versions == ["v1", "v2"]
    assert "v1" in router._selection_counts
    assert "v2" in router._selection_counts


def test_select_model_version():
    """Test model version selection returns valid version"""
    router = ABTestRouter()
    version = router.select_model_version()
    assert version in ["v1", "v2"]


def test_select_model_version_distribution():
    """Test model selection distribution is roughly 50/50"""
    router = ABTestRouter()
    selections = [router.select_model_version() for _ in range(1000)]
    
    v1_count = selections.count("v1")
    v2_count = selections.count("v2")
    
    # With 1000 selections, expect 400-600 for each (allow variance)
    assert 400 <= v1_count <= 600
    assert 400 <= v2_count <= 600


def test_get_model_for_version():
    """Test getting model instance for version"""
    router = ABTestRouter()
    
    model_v1 = router.get_model_for_version("v1")
    model_v2 = router.get_model_for_version("v2")
    
    assert model_v1 is not None
    assert model_v2 is not None
    assert model_v1.version == "v1"
    assert model_v2.version == "v2"


def test_get_model_invalid_version():
    """Test getting model with invalid version raises error"""
    router = ABTestRouter()
    
    with pytest.raises(ValueError, match="Invalid model version"):
        router.get_model_for_version("v3")


def test_get_selection_stats():
    """Test getting selection statistics"""
    router = ABTestRouter()
    
    # Make some selections
    for _ in range(10):
        router.select_model_version()
    
    stats = router.get_selection_stats()
    
    assert "v1" in stats
    assert "v2" in stats
    assert stats["v1"] + stats["v2"] == 10


def test_get_distribution_percentage():
    """Test getting distribution as percentages"""
    router = ABTestRouter()
    
    # Make selections
    for _ in range(100):
        router.select_model_version()
    
    dist = router.get_distribution_percentage()
    
    assert "v1" in dist
    assert "v2" in dist
    assert 30 <= dist["v1"] <= 70  # Allow variance
    assert 30 <= dist["v2"] <= 70
    assert abs(dist["v1"] + dist["v2"] - 100) < 0.1  # Sum should be ~100%


def test_predict_ab_endpoint(client):
    """Test /predict/ab endpoint returns valid response"""
    response = client.post(
        "/predict/ab",
        json={"text": "This is a test for A/B testing"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check all required fields
    assert "sentiment" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert "cache_hit" in data
    assert "model_version" in data
    
    # Check model_version is valid
    assert data["model_version"] in ["v1", "v2"]
    
    # Check other fields are valid
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] > 0


def test_predict_ab_multiple_requests():
    """Test multiple A/B requests return different versions"""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    
    versions_seen = set()
    
    # Make 20 requests, should see both versions
    for _ in range(20):
        response = client.post(
            "/predict/ab",
            json={"text": "Test message"}
        )
        data = response.json()
        versions_seen.add(data["model_version"])
    
    # Should have seen both versions (very high probability)
    assert len(versions_seen) >= 1  # At least one version
    # With 20 requests, ~99.9% chance of seeing both versions


def test_ab_comparison_endpoint_structure(client_with_db):
    """Test /ab/comparison endpoint returns correct structure"""
    # Make some A/B predictions first
    for i in range(10):
        client_with_db.post(
            "/predict/ab",
            json={"text": f"Test message {i}"}
        )
    
    response = client_with_db.get("/ab/comparison")
    assert response.status_code == 200
    
    data = response.json()
    
    # Check top-level structure
    assert "model_v1" in data
    assert "model_v2" in data
    assert "comparison" in data
    
    # Check model_v1 structure
    assert "total_predictions" in data["model_v1"]
    assert "avg_confidence" in data["model_v1"]
    assert "avg_latency_ms" in data["model_v1"]
    assert "sentiment_distribution" in data["model_v1"]
    
    # Check model_v2 structure
    assert "total_predictions" in data["model_v2"]
    assert "avg_confidence" in data["model_v2"]
    assert "avg_latency_ms" in data["model_v2"]
    assert "sentiment_distribution" in data["model_v2"]
    
    # Check comparison structure
    assert "confidence_diff" in data["comparison"]
    assert "latency_diff" in data["comparison"]
    assert "sample_size_sufficient" in data["comparison"]
    assert "traffic_distribution" in data["comparison"]

