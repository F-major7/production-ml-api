"""
Tests for health check endpoint
"""
import pytest


def test_health_endpoint_returns_200(client):
    """Test health endpoint returns 200 status code"""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client):
    """Test health response matches expected schema"""
    response = client.get("/health")
    data = response.json()
    
    # Check required fields exist
    assert "status" in data
    assert "model_loaded" in data
    
    # Check field types
    assert isinstance(data["status"], str)
    assert isinstance(data["model_loaded"], bool)


def test_health_model_loaded(client):
    """Test that model_loaded is True when healthy"""
    response = client.get("/health")
    data = response.json()
    
    # Model should be loaded on startup
    assert data["model_loaded"] is True


def test_health_status_healthy(client):
    """Test that status is 'healthy' when model is loaded"""
    response = client.get("/health")
    data = response.json()
    
    if data["model_loaded"]:
        assert data["status"] == "healthy"


def test_health_endpoint_no_parameters(client):
    """Test health endpoint accepts no parameters"""
    response = client.get("/health")
    assert response.status_code == 200
    
    # Ensure it's not affected by query parameters
    response_with_params = client.get("/health?test=1")
    assert response_with_params.status_code == 200

