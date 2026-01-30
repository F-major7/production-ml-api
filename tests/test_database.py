"""
Tests for database operations and batch endpoint
"""

import pytest


def test_batch_endpoint_with_db(client_with_db):
    """Test batch prediction endpoint logs to database"""
    response = client_with_db.post(
        "/batch", json={"texts": ["Great!", "Terrible!", "Okay"]}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["predictions"]) == 3

    # Verify each prediction has required fields
    for pred in data["predictions"]:
        assert "sentiment" in pred
        assert "confidence" in pred
        assert "latency_ms" in pred
        assert pred["sentiment"] in ["positive", "negative", "neutral"]
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["latency_ms"] > 0


def test_batch_endpoint_empty_list(client_with_db):
    """Test batch endpoint with empty list"""
    response = client_with_db.post("/batch", json={"texts": []})

    assert response.status_code == 422  # Validation error


def test_batch_endpoint_too_many_texts(client_with_db):
    """Test batch endpoint with too many texts"""
    response = client_with_db.post(
        "/batch", json={"texts": ["text"] * 101}  # Max is 100
    )

    assert response.status_code == 422  # Validation error


def test_batch_endpoint_invalid_text(client_with_db):
    """Test batch endpoint with invalid text"""
    response = client_with_db.post(
        "/batch",
        json={"texts": ["valid text", "   ", "another valid"]},  # Whitespace only
    )

    assert response.status_code == 422  # Validation error
