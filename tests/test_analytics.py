"""
Tests for analytics endpoints
"""

import pytest


def test_analytics_summary_empty_db(client_with_db):
    """Test analytics summary with empty database"""
    response = client_with_db.get("/analytics/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 0
    assert data["avg_confidence"] == 0.0
    assert data["avg_latency_ms"] == 0.0


def test_sentiment_distribution_empty_db(client_with_db):
    """Test sentiment distribution with empty database"""
    response = client_with_db.get("/analytics/sentiment-distribution")

    assert response.status_code == 200
    data = response.json()
    assert data["positive"] == 0
    assert data["negative"] == 0
    assert data["neutral"] == 0


def test_recent_predictions_empty_db(client_with_db):
    """Test recent predictions with empty database"""
    response = client_with_db.get("/analytics/recent")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_recent_predictions_invalid_limit(client_with_db):
    """Test recent predictions with invalid limit"""
    # Limit too high
    response = client_with_db.get("/analytics/recent?limit=2000")
    assert response.status_code == 422

    # Limit too low
    response = client_with_db.get("/analytics/recent?limit=0")
    assert response.status_code == 422
