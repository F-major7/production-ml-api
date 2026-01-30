"""
Tests for prediction endpoint
"""
import pytest


def test_predict_positive_sentiment(client, sample_positive_text):
    """Test prediction with positive text returns positive sentiment"""
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"


def test_predict_negative_sentiment(client, sample_negative_text):
    """Test prediction with negative text returns negative sentiment"""
    response = client.post(
        "/predict",
        json={"text": sample_negative_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"


def test_predict_empty_text_returns_422(client):
    """Test prediction with empty text returns 422 validation error"""
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    
    assert response.status_code == 422


def test_predict_long_text_returns_422(client, sample_long_text):
    """Test prediction with text exceeding max length returns 422"""
    response = client.post(
        "/predict",
        json={"text": sample_long_text}
    )
    
    assert response.status_code == 422


def test_predict_whitespace_only_returns_422(client, sample_whitespace_text):
    """Test prediction with whitespace-only text returns 422"""
    response = client.post(
        "/predict",
        json={"text": sample_whitespace_text}
    )
    
    assert response.status_code == 422


def test_predict_response_has_required_fields(client, sample_positive_text):
    """Test prediction response contains all required fields"""
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "sentiment" in data
    assert "confidence" in data
    assert "latency_ms" in data


def test_predict_confidence_in_valid_range(client, sample_positive_text):
    """Test confidence score is between 0 and 1"""
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    confidence = data["confidence"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_predict_latency_is_positive(client, sample_positive_text):
    """Test latency_ms is a positive number"""
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    latency = data["latency_ms"]
    assert isinstance(latency, float)
    assert latency > 0


def test_predict_missing_text_field_returns_422(client):
    """Test prediction without text field returns 422"""
    response = client.post(
        "/predict",
        json={}
    )
    
    assert response.status_code == 422


def test_predict_invalid_json_returns_422(client):
    """Test prediction with invalid JSON returns 422"""
    response = client.post(
        "/predict",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422


def test_predict_sentiment_values(client, sample_positive_text, sample_negative_text):
    """Test sentiment is one of expected values"""
    valid_sentiments = {"positive", "negative", "neutral"}
    
    # Test positive
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    assert response.json()["sentiment"] in valid_sentiments
    
    # Test negative
    response = client.post(
        "/predict",
        json={"text": sample_negative_text}
    )
    assert response.json()["sentiment"] in valid_sentiments


def test_predict_confidence_precision(client, sample_positive_text):
    """Test confidence is rounded to 4 decimal places"""
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Convert to string and check decimal places
    confidence_str = str(data["confidence"])
    if "." in confidence_str:
        decimal_places = len(confidence_str.split(".")[1])
        assert decimal_places <= 4


def test_predict_latency_precision(client, sample_positive_text):
    """Test latency_ms is rounded to 2 decimal places"""
    response = client.post(
        "/predict",
        json={"text": sample_positive_text}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Convert to string and check decimal places
    latency_str = str(data["latency_ms"])
    if "." in latency_str:
        decimal_places = len(latency_str.split(".")[1])
        assert decimal_places <= 2


def test_predict_multiple_requests(client, sample_positive_text):
    """Test multiple prediction requests work correctly"""
    for _ in range(3):
        response = client.post(
            "/predict",
            json={"text": sample_positive_text}
        )
        assert response.status_code == 200
        assert "sentiment" in response.json()

