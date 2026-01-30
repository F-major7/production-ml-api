"""
Pytest configuration and fixtures
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_positive_text():
    """Sample positive text for testing"""
    return "I love this product!"


@pytest.fixture
def sample_negative_text():
    """Sample negative text for testing"""
    return "This is terrible and awful"


@pytest.fixture
def sample_long_text():
    """Sample text exceeding max length"""
    return "a" * 5001


@pytest.fixture
def sample_whitespace_text():
    """Sample whitespace-only text"""
    return "   \n\t   "

