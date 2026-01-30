"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import os
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool
from datetime import datetime
import uuid
from unittest.mock import patch
import fakeredis.aioredis

from api.main import app
from db.models import Base, Prediction
from db.database import get_db
from cache.redis_client import RedisCache
from api.dependencies import get_redis

# Set test environment variables
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["TESTING"] = "1"  # Disable rate limiting for all tests

# Test database URL (using SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session", autouse=True)
def set_testing_env():
    """
    Automatically set TESTING=1 environment variable for all tests.
    This disables rate limiting to prevent 429 errors during test execution.
    """
    os.environ["TESTING"] = "1"
    yield
    # Cleanup after all tests
    os.environ.pop("TESTING", None)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def async_db_session():
    """Create a test database session"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    TestSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    async with TestSessionLocal() as session:
        yield session
        await session.rollback()

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def test_db(async_db_session):
    """Synchronous wrapper for async database session"""
    return async_db_session


@pytest.fixture
def client_with_db(redis_cache):
    """Test client with database dependency override (using in-memory SQLite)"""
    # For testing, we'll use a simpler approach - just skip DB operations
    # In a real production environment, you'd set up a test PostgreSQL database

    # Override to return None - endpoints handle this gracefully
    async def override_get_db():
        # Create in-memory SQLite for testing
        from sqlalchemy.ext.asyncio import (
            create_async_engine,
            async_sessionmaker,
            AsyncSession,
        )
        from sqlalchemy.pool import StaticPool

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create session
        TestSessionLocal = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

        await engine.dispose()

    # Don't override get_redis - use real Redis

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def redis_cache():
    """Get the real Redis cache instance (requires Docker Redis to be running)"""
    # Reset singleton to get fresh connection
    RedisCache._instance = None
    cache = RedisCache()
    yield cache
    # Reset for next test
    RedisCache._instance = None


@pytest.fixture
def client(redis_cache):
    """Test client for FastAPI app with Redis"""

    # Override get_db to return None (gracefully handle missing DB)
    async def override_get_db():
        yield None

    # Don't override get_redis - let it use the real Redis
    # (Tests will use the actual Docker Redis instance)

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


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


@pytest.fixture
async def sample_predictions(async_db_session):
    """Create sample predictions in test database"""
    predictions = [
        Prediction(
            id=uuid.uuid4(),
            input_text="I love this!",
            predicted_sentiment="positive",
            confidence_score=0.9998,
            latency_ms=45.23,
            model_version="distilbert-v1",
            cache_hit=False,
            timestamp=datetime(2026, 1, 29, 10, 0, 0),
        ),
        Prediction(
            id=uuid.uuid4(),
            input_text="This is terrible",
            predicted_sentiment="negative",
            confidence_score=0.9995,
            latency_ms=38.17,
            model_version="distilbert-v1",
            cache_hit=False,
            timestamp=datetime(2026, 1, 29, 11, 0, 0),
        ),
        Prediction(
            id=uuid.uuid4(),
            input_text="It's okay",
            predicted_sentiment="neutral",
            confidence_score=0.7234,
            latency_ms=42.50,
            model_version="distilbert-v1",
            cache_hit=False,
            timestamp=datetime(2026, 1, 29, 12, 0, 0),
        ),
    ]

    async_db_session.add_all(predictions)
    await async_db_session.commit()

    for p in predictions:
        await async_db_session.refresh(p)

    return predictions
