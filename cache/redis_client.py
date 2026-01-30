"""
Redis cache client for caching predictions
Implements singleton pattern for connection pooling
"""
import os
import hashlib
import logging
from typing import Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Async Redis cache client with connection pooling.
    Singleton pattern ensures single connection pool across application.
    """
    _instance: Optional['RedisCache'] = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_client()
        return cls._instance
    
    def _initialize_client(self) -> None:
        """Initialize Redis client with connection pooling"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            
            self._client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10
            )
            
            logger.info(f"Redis client initialized: {redis_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self._client = None
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or error
        """
        if self._client is None:
            return None
        
        try:
            value = await self._client.get(key)
            return value
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if self._client is None:
            return False
        
        try:
            await self._client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if self._client is None:
            return False
        
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        if self._client is None:
            return False
        
        try:
            result = await self._client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def get_cache_size(self) -> int:
        """
        Get total number of keys in cache.
        
        Returns:
            Number of keys or 0 on error
        """
        if self._client is None:
            return 0
        
        try:
            size = await self._client.dbsize()
            return size
        except Exception as e:
            logger.error(f"Redis DBSIZE error: {e}")
            return 0
    
    @staticmethod
    def generate_cache_key(text: str) -> str:
        """
        Generate deterministic cache key from text using SHA-256.
        
        Args:
            text: Input text
            
        Returns:
            Cache key string with "sentiment:" prefix
        """
        # Normalize text (strip whitespace, lowercase for consistency)
        normalized = text.strip().lower()
        
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Return with prefix
        return f"sentiment:{hash_hex}"
    
    @property
    def is_available(self) -> bool:
        """Check if Redis client is available"""
        return self._client is not None
    
    async def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if Redis is reachable, False otherwise
        """
        if self._client is None:
            return False
        
        try:
            await self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis PING error: {e}")
            return False

