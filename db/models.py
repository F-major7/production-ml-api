"""
SQLAlchemy ORM models for database tables
"""
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Boolean, DateTime, Text, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID


class Base(DeclarativeBase):
    """Base class for all ORM models"""
    pass


class Prediction(Base):
    """
    ORM model for predictions table.
    Stores all prediction requests and results for analytics and debugging.
    """
    __tablename__ = "predictions"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Timestamp (indexed for time-range queries)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
        nullable=False
    )
    
    # Input and prediction
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_sentiment: Mapped[str] = mapped_column(
        String(20),
        index=True,
        nullable=False
    )
    
    # Metrics
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Metadata
    model_version: Mapped[str] = mapped_column(
        String(50),
        default="distilbert-v1",
        nullable=False
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # Create composite indexes for common queries
    __table_args__ = (
        Index('ix_predictions_timestamp_sentiment', 'timestamp', 'predicted_sentiment'),
    )
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"<Prediction(id={self.id}, "
            f"sentiment={self.predicted_sentiment}, "
            f"confidence={self.confidence_score:.4f}, "
            f"timestamp={self.timestamp})>"
        )

