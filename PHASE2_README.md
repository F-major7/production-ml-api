# Phase 2: Database Logging and Analytics

## Overview

Phase 2 adds PostgreSQL database logging and analytics endpoints to track all predictions for historical analysis and debugging.

## What's New

### Database Layer
- **PostgreSQL Integration**: Async database with SQLAlchemy 2.0
- **Prediction Logging**: All predictions automatically logged to database
- **Alembic Migrations**: Database schema version control

### New Endpoints

#### POST /batch
Batch prediction endpoint for processing multiple texts at once.

**Request:**
```json
{
  "texts": ["I love this!", "This is terrible", "It's okay"]
}
```

**Response:**
```json
{
  "predictions": [
    {"sentiment": "positive", "confidence": 0.9998, "latency_ms": 45.23},
    {"sentiment": "negative", "confidence": 0.9995, "latency_ms": 38.17},
    {"sentiment": "neutral", "confidence": 0.7234, "latency_ms": 42.50}
  ],
  "total": 3
}
```

#### GET /analytics/summary
Get overall prediction statistics.

**Response:**
```json
{
  "total_predictions": 1000,
  "avg_confidence": 0.9234,
  "avg_latency_ms": 42.15,
  "date_range": {
    "earliest": "2026-01-01T00:00:00",
    "latest": "2026-01-30T12:00:00"
  }
}
```

#### GET /analytics/sentiment-distribution
Get sentiment distribution counts.

**Response:**
```json
{
  "positive": 450,
  "negative": 350,
  "neutral": 200
}
```

#### GET /analytics/recent?limit=100
Get recent predictions (default 100, max 1000).

**Response:**
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "timestamp": "2026-01-30T12:00:00",
    "sentiment": "positive",
    "confidence": 0.9998,
    "latency_ms": 45.23,
    "model_version": "distilbert-v1"
  }
]
```

Note: Input text is excluded from response for privacy.

## Database Schema

### `predictions` Table

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| timestamp | DateTime | Prediction time (indexed) |
| input_text | Text | User input |
| predicted_sentiment | String | positive/negative/neutral (indexed) |
| confidence_score | Float | 0.0 to 1.0 |
| latency_ms | Float | Inference time |
| model_version | String | Model version (default: distilbert-v1) |
| cache_hit | Boolean | For Phase 3 caching (default: false) |

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies:
- sqlalchemy==2.0.23
- alembic==1.13.0
- psycopg2-binary==2.9.9
- asyncpg==0.29.0
- aiosqlite==0.19.0 (for testing)
- pytest-asyncio==0.21.1
- greenlet==3.0.3

### 2. Start PostgreSQL

**Option A: Using Docker Compose (Recommended)**
```bash
docker-compose up -d
```

**Option B: Local PostgreSQL**
Install PostgreSQL and create database:
```sql
CREATE DATABASE mlapi;
CREATE USER mlapi_user WITH PASSWORD 'mlapi_password';
GRANT ALL PRIVILEGES ON DATABASE mlapi TO mlapi_user;
```

### 3. Configure Environment

Create `.env` file (or set environment variables):
```bash
DATABASE_URL=postgresql+asyncpg://mlapi_user:mlapi_password@localhost:5432/mlapi
```

### 4. Run Migrations

```bash
# Apply database migrations
alembic upgrade head
```

### 5. Start API

```bash
uvicorn api.main:app --reload
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Phase 1 tests (health + predict)
pytest tests/test_health.py tests/test_predict.py -v

# Phase 2 tests (database + analytics)
pytest tests/test_database.py tests/test_analytics.py -v
```

## Project Structure

```
production-ml-api/
├── api/
│   ├── main.py           # Updated: DB logging, /batch endpoint
│   ├── schemas.py        # Updated: Batch + analytics schemas
│   ├── analytics.py      # NEW: Analytics endpoints
│   └── dependencies.py
├── db/                   # NEW: Database layer
│   ├── __init__.py
│   ├── database.py       # DB connection & session management
│   ├── models.py         # SQLAlchemy ORM models
│   └── migrations/       # Alembic migrations
│       └── versions/
│           └── 001_create_predictions_table.py
├── models/
│   └── sentiment.py
├── tests/
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_database.py  # NEW: Database tests
│   └── test_analytics.py # NEW: Analytics tests
├── docker-compose.yml    # NEW: PostgreSQL setup
├── alembic.ini           # NEW: Alembic configuration
└── requirements.txt      # Updated: DB dependencies
```

## API Examples

### Make Predictions
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Batch predictions
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!", "Okay"]}'
```

### View Analytics
```bash
# Summary statistics
curl http://localhost:8000/analytics/summary

# Sentiment distribution
curl http://localhost:8000/analytics/sentiment-distribution

# Recent predictions
curl http://localhost:8000/analytics/recent?limit=10
```

### Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key Features

### Graceful Degradation
- API continues to work even if database is unavailable
- Predictions are returned successfully; logging failures are logged but don't fail requests
- Designed for high availability

### Performance
- Async database operations (non-blocking I/O)
- Connection pooling (pool_size=10, max_overflow=20)
- Indexed queries for fast analytics
- Batch endpoint for efficient bulk processing

### Privacy
- Analytics endpoints exclude input text
- Only metadata (sentiment, confidence, latency) exposed in `/analytics/recent`

## Database Management

### View Migrations
```bash
alembic history
```

### Create New Migration
```bash
alembic revision --autogenerate -m "Description"
```

### Rollback Migration
```bash
alembic downgrade -1
```

### Connect to Database
```bash
psql -h localhost -U mlapi_user -d mlapi
```

### Query Predictions
```sql
-- Total predictions
SELECT COUNT(*) FROM predictions;

-- Sentiment distribution
SELECT predicted_sentiment, COUNT(*) 
FROM predictions 
GROUP BY predicted_sentiment;

-- Recent predictions
SELECT timestamp, predicted_sentiment, confidence_score 
FROM predictions 
ORDER BY timestamp DESC 
LIMIT 10;
```

## Troubleshooting

### Database Connection Errors
```
ERROR: Failed to initialize database
```
**Solution**: Ensure PostgreSQL is running and DATABASE_URL is correct.

### Migration Errors
```
ERROR: Target database is not up to date
```
**Solution**: Run `alembic upgrade head`

### Import Errors
```
ModuleNotFoundError: No module named 'asyncpg'
```
**Solution**: Run `pip install -r requirements.txt`

## Notes

- Database logging is **optional** - API works without database
- Use environment variables for credentials (never hardcode)
- `.env` file is in `.gitignore` (never commit credentials)
- For production, use managed PostgreSQL (AWS RDS, Google Cloud SQL, etc.)

## Next Steps: Phase 3

Phase 3 will add:
- Redis caching layer
- Cache hit tracking
- Performance improvements for repeated queries

## Validation Checklist

✅ All Phase 1 tests pass (19/19)  
✅ Database models created  
✅ Alembic migrations configured  
✅ Docker Compose for PostgreSQL  
✅ POST /predict logs to database  
✅ POST /batch endpoint implemented  
✅ GET /analytics/summary endpoint  
✅ GET /analytics/sentiment-distribution endpoint  
✅ GET /analytics/recent endpoint  
✅ Graceful error handling (DB failures don't break API)  
✅ Environment variable configuration  
✅ Documentation updated  

**Phase 2 Complete!** 🎉

