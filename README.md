# Production ML API

![CI/CD](https://github.com/YOUR_USERNAME/production-ml-api/workflows/CI/CD%20Pipeline/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)

A production-grade sentiment analysis API built with FastAPI and HuggingFace Transformers, demonstrating best practices for deploying ML models at scale.

## Overview

This project showcases how to build a robust, observable ML API with:
- **FastAPI** for high-performance API serving (60-100ms latency)
- **DistilBERT** sentiment analysis model (PyTorch CPU-optimized)
- **PostgreSQL** for prediction logging and analytics
- **Redis** for intelligent caching (70%+ hit rate)
- **Prometheus + Grafana** for monitoring and dashboards
- **Rate limiting** with slowapi (100 req/min)
- **Docker Compose** for full stack deployment
- **CI/CD pipeline** with GitHub Actions
- **Comprehensive testing** (85% coverage, load tested to 100 RPS)

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### Running with Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd production-ml-api
```

2. Start all services:
```bash
docker-compose up -d
```

3. Wait ~60 seconds for services to be healthy, then access:
- **API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Local Development (Without Docker)

1. Start services (PostgreSQL, Redis):
```bash
docker-compose up -d postgres redis
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
export DATABASE_URL="postgresql+asyncpg://mlapi_user:mlapi_password@localhost:5432/mlapi"
export REDIS_URL="redis://localhost:6379/0"
```

5. Run the API:
```bash
uvicorn api.main:app --reload
```

## API Endpoints

### Root Endpoint
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Production ML API",
  "docs": "/docs"
}
```

### Health Check
Check if the API and model are ready.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Sentiment
Analyze the sentiment of text.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product! It works amazingly well."}'
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.9998,
  "latency_ms": 45.23
}
```

Example with negative sentiment:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is terrible and disappointing."}'
```

Response:
```json
{
  "sentiment": "negative",
  "confidence": 0.9995,
  "latency_ms": 38.17
}
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=api --cov=models --cov=db --cov=cache --cov=monitoring \
  --cov-report=term-missing --cov-report=html --cov-fail-under=80
```

### View Coverage Report
```bash
open htmlcov/index.html  # macOS
# Linux: xdg-open htmlcov/index.html
```

### Run Specific Test Files
```bash
pytest tests/test_integration.py -v  # Integration tests
pytest tests/test_cache.py -v        # Cache tests
pytest tests/test_rate_limit.py -v   # Rate limiting tests
```

### Load Testing
```bash
# Start services first
docker-compose up -d

# Run load test (100 users, 10 minutes)
locust -f loadtest/locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 10m --headless \
  --html loadtest/report.html

# View results
open loadtest/report.html
```

See [docs/test-coverage.md](docs/test-coverage.md) for detailed coverage information.  
See [loadtest/README.md](loadtest/README.md) for load testing guide.

## Project Structure

```
production-ml-api/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic models
│   └── dependencies.py   # Dependency injection
├── models/
│   ├── __init__.py
│   └── sentiment.py      # ML model wrapper
├── tests/
│   ├── __init__.py
│   ├── conftest.py       # Test fixtures
│   ├── test_health.py    # Health endpoint tests
│   └── test_predict.py   # Prediction endpoint tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **HuggingFace Transformers**: State-of-the-art NLP models
- **DistilBERT**: Efficient transformer model for sentiment analysis
- **Pydantic**: Data validation using Python type hints
- **Pytest**: Testing framework with fixtures and coverage
- **Uvicorn**: ASGI server for production deployment

## Features

### Model Architecture
- **Singleton pattern** for efficient model loading
- **DistilBERT** fine-tuned on SST-2 dataset
- Supports positive and negative sentiment classification

### API Features
- **Request validation**: Automatic validation with detailed error messages
- **Response models**: Strongly-typed responses with Pydantic
- **Error handling**: Comprehensive error handling with specific error types
- **Health checks**: Monitor API and model status
- **CORS enabled**: For frontend integration
- **Interactive docs**: Auto-generated Swagger UI

### Request Validation
- Text length: 1-5000 characters
- Whitespace handling: Automatic trimming
- Empty text rejection: Returns validation error

### Performance Metrics
- **Latency tracking**: Each prediction includes inference time
- **Confidence scores**: Model confidence for each prediction
- **Efficient inference**: Singleton pattern prevents model reloading

## Development

### Running in Development Mode
```bash
uvicorn api.main:app --reload --log-level debug
```

### Environment Variables
Currently no environment variables required for Phase 1.

## API Validation Criteria

✅ Server starts without errors  
✅ GET /health returns 200 with valid JSON  
✅ POST /predict with valid text returns sentiment  
✅ POST /predict with empty text returns 422  
✅ All pytest tests pass  
✅ /docs shows Swagger UI with both endpoints  
✅ curl commands work from terminal  

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

### Automated Checks
- ✅ Code formatting (black)
- ✅ Linting (flake8)
- ✅ Type checking (mypy)
- ✅ Test suite (pytest)
- ✅ Coverage enforcement (>80%)
- ✅ Docker build validation

### Running CI Checks Locally
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Format check
black --check api/ models/ db/ cache/ monitoring/

# Lint
flake8 api/ models/ db/ cache/ monitoring/

# Type check
mypy api/ models/ --ignore-missing-imports

# Tests with coverage
pytest tests/ --cov-fail-under=80
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Throughput** | 100+ RPS sustained |
| **Latency (p50)** | 65ms |
| **Latency (p95)** | 120ms |
| **Latency (p99)** | 180ms |
| **Cache Hit Rate** | 72% (mixed traffic) |
| **Error Rate** | <1% |

See [docs/load-test-results.md](docs/load-test-results.md) for detailed performance analysis.

## Project Status

**Current Phase:** Phase 6 Complete - CI/CD and Load Testing

### Completed Phases
- ✅ **Phase 1**: FastAPI + DistilBERT sentiment analysis
- ✅ **Phase 2**: PostgreSQL logging and analytics
- ✅ **Phase 3**: Redis caching + Prometheus metrics
- ✅ **Phase 4**: Grafana dashboards + A/B testing
- ✅ **Phase 5**: Docker containerization (full stack)
- ✅ **Phase 6**: CI/CD pipeline + load testing (85% coverage)

### Next Phase
- 🔄 **Phase 7**: Production deployment planning (AWS/GCP/Azure)

## License

MIT License

## Contributing

This is a portfolio project. Feedback and suggestions are welcome!

## Contact

For questions or feedback, please open an issue in the repository.

