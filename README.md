# Production ML API

A production-grade sentiment analysis API built with FastAPI and HuggingFace Transformers, demonstrating best practices for deploying ML models at scale.

## Overview

This project showcases how to build a robust, observable ML API with:
- **FastAPI** for high-performance API serving
- **DistilBERT** sentiment analysis model
- **Comprehensive testing** with pytest
- **Request/response validation** with Pydantic
- **Error handling** and health checks
- **Production-ready architecture** with dependency injection

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-api
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Server

Start the development server:
```bash
uvicorn api.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive API docs (Swagger)**: http://localhost:8000/docs
- **Alternative API docs (ReDoc)**: http://localhost:8000/redoc

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

Run the test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ -v --cov=api --cov=models --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # On macOS
# Or on Linux: xdg-open htmlcov/index.html
```

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

## Next Phases

This is **Phase 1** of a 7-phase project. Future phases will add:

- **Phase 2**: PostgreSQL for prediction logging
- **Phase 3**: Redis caching layer
- **Phase 4**: Prometheus/Grafana monitoring
- **Phase 5**: Docker containerization
- **Phase 6**: CI/CD pipeline
- **Phase 7**: Cloud deployment (AWS/GCP)

## License

MIT License

## Contributing

This is a portfolio project. Feedback and suggestions are welcome!

## Contact

For questions or feedback, please open an issue in the repository.

