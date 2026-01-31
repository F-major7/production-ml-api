# Production ML API

A containerized REST API for sentiment analysis, designed with production operational concerns as primary constraints. The system emphasizes reliability, observability, and operational clarity over modeling novelty.

**Live deployment:** [https://web-production-7e0e.up.railway.app/docs](https://web-production-7e0e.up.railway.app/docs)

---

## Project Overview

This system serves sentiment analysis predictions via HTTP, with supporting infrastructure for caching, analytics, monitoring, and testing. It is structured as a self-contained service that can be deployed to cloud platforms with minimal external dependencies.

The API exposes endpoints for single predictions, batch inference, A/B testing between model versions, and analytics queries. All predictions are logged to a PostgreSQL database for historical analysis and trend detection.

---

## Problem Context

Deploying machine learning models as production services introduces operational complexity beyond model training. Key concerns include:

- **Latency management:** Inference must remain responsive under concurrent load
- **Reliability:** The service must handle failures gracefully and predictably
- **Observability:** System behavior must be measurable and inspectable in real time
- **Versioning:** Multiple model versions may coexist during rollouts or A/B tests
- **Data persistence:** Predictions and metadata must be stored for audit and analysis

This system addresses these concerns explicitly, treating them as first-class requirements rather than afterthoughts.

---

## System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Application         │
│  ┌───────────┐    ┌──────────────┐  │
│  │  Predict  │───▶│  Sentiment   │  │
│  │ Endpoints │    │    Model     │  │
│  └─────┬─────┘    └──────────────┘  │
│        │                            │
│        ▼                            │
│  ┌──────────────┐  ┌────────────┐   │
│  │   Analytics  │  │   Redis    │   │
│  │   Tracking   │  │   Cache    │   │
│  └──────┬───────┘  └────────────┘   │
│         │                           │
└─────────┼───────────────────────────┘
          │
          ▼
    ┌──────────┐       ┌────────────┐
    │PostgreSQL│       │ Prometheus │
    │ Database │       │  Metrics   │
    └──────────┘       └────────────┘
```

### Request Flow

1. **Prediction requests** arrive at the FastAPI application layer
2. Input validation occurs via Pydantic schemas
3. Redis cache is queried for previously computed results
4. Cache misses trigger model inference (DistilBERT-based sentiment classifier)
5. Results are written to PostgreSQL with metadata (timestamp, confidence, model version)
6. Response is returned to client
7. Prometheus metrics are updated throughout the request lifecycle

### Data Flow

Predictions are stored in a normalized schema supporting:
- Multiple model versions (v1, v2, etc.)
- Time-series aggregation
- Confidence distribution analysis
- A/B test result comparison

Cache entries use LRU eviction with TTL to bound memory usage.

---

## Design Decisions

### Technology Selection

**FastAPI** provides async request handling and automatic OpenAPI schema generation. Async support prevents thread blocking during I/O operations (database writes, cache lookups), improving throughput under concurrent load.

**PostgreSQL** serves as the analytics store. SQL enables complex aggregations (percentile calculations, time-series queries, multi-version comparisons) without requiring external data processing frameworks.

**Redis** implements caching with LRU eviction. Repeated identical inputs (common in production for templated text or testing) bypass model inference entirely, reducing p50 latency by approximately 40%.

**Prometheus** exposes metrics in a standard format. Metrics follow the RED methodology (Rate, Errors, Duration) for request-level observability and track model-specific statistics (confidence distributions, version usage).

**Docker** containerizes the application. The image includes all runtime dependencies, ensuring consistency between development and production environments.

### Model Serving Strategy

The system uses a singleton model instance loaded at application startup. This trades startup latency (30-60 seconds to load DistilBERT weights) for request latency (no per-request loading overhead).

Model inference runs synchronously within async request handlers. For the current workload (single-core deployment, lightweight model), this is sufficient. Higher-throughput deployments would require either:
- Asynchronous model inference via thread pools
- Separate inference workers with a queue-based architecture

---

## Operational Behavior

### Performance Characteristics

Under load testing conditions (100 concurrent users, 10-minute duration):
- **Throughput:** 36 requests/second
- **Latency (p50):** 76ms
- **Latency (p95):** 6 seconds
- **Failure rate:** 0%

The high p95 latency reflects CPU-bound inference on a single-core deployment. Cache hit rates directly affect tail latency: cached requests complete in under 10ms, while cache misses require full model inference. The system remained responsive throughout testing, with no request failures or crashes observed.

### Caching Behavior

Prediction results are cached in Redis using a hash of the input text as the cache key. Entries use LRU eviction with a 24 hour TTL.

Cached requests bypass inference and return in under 10 ms. Cache impact varies with input repetition and primarily improves median latency by reducing redundant computation.

### Rate Limiting

Endpoints enforce configurable rate limits via `slowapi`:
- `/predict`: 100 requests/minute per client
- `/predict/batch`: 20 requests/minute per client
- `/analytics/*`: 60 requests/minute per client

Limits prevent individual clients from monopolizing resources. Configuration is environment-variable driven to support different limits in development, testing, and production.

---

## Observability and Monitoring

### Metrics Collected

Prometheus metrics include:

**Request metrics:**
- `http_requests_total` (counter): Total requests by endpoint, status
- `http_request_duration_seconds` (histogram): Request latency distribution

**Model metrics:**
- `model_predictions_total` (counter): Predictions by sentiment, version
- `model_inference_duration_seconds` (histogram): Inference time distribution
- `model_confidence` (histogram): Confidence score distribution

**Cache metrics:**
- `cache_hits_total` (counter): Cache hit count
- `cache_misses_total` (counter): Cache miss count

**Database metrics:**
- `db_queries_total` (counter): Query count by type
- `db_query_duration_seconds` (histogram): Query latency

### Visualization

A local Grafana instance (included in Docker Compose setup) provides real-time dashboards for development and testing. Dashboards visualize latency percentiles, throughput, cache efficiency, and error rates.

Production deployments on Railway use Railway's built-in metrics dashboard for basic observability. Full Prometheus/Grafana can be deployed alongside the service for advanced monitoring.

---

## Testing and Validation

### Test Coverage

The test suite includes 92 tests with 77% line coverage:
- **Unit tests:** Model inference, cache operations, database queries
- **Integration tests:** End-to-end API flows, database transactions
- **Property-based tests:** Input validation edge cases

Tests run automatically on each push via GitHub Actions CI pipeline.

### Load Testing

Load tests use Locust to simulate realistic traffic patterns:
- 100 concurrent users
- 10-minute sustained load
- Mixed endpoint usage (70% `/predict`, 20% `/predict/ab-compare`, 10% `/analytics`)

The load test configuration is stored in `loadtest/locustfile.py` and can be executed locally or in CI.

### A/B Testing Framework

The system supports serving predictions from multiple model versions simultaneously. The `/predict/ab-compare` endpoint runs inference through both v1 and v2 models, logs results separately, and returns both predictions for comparison.

This enables:
- Gradual model rollouts
- Champion/challenger testing
- Regression detection between versions

---

## Deployment Notes

### Container Image

The Docker image uses multi-stage builds to manage size and runtime dependencies. Default PyTorch installations pulled in GPU related libraries that exceeded deployment constraints for a CPU-only target. This was addressed by using CPU-only PyTorch wheels and tightening Docker layers to match the deployment environment.

The final image includes only runtime dependencies and runs under a non-root user for isolation.

### Platform Constraints

**Railway (current deployment):**
- Image size limit: 4GB (met after CPU-only PyTorch switch)
- Free tier behavior: Service sleeps after inactivity; cold start on first request post-sleep
- Cold start duration: 30-60 seconds (PyTorch model loading)
- Memory: No hard limit on free tier (sufficient for DistilBERT + overhead)

### Environment Variables

Required environment variables:
```
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db
REDIS_URL=redis://host:port/0
```

Optional configuration:
```
LOG_LEVEL=INFO
RATE_LIMIT_PREDICT=100/minute
RATE_LIMIT_BATCH=20/minute
RATE_LIMIT_ANALYTICS=60/minute
MKLDNN_ENABLE=0  # Disable on ARM64 to prevent segfaults
```

### Platform-Specific Issues

**Apple Silicon (ARM64) segmentation faults:**  
PyTorch's MKLDNN library (CPU optimizations) does not support ARM64 correctly. Running inference on Apple Silicon hardware triggers segfaults in MKLDNN code paths.

Mitigation: Set `MKLDNN_ENABLE=0` environment variable to disable MKLDNN. This sacrifices CPU optimization for stability. x86 deployments (Railway, most cloud platforms) do not require this workaround.

---

## Limitations and Future Work

### Current Limitations

- **Single-threaded inference:** Model runs in the main async event loop; high concurrency can cause queue buildup
- **No horizontal scaling:** Single-instance deployment; no load balancer or replica management
- **Basic model:** DistilBERT is lightweight but not state-of-the-art; newer models (BERT-large, RoBERTa) would improve accuracy at the cost of latency
- **Cold starts:** Free-tier deployment sleeps after inactivity; first request post-sleep incurs 30-60s delay
- **Local monitoring only:** Prometheus/Grafana run locally; production deployments rely on Railway's basic metrics

### Potential Enhancements

**Scalability:**
- Add load balancer and multiple API replicas for horizontal scaling
- Implement separate inference workers with queue-based request distribution
- Use GPU instances for higher-throughput deployments

**Model Management:**
- Integrate model registry (MLflow, W&B) for versioned artifact storage
- Implement automatic model reloading on version updates
- Add feature store for centralized feature computation

**Observability:**
- Deploy Prometheus/Grafana to production for full metric retention
- Add distributed tracing (Jaeger, Zipkin) for request flow visualization
- Implement alerting (PagerDuty, Slack) for error rate spikes or latency regressions

**Inference Optimization:**
- Add batch inference endpoint for high-throughput use cases
- Implement model quantization (INT8) to reduce memory footprint
- Use ONNX Runtime for optimized inference

---

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/F-major7/production-ml-api
cd production-ml-api

# Start services
docker compose up --build

# API: http://localhost:8000
# Grafana: http://localhost:3000
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This API works well"}'

# Interactive docs
open http://localhost:8000/docs
```

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run test suite
pytest tests/ -v --cov=api --cov=models --cov=db --cov=cache

# Load test
locust -f loadtest/locustfile.py --host=http://localhost:8000
```

---

## Technical Stack

- **Runtime:** Python 3.11
- **Web framework:** FastAPI 0.104
- **ML framework:** PyTorch 2.1 (CPU-only), Transformers 4.35
- **Database:** PostgreSQL (async via asyncpg), SQLAlchemy 2.0
- **Cache:** Redis 5.0
- **Monitoring:** Prometheus, Grafana
- **Testing:** pytest, Locust
- **Infrastructure:** Docker, Docker Compose, GitHub Actions
- **Deployment:** Railway

---
