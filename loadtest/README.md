# Load Testing with Locust

This directory contains load testing scenarios for the Production ML API using Locust.

## Prerequisites

```bash
pip install locust
```

Or use the main requirements.txt which includes locust.

## Running Load Tests

### 1. Start the API

First, ensure the API and all services are running:

```bash
docker-compose up -d
```

Wait for services to be healthy (~60 seconds).

### 2. Run Locust Tests

#### Web UI Mode (Interactive)

```bash
locust -f loadtest/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser and configure:
- Number of users
- Spawn rate
- Host (already set)

#### Headless Mode (Automated)

**Baseline Test (100 users, 10 min):**
```bash
locust -f loadtest/locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 10m --headless \
  --html loadtest/baseline-report.html
```

**Spike Test (500 users, 5 min):**
```bash
locust -f loadtest/locustfile.py --host=http://localhost:8000 \
  --users 500 --spawn-rate 50 --run-time 5m --headless \
  --html loadtest/spike-report.html
```

**Cache Effectiveness Test:**
```bash
locust -f loadtest/locustfile.py --host=http://localhost:8000 \
  --users 50 --spawn-rate 10 --run-time 5m --headless \
  --html loadtest/cache-report.html \
  --user-classes CacheTestUser
```

## Test Scenarios

### MLAPIUser (Default)
Simulates normal API usage with:
- 70% `/predict` requests (mix of cached and unique texts)
- 20% `/predict/ab` requests (A/B testing)
- 10% `/batch` requests (batch predictions)
- 5% `/analytics/summary` requests (read-only)
- 5% `/health` requests

### CacheTestUser
Specialized for testing cache effectiveness:
- 100% repeated texts to maximize cache hits
- Useful for measuring cache performance

## Monitoring During Tests

### Docker Stats
Monitor container resource usage:
```bash
docker stats
```

### Prometheus Metrics
View real-time metrics:
```
http://localhost:9090
```

### Grafana Dashboard
Visualize performance:
```
http://localhost:3000
```

## Expected Performance

Based on Phase 5 benchmarks:
- **Throughput:** 100+ RPS sustained
- **Latency (p50):** 60-100ms
- **Latency (p95):** <150ms
- **Latency (p99):** <200ms
- **Cache hit rate:** 50-70% (with mixed traffic)
- **Error rate:** <1%

## Interpreting Results

### Good Results
- ✅ RPS ≥ 100
- ✅ p95 latency < 150ms
- ✅ Error rate < 1%
- ✅ No container crashes

### Warning Signs
- ⚠️ RPS dropping over time (memory leak?)
- ⚠️ p99 latency > 500ms (outliers)
- ⚠️ Error rate 1-5% (approaching limits)

### Critical Issues
- ❌ Container crashes during test
- ❌ Error rate > 5%
- ❌ Latency increasing steadily (resource exhaustion)

## Troubleshooting

### Rate Limiting
If you see many 429 errors:
- Reduce number of users
- Increase spawn rate to spread requests
- Or temporarily increase rate limits for testing

### High Latency
- Check `docker stats` for CPU/memory usage
- Reduce concurrent users
- Check if cache is working (should see mix of fast/slow requests)

### Crashes
- Check `docker logs mlapi_api`
- May need to increase container memory limits
- Reduce load intensity

## Advanced Usage

### Custom Scenarios
Edit `locustfile.py` to add custom test scenarios or adjust task weights.

### CSV Output
```bash
locust ... --csv=loadtest/results
```

### Multiple Workers (Distributed)
```bash
# Master
locust -f loadtest/locustfile.py --master

# Worker
locust -f loadtest/locustfile.py --worker --master-host=localhost
```

