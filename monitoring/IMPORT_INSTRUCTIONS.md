# Grafana Dashboard Import Instructions

## Prerequisites
- Docker services running: `docker compose up -d`
- Grafana accessible at: http://localhost:3000
- Prometheus datasource auto-configured

## Import Steps

### 1. Access Grafana
```
URL: http://localhost:3000
Username: admin
Password: admin
```

On first login, you'll be prompted to change the password. You can skip this for development.

### 2. Import Dashboard

**Option A: Via UI Upload**
1. Click on **Dashboards** (left sidebar, four squares icon)
2. Click **New** → **Import**
3. Click **Upload JSON file**
4. Select: `monitoring/grafana-dashboard.json`
5. Dashboard will load automatically (datasource is pre-configured)
6. Click **Import**

**Option B: Via Direct JSON**
1. Click on **Dashboards** → **New** → **Import**
2. Copy the entire contents of `monitoring/grafana-dashboard.json`
3. Paste into the **Import via panel json** textbox
4. Click **Load**
5. Click **Import**

### 3. Verify Dashboard

After import, you should see:
- **Panel 1 (Top Left)**: API Request Rate
- **Panel 2 (Top Right)**: Request Latency (p50, p95, p99)
- **Panel 3 (Middle Left)**: Error Rate
- **Panel 4 (Middle Center)**: Cache Hit Rate (Gauge)
- **Panel 5 (Middle Right)**: Predictions by Sentiment (Pie Chart)
- **Panel 6 (Bottom Full Width)**: A/B Test Traffic Split

### 4. Generate Traffic to See Data

If panels show "No data", generate some API traffic:

```bash
# Generate regular predictions
for i in {1..20}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Test message '"$i"'"}' &
done

# Generate A/B test predictions
for i in {1..20}; do
  curl -X POST http://localhost:8000/predict/ab \
    -H "Content-Type: application/json" \
    -d '{"text": "A/B test '"$i"'"}' &
done

wait
```

Within 5-15 seconds, data should appear in all panels.

### 5. Dashboard Features

- **Auto-refresh**: 5 seconds (top-right)
- **Time range**: Last 15 minutes (adjustable)
- **Theme**: Dark mode (change in settings)
- **UID**: `mlapi-dashboard` (for API access)

### 6. Customize (Optional)

To customize the dashboard:
1. Click the gear icon (⚙️) in top-right → **Settings**
2. Edit panels by clicking panel title → **Edit**
3. After changes, click **Save dashboard** icon (💾) in top-right
4. To export: **Share** → **Export** → **Save to file**
5. Replace `monitoring/grafana-dashboard.json` with exported file

## Troubleshooting

### "No Data" in Panels

**Check Prometheus connection:**
1. Go to **Connections** → **Data sources**
2. Click **Prometheus**
3. Scroll down and click **Save & test**
4. Should show: "Successfully queried the Prometheus API"

**Check Prometheus targets:**
1. Open Prometheus: http://localhost:9090
2. Go to **Status** → **Targets**
3. Verify `mlapi` target is **UP**
4. If DOWN, ensure API is running: `uvicorn api.main:app --reload`

**Check metrics exist:**
1. In Prometheus, go to **Graph**
2. Query: `api_requests_total`
3. Click **Execute**
4. Should show metric data

### Datasource Not Found

If you see "Datasource not found" errors:
1. The datasource UID might be different
2. Go to **Connections** → **Data sources** → **Prometheus**
3. Check the **UID** field (should be "prometheus")
4. If different, edit `grafana-dashboard.json`:
   - Find all `"uid": "prometheus"` instances
   - Replace with actual UID
5. Re-import dashboard

### Panels Show Wrong Time Range

1. Click time picker (top-right, e.g., "Last 15 minutes")
2. Select desired range
3. Or use **Quick ranges** (Last 5m, Last 15m, etc.)

### Metrics Changed Names

If metric names changed in your API:
1. Edit panel (click title → **Edit**)
2. Update **Query** field with correct metric name
3. Click **Apply**
4. Save dashboard

## Prometheus Query Examples

Test queries directly in Prometheus (http://localhost:9090):

```promql
# Total requests per second
rate(api_requests_total[1m])

# Requests by endpoint
sum by (endpoint) (rate(api_requests_total[1m]))

# 95th percentile latency
histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m])) * 1000

# Cache hit rate
cache_hit_rate

# Error rate
rate(api_requests_total{status_code=~"5.."}[5m])

# A/B traffic distribution
rate(api_requests_total{endpoint=~"/predict/ab.*"}[1m])
```

## Accessing Dashboard via API

Dashboard JSON can be fetched via Grafana API:

```bash
# Get dashboard by UID
curl -u admin:admin \
  http://localhost:3000/api/dashboards/uid/mlapi-dashboard

# Export dashboard
curl -u admin:admin \
  http://localhost:3000/api/dashboards/uid/mlapi-dashboard \
  | jq '.dashboard' > dashboard-export.json
```

## Next Steps

- **Alerts**: Add alert rules for high error rates, slow latency
- **Variables**: Add dashboard variables for filtering by endpoint
- **Annotations**: Mark deployments on timeline
- **More panels**: Add memory usage, CPU usage (requires node exporter)

## Useful Links

- Grafana Docs: https://grafana.com/docs/
- PromQL Docs: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Dashboard Examples: https://grafana.com/grafana/dashboards/

