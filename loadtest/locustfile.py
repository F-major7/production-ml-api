"""
Locust load testing scenarios for Production ML API
"""
from locust import HttpUser, task, between, events
import random
import json
import time

# Sample texts for testing (mix of repeated and unique)
REPEATED_TEXTS = [
    "This product is amazing!",
    "Terrible service, very disappointed",
    "It's okay, nothing special",
    "Absolutely love it!",
    "Worst experience ever",
    "Pretty good overall",
    "Not bad for the price",
    "Excellent quality",
    "Could be better",
    "Fantastic purchase!"
]

SENTIMENT_WORDS = {
    "positive": ["amazing", "excellent", "fantastic", "wonderful", "great", "love", "perfect"],
    "negative": ["terrible", "horrible", "awful", "worst", "disappointed", "hate", "bad"],
    "neutral": ["okay", "fine", "adequate", "acceptable", "decent", "moderate"]
}


class MLAPIUser(HttpUser):
    """Simulates a user interacting with the ML API"""
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts"""
        self.request_count = 0
    
    @task(7)
    def predict_endpoint(self):
        """Test /predict endpoint (70% of traffic)"""
        # 50% repeated (cache hits), 50% unique (cache misses)
        if random.random() < 0.5:
            text = random.choice(REPEATED_TEXTS)
        else:
            # Generate unique text
            sentiment_type = random.choice(list(SENTIMENT_WORDS.keys()))
            word = random.choice(SENTIMENT_WORDS[sentiment_type])
            text = f"Test {sentiment_type} {word} {time.time()}"
        
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "sentiment" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Missing required fields")
            elif response.status_code == 429:
                # Rate limited - expected under load
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(2)
    def predict_ab_endpoint(self):
        """Test /predict/ab endpoint (20% of traffic)"""
        text = random.choice(REPEATED_TEXTS)
        
        with self.client.post(
            "/predict/ab",
            json={"text": text},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "model_version" in data and data["model_version"] in ["v1", "v2"]:
                    response.success()
                else:
                    response.failure("Invalid A/B response")
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def batch_endpoint(self):
        """Test /batch endpoint (10% of traffic)"""
        # Send 3 texts in a batch
        texts = random.sample(REPEATED_TEXTS, 3)
        
        with self.client.post(
            "/batch",
            json={"texts": texts},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("total") == 3 and len(data.get("predictions", [])) == 3:
                    response.success()
                else:
                    response.failure("Invalid batch response")
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def analytics_summary(self):
        """Test /analytics/summary endpoint (read-only)"""
        with self.client.get(
            "/analytics/summary",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "total_predictions" in data:
                    response.success()
                else:
                    response.failure("Invalid analytics response")
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test /health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("System unhealthy")
            else:
                response.failure(f"Health check failed: {response.status_code}")


class CacheTestUser(HttpUser):
    """Specialized user for testing cache effectiveness"""
    
    wait_time = between(0.5, 1.5)
    
    @task
    def repeated_requests(self):
        """Make repeated requests to measure cache performance"""
        # Always use the same texts to maximize cache hits
        text = random.choice(REPEATED_TEXTS[:5])  # Only use first 5 for high hit rate
        
        self.client.post("/predict", json={"text": text})


# Track custom metrics
cache_hit_count = 0
cache_miss_count = 0
response_times = []


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Track custom metrics from responses"""
    global cache_hit_count, cache_miss_count, response_times
    
    if exception is None and request_type == "POST" and "/predict" in name:
        response_times.append(response_time)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics when test completes"""
    print("\n" + "="*50)
    print("LOAD TEST SUMMARY")
    print("="*50)
    
    if response_times:
        response_times.sort()
        p50 = response_times[len(response_times)//2]
        p95 = response_times[int(len(response_times)*0.95)]
        p99 = response_times[int(len(response_times)*0.99)]
        
        print(f"Response times (ms):")
        print(f"  p50: {p50:.2f}")
        print(f"  p95: {p95:.2f}")
        print(f"  p99: {p99:.2f}")
        print(f"  min: {min(response_times):.2f}")
        print(f"  max: {max(response_times):.2f}")
    
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average RPS: {stats.total.total_rps:.2f}")
    print(f"Failure rate: {(stats.total.num_failures / max(stats.total.num_requests, 1) * 100):.2f}%")
    print("="*50 + "\n")

